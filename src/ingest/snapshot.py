"""Freeze the source CSVs into data/raw with checksums.

    python -m src.ingest.snapshot            copy anything not yet snapshotted
    python -m src.ingest.snapshot --verify   check data/raw against the manifest
    python -m src.ingest.snapshot --refresh  replace the snapshot on purpose

Why a snapshot
--------------
The city refreshes these extracts daily. A re-download is a different dataset,
so every number in the README and figures is only reproducible against the
exact files the analysis ran on. This step copies them into the project (which
OneDrive syncs; they are too large for git) and records a SHA-256 for each in
``data/raw/MANIFEST.json``. The manifest is small and committed, so the repo
always says which data produced its results.

An existing snapshot is never overwritten implicitly. If the file in the
source folder has changed, the step reports the difference and keeps the
frozen copy; ``--refresh`` replaces it and logs the change in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src import config

CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class Entry:
    key: str
    filename: str
    bytes: int
    sha256: str
    snapshotted_at: str
    source_modified_at: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def load_manifest(path: Path = config.SNAPSHOT_MANIFEST) -> dict:
    if not path.exists():
        return {"files": {}, "history": []}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(manifest: dict, path: Path = config.SNAPSHOT_MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)  # atomic: a crash never leaves a half-written manifest


def _copy_verified(src: Path, dst: Path) -> str:
    """Copy via a temp file and confirm the copy matches byte-for-byte."""
    expected = sha256(src)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    shutil.copy2(src, tmp)
    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        raise OSError(f"checksum mismatch copying {src.name}; copy discarded")
    tmp.replace(dst)
    return actual


def snapshot(
    *,
    source_dir: Path = config.SOURCE_DIR,
    raw_dir: Path = config.DATA_RAW,
    manifest_path: Path = config.SNAPSHOT_MANIFEST,
    refresh: bool = False,
) -> list[str]:
    """Copy missing sources into ``raw_dir``. Returns a line per file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = []

    for key, name in config.SOURCE_FILES.items():
        src, dst = source_dir / name, raw_dir / name
        recorded = manifest["files"].get(key)

        if dst.exists() and not refresh:
            if src.exists() and recorded and sha256(src) != recorded["sha256"]:
                report.append(f"kept      {name}  (source has changed since the "
                              "snapshot; use --refresh to replace)")
            else:
                report.append(f"present   {name}")
            if not recorded:
                # A file placed by hand: record it so --verify can check it.
                stat = dst.stat()
                manifest["files"][key] = asdict(Entry(
                    key, name, stat.st_size, sha256(dst), now, _utc(stat.st_mtime),
                ))
            continue

        if not src.exists():
            report.append(f"MISSING   {name}  (not in {source_dir})")
            continue

        digest = _copy_verified(src, dst)
        entry = Entry(key, name, dst.stat().st_size, digest, now,
                      _utc(src.stat().st_mtime))
        if recorded and recorded["sha256"] != digest:
            manifest["history"].append({
                "key": key, "replaced_at": now,
                "old_sha256": recorded["sha256"], "new_sha256": digest,
            })
        manifest["files"][key] = asdict(entry)
        report.append(f"copied    {name}  ({entry.bytes / 1e6:.1f} MB)")

    write_manifest(manifest, manifest_path)
    return report


def verify(
    *,
    raw_dir: Path = config.DATA_RAW,
    manifest_path: Path = config.SNAPSHOT_MANIFEST,
) -> list[str]:
    """Return a list of problems; empty means the snapshot is intact."""
    manifest = load_manifest(manifest_path)
    problems = []
    for key, name in config.SOURCE_FILES.items():
        recorded = manifest["files"].get(key)
        path = raw_dir / name
        if recorded is None:
            problems.append(f"{name}: not in manifest")
        elif not path.exists():
            problems.append(f"{name}: missing from {raw_dir}")
        elif sha256(path) != recorded["sha256"]:
            problems.append(f"{name}: checksum differs from manifest")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if args.verify:
        problems = verify()
        for p in problems:
            print(f"   {p}")
        print("snapshot intact" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0

    print(f">> snapshot {config.SOURCE_DIR} -> {config.DATA_RAW}")
    lines = snapshot(refresh=args.refresh)
    for line in lines:
        print(f"   {line}")
    missing = [l for l in lines if l.startswith("MISSING")]
    problems = verify()
    print("snapshot intact" if not problems else f"{len(problems)} problem(s)")
    return 1 if (missing or problems) else 0


if __name__ == "__main__":
    sys.exit(main())
