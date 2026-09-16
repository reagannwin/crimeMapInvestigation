"""Tests for the source-data snapshot.

The snapshot exists so published numbers stay tied to the exact files they
came from. The failure modes that matter are the silent ones: a changed
download quietly replacing the frozen copy, a corrupted copy passing as good,
and a loader falling back to Downloads when the snapshot is missing.
"""
from __future__ import annotations

import json

import pytest

from src import config
from src.ingest import snapshot as snap


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """A fake source folder holding every expected file, and an empty raw dir."""
    source, raw = tmp_path / "downloads", tmp_path / "raw"
    source.mkdir()
    for key, name in config.SOURCE_FILES.items():
        (source / name).write_text(f"{key},v1\n", encoding="utf-8")
    monkeypatch.setattr(config, "DATA_RAW", raw)
    return source, raw, raw / "MANIFEST.json"


def _run(dirs, **kw):
    source, raw, manifest = dirs
    return snap.snapshot(source_dir=source, raw_dir=raw, manifest_path=manifest, **kw)


def _verify(dirs):
    _, raw, manifest = dirs
    return snap.verify(raw_dir=raw, manifest_path=manifest)


class TestSnapshot:
    def test_copies_every_file_and_records_checksums(self, dirs):
        _run(dirs)
        _, raw, manifest = dirs
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert set(data["files"]) == set(config.SOURCE_FILES)
        for key, name in config.SOURCE_FILES.items():
            assert (raw / name).exists()
            assert data["files"][key]["sha256"] == snap.sha256(raw / name)
        assert _verify(dirs) == []

    def test_second_run_is_a_no_op(self, dirs):
        _run(dirs)
        lines = _run(dirs)
        assert all(line.startswith("present") for line in lines)

    def test_changed_download_does_not_replace_the_snapshot(self, dirs):
        """The whole point: a fresh download must not swap the data silently."""
        source, raw, _ = dirs
        _run(dirs)
        name = config.SOURCE_FILES["stops"]
        (source / name).write_text("stops,v2\n", encoding="utf-8")

        lines = _run(dirs)
        assert (raw / name).read_text(encoding="utf-8") == "stops,v1\n"
        assert any(line.startswith("kept") and name in line for line in lines)

    def test_refresh_replaces_and_logs_history(self, dirs):
        source, raw, manifest = dirs
        _run(dirs)
        name = config.SOURCE_FILES["stops"]
        (source / name).write_text("stops,v2\n", encoding="utf-8")

        _run(dirs, refresh=True)
        assert (raw / name).read_text(encoding="utf-8") == "stops,v2\n"
        history = json.loads(manifest.read_text(encoding="utf-8"))["history"]
        assert [h["key"] for h in history] == ["stops"]
        assert _verify(dirs) == []

    def test_missing_source_is_reported(self, dirs):
        source, _, _ = dirs
        (source / config.SOURCE_FILES["crime_stats"]).unlink()
        lines = _run(dirs)
        assert any(line.startswith("MISSING") for line in lines)

    def test_no_partial_files_left_behind(self, dirs):
        _run(dirs)
        _, raw, _ = dirs
        assert not list(raw.glob("*.partial"))


class TestVerify:
    def test_detects_a_modified_snapshot_file(self, dirs):
        _, raw, _ = dirs
        _run(dirs)
        (raw / config.SOURCE_FILES["calls_911"]).write_text("tampered", encoding="utf-8")
        problems = _verify(dirs)
        assert any("checksum differs" in p for p in problems)

    def test_detects_a_deleted_snapshot_file(self, dirs):
        _, raw, _ = dirs
        _run(dirs)
        (raw / config.SOURCE_FILES["calls_911"]).unlink()
        assert any("missing" in p for p in _verify(dirs))

    def test_empty_manifest_is_a_problem_not_a_pass(self, dirs):
        assert len(_verify(dirs)) == len(config.SOURCE_FILES)


class TestRawPath:
    def test_no_fallback_to_downloads(self, dirs):
        """A missing snapshot must fail loudly, not read a fresh download."""
        with pytest.raises(config.SourceNotSnapshotted, match="src.ingest.snapshot"):
            config.raw_path("stops")

    def test_resolves_inside_the_snapshot(self, dirs):
        _run(dirs)
        _, raw, _ = dirs
        assert config.raw_path("stops") == raw / config.SOURCE_FILES["stops"]
