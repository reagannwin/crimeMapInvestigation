#!/usr/bin/env bash
# Run any pipeline command inside the Linux container.
#
#   scripts/docker-run.sh python -m src.ingest.build_census
#   scripts/docker-run.sh python -m pytest tests/ -q
#
# Mounts the project read-write at /app and the source CSV folder read-only at
# /data/source. CENSUS_API_KEY is forwarded from the calling shell if set; it
# is never written into the image or any file.
set -euo pipefail

IMAGE="crimemap:latest"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${CRIME_SOURCE_DIR_HOST:-$HOME/Downloads}"

# Docker Desktop wants C:/... host paths; Git Bash produces /c/.... Convert when
# cygpath exists (Git Bash), and leave paths alone on Linux or macOS.
if command -v cygpath >/dev/null 2>&1; then
  PROJECT_DIR="$(cygpath -m "$PROJECT_DIR")"
  SOURCE_DIR="$(cygpath -m "$SOURCE_DIR")"
fi

# Git Bash on Windows rewrites /app-style arguments into Windows paths; this
# stops it doing so for the container paths below.
export MSYS_NO_PATHCONV=1

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">> building $IMAGE"
  docker build -t "$IMAGE" "$PROJECT_DIR"
fi

# -i forwards stdin, so `scripts/docker-run.sh python - <<EOF` works.
exec docker run --rm -i \
  -v "$PROJECT_DIR:/app" \
  -v "$SOURCE_DIR:/data/source:ro" \
  ${CENSUS_API_KEY:+-e CENSUS_API_KEY} \
  "$IMAGE" "$@"
