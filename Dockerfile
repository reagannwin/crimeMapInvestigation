# Linux runtime for the pipeline.
#
# Why this exists: on the development machine, Windows Smart App Control
# blocks the unsigned compiled extensions in pyproj, pyogrio, scikit-learn,
# libpysal, esda and spreg. Linux binaries inside a container are not subject
# to that policy, so the pipeline runs here unchanged without weakening a
# system security setting.
#
# The image holds dependencies only. Code and data are bind-mounted at run
# time (see scripts/docker-run.sh), so edits need no rebuild, and secrets such
# as CENSUS_API_KEY are passed as environment variables, never baked in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    CRIME_SOURCE_DIR=/data/source

WORKDIR /app

# A real typeface for the figures; slim images ship only DejaVu, which has no
# semibold and makes every chart render differently from the Windows build.
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-inter \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as an unprivileged user; bind-mounted output stays writable because
# Docker Desktop maps ownership on Windows mounts.
RUN useradd --create-home analyst
USER analyst

CMD ["python", "-m", "pytest", "tests/", "-q"]
