#!/usr/bin/env bash
# Fallback runner. `python3 -m pytest` works from the repo root (pytest.ini pins
# testpaths), so use this only if the bracketed path breaks collection again.
# It copies the code, config, courses and tests to a bracket-free temp dir.
# Code finds its config through Path(__file__), so the copy resolves correctly.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="$(mktemp -d)"
trap 'rm -rf "$DST"' EXIT

cp -R "$SRC/guided_practice" "$SRC/pedagogy_server" "$SRC/eval" "$SRC/scripts" "$SRC/config" "$SRC/courses" "$SRC/tests" "$SRC/samples" "$SRC/pytest.ini" "$DST/"
cd "$DST"
python3 -m pytest "$@"
