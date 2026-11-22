#!/usr/bin/env bash
# Run the same container locally against a local NVIDIA GPU.
# This is the control run: whatever number a rented node reports has to be
# comparable to this one, or the comparison is meaningless.
#
#   scripts/run_local.sh examples/ out/
set -euo pipefail

IN_DIR="${1:-examples}"
OUT_DIR="${2:-out}"
IMAGE="${I2V_IMAGE:-nosana-i2v:local}"
CACHE_DIR="${I2V_CACHE:-$HOME/.cache/nosana-i2v}"

if [ ! -d "$IN_DIR" ]; then
  echo "input directory not found: $IN_DIR" >&2
  exit 1
fi
mkdir -p "$OUT_DIR" "$CACHE_DIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE"
  docker build -t "$IMAGE" .
fi

# --gpus all is required; without it torch falls back to CPU and a 5 s clip
# takes hours instead of minutes, which looks like a hang rather than an error.
exec docker run --rm --gpus all \
  -v "$(realpath "$IN_DIR")":/work/in:ro \
  -v "$(realpath "$OUT_DIR")":/work/out \
  -v "$(realpath "$CACHE_DIR")":/work/cache \
  -e I2V_STEPS="${I2V_STEPS:-30}" \
  -e I2V_FRAMES="${I2V_FRAMES:-81}" \
  -e I2V_GUIDANCE="${I2V_GUIDANCE:-3.5}" \
  -e I2V_TIMEOUT="${I2V_TIMEOUT:-2700}" \
  "$IMAGE"
