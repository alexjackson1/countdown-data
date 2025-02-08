#!/bin/bash

set -euo pipefail

OUTPUT_DIR="data"
GENERATE_MODE="exhaustive"
FILENAME_PREFIX="exhaustive"

mkdir -p ${OUTPUT_DIR}

cargo run --release -- \
  --mode ${GENERATE_MODE} \
  --num-count 2 \
  --outfile "${OUTPUT_DIR}/${FILENAME_PREFIX}_2.jsonl.zst"

echo "Generation complete"