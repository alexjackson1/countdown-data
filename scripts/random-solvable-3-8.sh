#!/bin/bash

set -euo pipefail

OUTPUT_DIR="data"
GENERATE_MODE="random-near"
FILENAME_PREFIX="random_solvable"

mkdir -p ${OUTPUT_DIR}

cargo run --release -- \
  --mode ${GENERATE_MODE} \
  --num-instances 1000000 \
  --min-size 3 \
  --max-size 8 \
  --outfile "${OUTPUT_DIR}/${FILENAME_PREFIX}_3_8_1m.jsonl.zst"

echo "Generation complete"