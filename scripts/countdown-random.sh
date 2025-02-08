#!/bin/bash

set -euo pipefail

OUTPUT_DIR="data"
GENERATE_MODE="countdown-random"
FILENAME_PREFIX="countdown_random"

mkdir -p ${OUTPUT_DIR}

cargo run --release -- \
  --mode ${GENERATE_MODE} \
  --num-instances 1000000 \
  --outfile "${OUTPUT_DIR}/${FILENAME_PREFIX}_6_1m.jsonl.zst"

echo "Generation complete"