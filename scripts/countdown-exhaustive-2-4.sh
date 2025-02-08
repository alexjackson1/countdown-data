#!/bin/bash

set -euo pipefail

OUTPUT_DIR="data"
GENERATE_MODE="countdown-exhaustive"
FILENAME_PREFIX="countdown_exhaustive"

mkdir -p ${OUTPUT_DIR}

for NUM_COUNT in {2..4}; do
  cargo run --release -- \
    --mode ${GENERATE_MODE} \
    --num-count ${NUM_COUNT} \
    --outfile "${OUTPUT_DIR}/${FILENAME_PREFIX}_${NUM_COUNT}.jsonl.zst"

  # Decompress the output file
  zstd -d "${OUTPUT_DIR}/${FILENAME_PREFIX}_${NUM_COUNT}.jsonl.zst"

  # Combine into single file
  cat "${OUTPUT_DIR}/${FILENAME_PREFIX}_${NUM_COUNT}.jsonl" >> "${OUTPUT_DIR}/${FILENAME_PREFIX}.jsonl"

  # Remove the decompressed file
  rm "${OUTPUT_DIR}/${FILENAME_PREFIX}_${NUM_COUNT}.jsonl"
done

# Compress the combined file
zstd "${OUTPUT_DIR}/${FILENAME_PREFIX}.jsonl"

# Remove the combined file
rm "${OUTPUT_DIR}/${FILENAME_PREFIX}.jsonl"

echo "Extraction complete"