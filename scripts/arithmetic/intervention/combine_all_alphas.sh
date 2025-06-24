#!/bin/bash
# combine_all_alphas.sh
# This script combines results for all alphas of a specific base

set -e

# Default settings
BASE=10
INPUT_BASE_DIR="./outputs/arithmetic/intervention"
OUTPUT_BASE_DIR="./results/arithmetic/intervention/liref"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --base)
      BASE="$2"
      shift 2
      ;;
    --input_base_dir)
      INPUT_BASE_DIR="$2"
      shift 2
      ;;
    --output_base_dir)
      OUTPUT_BASE_DIR="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Verify arguments
if [[ -z "$BASE" ]]; then
    echo "ERROR: Base must be specified with --base"
    echo "Usage: $0 --base <base> [--input_base_dir <input_dir>] [--output_base_dir <output_dir>]"
    exit 1
fi

# Find all intervention directories for the specified base
INTERVENTION_DIRS=$(find "$INPUT_BASE_DIR" -maxdepth 1 -type d -name "intervention_*_base${BASE}_alpha*" | sort)

if [[ -z "$INTERVENTION_DIRS" ]]; then
    echo "ERROR: No intervention directories found for base=${BASE}"
    exit 1
fi

# Count the number of directories
NUM_DIRS=$(echo "$INTERVENTION_DIRS" | wc -l)
echo "Found $NUM_DIRS intervention directories for base=${BASE}"

# Process each directory
for DIR in $INTERVENTION_DIRS; do
    # Extract alpha value from directory name
    DIR_NAME=$(basename "$DIR")
    ALPHA_PART=$(echo "$DIR_NAME" | sed -n 's/.*_alpha\([-0-9.]*\).*/\1/p')
    
    if [[ -z "$ALPHA_PART" ]]; then
        echo "WARNING: Could not extract alpha value from directory: $DIR"
        continue
    fi
    
    echo "Processing directory: $DIR (alpha=${ALPHA_PART})"
    
    # Run the combine_intervention_chunks.sh script for this alpha
    ./scripts/arithmetic/intervention/combine_intervention_chunks.sh \
        --base "$BASE" \
        --alpha "${ALPHA_PART}" \
        --input_base_dir "$INPUT_BASE_DIR" \
        --output_base_dir "$OUTPUT_BASE_DIR"
    
    echo "Completed processing for alpha=${ALPHA_PART}"
    echo "----------------------------------------"
done

echo "All intervention directories for base=${BASE} have been processed"
echo "Results are saved in ${OUTPUT_BASE_DIR}/base${BASE}/" 