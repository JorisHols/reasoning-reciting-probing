#!/bin/bash
# run_arithmetic_intervention_for_different_alphas.sh
# This script runs intervention experiments with different alpha values for a specific base

# Default values - can be overridden with command line parameters
INPUT_PATH="${1:-./inputs/arithmetic/data/base10.txt}"
OUTPUT_BASE="${2:-./outputs/arithmetic/intervention}"
CHUNK_SIZE="${3:-500}"
BASE="${4:-10}"
INTERVENTION_PATH="${5:-./inputs/chess/interventions/liref_reasoning_directions.json}"

# List of alpha values to test
ALPHAS=(-0.15 -0.10 -0.05 0.0 0.05 0.15)

echo "Starting intervention experiments for base $BASE with multiple alpha values"
echo "Input path: $INPUT_PATH"
echo "Output base: $OUTPUT_BASE"
echo "Intervention vector: $INTERVENTION_PATH"

for ALPHA in "${ALPHAS[@]}"; do
    echo "Scheduling job with base=$BASE, alpha=$ALPHA"
    ./scripts/arithmetic/intervention/run_arithmetic_intervention_chunks_job.sh \
        "$INPUT_PATH" \
        "$OUTPUT_BASE" \
        "$CHUNK_SIZE" \
        "$BASE" \
        "$ALPHA" \
        "$INTERVENTION_PATH"
    
    # Add a small delay between job submissions
    sleep 2
done

echo "All jobs scheduled. Check status with 'squeue -u $USER'" 