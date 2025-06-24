#!/bin/bash
# run_all_bases_alphas.sh
# This script runs intervention experiments with different bases and alpha values

set -e

# Default settings - can be overridden with command line parameters
INPUT_BASE_DIR="${1:-./inputs/arithmetic/data}"
OUTPUT_BASE_DIR="${2:-./outputs/arithmetic/intervention}"
CHUNK_SIZE="${3:-500}"
INTERVENTION_PATH="${4:-./inputs/chess/interventions/liref_reasoning_directions.json}"

# List of bases to test
BASES=(8 9 10 11 16)

# List of alpha values to test
ALPHAS=(-0.15 -0.10 -0.05 0.0 0.05 0.10 0.15)

echo "Starting intervention experiments for all bases and alphas"
echo "Input base directory: $INPUT_BASE_DIR"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo "Intervention vector: $INTERVENTION_PATH"

for BASE in "${BASES[@]}"; do
    echo "Processing base $BASE"
    INPUT_PATH="${INPUT_BASE_DIR}/base${BASE}.txt"
    
    # Verify input file exists
    if [[ ! -f "$INPUT_PATH" ]]; then
        echo "WARNING: Input file not found: $INPUT_PATH. Skipping base $BASE."
        continue
    fi
    
    for ALPHA in "${ALPHAS[@]}"; do
        echo "Scheduling job with base=$BASE, alpha=$ALPHA"
        ./scripts/arithmetic/intervention/run_arithmetic_intervention_chunks_job.sh \
            "$INPUT_PATH" \
            "$OUTPUT_BASE_DIR" \
            "$CHUNK_SIZE" \
            "$BASE" \
            "$ALPHA" \
            "$INTERVENTION_PATH"
        
        # Add a small delay between job submissions
        sleep 2
    done
    
    echo "All alphas scheduled for base $BASE"
    echo "----------------------------------------"
done

echo "All jobs scheduled. Check status with 'squeue -u $USER'" 