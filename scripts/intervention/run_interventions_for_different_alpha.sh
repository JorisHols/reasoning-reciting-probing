#!/bin/bash
# schedule_intervention_chunks_jobs.sh

# Set these variables as needed
INPUT_PATH="./inputs/chess/data/"
OUTPUT_BASE="outputs/chess/interventions/liref"
CHUNK_SIZE=80
INTERVENTION_VECTOR_FILENAME="liref_reasoning_directions.json"

# List of alpha values
ALPHAS=(-0.15 -0.20 -0.30)

for ALPHA in "${ALPHAS[@]}"; do
    echo "Scheduling job with alpha=$ALPHA"
    ./scripts/intervention/run_intervention_chunks_job.sh "$INPUT_PATH" "$OUTPUT_BASE" "$CHUNK_SIZE" "$ALPHA" "$INTERVENTION_VECTOR_FILENAME"
done