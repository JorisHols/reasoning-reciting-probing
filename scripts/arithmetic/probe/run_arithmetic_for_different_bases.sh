#!/bin/bash
# run_arithmetic_for_different_bases.sh

# Set these variables as needed

CHUNK_SIZE=100

# List of bases to test
BASES=(1 2 3 4 5 6 7 9)

for BASE in "${BASES[@]}"; do
    echo "Scheduling job with base=$BASE"
    ./scripts/arithmetic/probe/run_arithmetic_chunks_job.sh "$CHUNK_SIZE" "$BASE"
done 