#!/bin/bash
# combine_all_bases_alphas.sh
# This script combines results for all bases and alphas into the results folder

set -e

# Default settings
INPUT_BASE_DIR="${1:-./outputs/arithmetic/intervention}"
OUTPUT_BASE_DIR="${2:-./results/arithmetic/intervention/liref}"

# List of bases to process
BASES=(8 9 10 11 16)

echo "Starting combination of results for all bases and alphas"
echo "Input base directory: $INPUT_BASE_DIR"
echo "Output base directory: $OUTPUT_BASE_DIR"

# Process each base
for BASE in "${BASES[@]}"; do
    echo "Processing base $BASE"
    
    # Find all intervention directories for this base
    INTERVENTION_DIRS=$(find "$INPUT_BASE_DIR" -maxdepth 1 -type d -name "intervention_*_base${BASE}_alpha*" | sort)
    
    if [[ -z "$INTERVENTION_DIRS" ]]; then
        echo "No intervention directories found for base=${BASE}. Skipping."
        continue
    fi
    
    # Count the number of directories
    NUM_DIRS=$(echo "$INTERVENTION_DIRS" | wc -l)
    echo "Found $NUM_DIRS intervention directories for base=${BASE}"
    
    # Process each directory for this base
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
    done
    
    echo "All intervention directories for base=${BASE} have been processed"
    echo "----------------------------------------"
done

echo "All bases and alphas have been processed"
echo "Results are saved in ${OUTPUT_BASE_DIR}/"

# Create a summary of all processed results
SUMMARY_FILE="${OUTPUT_BASE_DIR}/all_results_summary.txt"
echo "Creating summary file: $SUMMARY_FILE"

cat > "$SUMMARY_FILE" << EOF
Combined results for all arithmetic intervention experiments
Date processed: $(date)
Bases processed: ${BASES[@]}
EOF

# Add information about each base and the alphas found
for BASE in "${BASES[@]}"; do
    echo -e "\nBase ${BASE}:" >> "$SUMMARY_FILE"
    # List all alpha subdirectories for this base
    if [[ -d "${OUTPUT_BASE_DIR}/base${BASE}" ]]; then
        ls -1 "${OUTPUT_BASE_DIR}/base${BASE}" | sort >> "$SUMMARY_FILE"
    else
        echo "  No results found" >> "$SUMMARY_FILE"
    fi
done

echo "Summary created at $SUMMARY_FILE"
echo "Done!" 