#!/bin/bash
# combine_all_alphas.sh

# List of alphas to process
ALPHAS=(-0.15 -0.20 -0.30)

# Base input/output paths
IN_BASE="/home/jholshuijsen/reasoning-reciting-probing/outputs/chess/interventions/liref"
OUT_BASE="/home/jholshuijsen/reasoning-reciting-probing/results/chess/intervention/liref"

for ALPHA in "${ALPHAS[@]}"; do
    # Find the folder matching the pattern for this alpha
    MATCHED_DIR=$(find "$IN_BASE" -maxdepth 1 -type d -name "temp_*_${ALPHA}_liref_reasoning_directions.json" | head -n 1)
    OUT_DIR="${OUT_BASE}/${ALPHA}"

    if [[ -z "$MATCHED_DIR" ]]; then
        echo "No folder found for alpha=${ALPHA}, skipping."
        continue
    fi

    echo "Combining chunks for alpha=${ALPHA} from $MATCHED_DIR"
    python /home/jholshuijsen/reasoning-reciting-probing/code/utilities/combine_chunks.py \
        --input_dir "$MATCHED_DIR" \
        --output_dir "$OUT_DIR"
done