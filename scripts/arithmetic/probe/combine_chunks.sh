#!/bin/bash
# combine_chunks.sh for arithmetic experiment

# Default values
EXPERIMENT_TYPE="probe"
BASES=(9)
TEMP_FOLDER=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --temp_folder)
      TEMP_FOLDER="$2"
      shift 2
      ;;
    --base)
      BASES=("$2")
      shift 2
      ;;
    --experiment_type)
      EXPERIMENT_TYPE="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Base input/output paths
IN_BASE="/home/jholshuijsen/reasoning-reciting-probing/outputs/arithmetic/${EXPERIMENT_TYPE}/llama3_base"
OUT_BASE="/home/jholshuijsen/reasoning-reciting-probing/results/arithmetic/${EXPERIMENT_TYPE}/llama3_base"

for BASE in "${BASES[@]}"; do
    # Use provided temp folder if specified, otherwise find it
    if [[ -z "$TEMP_FOLDER" ]]; then
        MATCHED_DIR=$(find "$IN_BASE" -maxdepth 1 -type d -name "temp_*_base${BASE}" | head -n 1)
    else
        MATCHED_DIR="${IN_BASE}/${TEMP_FOLDER}"
    fi
    
    OUT_DIR="${OUT_BASE}/base${BASE}"

    if [[ -z "$MATCHED_DIR" || ! -d "$MATCHED_DIR" ]]; then
        echo "No folder found for base=${BASE}, skipping."
        continue
    fi

    echo "Combining chunks for base=${BASE} from $MATCHED_DIR"
    echo "Output will be saved to $OUT_DIR"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUT_DIR"
    
    python /home/jholshuijsen/reasoning-reciting-probing/code/utilities/combine_chunks.py \
        --input_dir "$MATCHED_DIR" \
        --output_dir "$OUT_DIR"
done