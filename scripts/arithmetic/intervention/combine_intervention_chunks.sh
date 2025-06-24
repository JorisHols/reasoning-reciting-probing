#!/bin/bash
# combine_intervention_chunks.sh
# This script combines the results from chunked intervention experiments for a specific base and alpha

set -e

# Default settings
BASE=10
ALPHA=0.1
INPUT_BASE_DIR="./outputs/arithmetic/intervention"
OUTPUT_BASE_DIR="./results/arithmetic/intervention/liref"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --base)
      BASE="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
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
    exit 1
fi

if [[ -z "$ALPHA" ]]; then
    echo "ERROR: Alpha must be specified with --alpha"
    exit 1
fi

# Find the matching directory for the given base and alpha
ALPHA_PATTERN="${ALPHA//-/\\-}"  # Escape minus sign for regex matching
TARGET_DIR=$(find "$INPUT_BASE_DIR" -maxdepth 1 -type d -name "intervention_*_base${BASE}_alpha${ALPHA_PATTERN}" | head -n 1)

if [[ -z "$TARGET_DIR" ]]; then
    echo "ERROR: No matching directory found for base=${BASE}, alpha=${ALPHA}"
    echo "Available directories:"
    find "$INPUT_BASE_DIR" -maxdepth 1 -type d -name "intervention_*" | sort
    exit 1
fi

# Create output directory structure
OUTPUT_DIR="${OUTPUT_BASE_DIR}/base${BASE}/${ALPHA}"
mkdir -p "$OUTPUT_DIR"

echo "Found matching directory: $TARGET_DIR"
echo "Will save combined results to: $OUTPUT_DIR"

# Set working directory to project root
WORK_DIR=$(pwd)
echo "Working in: $WORK_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source ".venv/bin/activate"
    echo "Activated virtual environment: $(which python)"
else
    # Try to find Poetry environment
    VENV_PATH=$(poetry env info --path 2>/dev/null || echo "")
    if [ -n "$VENV_PATH" ]; then
        source "${VENV_PATH}/bin/activate"
        echo "Activated Poetry environment: $(which python)"
    else
        echo "WARNING: No virtual environment found. Using system Python."
    fi
fi

# Check for Python
if ! command -v python &>/dev/null; then
    echo "ERROR: Python not found"
    exit 1
fi

echo "Using Python: $(which python) version $(python --version)"

# Run the combine_chunks.py script
echo "Combining chunks from $TARGET_DIR to $OUTPUT_DIR..."
python code/utilities/combine_chunks.py --input_dir "$TARGET_DIR" --output_dir "$OUTPUT_DIR"

echo "Combination complete. Results saved to $OUTPUT_DIR"

# Create a summary file
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"
echo "Creating summary file: $SUMMARY_FILE"

cat > "$SUMMARY_FILE" << EOF
Combined results for arithmetic intervention experiment
Base: $BASE
Alpha: $ALPHA
Source directory: $TARGET_DIR
Date processed: $(date)
EOF

echo "Done!" 