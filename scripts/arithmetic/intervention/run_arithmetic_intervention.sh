#!/bin/bash
# This script runs the arithmetic intervention experiment on a compute node

# Load required modules
module purge
module load 2024

# Default values
CHUNK_ID=0
CHUNK_SIZE=100
BASE=10
ALPHA=0.1
INTERVENTION_PATH="./inputs/chess/interventions/liref_reasoning_directions.json"
INPUT_PATH="./inputs/arithmetic/data/base10.txt"
OUTPUT_DIR="./outputs/arithmetic/intervention/"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --chunk_id)
      CHUNK_ID="$2"
      shift 2
      ;;
    --chunk_size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    --input_path)
      INPUT_PATH="$2"
      shift 2
      ;;
    --output_path)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --base)
      BASE="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
      shift 2
      ;;
    --intervention_path)
      INTERVENTION_PATH="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$INPUT_PATH" ]; then
    echo "ERROR: Input path must be specified with --input_path"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory must be specified with --output_path"
    exit 1
fi

# Throw an error if chunk_id is not passed
if [ -z "$CHUNK_ID" ]; then
    echo "ERROR: Chunk ID must be specified with --chunk_id"
    exit 1
fi

# Set working directory to current directory
WORK_DIR=$(pwd)
echo "Working in: $WORK_DIR"

# Install Poetry if needed
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    echo "Poetry installed at: $(which poetry)"
fi

# Configure Poetry
poetry config virtualenvs.in-project true --local

# Install dependencies with --no-root flag to avoid package installation issue
echo "Installing project dependencies..."
poetry install --no-interaction --no-root

# Get and activate virtual environment
VENV_PATH=$(poetry env info --path)
if [ -z "$VENV_PATH" ]; then
    echo "ERROR: Poetry virtual environment not found"
    exit 1
fi

source "${VENV_PATH}/bin/activate"
echo "Using Python: $(which python) version $(python --version)"

# Load HF_TOKEN from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v "^#" .env | xargs)
else
    echo "ERROR: .env file not found"
    exit 1
fi

# Print the output directory
echo "Output directory: $OUTPUT_DIR"

# Run the main script - output will appear in real-time
echo "Starting arithmetic intervention experiment..."

echo "Chunk ID: $CHUNK_ID"
echo "Base: $BASE"
echo "Alpha: $ALPHA"
echo "Intervention vector path: $INTERVENTION_PATH"

# Build command
CMD="python code/main.py \
    --experiment arithmetic \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_DIR \
    --batch_size 8 \
    --max_new_tokens 2048 \
    --chunk_size $CHUNK_SIZE \
    --chunk_id $CHUNK_ID \
    --experiment_type intervention \
    --base $BASE \
    --alpha $ALPHA \
    --intervention_vector_path $INTERVENTION_PATH"

# Execute the command
$CMD 2>&1 

echo "Job complete. Results saved to $OUTPUT_DIR" 