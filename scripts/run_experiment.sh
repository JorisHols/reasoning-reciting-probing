#!/bin/bash
# This script runs the experiment on a compute node
# It is called by run_gpu_job_live.sh

# Load required modules
module purge
module load 2024

# Default values
CHUNK_ID=""
CHUNK_SIZE=500

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

# # Ensure PyYAML is installed (in case poetry install missed it)
# pip install pyyaml

# # Check for yaml module
# python -c "import yaml; print(f\"PyYAML version: {yaml.__version__}\")"

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

# # Save run information to output directory
# cat > "$OUTPUT_DIR/run_info.txt" << EOL
# Job ID: ${SLURM_JOB_ID:-"local"}
# Run timestamp: $(date)
# Input path: $INPUT_PATH
# Chunk ID: ${CHUNK_ID:-"None"}
# Chunk size: $CHUNK_SIZE
# Python version: $(python --version 2>&1)
# Node: $(hostname)
# CPU cores: ${SLURM_CPUS_PER_TASK:-"N/A"}
# GPUs: ${SLURM_GPUS:-"N/A"}
# EOL

# Run the main script - output will appear in real-time
echo "Starting experiment..."

echo "Chunk ID: $CHUNK_ID"

# Build command with conditional chunk_id
# Setting checkpoint to 100 so checkpointing is actually not used. And each chunk doesn't also contain checkpoints
CMD="python code/main.py \
    --experiment chess \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_DIR \
    --batch_size 8 \
    --max_new_tokens 2048 \
    --chunk_size $CHUNK_SIZE \
    --chunk_id $CHUNK_ID \
    --invervention_vector_path ./inputs/chess/interventions/world_detection_coefficients.json \
    --experiment_type intervention"

# Add chunk_id only if it's provided
if [ ! -z "$CHUNK_ID" ]; then
    CMD="$CMD --chunk_id $CHUNK_ID"
fi

# Execute the command
$CMD 2>&1 | tee "$OUTPUT_DIR/run_log.txt"

# # Copy script that was used for this run for reference
# mkdir -p "$OUTPUT_DIR/scripts_used"
# cp "$(dirname "$0")/run_experiment.sh" "$OUTPUT_DIR/scripts_used/"
# cp "$(dirname "$0")/create_output_dir.sh" "$OUTPUT_DIR/scripts_used/"

echo "Job complete. Results saved to $OUTPUT_DIR" 