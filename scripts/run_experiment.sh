#!/bin/bash
# This script runs the experiment on a compute node
# It is called by run_gpu_job_live.sh

# Load required modules
module purge
module load 2024

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

# Define input path
INPUT_PATH="./inputs/k_and_k/3ppl"

# Create a unique output directory for this run
# Source the create_output_dir script to get the function
source "$(dirname "$0")/create_output_dir.sh"
# Call the function to create a unique output directory for k_and_k experiment
OUTPUT_DIR=$(create_output_dir "k_and_k" | tail -n 1)

# Print the output directory
echo "Output directory: $OUTPUT_DIR"

# First check if the clean data exists
CLEAN_PATH="$INPUT_PATH/clean/clean_people3_num5000.jsonl"
if [ ! -f "$CLEAN_PATH" ]; then
    echo "WARNING: Clean data file not found at $CLEAN_PATH"
    # Check if any JSONL files exist to verify structure
    JSONL_COUNT=$(find "$INPUT_PATH" -name "*.jsonl" | wc -l)
    echo "Found $JSONL_COUNT JSONL files in input directory"
fi

# Save run information to output directory
cat > "$OUTPUT_DIR/run_info.txt" << EOL
Job ID: ${SLURM_JOB_ID:-"local"}
Run timestamp: $(date)
Input path: $INPUT_PATH
Python version: $(python --version 2>&1)
Node: $(hostname)
CPU cores: ${SLURM_CPUS_PER_TASK:-"N/A"}
GPUs: ${SLURM_GPUS:-"N/A"}
EOL

# Run the main script - output will appear in real-time
echo "Starting experiment..."
python code/main.py \
    --experiment kk \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_DIR \
    --batch_size 32 \
    --max_new_tokens 2048 \
    --checkpoint_every 25 \
    2>&1 | tee "$OUTPUT_DIR/run_log.txt"

# Copy script that was used for this run for reference
mkdir -p "$OUTPUT_DIR/scripts_used"
cp "$(dirname "$0")/run_experiment.sh" "$OUTPUT_DIR/scripts_used/"
cp "$(dirname "$0")/create_output_dir.sh" "$OUTPUT_DIR/scripts_used/"

echo "Job complete. Results saved to $OUTPUT_DIR" 