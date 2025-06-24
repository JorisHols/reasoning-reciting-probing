#!/bin/bash
# Usage: ./run_arithmetic_chunks_job.sh <output_base> [<chunk_size>] [<base>]
#
# This script handles both setup and execution:
# 1. When run initially, it sets up directories and submits the SLURM array job.
# 2. When run as a SLURM array task, it processes only its assigned chunk.
#
# Arguments:
#   [<chunk_size>] Number of lines per chunk (default: 500)
#   [<base>]       Base for arithmetic operations (default: 10)

set -e

# SLURM job parameters
PARTITION="gpu_a100"
GPUS_PER_TASK=1
CPUS_PER_TASK=8
TIME="2:00:00"

# Function to set up the job and submit the array
setup_and_submit() {
    local chunk_size="$1"
    local base="$2"
    
    # Set the input path to the fixed location
    local input_path="./inputs/arithmetic/data/base${base}.txt"
    local output_base="./outputs/arithmetic/probe/llama3_base"
    
    # Check if the input file exists
    if [[ ! -f "$input_path" ]]; then
        echo "ERROR: Input file not found: $input_path"
        exit 1
    fi
    
    # Count total lines in the input file
    local total_lines=$(wc -l < "$input_path")
    
    local num_chunks=$(( (total_lines + chunk_size - 1) / chunk_size ))
    echo "Total examples: $total_lines, Chunk size: $chunk_size, Number of chunks: $num_chunks"

    # Create a timestamp-based temporary directory
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local temp_dir="${output_base}/temp_${timestamp}_base${base}"
    echo "Creating temporary directory: $temp_dir"
    mkdir -p "$temp_dir/logs"  # Create the main logs directory

    # Create all chunk directories in the temporary directory
    for chunk_id in $(seq 0 $((num_chunks-1))); do
        mkdir -p "$temp_dir/chunk_${chunk_id}/logs"
        mkdir -p "$temp_dir/chunk_${chunk_id}/data"
        echo "Chunk $chunk_id will be processed." > "$temp_dir/chunk_${chunk_id}/logs/chunk_info.txt"
    done

    # Save job info (will update with real job ID later)
    cat > "$temp_dir/job_info.txt" << EOF
Temporary directory: $temp_dir
Input file: $input_path
Chunk size: $chunk_size
Number of chunks: $num_chunks
Base: $base
Setup completed: $(date)
EOF

    # Submit the array job
    echo "Submitting SLURM array job with $num_chunks tasks..."
    local script_path="$(realpath "$0")"
    local job_id=$(sbatch --parsable \
        --job-name=arithmetic_chunks \
        --partition=$PARTITION \
        --gpus=$GPUS_PER_TASK \
        --cpus-per-task=$CPUS_PER_TASK \
        --time=$TIME \
        --output="$temp_dir/chunk_%a/logs/slurm_%A_%a.out" \
        --error="$temp_dir/chunk_%a/logs/slurm_%A_%a.err" \
        --array=0-$((num_chunks-1)) \
        --export=ALL,MAIN_DIR="$temp_dir",INPUT_PATH="$input_path",CHUNK_SIZE="$chunk_size",BASE="$base" \
        "$script_path")
        
    echo "Job submitted with ID: $job_id"
    
    # Update the job info file with the real job ID
    cat > "$temp_dir/logs/job_info.txt" << EOF
Job ID: $job_id
Input file: $input_path
Chunk size: $chunk_size
Number of chunks: $num_chunks
Base: $base
Setup completed: $(date)
Directory: $temp_dir
EOF

    echo "Job submitted. Output will be in $temp_dir"
    echo "Monitor with: squeue -u $USER"
}

# Function to process a single chunk (run by array task)
process_chunk() {
    local chunk_id=$SLURM_ARRAY_TASK_ID
    local logs_dir="$MAIN_DIR/chunk_${chunk_id}/logs"
    local data_dir="$MAIN_DIR/chunk_${chunk_id}/data"
    
    echo "Processing chunk $chunk_id..."
    echo "Started at: $(date)" >> "$logs_dir/chunk_info.txt"
    
    bash ./scripts/arithmetic/probe/run_arithmetic_experiment.sh \
        --input_path "$INPUT_PATH" \
        --output_path "$data_dir" \
        --chunk_id $chunk_id \
        --chunk_size $CHUNK_SIZE \
        --base $BASE \
        2>&1 | tee -a "$logs_dir/run_log.txt"
    
    echo "Finished at: $(date)" >> "$logs_dir/chunk_info.txt"
}

# Main execution logic
if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    # Running as an array task - process the assigned chunk
    process_chunk
else
    # Initial run - set up and submit the job
    # Default values
    CHUNK_SIZE="${1:-500}"
    BASE="${2:-10}"
    setup_and_submit "$CHUNK_SIZE" "$BASE"
fi 