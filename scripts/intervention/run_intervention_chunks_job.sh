#!/bin/bash
# Usage: ./run_chunks_job.sh <input_path> <output_base> [<chunk_size>]
#
# This script handles both setup and execution:
# 1. When run initially, it sets up directories and submits the SLURM array job.
# 2. When run as a SLURM array task, it processes only its assigned chunk.
#
# Arguments:
#   <input_path>   Path to the input file or directory
#   <output_base>  Base directory for outputs
#   [<chunk_size>] Number of lines per chunk (default: 500)

set -e

# SLURM job parameters
PARTITION="gpu_a100"
GPUS_PER_TASK=1
CPUS_PER_TASK=8
TIME="2:00:00"

# Function to set up the job and submit the array
setup_and_submit() {
    local input_path="$1"
    local output_base="$2"
    local chunk_size="$3"
    local alpha="$4"
    local intervention_vector_filename="$5"

    # If INPUT_PATH is a directory, find the first .jsonl file inside
    if [[ -d "$input_path" ]]; then
        local jsonl_file=$(find "$input_path" -maxdepth 1 -type f -name '*.jsonl' | head -n 1)
        if [[ -z "$jsonl_file" ]]; then
            echo "ERROR: No .jsonl file found in directory: $input_path"
            exit 1
        fi
        echo "Using $jsonl_file as the input file."
        input_file_path="$jsonl_file"
    fi
    

    if [[ ! -f "$input_file_path" ]]; then
        echo "ERROR: Input folder not found: $input_path"
        exit 1
    fi

    local total_lines=$(wc -l < "$input_file_path")
    if [[ "$total_lines" -eq 0 ]]; then
        echo "ERROR: No lines found in $input_file_path"
        exit 1
    fi

    local num_chunks=$(( (total_lines + chunk_size - 1) / chunk_size ))
    echo "Total examples: $total_lines, Chunk size: $chunk_size, Number of chunks: $num_chunks"

    # Create a timestamp-based temporary directory
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local temp_dir="${output_base}/temp_${timestamp}_${alpha}_${intervention_vector_filename}"
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
Setup completed: $(date)
EOF

    # Submit the array job
    echo "Submitting SLURM array job with $num_chunks tasks..."
    local script_path="$(realpath "$0")"
    local job_id=$(sbatch --parsable \
        --job-name=llama_chunks \
        --partition=$PARTITION \
        --gpus=$GPUS_PER_TASK \
        --cpus-per-task=$CPUS_PER_TASK \
        --time=$TIME \
        --output="$temp_dir/chunk_%a/logs/slurm_%A_%a.out" \
        --error="$temp_dir/chunk_%a/logs/slurm_%A_%a.err" \
        --array=0-$((num_chunks-1)) \
        --export=ALL,MAIN_DIR="$temp_dir",INPUT_PATH="$input_path",CHUNK_SIZE="$chunk_size",ALPHA="$alpha",INTERVENTION_VECTOR_FILENAME="$intervention_vector_filename" \
        "$script_path")
        
    echo "Job submitted with ID: $job_id"
    
    # # Rename the temporary directory to include the job ID
    # local job_dir="${output_base}/${job_id}"
    # mv "$temp_dir" "$job_dir"
    # echo "Renamed directory: $temp_dir -> $job_dir"
    
    # Update the job info file with the real job ID
    cat > "$temp_dir/logs/job_info.txt" << EOF
Job ID: $job_id
Input file: $input_path
Chunk size: $chunk_size
Number of chunks: $num_chunks
Setup completed: $(date)
Directory: $temp_dir
Alpha: $alpha
Intervention vector filename: $intervention_vector_filename
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
    
    bash ./scripts/intervention/run_intervention_experiment.sh \
        --input_path "$INPUT_PATH" \
        --output_path "$data_dir" \
        --chunk_id $chunk_id \
        --chunk_size $CHUNK_SIZE \
        --intervention_vector_filename $INTERVENTION_VECTOR_FILENAME \
        --alpha $ALPHA \
        2>&1 | tee -a "$logs_dir/run_log.txt"
    
    echo "Finished at: $(date)" >> "$logs_dir/chunk_info.txt"
}

# Main execution logic
if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    # Running as an array task - process the assigned chunk
    process_chunk
else
    # Initial run - set up and submit the job
    if [[ -z "$1" || -z "$2" ]]; then
        echo "Usage: $0 <input_path> <output_base> [<chunk_size>]"
        exit 1
    fi
    
    INPUT_PATH="$1"
    OUTPUT_BASE="$2"
    CHUNK_SIZE="${3:-500}"
    ALPHA="${4:-0.05}"
    INTERVENTION_VECTOR_FILENAME="${5:-liref_reasoning_directions.json}"
    
    setup_and_submit "$INPUT_PATH" "$OUTPUT_BASE" "$CHUNK_SIZE" "$ALPHA" "$INTERVENTION_VECTOR_FILENAME"
fi