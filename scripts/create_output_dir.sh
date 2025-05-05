#!/bin/bash
# This script creates a unique output directory for each run
# based on job ID, experiment name, and timestamp

create_output_dir() {
    # Base output directory
    BASE_OUTPUT_DIR="./outputs"
    
    # Get current date and time in format YYYYMMDD_HHMMSS
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    
    # Get experiment name (default to "experiment" if not provided)
    EXPERIMENT_NAME=${1:-"experiment"}
    
    # Get job ID from SLURM if available, otherwise use "local"
    if [ -n "$SLURM_JOB_ID" ]; then
        JOB_ID=$SLURM_JOB_ID
    else
        JOB_ID="local"
    fi
    
    # Create the unique output directory path
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}/job_${JOB_ID}_${TIMESTAMP}"
    
    # Create the directory
    mkdir -p "$OUTPUT_DIR"
    
    # Create a symlink to the latest run for convenience
    LATEST_LINK="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}/latest"
    
    # Remove existing symlink if it exists
    if [ -L "$LATEST_LINK" ]; then
        rm "$LATEST_LINK"
    fi
    
    # Create new symlink
    ln -sf "$OUTPUT_DIR" "$LATEST_LINK"
    
    # Print the directory path
    echo "Created output directory: $OUTPUT_DIR"
    echo "Created symbolic link to latest run: $LATEST_LINK"
    
    # Return the directory path
    echo "$OUTPUT_DIR"
}

# If this script is sourced, the function will be available
# If this script is run directly, create the directory with provided args
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    create_output_dir "$@"
fi 