#!/bin/bash

# Interactive job script for live log output
# Run this script with: bash scripts/run_gpu_job_live.sh

echo "Requesting GPU resources for interactive session..."
echo "This might take some time depending on the queue."

# Request resources and run the job interactively
srun --job-name=llama_probe_live \
     --partition=gpu_a100 \
     --gpus=1 \
     --ntasks=1 \
     --cpus-per-task=16 \
     --time=4:00:00 \
     --pty \
     bash ./scripts/run_experiment.sh

echo "Interactive session completed." 