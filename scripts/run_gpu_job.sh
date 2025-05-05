#!/bin/bash
#SBATCH --job-name=llama_probe
#SBATCH --partition=gpu_a100
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=snellius_logs/slurm_%j.out
#SBATCH --error=snellius_logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.e.holshuijsen@student.vu.nl

# This script submits a batch job that executes run_experiment.sh
# This provides a consistent experiment execution between batch and interactive jobs

echo "Starting LLaMA probe batch job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated node(s): $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS"

# Execute the run_experiment.sh script
# This is the same script used by run_gpu_job_live.sh to ensure consistency
# Change to the project root directory first to ensure proper path resolution
bash "./scripts/run_experiment.sh"

echo "Batch job completed." 