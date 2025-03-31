#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:00:15
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Setup the virtual environment with poetry
export PATH="$HOME/.local/bin:$PATH"
cd $HOME/reasoning-reciting-probing
poetry install --no-root
poetry env activate
echo "Activated poetry environment"

# Base directory for the experiment
mkdir $HOME/experiments
cd $HOME/experiments

# Simple trick to create a unique directory for each run of the script
echo $$
mkdir o`echo $$`
cd o`echo $$`

# Run the actual experiment. 
python /home/jholshuijsen/test_job.py
