# Experiment Scripts Guide

This directory contains scripts for running LLaMA probing experiments on Snellius.

## Scripts Overview

- `run_experiment.sh`: The core experiment script that handles all experiment logic
- `run_gpu_job.sh`: Submits a batch job to the GPU queue using SLURM (uses run_experiment.sh)
- `run_gpu_job_live.sh`: Starts an interactive GPU session with live output (uses run_experiment.sh) 
- `activate_venv.sh`: Utility to set up and activate the Poetry virtual environment
- `create_output_dir.sh`: Utility to create unique output directories for each run

## Improved Script Architecture

The scripts now follow a "single source of truth" pattern:
1. `run_experiment.sh` contains all core logic for running experiments
2. Both batch jobs (`run_gpu_job.sh`) and interactive sessions (`run_gpu_job_live.sh`) use this same script
3. This ensures consistent behavior between batch and interactive runs

## Output Directory Structure

Experiments save results in uniquely named directories with the pattern:
```
outputs/
└── experiment_name/
    ├── job_JOBID_TIMESTAMP/   # Results from a specific run
    │   ├── clean/             # Final dataset for clean examples
    │   ├── clean_checkpoints/ # Checkpoints for clean examples
    │   ├── run_info.txt       # Run information (job ID, timestamp, configs)
    │   ├── run_log.txt        # Complete log of the run
    │   └── scripts_used/      # Copy of scripts used for this run
    └── latest -> job_JOBID_TIMESTAMP/  # Symlink to most recent run
```

## Unique Features

- **Single Source of Truth**: All experiment logic is in one script regardless of execution mode
- **Job-specific Output**: Each run creates a new directory named with job ID and timestamp
- **Run Information**: Each output directory includes details about the run configuration
- **Latest Symlink**: A `latest` symlink points to the most recent run for quick access
- **Script Preservation**: Copies of the scripts used are saved with results for reproducibility
- **Comprehensive Logging**: All output is captured to `run_log.txt`

## Running Experiments

### Batch Job (for long runs)
```bash
cd ~/reasoning-reciting-probing
sbatch scripts/run_gpu_job.sh
```

### Interactive Session (for development/debugging)
```bash
cd ~/reasoning-reciting-probing
bash scripts/run_gpu_job_live.sh
```

### Default Resource Allocation

- **Batch Job**: 4 GPUs, 32 CPUs, 128GB memory, 12-hour time limit
- **Interactive Session**: 3 GPUs, 16 CPUs, 4-hour time limit

### Accessing Results

The latest results are always available at:
```
./outputs/experiment_name/latest/
```

For example, for the K&K experiment:
```
./outputs/k_and_k/latest/
```

You can also access specific runs by their job ID and timestamp:
```
./outputs/k_and_k/job_12345_20230401_120000/
``` 