import subprocess
import os
import time

# Path to your script
script_path = "/gpfs/home5/jholshuijsen/reasoning-reciting-probing/code/find_best_intervention_dir.py"

# Base Slurm parameters
slurm_params = {
    'partition': 'gpu_a100',
    'nodes': 1,
    'ntasks': 1,
    'cpus-per-task': 8,
    'gpus': 1,
    'mem': '32G',
    'time': '0:20:00'
}

# Function to submit a job
def tune_intervention_dir_job(alpha, layer):
    if EXPERIMENT == 'GSM-symbolic':
        output_dir = f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/gsm-symbolic/tuning/"
    elif EXPERIMENT == 'chess':
        output_dir = f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/chess/tuning/"

    os.makedirs(output_dir, exist_ok=True)
    if layer is None:
        run_dir_name = f"no_intervention/"
    else:
        run_dir_name = f"with_intervention/alpha_{alpha:.2f}/layer_{layer}"

    os.makedirs(f"{output_dir}/{run_dir_name}", exist_ok=True)
    log_dir = f"{output_dir}/{run_dir_name}/logs"
    os.makedirs(log_dir, exist_ok=True)
    job_name = f"{EXPERIMENT}_alpha_{alpha:.2f}_layer_{layer}"
    # Build the sbatch command
    sbatch_cmd = "sbatch "
    for param, value in slurm_params.items():
        sbatch_cmd += f"--{param}={value} "
    
    sbatch_cmd += f"--job-name={run_dir_name} --output={log_dir}/output.out --error={log_dir}/output.err "

    if layer is None:
        sbatch_cmd += f"--wrap=\"poetry run python {script_path} \
        --alpha {alpha} \
        --run_dir {run_dir_name} \
        --experiment {EXPERIMENT}\""
    else:
        sbatch_cmd += f"--wrap=\"poetry run python {script_path} \
        --alpha {alpha} \
        --layer {layer} \
        --run_dir {run_dir_name} \
        --experiment {EXPERIMENT}\""
    # Submit the job
    result = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job for alpha={alpha}, job ID: {job_id}")
        return job_id
    else:
        print(f"Error submitting job for alpha={alpha}: {result.stderr}")
        return None

# Submit a job for each alpha value
# for layer in range(3, 32):
#     if layer in layers:
#         continue


EXPERIMENT = 'chess'

alpha = 0.1

# job_id = tune_intervention_dir_job(alpha, None)

# for layer in range(3, 32):
#     job_id = tune_intervention_dir_job(alpha, layer)
#     time.sleep(1)

for alpha in [-0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
    job_id = tune_intervention_dir_job(alpha, 23)
    time.sleep(1)