import subprocess
import os
import time

# Alpha values to process
alphas = [0.1]
base = 8

# Path to your script
script_path = "/gpfs/home5/jholshuijsen/reasoning-reciting-probing/code/liref_style_intervention.py"

# Base Slurm parameters
slurm_params = {
    'partition': 'gpu_a100',
    'nodes': 1,
    'ntasks': 1,
    'cpus-per-task': 8,
    'gpus': 1,
    'mem': '32G',
    'time': '2:00:00'
}

# Function to submit a job
def submit_job_for_alpha(alpha, layer):

    # Check if base8 directory exists
    os.makedirs(f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/arithmetic/intervention/base{base}", exist_ok=True)
    os.makedirs(f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/arithmetic/intervention/base{base}/alpha_{alpha}", exist_ok=True)

    #run_dir_name = f"base{base}_alpha_{alpha:.2f}_layer_dof_{layer}"
    #run_dir_name = f"base{base}_alpha_{alpha:.2f}_layer_dof_{layer}_full_dataset"
    run_dir_name = f"base{base}_alpha_{alpha:.2f}_respective_diff_of_means"
    output_dir = f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/arithmetic/intervention/base{base}/alpha_{alpha}/runs/"
    os.makedirs(output_dir, exist_ok=True)

    # create dir for output file
    run_dir = f"{output_dir}/{run_dir_name}"
    os.makedirs(run_dir, exist_ok=True)

    # create dir for logs
    
    # Build the sbatch command
    sbatch_cmd = "sbatch "
    for param, value in slurm_params.items():
        sbatch_cmd += f"--{param}={value} "
    
    sbatch_cmd += f"--job-name={run_dir_name} --output={run_dir}/output.out --error={run_dir}/output.err "

    sbatch_cmd += f"--wrap=\"poetry run python {script_path} --alpha {alpha} --layer {layer} --base {base} --run_dir {run_dir_name}\""
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
layer = 9
for alpha in alphas:
    job_id = submit_job_for_alpha(alpha, layer)
    time.sleep(1)