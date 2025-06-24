import subprocess
import os
import time

# Path to your script
script_path = "/gpfs/home5/jholshuijsen/reasoning-reciting-probing/code/probe_intervention.py"

# Base Slurm parameters
slurm_params = {
    'partition': 'gpu_a100',
    'nodes': 1,
    'ntasks': 1,
    'cpus-per-task': 8,
    'gpus': 1,
    'mem': '32G',
    'time': '1:00:00'
}

# Function to submit a job
def submit_probe_intervention_chunk_job(alpha, layer, chunk_id):

    # Check if base8 directory exists
    if EXPERIMENT == 'arithmetic':
        os.makedirs(f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/arithmetic/probe/base{BASE}", exist_ok=True)

        run_dir_name = f"intervention_probe_base{BASE}_alpha_{alpha:.2f}_layer_dofm_{layer}"
        #run_dir_name = f"base{BASE}_alpha_{alpha:.2f}_layer_dof_{layer}_full_dataset"
        #run_dir_name = f"base{BASE}_alpha_{alpha:.2f}_respective_diff_of_means"
        output_dir = f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/arithmetic/probe/base{BASE}/with_intervention"
    
    elif EXPERIMENT == 'chess':
        os.makedirs(f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/chess/intervention/", exist_ok=True)

        run_dir_name = f"chess_probe_intervention_alpha_{alpha:.2f}_layer_dofm_{layer}"
        output_dir = f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/chess/intervention/"
    
    elif EXPERIMENT == 'GSM-symbolic':
        os.makedirs(f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/gsm-symbolic/intervention/", exist_ok=True)

        run_dir_name = f"gsm-symbolic_probe_intervention_alpha_{alpha:.2f}_layer_dofm_{layer}"
        output_dir = f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/gsm-symbolic/intervention/"
    
    os.makedirs(output_dir, exist_ok=True)
    

    # create dir for output file
    run_dir = f"{output_dir}/{run_dir_name}"
    os.makedirs(run_dir, exist_ok=True)

    chunk_dir = f"{run_dir}/chunks/"
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_id_dir = f"{chunk_dir}/{chunk_id}"
    os.makedirs(chunk_id_dir, exist_ok=True)

    chunk_id_logs_dir = f"{chunk_id_dir}/logs"
    os.makedirs(chunk_id_logs_dir, exist_ok=True)
    # Build the sbatch command
    sbatch_cmd = "sbatch "
    for param, value in slurm_params.items():
        sbatch_cmd += f"--{param}={value} "
    
    sbatch_cmd += f"--job-name={run_dir_name} --output={chunk_id_logs_dir}/output.out --error={chunk_id_logs_dir}/output.err "

    sbatch_cmd += f"--wrap=\"poetry run python {script_path} \
    --alpha {alpha} \
    --layer {layer} \
    --base {BASE} \
    --run_dir {run_dir_name} \
    --chunk_id {chunk_id} \
    --chunk_size {CHUNK_SIZE} \
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

CHUNK_SIZE = 200
DATASET_SIZE = 200
BASE = 10
EXPERIMENT = 'chess'

intervention_layer = 9
alpha = 0.05
INTERVENTION = False

for chunk_id in range(DATASET_SIZE // CHUNK_SIZE):
    job_id = submit_probe_intervention_chunk_job(alpha, intervention_layer, chunk_id)
    time.sleep(1)