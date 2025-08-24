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
    'time': '01:30:00'
}

# Function to submit a job
def submit_probe_intervention_chunk_job(alpha, layer, chunk_id):
    # Check if base8 directory exists
    if 'base' in EXPERIMENT:
        base = int(EXPERIMENT.split('base')[1])
        run_dir_name = f"base{base}_probe_intervention_alpha_{alpha:.2f}_layer_dofm_{layer}"
        output_dir = f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/arithmetic/intervention/base{base}/{INTERVENTION_TYPE}"
    
    elif EXPERIMENT == 'chess':
        run_dir_name = f"chess_probe_intervention_alpha_{alpha:.2f}_layer_dofm_{layer}"
        output_dir = f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/chess/intervention/{INTERVENTION_TYPE}"
    
    elif EXPERIMENT == 'gsm-symbolic':
        run_dir_name = f"gsm-symbolic_probe_intervention_alpha_{alpha:.2f}_layer_dofm_{layer}"
        output_dir = f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/gsm-symbolic/intervention/{INTERVENTION_TYPE}"

    elif EXPERIMENT == 'programming':
        run_dir_name = f"programming_probe_intervention_alpha_{alpha:.2f}_layer_dofm_{layer}"
        output_dir = f"/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/programming/intervention/{INTERVENTION_TYPE}"
    
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
    --run_dir {run_dir_name} \
    --chunk_id {chunk_id} \
    --chunk_size {CHUNK_SIZE} \
    --experiment {EXPERIMENT} \
    --intervention_type {INTERVENTION_TYPE} \
    --intervention_only yes\"" # Comment out to also run base intervention
    # Submit the job
    result = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job for alpha={alpha}, layer={layer}, job ID: {job_id}")
        return job_id
    else:
        print(f"Error submitting job for alpha={alpha}: {result.stderr}")
        return None

# Submit a job for each alpha value
# for layer in range(3, 32):
#     if layer in layers:
#         continue

# Set this to larger than larges dataset to avoid chunking
# DATASET_SIZE = 2000
CHUNK_SIZE = 2000
CHUNK_ID = 0

EXPERIMENT = 'gsm-symbolic'
# intervention_args = ('exclusive_chess', 23)
intervention_args = ('gibberish_chess', 5)
# intervention_args = ('gibberish_arithmetic', 18)
# intervention_args = ('exclusive_arithmetic', 9)
#intervention_args = ('combined_arithmetic_chess_programming', 7)
INTERVENTION_TYPE = intervention_args[0]
INTERVENTION_LAYER = intervention_args[1]
# Define the intervention type and layer
# intervention_args = ('programming', 9)
# INTERVENTION_TYPE = intervention_args[0]
# INTERVENTION_LAYER = intervention_args[1]

INTERVENTION = True

alphas = [-0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
for alpha in alphas:
    job_id = submit_probe_intervention_chunk_job(alpha, INTERVENTION_LAYER, CHUNK_ID)
    time.sleep(1)

# job_id = submit_probe_intervention_chunk_job(0.1, INTERVENTION_LAYER, CHUNK_ID)


# # UNCOMMENT TO FIND THE BEST LAYER ON THE TASK
# for layer in range(3, 32):
#     job_id = submit_probe_intervention_chunk_job(0.1, layer, CHUNK_ID)
#     time.sleep(1)

# for chunk_id in range(DATASET_SIZE // CHUNK_SIZE):
#     job_id = submit_probe_intervention_chunk_job(alpha, intervention_layer, chunk_id)
#     time.sleep(1)