"""Adaption of LinearReasonigFeature paper to calculate and find best intervention"""

import pickle
import code.utils.arithmetic_utils as arithmetic_utils
import torch
from code.experiments.arithmetic import ArithmeticExperiment


def get_candidate_directions(hs_cache_no_cot, model_layers_num, mlp_dim_num, reason_indices, memory_indices):

    candidate_directions = torch.zeros((model_layers_num, mlp_dim_num), dtype=torch.float64, device='cuda')

    # calculating candidate reasoning features
    for layer in range(model_layers_num):
            
        activations = hs_cache_no_cot[layer]

        #  we store the mean activations in high-precision to avoid numerical issues
        reason_activations = activations[reason_indices, :].to(torch.float64)
        #print('reason_hs_no_cot.shape: ',reason_hs_no_cot.shape) reason有点多，memory有点少，需要进一步把数据集做scale up    
        memory_activations = activations[memory_indices, :].to(torch.float64)

        mean_reason_activations = reason_activations.mean(dim=0)
        mean_memory_activations = memory_activations.mean(dim=0)

        mean_diff = mean_reason_activations - mean_memory_activations  #Reasoning features shape: [bsz, dims] 
        candidate_directions[layer] = mean_diff

    return candidate_directions

def get_indices(llm_responses, expressions, base):
    base_indices = []
    base10_indices = []
    other_indices = []
    for i in range(len(llm_responses)):
        base_response = arithmetic_utils.get_label(expressions[i], base)
        base10_response = arithmetic_utils.get_label(expressions[i], 10)

        llm_response = llm_responses[i].split("=")[1].strip()
        llm_response = llm_response.split("\n")[0].strip()
        pred = arithmetic_utils.parse_output(llm_response).upper()
        
        if pred == base_response:
            base_indices.append(i)
        elif pred == base10_response:
            base10_indices.append(i)
        else:
            other_indices.append(i)

    return base_indices, base10_indices, other_indices
        

def get_diff_means_all_layers(base, base_indices, selected_base_10_indices):
    """Loads the activations for the base and calculates the diff of means for all layers"""
    print(f'****Loading activations for base: {base}')
    input_path = f'./code/probes/notebooks/utils/Llama-3.1-8B/base{base}/'
    layer_wise_activations = pickle.load(open(input_path + f'layer_activations_base{base}.pkl', 'rb'))


    layerwise_diff_means = get_candidate_directions(
        hs_cache_no_cot=layer_wise_activations,
        model_layers_num=32,
        mlp_dim_num=4096,
        reason_indices=base_indices,
        memory_indices=selected_base_10_indices
    )

    return layerwise_diff_means

def format_intervention_vectors(layer_diff_means: torch.Tensor):
    # Prepare the intervention vectors
    intervention_vectors = [layer_diff_means] * 32
    # Replace the first 3 vectors with 0 
    intervention_vectors[:3] = torch.zeros(3, 4096)
    return intervention_vectors


def run_intervention_experiment_from_liref_paper(layer, alpha, respective_layer_diff_of_means=False):
    """
    Runs the intervention experiment for the difference of means vector for a given layer and alpha

    Uses the indices not selected for diff of means vector as validation set on which the intervention is applied 
    """

    if not respective_layer_diff_of_means:
        print(f'****Running on difference of means vector for layer: {layer} with alpha: {alpha}')
    else:
        print(f'****Running on difference of means vector of all respective layers with alpha: {alpha}')

    input_path = f'./code/probes/notebooks/utils/Llama-3.1-8B/base{BASE}/'

    llm_responses = pickle.load(open(input_path + f'llm_responses_base{BASE}.pkl', 'rb'))
    expressions = pickle.load(open(input_path + f'expressions_base{BASE}.pkl', 'rb'))

    base_indices, base10_indices, other_indices = get_indices(llm_responses, expressions, BASE)
    
    print(f'****Base indices: {len(base_indices)}')
    print(f'****Base10 indices: {len(base10_indices)}')
    print(f'****Other indices: {len(other_indices)}')

    selected_base_10_indices = random.sample(base10_indices, len(base_indices))
    
    # Get diff of means vector for all layers
    layerwise_diff_means = get_diff_means_all_layers(BASE, base_indices, selected_base_10_indices)
    

    print(f'****Candidate directions calculated')
    print("Shape of candidate directions: ", layerwise_diff_means.shape)

    # Format the intervention vectors
    if not respective_layer_diff_of_means:
        intervention_vectors = format_intervention_vectors(layerwise_diff_means[layer])
    else:
        # Apply the dof to each layer respectively
        intervention_vectors = layerwise_diff_means

    # Load the arithmetic experiment
    output_path = f'./outputs/arithmetic/intervention/base{BASE}/alpha_{alpha}/'
    os.makedirs(output_path, exist_ok=True)
    
    experiment = ArithmeticExperiment(
        input_path=f'./inputs/arithmetic/data/base{BASE}.txt',
        output_path=output_path+f'runs/{RUN_DIR}',
        chunk_size=0, # No chunking
        chunk_id=0,
        base=BASE
    )

    exclude_indices = base_indices + selected_base_10_indices
    results = experiment.run_intervention_study(
        intervention_vectors=intervention_vectors,
        alpha=alpha,
        exclude_indices=exclude_indices if 'full_dataset' not in RUN_DIR else None
    )

    print("Number of results: ", len(results), "Should equal: ", len(expressions) - len(exclude_indices))

    base_indices = []
    base10_indices = []
    other_indices = []
    llm_responses = results['intervention_response']
    for ix in range(len(llm_responses)):
        base_response = arithmetic_utils.get_label(expressions[ix], BASE)
        base10_response = arithmetic_utils.get_label(expressions[ix], 10)    

        llm_response = llm_responses[ix].split("=")[1].strip()
        llm_response = llm_response.split("\n")[0].strip()
        pred = arithmetic_utils.parse_output(llm_response).upper()

        if pred == base_response:
            base_indices.append(ix)
        elif pred == base10_response:
            base10_indices.append(ix)
        else:
            other_indices.append(ix)

    print(f"Base {BASE} indices: {len(base_indices)}")
    print(f"Base10 indices: {len(base10_indices)}")
    print(f"Other indices: {len(other_indices)}")

    print(f"Base {BASE} accuracy (correct): {len(base_indices) / len(llm_responses)}")
    print(f"Base10 accuracy (incorrect): {len(base10_indices) / len(llm_responses)}")
    print(f"Other accuracy (incorrect): {len(other_indices) / len(llm_responses)}")

    # Save results to json file
    results_dict = {
        'layer': layer,
        'base_accuracy': len(base_indices) / len(llm_responses),
        'base10_accuracy': len(base10_indices) / len(llm_responses), 
        'other_accuracy': len(other_indices) / len(llm_responses)
    }

    # Create output path for the alpha results
    if respective_layer_diff_of_means:
        output_file = os.path.join(output_path, 'respective_diff_of_means_accuracy.json')
    elif 'full_dataset' in RUN_DIR:
        output_file = os.path.join(output_path, 'layer_result_full_dataset.json')
    else:
        output_file = os.path.join(output_path, 'layer_results.json')

    # Load existing results if any
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Add/update results for this layer
    if not respective_layer_diff_of_means:
        all_results[layer] = results_dict
    else:
        all_results['respective_diff_of_means'] = results_dict

    # Save updated results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)




if __name__ == "__main__":
    import subprocess
    import sys
    import os

    print("Setting up Poetry environment")
    os.chdir('/gpfs/home5/jholshuijsen/reasoning-reciting-probing')
    
    # Configure Poetry to create venv in project
    subprocess.run(['poetry', 'config', 'virtualenvs.in-project', 'true', '--local'], check=True)

    # Install dependencies
    print("Installing dependencies...")
    subprocess.run(['poetry', 'install', '--no-interaction', '--no-root'], check=True)
    
    # Get the virtual environment path
    result = subprocess.run(['poetry', 'env', 'info', '--path'], 
                           capture_output=True, text=True, check=True)
    venv_path = result.stdout.strip()
    
    # Activate the virtual environment by modifying sys.path
    venv_site_packages = os.path.join(venv_path, 'lib', 
                                     f'python{sys.version_info.major}.{sys.version_info.minor}', 
                                     'site-packages')
    sys.path.insert(0, venv_site_packages)
    
    print(f"Poetry virtual environment activated at: {venv_path}")


    import argparse
    import os
    import json
    import re
    import torch
    from tqdm import tqdm
    import random

    from probe_llama import ProbeLlamaModel

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--base", type=int, default=8)
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(8888)
    random.seed(8888)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(8888)
        torch.cuda.manual_seed_all(8888)

    BASE = args.base
    RUN_DIR = args.run_dir

    respective_layer_diff_of_means = False

    if 'respective_diff_of_means' in RUN_DIR:
        respective_layer_diff_of_means = True

    accuracies = run_intervention_experiment_from_liref_paper(args.layer, args.alpha, respective_layer_diff_of_means=respective_layer_diff_of_means)



