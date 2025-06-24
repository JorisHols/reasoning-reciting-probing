"""Adaption of LinearReasonigFeature paper to calculate and find best intervention"""

# No imports here
def _format_activations(dataset):
    """Format the activations for the probe model."""
    activations = dataset['residual_activations']
    model_layers_num = 32
    mlp_dim_num = 4096
    layer_activation_dict={i: torch.zeros(len(activations), mlp_dim_num) for i in range(model_layers_num)}
    for i in range(len(activations)):
        for j in range(model_layers_num):
            layer_activation_dict[j][i] = torch.tensor(activations[i][j])
    return layer_activation_dict

        
def _format_intervention_vectors(layer_diff_means):
    """
    Format the intervention vectors

    Args:
        layer_diff_means: torch.Tensor
    """
    # Prepare the intervention vectors
    intervention_vectors = [layer_diff_means] * 32
    # Replace the first 3 vectors with 0 
    intervention_vectors[:3] = torch.zeros(3, 4096)
    return intervention_vectors


def _calculate_diff_means_directions(layer_activation_dict, first_indices, second_indices):
    """Calculate the diff means direction"""
    # Get the activations for the correct and incorrect predictions
    model_layers_num = 32
    mlp_dim_num = 4096

    candidate_directions = torch.zeros((model_layers_num, mlp_dim_num), dtype=torch.float64, device='cuda')
    
    # calculating candidate reasoning features
    for layer in range(model_layers_num):
        activations = layer_activation_dict[layer]
        #  we store the mean activations in high-precision to avoid numerical issues
        correct_activations = activations[first_indices, :].to(torch.float64)
        incorrect_activations = activations[second_indices, :].to(torch.float64)

        mean_correct_activations = correct_activations.mean(dim=0)
        mean_incorrect_activations = incorrect_activations.mean(dim=0)

        mean_diff = mean_correct_activations - mean_incorrect_activations 
        candidate_directions[layer] = mean_diff

    return candidate_directions


def find_best_intervention_direction(layer, alpha=0.1):
    """Find the best intervention direction"""
    print(f"**** Finding best intervention direction for layer {layer} with alpha {alpha} for {EXPERIMENT} experiment")
    results_path = f'./outputs/{EXPERIMENT}/tuning/'
    # load the directions
    print(f'**** Loading diff means directions')
    layer_wise_diff_means = torch.load(results_path+'diff_means_directions.pth')

    # Select the layer
    intervention_direction = layer_wise_diff_means[layer]

    print(f'**** Formatting intervention vectors')
    intervention_vectors = _format_intervention_vectors(intervention_direction)

    # Run the experiment
    print(f'**** Running experiment')
    if EXPERIMENT == 'gsm-symbolic':
        sample_size = 200
        experiment = GSMSymbolicExperiment(
            input_path=f'./inputs/liref/gsm-symbolic_data/',
            output_path=f'./outputs/gsm-symbolic/tuning/{RUN_DIR}/',
        chunk_size=sample_size,
        chunk_id=0,
        model_name='meta-llama/Llama-3.1-8B',
        sample_size=sample_size
        )

        # Run the experiment
        dataset = experiment.run_experiment(
            intervention_vectors=intervention_vectors,
            alpha=alpha,
            collect_activations=False,
            seed=42
        )

        print(f'**** Evaluating experiment')
        # Evaluate the experiment
        accuracy, correct_prediction_indices, _ = experiment.evaluate_llm_responses(dataset, seed=42)

        print(f"**** Layer {layer} intervention direction --- accuracy: {accuracy}")


    elif EXPERIMENT == 'chess':
        experiment = ChessExperiment(
            input_path=f'./inputs/chess/data/',
            output_path=results_path+RUN_DIR,
            chunk_size=800,
            chunk_id=0,
        )
        dataset = experiment.run_experiment(
            intervention_vectors=intervention_vectors,
            alpha=alpha,
            collect_activations=False,
            attach_control_prompts=False,
        )
        print(f'**** Evaluating experiment')
        results, correct_prediction_indices, _ = experiment.evaluate_llm_responses(dataset)
        rw_accuracy, counter_factual_accuracy, num_rw_instances, num_counter_factual_instances = experiment.get_chess_accuracies(results)
        print(f"Real world accuracy: {rw_accuracy}")
        print(f"Number of real world instances: {num_rw_instances}")
        print(f"Counterfactual accuracy: {counter_factual_accuracy}")
        print(f"Number of counterfactual instances: {num_counter_factual_instances}")
        
    
 
    if os.path.exists(results_path+f'results_alpha_{alpha}.json'):
        with open(results_path+f'results_alpha_{alpha}.json', 'r') as f:
            results = json.load(f)
    else:
        results = {}

    print(f'**** Storing results')
    # Write to results accuracy file json
    if EXPERIMENT == 'chess':
        results[f'layer_{layer}'] = {
            'rw_accuracy': rw_accuracy,
            'counter_factual_accuracy': counter_factual_accuracy,
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'correct_prediction_indices': correct_prediction_indices,
        }
    else:
        results[f'layer_{layer}'] = {
            'accuracy': accuracy,
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'correct_prediction_indices': correct_prediction_indices,
        }

    with open(results_path+f'results_alpha_{alpha}.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f'**** Finished')


def collect_activations_and_store_diff_means(alpha=0.1):
    """Collect the activations and store the diff means"""
    if EXPERIMENT == 'gsm-symbolic':
        results_path = f'./outputs/gsm-symbolic/tuning/'

        sample_size = 400

        experiment = GSMSymbolicExperiment(
            input_path=f'./inputs/liref/gsm-symbolic_data/',
            output_path=results_path+RUN_DIR,
            chunk_size=sample_size,
            chunk_id=0,
            model_name='meta-llama/Llama-3.1-8B',
            sample_size=sample_size
        )

        print(f'**** Running experiment')
        dataset = experiment.run_experiment(
            collect_activations=True,
            seed=8888
        )
        # dataset = load_from_disk('./outputs/gsm-symbolic/tuning/no_intervention/')

        # Get the indices
        accuracy, correct_prediction_indices, incorrect_prediction_indices = experiment.evaluate_llm_responses(dataset)

    elif EXPERIMENT == 'chess':
        results_path = f'./outputs/chess/tuning/'
        # os.makedirs(results_path+RUN_DIR, exist_ok=True)
        experiment = ChessExperiment(
            input_path=f'./inputs/chess/data/',
            output_path=results_path+RUN_DIR,
            chunk_size=800,
            chunk_id=0,
        )
        dataset = experiment.run_experiment(
            collect_activations=True,
            attach_control_prompts=False,
        )
        #dataset = load_from_disk(results_path+RUN_DIR)
        correct_sum = 0
        results, correct_prediction_indices, incorrect_prediction_indices = experiment.evaluate_llm_responses(dataset)
        # Evaluate only the counterfactual predictions
        correct_prediction_indices = results['counter_factual']['correct']['yes'] + results['counter_factual']['correct']['no']
        incorrect_prediction_indices = (
            results['counter_factual']['incorrect']['yes'] + 
            results['counter_factual']['incorrect']['no'] + 
            results['counter_factual']['incorrect']['invalid']
        )

        # for mode in results.keys():
        #     correct_sum += results[mode]['correct']['yes'] + results[mode]['correct']['no']
        
        accuracy = len(correct_prediction_indices) / (len(correct_prediction_indices) + len(incorrect_prediction_indices))

    if os.path.exists(results_path+f'results_alpha_{alpha}.json'):
        with open(results_path+f'results_alpha_{alpha}.json', 'r') as f:
            results = json.load(f)
    else:
        results = {}

    # Store the base accuracy
    print(f'Base accuracy: {accuracy}')
    results[f'base'] = {
        'accuracy': accuracy,
        'correct_prediction_indices': correct_prediction_indices,
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


    print(f'**** Storing results at {results_path + f"results_alpha_{alpha}.json"}')
    with open(results_path+f'results_alpha_{alpha}.json', 'w') as f:
        json.dump(results, f, indent=4)


    layer_activation_dict = _format_activations(dataset)

    # Calculate the diff means direction
    print(f'**** Calculating diff means direction')
    diff_means_directions = _calculate_diff_means_directions(layer_activation_dict, correct_prediction_indices, incorrect_prediction_indices)

    # Store the diff means direction (in torch file)
    print(f'**** Storing diff means direction at {results_path + "diff_means_directions.pth"}')
    torch.save(diff_means_directions, results_path+'diff_means_directions.pth')

    print(f'**** Finished')


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
    import pickle
    torch.manual_seed(8888)
    random.seed(8888)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(8888)
        torch.cuda.manual_seed_all(8888)

    from probe_llama import ProbeLlamaModel
    from experiments.arithmetic import ArithmeticExperiment
    from experiments.chess import ChessExperiment
    from experiments.gsm_symbolic import GSMSymbolicExperiment
    from datasets import load_from_disk
    import utils.arithmetic_utils as arithmetic_utils
    from datetime import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--experiment", type=str, default='gsm-symbolic')
    args = parser.parse_args()


    # Base for the expressionsthe intervention is applied on
    # Base the intervention directions are found on
    # Experiment settings
    EXPERIMENT = args.experiment
    RUN_DIR = args.run_dir

    if args.layer is not None:
        accuracies = find_best_intervention_direction(args.layer, args.alpha)
    else:
        accuracies = collect_activations_and_store_diff_means(args.alpha)




