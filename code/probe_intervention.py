"""Adaption of LinearReasonigFeature paper to calculate and find best intervention"""

# No imports here

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
    input_path = f'./notebooks/notebooks/utils/Llama-3.1-8B/base{base}/'
    layer_wise_activations = pickle.load(open(input_path + f'layer_activations_base{base}.pkl', 'rb'))


    layerwise_diff_means = get_candidate_directions(
        hs_cache_no_cot=layer_wise_activations,
        model_layers_num=32,
        mlp_dim_num=4096,
        reason_indices=base_indices,
        memory_indices=selected_base_10_indices
    )

    return layerwise_diff_means

def format_intervention_vectors(layer_diff_means):
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

def save_intervention_result(experiment, intervention_type, alpha, layer, result_data, mode):
    """
    Save intervention result to a structured JSON file organized by experiment, intervention type, and layer
    
    Args:
        experiment: str - experiment name (e.g., 'chess', 'base8', etc.)
        intervention_type: str - type of intervention
        alpha: float - intervention strength
        layer: int - layer number
        result_data: dict - experiment-specific results to save
    """
    import json
    import os
    
    # Create unique ID for this specific alpha configuration
    config_id = f"alpha{alpha}"
    
    # Create the result object
    result = {
        'alpha': alpha,
        'layer': layer,
        'config_id': config_id,
        **result_data  # Merge in experiment-specific data
    }
    if intervention_type == 'liref':
        mode = 'liref'
    elif intervention_type in ['exclusive_chess', 'gibberish_chess', 'exclusive_arithmetic', 'gibberish_arithmetic']:
        mode = intervention_type.split('_')[0]
    # Load existing results structure
    results_file = f'./outputs/{experiment}/results/{mode}/intervention_results.json'
     # Ensure output directory exists
    os.makedirs(f'./outputs/{experiment}/results/{mode}/', exist_ok=True)
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(result_data)
            raise ValueError(f"Warning: {results_file} exists but contains invalid JSON. Starting with empty structure.")
    else:
        all_results = {}
    
    # Initialize nested structure if needed
    if experiment not in all_results:
        all_results[experiment] = {}
    if intervention_type not in all_results[experiment]:
        all_results[experiment][intervention_type] = {}
    if str(layer) not in all_results[experiment][intervention_type]:
        all_results[experiment][intervention_type][str(layer)] = {}
    
    # Check if this configuration already exists
    if config_id in all_results[experiment][intervention_type][str(layer)]:
        print(f"Overwriting existing result for {experiment} -> {intervention_type} -> layer{layer} -> {config_id}")
    else:
        print(f"Adding new result for {experiment} -> {intervention_type} -> layer{layer} -> {config_id}")
    
    # Save the result
    all_results[experiment][intervention_type][str(layer)][config_id] = result
    
   
    # Save back to file with pretty formatting
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, sort_keys=True)
    
    return result


def collect_activations_for_intervention(layer, alpha, intervention_type, mode='counter_factual'):
    """
    Runs the intervention experiment for the difference of means vector for a given layer and alpha

    Uses the indices not selected for diff of means vector as validation set on which the intervention is applied 
    """

    if INTERVENTION:
        if intervention_type in ['chess', 'gsm-symbolic', 'programming', 'arithmetic']:
            results_path = f'./outputs/{intervention_type}/tuning/{mode}/'
            # load the directions
            print(f'**** Loading diff means directions')
            layer_wise_diff_means = torch.load(results_path+'diff_means_directions.pth')

            # Select the layer
            intervention_direction = layer_wise_diff_means[layer]

            print(f'**** Formatting intervention vectors')
            intervention_vectors = format_intervention_vectors(intervention_direction)

        elif 'combined_' in intervention_type:
            vector = intervention_type.split('combined_')[1]
            results_path = f'./outputs/combined_directions/{mode}/'

            print(f'**** Loading intervention vectors for {vector}')
            layer_wise_diff_means = torch.load(results_path+f'diff_means_directions_{vector}.pth')
            # Select the layer
            intervention_direction = layer_wise_diff_means[layer]

            print(f'**** Formatting intervention vectors')
            intervention_vectors = format_intervention_vectors(intervention_direction)
        elif intervention_type == 'liref':
            print(f'**** Loading intervention vectors for liref')
            layer_wise_diff_means = torch.load('./outputs/liref/tuning/diff_means_directions.pth')
            intervention_direction = layer_wise_diff_means[layer]
            print(f'**** Formatting intervention vectors')
            intervention_vectors = format_intervention_vectors(intervention_direction)
        elif intervention_type in ['exclusive_chess', 'gibberish_chess', 'exclusive_arithmetic', 'gibberish_arithmetic']:
            type = intervention_type.split('_')[0]
            task = intervention_type.split('_')[1]
            print(f'**** Loading intervention vectors for {type}')
            results_path = f'./outputs/{task}/tuning/{type}/'
            layer_wise_diff_means = torch.load(results_path+f'diff_means_directions.pth')
            intervention_direction = layer_wise_diff_means[layer]
            print(f'**** Formatting intervention vectors')
            intervention_vectors = format_intervention_vectors(intervention_direction)

    if EXPERIMENT == 'base8':
    # Load the arithmetic experiment
        base = int(EXPERIMENT.split('base')[1])

        output_path = f'./outputs/arithmetic/probe/base{base}/with_intervention/{RUN_DIR}/chunks/{CHUNK_ID}/'
        #output_path = f'./outputs/arithmetic/logit_lens/base{base}/combine_intervention/'
        os.makedirs(output_path, exist_ok=True)
        # Get the digit from the experiment name
        experiment = ArithmeticExperiment(
            input_path=f'./inputs/arithmetic/data/base{base}.txt',
            output_path=output_path,
            chunk_size=CHUNK_SIZE,
            chunk_id=CHUNK_ID,
            base=base
        )
        if INTERVENTION:
            # Run the experiment with the intervention vectors
            dataset = experiment.run_experiment(
                intervention_vectors=intervention_vectors,
                alpha=alpha,
                collect_activations=COLLECT_ACTIVATIONS
            )
            accuracy, correct_indices, real_world_indices, unparsable_indices = experiment.evaluate_llm_responses(dataset)

            print(len(real_world_indices))
            print(len(correct_indices))
            print(len(unparsable_indices))

            print(f'****Accuracy: {accuracy}')
            print(f'****Real world accuracy: {len(real_world_indices) / len(dataset)}')

            result_data = {
                'accuracy': accuracy,
                'real_world_accuracy': len(real_world_indices) / len(dataset),
                'alpha': alpha,
                'layer': layer,
                'base': base,
                'intervention_type': intervention_type,
                'experiment': EXPERIMENT,
                'correct_indices': correct_indices,
            }
            save_intervention_result(EXPERIMENT, intervention_type, alpha, layer, result_data, mode)

        
        return

    elif EXPERIMENT == 'chess':
        output_path = f'./outputs/chess/intervention/{RUN_DIR}/chunks/{CHUNK_ID}/'
        os.makedirs(output_path, exist_ok=True)

        experiment = ChessExperiment(
            model_name='meta-llama/Llama-3.1-8B',
            input_path=f'./inputs/chess/data/',
            output_path=output_path,
            chunk_size=CHUNK_SIZE,
            chunk_id=CHUNK_ID,
            seed=8888
        )

        if not INTERVENTION_ONLY:
            print("**** Running experiment WITHOUT intervention")
            # Run the experiment wthout intervention
            dataset = experiment.run_experiment(
                attach_control_prompts=False,
                collect_activations=COLLECT_ACTIVATIONS
            )

    
            results, _, _ = experiment.evaluate_llm_responses(dataset)

            rw_accuracy, counter_factual_accuracy, num_rw_instances, num_counter_factual_instances = experiment.get_chess_accuracies(results)
            print(f"Real world accuracy: {rw_accuracy}")
            print(f"Number of real world instances: {num_rw_instances}")
            print(f"Counterfactual accuracy: {counter_factual_accuracy}")
            print(f"Number of counterfactual instances: {num_counter_factual_instances}")
           
        # Print the accuracy of the LLM responses on the chess data
        if INTERVENTION:
            print("**** Running experiment WITH intervention")

            # Run the experiment with intervention
            dataset = experiment.run_experiment(
                intervention_vectors=intervention_vectors,
                alpha=alpha,
                attach_control_prompts=False,
                collect_activations=COLLECT_ACTIVATIONS
            )

            results, _, _ = experiment.evaluate_llm_responses(dataset)
            rw_accuracy, counter_factual_accuracy, num_rw_instances, num_counter_factual_instances = experiment.get_chess_accuracies(results)
            print(f"Real world accuracy: {rw_accuracy}")
            print(f"Number of real world instances: {num_rw_instances}")
            print(f"Counterfactual accuracy: {counter_factual_accuracy}")
            print(f"Number of counterfactual instances: {num_counter_factual_instances}")

            result_data = {
                'real_world_accuracy': rw_accuracy,
                'counterfactual_accuracy': counter_factual_accuracy,
                'num_rw_instances': num_rw_instances,
                'num_counter_factual_instances': num_counter_factual_instances,
                'alpha': alpha,
                'layer': layer,
                'intervention_type': intervention_type,
                'experiment': EXPERIMENT                
            }
            save_intervention_result(EXPERIMENT, intervention_type, alpha, layer, result_data, mode)

        return
    elif EXPERIMENT == 'gsm-symbolic':
        sample_size = 400
        output_path = f'./outputs/gsm-symbolic/intervention/{RUN_DIR}/chunks/{CHUNK_ID}/'
        os.makedirs(output_path, exist_ok=True)

        experiment = GSMSymbolicExperiment(
            input_path=f'./inputs/liref/gsm-symbolic_data/',
            output_path=output_path,
            chunk_size=CHUNK_SIZE,
            chunk_id=CHUNK_ID,
            model_name='meta-llama/Llama-3.1-8B',
            sample_size=sample_size
        )

        if not INTERVENTION_ONLY:
            print("**** Running experiment WITHOUT intervention")
            # Run the experiment wthout intervention
            dataset = experiment.run_experiment(
                collect_activations=COLLECT_ACTIVATIONS,
            )

            reason_accuracy, _, _ = experiment.evaluate_llm_responses(dataset)
            print(f"Reason accuracy: {reason_accuracy:.4f}")

        if INTERVENTION:
            test_set = False
            print("**** Running experiment WITH intervention")
            # Run the experiment with intervention
            dataset = experiment.run_experiment(
                intervention_vectors=intervention_vectors,
                alpha=alpha,
                collect_activations=COLLECT_ACTIVATIONS,
                test_set=test_set
            )

            reason_accuracy, _, _ = experiment.evaluate_llm_responses(dataset, test_set=test_set)
            print(f"Reason accuracy: {reason_accuracy:.4f}")

            result_data = {
                'reason_accuracy': reason_accuracy,
                'alpha': alpha,
                'layer': layer,
                'intervention_type': intervention_type,
                'experiment': EXPERIMENT,
                'test_set': test_set
            }
            save_intervention_result(EXPERIMENT, intervention_type, alpha, layer, result_data, mode)

    elif EXPERIMENT == 'programming':
        print(f"Running experiment for programming")
        output_path = f'./outputs/programming/intervention/{RUN_DIR}/chunks/{CHUNK_ID}/'
        os.makedirs(output_path, exist_ok=True)

        experiment = ProgrammingExperiment(
            input_path=f'./inputs/programming/',
            output_path=output_path,
            chunk_size=CHUNK_SIZE,
            chunk_id=CHUNK_ID,
            model_name='meta-llama/Llama-3.1-8B',
            mode='counter_factual',
        )

        if not INTERVENTION_ONLY:
            print("**** Running experiment WITHOUT intervention")
            # Run the experiment wthout intervention
            dataset = experiment.run_experiment(
                collect_activations=COLLECT_ACTIVATIONS,
            )
            pass_at_k_result, _, _ = experiment.evaluate_llm_responses(dataset)
            print(f"Pass@1 for index_from={1 if experiment.mode == 'counter_factual' else 0}: {pass_at_k_result['pass@1']}")

        if INTERVENTION:
            print("**** Running experiment WITH intervention")
            # Run the experiment with intervention
            dataset = experiment.run_experiment(
                intervention_vectors=intervention_vectors,
                alpha=alpha,
            )
            pass_at_k_result, _, _ = experiment.evaluate_llm_responses(dataset)
            print(f"**** INTERVENTION RESULT:Pass@1 for index_from={1 if experiment.mode == 'counter_factual' else 0}: {pass_at_k_result['pass@1']}")
            result_data = {
                'pass@1': pass_at_k_result['pass@1'],
                'alpha': alpha,
                'layer': layer,
                'intervention_type': intervention_type,
                'experiment': EXPERIMENT                
            }
            save_intervention_result(EXPERIMENT, intervention_type, alpha, layer, result_data, mode)
        return

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

    from experiments.arithmetic import ArithmeticExperiment
    from experiments.chess import ChessExperiment
    from experiments.gsm_symbolic import GSMSymbolicExperiment
    from experiments.programming import ProgrammingExperiment
    import utils.arithmetic_utils as arithmetic_utils

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=250)
    parser.add_argument("--experiment", type=str, default='base8')
    parser.add_argument("--intervention_type", type=str, default='base8')
    parser.add_argument("--intervention_only", type=bool, default=False)
    args = parser.parse_args()


    # Base the intervention directions are found on
    RUN_DIR = args.run_dir
    CHUNK_ID = args.chunk_id
    CHUNK_SIZE = args.chunk_size
    INTERVENTION = True
    SAVE_ACTIVATIONS = True
    COLLECT_ACTIVATIONS = True
    INTERVENTION_ONLY = args.intervention_only
    # Experiment settings
    EXPERIMENT = args.experiment
    INTERVENTION_TYPE = args.intervention_type
    INTERVENTION_MODE = 'counter_factual'
    accuracies = collect_activations_for_intervention(args.layer, args.alpha, intervention_type=INTERVENTION_TYPE, mode=INTERVENTION_MODE)



