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


# def find_best_intervention_direction(layer, alpha=0.1):
#     """Find the best intervention direction"""
#     print(f"**** Finding best intervention direction for layer {layer} with alpha {alpha} for {EXPERIMENT} experiment")
#     if 'combined' in EXPERIMENT:
#         type = EXPERIMENT.split('_', 1)[1]
#         results_path = f'./outputs/combined_directions/{type}/tuning/'
#     else:
#         results_path = f'./outputs/{type}/tuning/'
#     # load the directions
#     print(f'**** Loading diff means directions')
#     if 'combined' in EXPERIMENT:
#         layer_wise_diff_means = torch.load(f'./outputs/combined_directions/diff_means_directions_{type}.pth')
#     else:
#         layer_wise_diff_means = torch.load(results_path+'diff_means_directions.pth')

#     # Select the layer
#     intervention_direction = layer_wise_diff_means[layer]

#     print(f'**** Formatting intervention vectors')
#     intervention_vectors = _format_intervention_vectors(intervention_direction)

#     # Run the experiment
#     print(f'**** Running experiment')
#     if EXPERIMENT == 'gsm-symbolic' or 'combined' in EXPERIMENT:
#         sample_size = 300
#         experiment = GSMSymbolicExperiment(
#             input_path=f'./inputs/liref/gsm-symbolic_data/',
#             output_path=f'./outputs/gsm-symbolic/tuning/{RUN_DIR}/' if 'combined' not in EXPERIMENT else results_path,
#             chunk_size=sample_size,
#             chunk_id=0,
#             model_name='meta-llama/Llama-3.1-8B',
#             sample_size=sample_size
#         )

#         # Run the experiment
#         dataset = experiment.run_experiment(
#             intervention_vectors=intervention_vectors,
#             alpha=alpha,
#             collect_activations=False,
#             seed=42
#         )

#         print(f'**** Evaluating experiment')
#         # Evaluate the experiment
#         accuracy, correct_prediction_indices, incorrect_prediction_indices = experiment.evaluate_llm_responses(dataset, seed=42)

#         print(f"**** Layer {layer} intervention direction --- accuracy: {accuracy}")


#     elif EXPERIMENT == 'chess':
#         experiment = ChessExperiment(
#             input_path=f'./inputs/chess/data/',
#             output_path=results_path+RUN_DIR,
#             chunk_size=800,
#             chunk_id=0,
#         )
#         dataset = experiment.run_experiment(
#             intervention_vectors=intervention_vectors,
#             alpha=alpha,
#             collect_activations=False,
#             attach_control_prompts=False,
#         )
#         print(f'**** Evaluating experiment')
#         results, correct_prediction_indices, incorrect_prediction_indices = experiment.evaluate_llm_responses(dataset)
#         rw_accuracy, counter_factual_accuracy, num_rw_instances, num_counter_factual_instances = experiment.get_chess_accuracies(results)
#         print(f"Real world accuracy: {rw_accuracy}")
#         print(f"Number of real world instances: {num_rw_instances}")
#         print(f"Counterfactual accuracy: {counter_factual_accuracy}")
#         print(f"Number of counterfactual instances: {num_counter_factual_instances}")

#     elif EXPERIMENT == 'programming':
#         print(f'**** Running {EXPERIMENT} experiment')
#         experiment = ProgrammingExperiment(
#             input_path=f'./inputs/programming/',
#             output_path=results_path+RUN_DIR,
#             chunk_size=1000,
#             chunk_id=0,
#             mode='counter_factual',
#         )
#         dataset = experiment.run_experiment(
#             intervention_vectors=intervention_vectors,
#             alpha=alpha,
#             collect_activations=False,
#         )
#         print(f'**** Evaluating {EXPERIMENT} experiment')
#         accuracy, correct_prediction_indices, incorrect_prediction_indices = experiment.evaluate_llm_responses(dataset)
#         print(f"**** Layer {layer} intervention direction, alpha {alpha} --- accuracy: {accuracy}")
 
#     if os.path.exists(results_path+f'results.json'):
#         with open(results_path+f'results.json', 'r') as f:
#             results = json.load(f)
#     else:
#         results = {}

#     print(f'**** Storing results')
#     if f'alpha_{alpha}' not in results:
#         results[f'alpha_{alpha}'] = {}
#     if f'layer_{layer}' not in results[f'alpha_{alpha}']:
#         results[f'alpha_{alpha}'][f'layer_{layer}'] = {}

#     # Write to results accuracy file json
#     if EXPERIMENT == 'chess':
#         results[f'alpha_{alpha}'][f'layer_{layer}'] = {
#             'rw_accuracy': rw_accuracy,
#             'counter_factual_accuracy': counter_factual_accuracy,
#             'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'correct_prediction_indices': correct_prediction_indices,
#             'incorrect_prediction_indices': incorrect_prediction_indices,
#             'alpha': alpha,
#             'layer': layer,
#         }
#     else:
#         results[f'alpha_{alpha}'][f'layer_{layer}'] = {
#             'accuracy': accuracy,
#             'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'correct_prediction_indices': correct_prediction_indices,
#             'incorrect_prediction_indices': incorrect_prediction_indices,
#             'alpha': alpha,
#             'layer': layer,
#         }

#     with open(results_path+f'results.json', 'w') as f:
#         json.dump(results, f, indent=4)
#     print(f'**** Finished')


def collect_activations_and_store_diff_means(mode='counter_factual'):
    """Collect the activations and store the diff means"""
    if EXPERIMENT == 'gsm-symbolic':
        results_path = f'./outputs/gsm-symbolic/tuning/{mode}/'

        sample_size = 400

        experiment = GSMSymbolicExperiment(
            input_path=f'./inputs/liref/gsm-symbolic_data/',
            output_path=results_path+"activations/",
            chunk_size=sample_size,
            chunk_id=0,
            model_name='meta-llama/Llama-3.1-8B',
            sample_size=sample_size
        )

        print(f'**** Running experiment')
        # UNCOMMENT TO COLLECT ACTIVATIONS AGAIN
        # dataset = experiment.run_experiment(
        #     collect_activations=True,
        #     seed=8888
        # )
        dataset = load_from_disk(results_path+"activations/")
        # dataset = load_from_disk('./outputs/gsm-symbolic/tuning/no_intervention/')

        # Get the indices
        accuracy, correct_prediction_indices, incorrect_prediction_indices = experiment.evaluate_llm_responses(dataset)

    elif EXPERIMENT == 'chess':
        results_path = f'./outputs/chess/tuning/{mode}/'
        os.makedirs(results_path+"activations/", exist_ok=True)
        # os.makedirs(results_path+RUN_DIR, exist_ok=True)
        experiment = ChessExperiment(
            input_path=f'./inputs/chess/data/',
            output_path=results_path+"activations/",
            chunk_size=800,
            chunk_id=0,
        )
        # UNCOMMENT TO COLLECT ACTIVATIONS AGAIN
        # dataset = experiment.run_experiment(
        #     collect_activations=True,
        #     attach_control_prompts=False,
        # )
        dataset = load_from_disk(results_path+"activations/")
        #dataset = load_from_disk(results_path+RUN_DIR)
        correct_sum = 0
        results, correct_prediction_indices, incorrect_prediction_indices = experiment.evaluate_llm_responses(dataset)
        # Evaluate only the counterfactual predictions
        correct_prediction_indices = results[mode]['correct']['yes'] + results[mode]['correct']['no']
        incorrect_prediction_indices = (
            results[mode]['incorrect']['yes'] + 
            results[mode]['incorrect']['no'] + 
            results[mode]['incorrect']['invalid']
        )

        # for mode in results.keys():
        #     correct_sum += results[mode]['correct']['yes'] + results[mode]['correct']['no']
        
        accuracy = len(correct_prediction_indices) / (len(correct_prediction_indices) + len(incorrect_prediction_indices))

    elif EXPERIMENT == 'programming':
        results_path = f'./outputs/programming/tuning/{mode}/'
        os.makedirs(results_path+"activations/", exist_ok=True)
        experiment = ProgrammingExperiment(
            input_path=f'./inputs/programming/',
            output_path=results_path+"activations/",
            chunk_size=800,
            chunk_id=0,
            mode=mode,
        )
        # UNCOMMENT TO COLLECT ACTIVATIONS AGAIN
        # dataset = experiment.run_experiment(
        #     collect_activations=True,
        # )
        dataset = load_from_disk(results_path+"activations/")
        accuracy, correct_prediction_indices, incorrect_prediction_indices = experiment.evaluate_llm_responses(dataset)

    elif EXPERIMENT == 'arithmetic':
        if mode == 'counter_factual':
            base = 8
        else:
            base = 10

        results_path = f'./outputs/arithmetic/tuning/{mode}/'
        os.makedirs(results_path+"activations/", exist_ok=True)

        chunk_size = 1000

        experiment = ArithmeticExperiment(
            input_path=f'./inputs/arithmetic/data/base{base}.txt',
            output_path=results_path+"activations/",
            chunk_size=chunk_size,
            chunk_id=0,
            base=base,
        )
        # UNCOMMENT TO COLLECT ACTIVATIONS AGAIN
        dataset = experiment.run_experiment(
            collect_activations=True,
            mode=mode,
        )
        # dataset = load_from_disk(results_path+"activations/")
        if mode == 'counter_factual':
            _, correct_prediction_indices, real_world_prediction_indices, unparsable_prediction_indices = experiment.evaluate_llm_responses(dataset)
            incorrect_prediction_indices = unparsable_prediction_indices + real_world_prediction_indices
        else:
            # First 1000 are the arithmetic problems which the model answers correctly 100% of the time
            correct_prediction_indices = [i for i in range(chunk_size)]
            # Next 1000 are the gibberish problems for which there is no correct answer
            incorrect_prediction_indices = [i for i in range(chunk_size, 2*chunk_size)]


    layer_activation_dict = _format_activations(dataset)

    # Calculate the diff means direction
    print(f'**** Calculating diff means direction')
    diff_means_directions = _calculate_diff_means_directions(layer_activation_dict, correct_prediction_indices, incorrect_prediction_indices)

    # Store the diff means direction (in torch file)
    diff_means_directions_cpu = diff_means_directions.cpu().float()

    print(f'**** Storing diff means direction at {results_path + "diff_means_directions.pth"}')
    torch.save(diff_means_directions_cpu, results_path+'diff_means_directions.pth')

    print(f'**** Finished')


def get_gibberish_directions():
    """Get the gibberish directions"""
    if EXPERIMENT == 'chess':
        results_path = f'./outputs/chess/tuning/gibberish/'
        os.makedirs(results_path+"activations/", exist_ok=True)
        # os.makedirs(results_path+RUN_DIR, exist_ok=True)
        experiment = ChessExperiment(
            input_path=f'./inputs/chess/data/',
            output_path=results_path+"activations/",
            chunk_size=800,
            chunk_id=0,
        )
        #UNCOMMENT TO COLLECT ACTIVATIONS AGAIN
        dataset = experiment.run_experiment(
            collect_activations=True,
            save_activations=False,
            attach_control_prompts=False,
            gibberish=True,
        )
        # dataset = load_from_disk(results_path+"activations/")
        #dataset = load_from_disk(results_path+RUN_DIR)
        results, _, _ = experiment.evaluate_llm_responses(dataset)
        # Evaluate only the counterfactual predictions
        correct_prediction_indices = results['counter_factual']['correct']['yes'] + results['counter_factual']['correct']['no']
        incorrect_prediction_indices = (
            results['real_world']['incorrect']['yes'] + 
            results['real_world']['incorrect']['no'] + 
            results['real_world']['incorrect']['invalid']
        )

        incorrect_prediction_indices = random.sample(incorrect_prediction_indices, 400-len(correct_prediction_indices))

        # for mode in results.keys():
        #     correct_sum += results[mode]['correct']['yes'] + results[mode]['correct']['no']
        
        accuracy = len(correct_prediction_indices) / (len(correct_prediction_indices) + len(incorrect_prediction_indices))
    elif EXPERIMENT == 'arithmetic':
        base = 8

        results_path = f'./outputs/arithmetic/tuning/gibberish/'
        os.makedirs(results_path+"activations/", exist_ok=True)

        chunk_size = 1000

        experiment = ArithmeticExperiment(
            input_path=f'./inputs/arithmetic/data/base{base}.txt',
            output_path=results_path+"activations/",
            chunk_size=chunk_size,
            chunk_id=0,
            base=base,
        )
        # UNCOMMENT TO COLLECT ACTIVATIONS AGAIN
        dataset = experiment.run_experiment(
            collect_activations=True,
            mode='counter_factual',
            gibberish=True,
        )
        # dataset = load_from_disk(results_path+"activations/")
        
        _, correct_prediction_indices, _, _ = experiment.evaluate_llm_responses(dataset)
        incorrect_prediction_indices = [i for i in range(chunk_size, 2*chunk_size)]
        # Remove the number of correct predictions from the incorrect predictions - randomly - to keep the same number of indices as the original dataset
        incorrect_prediction_indices = random.sample(incorrect_prediction_indices, chunk_size-len(correct_prediction_indices))

    layer_activation_dict = _format_activations(dataset)

    # Calculate the diff means direction
    print(f'**** Calculating diff means direction')
    diff_means_directions = _calculate_diff_means_directions(layer_activation_dict, correct_prediction_indices, incorrect_prediction_indices)

    # Store the diff means direction (in torch file)
    diff_means_directions_cpu = diff_means_directions.cpu().float()
    print(f'**** Storing diff means direction at {results_path + "diff_means_directions.pth"}')
    torch.save(diff_means_directions_cpu, results_path+'diff_means_directions.pth')

    print(f'**** Finished')


def get_liref_paper_directions():
    """Get the liref paper approach directions - based on question type"""
    activations = torch.load('./inputs/liref/activations/llama3-8B-res-stream-3000-mmlu-pro.pt')

    print(f'****Activations loaded')

    with open("./inputs/pca/liref_mmlu_activations/mmlu-pro-3000samples.json", 'r') as f:
        data = json.load(f)
    print(f'****MMLU data loaded for reasoning and memory indices')
    
    reasoning_indices = [ix for ix, sample in enumerate(data) if sample['memory_reason_score'] > 0.5]
    memory_indices = [ix for ix, sample in enumerate(data) if sample['memory_reason_score'] <= 0.5]

    candidate_directions = _calculate_diff_means_directions(
        layer_activation_dict=activations,
        first_indices=reasoning_indices,
        second_indices=memory_indices
    )
    print(f'****Candidate directions calculated')

    diff_means_directions_cpu = candidate_directions.cpu().float()
    os.makedirs('./outputs/liref/tuning/', exist_ok=True)
    torch.save(diff_means_directions_cpu, './outputs/liref/tuning/diff_means_directions.pth')

    return

def get_exclusive_rw_vs_cf_directions():
    """Get the exclusive rw vs cf directions"""
    if EXPERIMENT == 'chess':
        results_path = f'./outputs/chess/tuning/exclusive/'
        os.makedirs(results_path+"activations/", exist_ok=True)
        # os.makedirs(results_path+RUN_DIR, exist_ok=True)
        experiment = ChessExperiment(
            input_path=f'./inputs/chess/data/',
            output_path=results_path+"activations/",
            chunk_size=800,
            chunk_id=0,
        )
        #UNCOMMENT TO COLLECT ACTIVATIONS AGAIN
        dataset = experiment.run_experiment(
            collect_activations=True,
            save_activations=False,
            attach_control_prompts=False,
        )
        # dataset = load_from_disk(results_path+"activations/")
        #dataset = load_from_disk(results_path+RUN_DIR)
        results, _, _ = experiment.evaluate_llm_responses(dataset)
        # Evaluate only the counterfactual predictions
        correct_prediction_indices = results['counter_factual']['correct']['yes'] + results['counter_factual']['correct']['no']
        incorrect_prediction_indices = (
            results['counter_factual']['incorrect']['yes'] + 
            results['counter_factual']['incorrect']['no']
        )

        
        accuracy = len(correct_prediction_indices) / (len(correct_prediction_indices) + len(incorrect_prediction_indices))
    elif EXPERIMENT == 'arithmetic':
        base = 8

        results_path = f'./outputs/arithmetic/tuning/exclusive/'
        os.makedirs(results_path+"activations/", exist_ok=True)

        chunk_size = 1000

        experiment = ArithmeticExperiment(
            input_path=f'./inputs/arithmetic/data/base{base}.txt',
            output_path=results_path+"activations/",
            chunk_size=chunk_size,
            chunk_id=0,
            base=base,
        )
        # UNCOMMENT TO COLLECT ACTIVATIONS AGAIN
        dataset = experiment.run_experiment(
            collect_activations=True,
            save_activations=False,
            mode='counter_factual',
        )
        
        _, correct_prediction_indices, real_world_prediction_indices, _ = experiment.evaluate_llm_responses(dataset)
        incorrect_prediction_indices = real_world_prediction_indices
        # Remove the number of correct predictions from the incorrect predictions - randomly - to keep the same number of indices as the original dataset

    layer_activation_dict = _format_activations(dataset)

    # Calculate the diff means direction
    print(f'**** Calculating diff means direction')
    diff_means_directions = _calculate_diff_means_directions(layer_activation_dict, correct_prediction_indices, incorrect_prediction_indices)

    # Store the diff means direction (in torch file)
    diff_means_directions_cpu = diff_means_directions.cpu().float()
    print(f'**** Storing diff means direction at {results_path + "diff_means_directions.pth"}')
    torch.save(diff_means_directions_cpu, results_path+'diff_means_directions.pth')

    print(f'**** Finished')
        
    


def create_diff_means_vectors_across_experiments(experiments: list[str], mode='counter_factual'):
    """Create the diff means vectors across experiments"""
    all_correct_activations = {layer: [] for layer in range(32)}
    all_incorrect_activations = {layer: [] for layer in range(32)}

    candidate_directions = torch.zeros((32, 4096), dtype=torch.float64, device='cuda')

    for experiment in experiments:
        results_path = f'./outputs/{experiment}/tuning/{mode}/'
        if experiment == 'programming':
            experiment_obj = ProgrammingExperiment(
                input_path=f'./inputs/programming/',
                output_path=results_path,
                chunk_size=200,
                chunk_id=0,
                mode=mode,
            )
        elif experiment == 'chess':
            experiment_obj = ChessExperiment(
                input_path=f'./inputs/chess/data/',
                output_path=results_path,
                chunk_size=800,
                chunk_id=0,
            )
        elif experiment == 'arithmetic':
            if mode == 'counter_factual':
                base = 8
            else:
                base = 10
            experiment_obj = ArithmeticExperiment(
                input_path=f'./inputs/arithmetic/data/base{base}.txt',
                output_path=results_path,
                chunk_size=1000,
                chunk_id=0,
                base=base,
            )
        else:
            raise ValueError(f'Experiment {experiment} not supported')
        
        print(f'**** Loading dataset for {experiment} experiment')
        if experiment == 'arithmetic':
            dataset = load_from_disk(results_path+'activations/')
            if mode == 'counter_factual':
                _, correct_prediction_indices, real_world_prediction_indices, unparsable_prediction_indices = experiment_obj.evaluate_llm_responses(dataset)
                incorrect_prediction_indices = unparsable_prediction_indices + real_world_prediction_indices
            else:
               # _, correct_prediction_indices, _, incorrect_prediction_indices = experiment_obj.evaluate_llm_responses(dataset)
               correct_prediction_indices = [i for i in range(1000)]
               incorrect_prediction_indices = [i for i in range(1000, 2000)]

        elif experiment == 'chess':
            dataset = load_from_disk(results_path+'activations/')
            results, correct_prediction_indices, incorrect_prediction_indices = experiment_obj.evaluate_llm_responses(dataset)
            correct_prediction_indices = results[mode]['correct']['yes'] + results[mode]['correct']['no']
            incorrect_prediction_indices = (
                results[mode]['incorrect']['yes'] + 
                results[mode]['incorrect']['no'] + 
                results[mode]['incorrect']['invalid']
            )
        else:
            dataset = load_from_disk(results_path+'activations/')
            accuracy, correct_prediction_indices, incorrect_prediction_indices = experiment_obj.evaluate_llm_responses(dataset)

        layer_activation_dict = _format_activations(dataset)

        for layer in range(32):
            all_correct_activations[layer].append(layer_activation_dict[layer][correct_prediction_indices, :])
            all_incorrect_activations[layer].append(layer_activation_dict[layer][incorrect_prediction_indices, :])

    results_path = f'./outputs/combined_directions/{mode}/'
    os.makedirs(results_path, exist_ok=True)
    for layer in range(32):
        all_correct_activations[layer] = torch.cat(all_correct_activations[layer], dim=0)
        all_incorrect_activations[layer] = torch.cat(all_incorrect_activations[layer], dim=0)

        mean_correct_activations = all_correct_activations[layer].mean(dim=0)
        mean_incorrect_activations = all_incorrect_activations[layer].mean(dim=0)

        candidate_directions[layer] = mean_correct_activations - mean_incorrect_activations

    # Store the diff means directions
    file_name = f'diff_means_directions_{"_".join(experiments)}.pth'
    print(f'**** Storing diff means directions at {results_path + file_name}')
    torch.save(candidate_directions, results_path+file_name)
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

    from probe_llama import ProbeLlamaModel
    from experiments.arithmetic import ArithmeticExperiment
    from experiments.chess import ChessExperiment
    from experiments.gsm_symbolic import GSMSymbolicExperiment
    from experiments.programming import ProgrammingExperiment
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
    # RUN_DIR = args.run_dir
    # MODE = 'real_world'
    #create_diff_means_vectors_across_experiments(['arithmetic', 'chess', 'programming'])
    # if args.layer is not None:
    #     # accuracies = find_best_intervention_direction(args.layer, args.alpha)
    # else:
    MODE = 'counter_factual'
    #accuracies = collect_activations_and_store_diff_means(MODE)
    #create_diff_means_vectors_across_experiments(['arithmetic', 'chess', 'programming'], mode=MODE)
    #get_gibberish_directions()
    #get_liref_paper_directions()
    get_exclusive_rw_vs_cf_directions()

