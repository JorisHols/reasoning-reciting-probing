"""Adaption of LinearReasonigFeature paper to check if the direction we extracted is correct"""

gsm8k_prompt_template = """As an expert problem solver, solve step by step the following mathematical questions.

  Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
  A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. The final answer is 6.

  Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
  A: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. The final answer is 5.

  Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
  A: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. The final answer is 39.

  Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
  A: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. The final answer is 8.

  Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
  A: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. The final answer is 9.

  Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
  A: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. The final answer is 29.

  Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
  A: Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. The final answer is 33.

  Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
  A: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8. The final answer is 8.

  Q: {TARGET_QUESTION}
  A: Let's think step by step. """


def load_prompt_template(ds_name):
  
  if ds_name in ['GSM8k', 'GSM-symbolic', "MGSM"]:

    template = gsm8k_prompt_template
    template_no_cot = gsm8k_prompt_template

    return template, template_no_cot
  

def extract_final_answer(model_resp: str) -> float:
    # Remove commas so for example 5,000 becomes 5000
    model_resp = model_resp.replace(",", "")
    # Find the last number
    extracted_num = re.findall(r"-?\d+\.?\d*", model_resp)[-1]
    # Use float to ensure 3.0 and 3 are the same.
    return float(extracted_num)


def load_dataset(ds_name, dataset_dir, sample_num=3000, split='test'):
  if ds_name == 'GSM-symbolic':
    with open(os.path.join(dataset_dir, f'gsm-symbolic_data/GSM_symbolic.jsonl'), 'r') as file:
        ds_data = []
        for line in file:
            ds_data.append(json.loads(line))  # 解析每一行的JSON对象

    for entry in ds_data:
        entry['final_answer'] = extract_final_answer(model_resp=entry['answer'])
  return ds_data



def get_prediction(output=None, ds_name='GSM-symbolic'):
            
    if ds_name in ['GSM8k', 'GSM-symbolic', 'MGSM']:
        
        model_resp = output 
        #match = re.search(r"\\boxed\{(\d+\.?\d*)\}", model_resp)
        match = re.search(r"The final answer is (\d+\.?\d*)", model_resp)
        if match:
            return float(match.group(1)) 
        else:
            return None  
        
def parse_model_response(model_resp: str) -> float:
        pattern = "Let's think step by step"

        # Find all occurrences of the pattern
        occurrences = [m.start() for m in re.finditer(pattern, model_resp)]

        # Get the ninth occurrence if it exists
        ninth_idx = occurrences[8] if len(occurrences) >= 9 else -1
        if ninth_idx != -1:
            response_after_prompt = model_resp[ninth_idx:]
            return response_after_prompt
        else:
            return model_resp

def evaluation_on_dataset(val_sampled_data=None, prompts_cot=None, prompts_no_cot=None, run_in_fewshot=True, run_in_cot=True, 
                           ablation_dir=None, ds_name='GSM-symbolic', alpha=0.0, prober=None, layer=None):
    
    queries_batch = []  
    entry_batch = []
        
    for ix, entry in tqdm(enumerate(val_sampled_data)):
        
        if ds_name in ['GSM8k', 'GSM-symbolic', 'MGSM']:
            if run_in_fewshot:
                if run_in_cot:
                    query = prompts_cot.format(TARGET_QUESTION=entry['question'])
                else:
                    query = prompts_no_cot.format(TARGET_QUESTION=entry['question'])
            else:
                query = 'You are an expert problem solver. Solve the following problem and give your final answer in the following format:  \"\\boxed{yes}\".\n\nQ: ' + entry['question'] + "\n\nA: "


        queries_batch.append(query)
        entry_batch.append(entry)

    responses_dataset = prober.process_intervention(
        queries_batch,
        ablation_dir,
        alpha=alpha,
    )

    for entry, response in zip(entry_batch, responses_dataset):
        answer = parse_model_response(response['intervention_response'])
        entry['solution'] = answer
        prediction = get_prediction(answer, ds_name)

        if entry["final_answer"] == prediction:
            entry['model_predict_correctness'] = True
        else:
            entry['model_predict_correctness'] = False

    # Save the entry list to a json file
    if layer is not None:
        output_file = os.path.join(f'./outputs/liref-check/all-layers/{alpha}-liref_gsm_symbolic_layer_{layer}.json')
    else:
        output_file = os.path.join(f'./outputs/liref-check/all-layers/{alpha}-liref_gsm_symbolic.json')
    with open(output_file, 'w') as f:
        json.dump(entry_batch, f, indent=2)


def compute_performance_on_reason_subset(val_sampled_data=None, intervention=False, ds_name=None, alpha=None, layer=None):

    
    correct_predictions = 0
    total_predictions = len(val_sampled_data)

    for ix, entry in tqdm(enumerate(val_sampled_data)):
        if entry['model_predict_correctness'] == True:
            correct_predictions += 1

    reason_accuracy = correct_predictions / total_predictions
    
    if intervention:
        print(f"***Reason Subset {ds_name} Accuracy: {reason_accuracy:.4f} for alpha: {alpha} and layer: {layer}")
    else:
        print(f"***Original performance of Reason Subset {ds_name} Accuracy: {reason_accuracy:.4f}")

    return reason_accuracy
# loaded_dict = torch.load('./inputs/pca/liref_mmlu_activations/mmlu-pro-3000samples.pt')
# hs_cache_no_cot = loaded_dict['mmlu-pro_3000samples'] 

# with open('./inputs/pca/liref_mmlu_activations/mmlu-pro-3000samples.json', 'r', encoding='utf-8') as f:
#       sampled_data = json.load(f)

# reason_indices = [ix for ix, sample in enumerate(sampled_data) if sample['memory_reason_score'] > 0.5]
# memory_indices = [ix for ix, sample in enumerate(sampled_data) if sample['memory_reason_score'] <= 0.5]

#candidate_directions = get_candidate_directions(hs_cache_no_cot, model_layers_num, mlp_dim_num, reason_indices, memory_indices)

def get_candidate_directions(hs_cache_no_cot, model_layers_num, mlp_dim_num, reason_indices, memory_indices):

    candidate_directions = torch.zeros((model_layers_num, mlp_dim_num), dtype=torch.float64, device='cuda')

    # calculating candidate reasoning features
    for layer in range(model_layers_num):
            
        hs_no_cot = hs_cache_no_cot[layer]

        #  we store the mean activations in high-precision to avoid numerical issues
        reason_hs_no_cot = hs_no_cot[reason_indices, :].to(torch.float64)
        #print('reason_hs_no_cot.shape: ',reason_hs_no_cot.shape) reason有点多，memory有点少，需要进一步把数据集做scale up    
        memory_hs_no_cot = hs_no_cot[memory_indices, :].to(torch.float64)

        mean_reason_hs_no_cot = reason_hs_no_cot.mean(dim=0)
        mean_memory_hs_no_cot = memory_hs_no_cot.mean(dim=0)

        mean_diff = mean_reason_hs_no_cot - mean_memory_hs_no_cot  #Reasoning features shape: [bsz, dims] 
        candidate_directions[layer] = mean_diff

    return candidate_directions


def run_intervention_experiment_from_liref_paper(alpha):
    activations = torch.load('./inputs/liref/activations/llama3-8B-res-stream-3000-mmlu-pro.pt')
    print(f'****Activations loaded')

    with open("./inputs/pca/liref_mmlu_activations/mmlu-pro-3000samples.json", 'r') as f:
        data = json.load(f)
    print(f'****MMLU data loaded for reasoning and memory indices')
    
    reasoning_indices = [ix for ix, sample in enumerate(data) if sample['memory_reason_score'] > 0.5]
    memory_indices = [ix for ix, sample in enumerate(data) if sample['memory_reason_score'] <= 0.5]

    candidate_directions = get_candidate_directions(
        hs_cache_no_cot=activations,
        model_layers_num=32,
        mlp_dim_num=4096,
        reason_indices=reasoning_indices,
        memory_indices=memory_indices
    )
    print(f'****Candidate directions calculated')

    ds_name = 'GSM-symbolic'
    dataset_dir = "./inputs/liref"

    ds_data = load_dataset(ds_name=ds_name, dataset_dir=dataset_dir, split='test')

    print(f'****Data loaded from {ds_name}')

    prompt_template, prompt_template_no_cot = load_prompt_template(ds_name=ds_name)


    print(f'****Sampling 200 data points from {ds_name}')
    ds_data = random.sample(ds_data, 200)

    model_name = 'meta-llama/Llama-3.1-8B'

    prober = ProbeLlamaModel(max_new_tokens=200, generate_response=True, model_name=model_name)
    
    accuracies = []

    print(f'****Running on {ds_name} with Features Intervention and alpha: {alpha} and model: {model_name}')

    # Get accuracy of original model
    # print(f'****Running without intervention')
    # intervention_vectors = torch.zeros(32, 4096)
    # evaluation_on_dataset(val_sampled_data=ds_data, prompts_cot=prompt_template, prompts_no_cot=prompt_template_no_cot, 
    #                         ds_name=ds_name, run_in_fewshot=True, run_in_cot=True,
    #                         ablation_dir=intervention_vectors, alpha=0, prober=prober)
    # original_accuracy = compute_performance_on_reason_subset(val_sampled_data=ds_data, intervention=False, ds_name=ds_name, alpha=0)
    # accuracies.append(('original', original_accuracy))


    # for layer in range(15, 32):
    #     print(f'****Running on layer {layer} with intervention')
    #     if layer <= 2:
    #         continue
    #     intervention_vectors = [candidate_directions[layer]] * 32
    #     # Replace the first 3 vectors with 0 
    #     intervention_vectors[:3] = torch.zeros(3, 4096)
    #     evaluation_on_dataset(val_sampled_data=ds_data, prompts_cot=prompt_template, prompts_no_cot=prompt_template_no_cot, 
    #                         ds_name=ds_name, run_in_fewshot=True, run_in_cot=True,
    #                         ablation_dir=intervention_vectors, alpha=alpha, prober=prober, layer=layer)

        
    #     reason_accuracy = compute_performance_on_reason_subset(val_sampled_data=ds_data, intervention=True, ds_name=ds_name, alpha=alpha, layer=layer)
    #     accuracies.append((layer, reason_accuracy))
    
    layer = 6
    intervention_vectors = [candidate_directions[layer]] * 32
        # Replace the first 3 vectors with 0 
    intervention_vectors[:3] = torch.zeros(3, 4096)
    evaluation_on_dataset(val_sampled_data=ds_data, prompts_cot=prompt_template, prompts_no_cot=prompt_template_no_cot, 
                            ds_name=ds_name, run_in_fewshot=True, run_in_cot=True,
                            ablation_dir=intervention_vectors, alpha=alpha, prober=prober, layer=layer)

        
    reason_accuracy = compute_performance_on_reason_subset(val_sampled_data=ds_data, intervention=True, ds_name=ds_name, alpha=alpha, layer=layer)
    accuracies.append((layer, reason_accuracy))

    # Save the accuracies to a json file
    # Create directory if it doesn't exist
    os.makedirs(f'./outputs/liref/final/{alpha}', exist_ok=True)
    with open(f'./outputs/liref/final/{alpha}/liref_results_{model_name.replace("/", "_")}.json', 'w') as f:
        json.dump(accuracies, f, indent=2)
    return accuracies

    #print(f'****Finished running on layer {layer} with alpha {alpha}')

# def test_all_layers(alpha):
#     print(f'****Running on alpha: {alpha}')
#     candidate_directions = json.load(open('./inputs/chess/interventions/liref_reasoning_directions.json'))

#     ds_name = 'GSM-symbolic' 
#     dataset_dir = "./inputs/liref"

#     ds_data = load_dataset(ds_name=ds_name, dataset_dir=dataset_dir, split='test')
#     prompt_template, prompt_template_no_cot = load_prompt_template(ds_name=ds_name)

#     ds_data = random.sample(ds_data, 100)

#     print(f'****Running on {ds_name} with Features Intervention and alpha: {alpha}')

#     prober = ProbeLlamaModel(max_new_tokens=200, generate_response=True, model_name='meta-llama/Llama-3.1-8B')
    

#     evaluation_on_dataset(val_sampled_data=ds_data, prompts_cot=prompt_template, prompts_no_cot=prompt_template_no_cot,
#                          ds_name=ds_name, run_in_fewshot=True, run_in_cot=True,
#                          ablation_dir=candidate_directions, alpha=alpha, prober=prober)

#     compute_performance_on_reason_subset(val_sampled_data=ds_data, intervention=True, ds_name=ds_name, alpha=alpha)
#     print(f'****Finished running on alpha: {alpha}')

# def main(alpha, use_single_layer_direction):
#     if use_single_layer_direction:
#         for layer in range(32):
#             test_single_layer(layer, alpha)
#     else:
#         test_all_layers(alpha)

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
    args = parser.parse_args()

    torch.manual_seed(8888)
    random.seed(8888)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(8888)
        torch.cuda.manual_seed_all(8888)

    
    print(f'****Running on alpha: {args.alpha}')

    accuracies = run_intervention_experiment_from_liref_paper(args.alpha)



