import random
import pandas as pd
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer, 
    AutoModelForCausalLM, 
    AutoTokenizer
)
import torch
import numpy as np
from tqdm import tqdm

random.seed(8888)
torch.manual_seed(8888)
random.seed(8888)
np.random.seed(8888)

if torch.cuda.is_available():
    torch.cuda.manual_seed(8888)
    torch.cuda.manual_seed_all(8888)


def load_mmlu_pro_data():
    data = pd.read_json("./inputs/pca/liref_mmlu_activations/mmlu-pro-3000samples.json")
    return data

def generate_outputs(questions, model, tokenizer):
    
    inputs = tokenizer(questions, return_tensors="pt", padding="longest", return_token_type_ids=False).to('cuda')
    with torch.no_grad():
        output = model(**inputs, output_hidden_states = True)

    return output




def main():
    # Assert cuda is available
    assert torch.cuda.is_available(), "CUDA is not available"

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B").to('cuda')
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    data = load_mmlu_pro_data()

    model_layers_num = 32 # Llama-3.1-8B has 32 layers
    batch_size = 4

    total_iterations = len(data)
    batch_count = (total_iterations + batch_size - 1) // batch_size

    layers_to_cache = list(range(model_layers_num))
    print('layers_to_cache: ',layers_to_cache)
    hs_cache_no_cot = {}
    
    queries_batch = []

    with tqdm(total=batch_count, desc="Processing batches") as pbar:
        for ix, row in data.iterrows():
            query = 'Q: ' + row['question'] + "\nA: "
            queries_batch.append(query)
            if len(queries_batch) == batch_size or ix == len(data) - 1:
                output = generate_outputs(queries_batch, model, tokenizer)
                
                for layer in layers_to_cache:
                    if layer not in hs_cache_no_cot:
                        hs_cache_no_cot[layer] = output["hidden_states"][layer][: ,-1 , :].detach().cpu() #bs * tok * dims
                    else:
                        hs_cache_no_cot[layer] = torch.cat((hs_cache_no_cot[layer], output["hidden_states"][layer][: ,-1 , :].detach().cpu()), dim=0)

                del output
                queries_batch = []
                pbar.update(1)

            torch.cuda.empty_cache()


    torch.save(hs_cache_no_cot, "./inputs/liref/activations/llama3-8B-res-stream-3000-mmlu-pro.pt")


if __name__ == "__main__":
    main()


