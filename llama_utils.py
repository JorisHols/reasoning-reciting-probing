from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F

import os
import torch
from tqdm import tqdm

def load_model_and_tokenizer(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the model from the model name.
    """
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"), padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_simple_dataset(file_path: str) -> list[str]:
    """
    Load the simple dataset from a CSV file.
    """
    data = []
    labels = []
    with open(file_path, "r") as f:
        # Skip header row
        next(f)
        for line in f:
            # Split on comma and remove quotes if present
            id, example = line.strip().split(",", 1)
            # Split the example into text and label
            splits = example.rsplit(" ", 1)
            data.append(splits[0])
            labels.append(splits[1].replace(".", ""))

    return data, labels


def query_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt_batch: list[str], max_new_tokens=250):
    """
    Query the model with a batch of prompts.
    """
    prompt_tokens = tokenizer(prompt_batch, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model.generate(
            prompt_tokens.input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)



def query_with_hidden_states(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt_batch: list[str], max_new_tokens=250):
    """
    Generate responses for a batch of prompts while saving hidden states for the last tokens.
    
    Args:
        model: The pretrained model
        tokenizer: The tokenizer
        prompt_batch: Batch of input text prompts
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        full_responses: List of complete generated texts
        last_token_states: Hidden states for the last token of each prompt
    """
    # Tokenize all prompts
    prompt_tokens = tokenizer(prompt_batch, return_tensors="pt", padding=True)
    
    # Find positions of last tokens for each prompt
    last_positions = prompt_tokens.input_ids.ne(tokenizer.pad_token_id).sum(dim=1) - 1
    
    # First run: Capture hidden states for prompts only
    with torch.no_grad():
        prompt_outputs = model(**prompt_tokens, output_hidden_states=True)
    
    # Extract and save hidden states for the last token of each prompt
    batch_size = len(prompt_batch)
    last_token_states = [{} for _ in range(batch_size)]
    
    for layer_idx, layer_states in enumerate(prompt_outputs.hidden_states):
        for batch_idx in range(batch_size):
            last_pos = last_positions[batch_idx]
            states = layer_states[batch_idx, last_pos, :].detach().cpu()
            last_token_states[batch_idx][layer_idx] = states
    
    # Second run: Generate full responses
    with torch.no_grad():
        generation_outputs = model.generate(
            prompt_tokens.input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            # Add any other generation parameters you need
        )
    
    # Decode all responses
    full_responses = tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)
    
    return full_responses, last_token_states


def batch_query(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompts: list[str], batch_size=32, max_new_tokens=250, return_hidden_states=False):
    """Process multiple prompts efficiently while saving hidden states."""
    # Store results
    full_responses = []
    all_last_token_states = []
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size)):
        prompt_batch = prompts[i:i+batch_size]
        if return_hidden_states:
            response, states = query_with_hidden_states(
                model, tokenizer, prompt_batch, max_new_tokens
            )
            all_last_token_states.append(states)
        else:
            response = query_model(
                model, tokenizer, prompt_batch, max_new_tokens
            )
        full_responses.append(response)
    
    return full_responses, all_last_token_states

def calculate_accuracy(responses: list[str], labels: list[str]) -> float:
    """Calculate the accuracy of the responses."""
    correct = 0
    correct_instances = []
    total = len(responses)
    for response, label in zip(responses, labels, strict=True):
        response = response.replace("\n", "")
        if label in response.split()[-1]:
            correct += 1
            correct_instances.append((response, label))

    return correct / total, correct_instances


def probe_hidden_states(results, num_layers=7):
    """Probe hidden states to predict response correctness using logistic regression.
    
    Args:
        results: List of tuples containing (response, hidden_state, data, label, correct)
        num_layers: Number of layers in the model
        
    Returns:
        None, displays plot of model performance metrics by layer
    """
    # Prepare data for training per layer
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []
    for layer in range(num_layers):
        X = [result[1][layer] for result in results]
        y = [result[4] for result in results]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(X_train, y_train)

        # Test the model on the test set
        # Accuracy isn't a good metric here since the dataset is imbalanced
        # Instead, we can use precision, recall, and F1 score
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Layer {layer}:")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1}")
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
    
    # Plot the precision, recall, F1 score and accuracy for each layer as bars
    x = range(num_layers)
    width = 0.2

    plt.bar([i - 1.5*width for i in x], precision_scores, width, label='Precision')
    plt.bar([i - 0.5*width for i in x], recall_scores, width, label='Recall')
    plt.bar([i + 0.5*width for i in x], f1_scores, width, label='F1 score')
    plt.bar([i + 1.5*width for i in x], accuracy_scores, width, label='Accuracy')

    plt.xlabel('Layer')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics by Layer')
    plt.legend()
    plt.xticks(x)
    plt.show()

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer("distilgpt2")

    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    batch_size = 32
    data, labels = load_simple_dataset("datasets/simple_test_set/gift_dataset.csv")

    responses, hidden_states = batch_query(model, tokenizer, data[:31], batch_size=batch_size, max_new_tokens=20, return_hidden_states=True)
    # Flatten responses list since it contains batches
    flattened_responses = [r for batch in responses for r in batch]
    flattened_hidden_states = [h for batch in hidden_states for h in batch]

    results = []
    for i, (response, hidden_state) in enumerate(zip(flattened_responses, flattened_hidden_states, strict=True)):
        # Check if the last word is correct and label the response, hidden_state as correct
        if labels[i] in response.split()[-1]:
            results.append((response, hidden_state, data[i], labels[i], 1))
        else:
            results.append((response, hidden_state, data[i], labels[i], 0))


    for result in results:
        print(result[1][6].shape)
        
        # Get the hidden state and ensure it's on the right device
        hidden_state = result[1][6]

        # Get device of model and move tensors there
        with torch.no_grad():
            # Find model device
            device = next(model.parameters()).device
            print(f"Model is on device: {device}")
            
            # Ensure hidden state has the right shape and type
            if hidden_state.dim() == 1:
                hidden_state = hidden_state.unsqueeze(0)
            
            # Move hidden state to the same device as model
            hidden_state = hidden_state.to(device, dtype=torch.float32)
            print(f"Moved hidden state to device: {hidden_state.device}")
            
            # Apply layer norm first (generally recommended before lm_head)
            normalized = model.transformer.ln_f(hidden_state)
            print(normalized.shape)
            
            
            # Approach 2: Manually use the embedding weight matrix transposed
            # Get the embedding weight matrix
            embed_weight = model.transformer.wte.weight
            print(f"Embedding weight shape: {embed_weight.shape}")

            # Set the weights to correct devices
            embed_weight = embed_weight.to(device, dtype=torch.float32)
            # Transpose and apply as the "unembedding" matrix
            # Linear transformation: x @ W.T (matrix multiplication)
            transposed_logits = torch.matmul(normalized, embed_weight.t())
            print(f"Logits shape using transposed embedding: {transposed_logits.shape}")

            
            # Use the direct lm_head output for predictions
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get top prediction
            top_token_id = torch.argmax(probabilities[0]).item()
            top_token = tokenizer.decode(top_token_id)
            print(f"Top predicted token: '{top_token}'")
            
            # Move back to CPU for further processing if needed
            logits_cpu = logits.cpu()
        


