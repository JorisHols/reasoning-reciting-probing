from transformers import AutoModelForCausalLM, AutoTokenizer
import os
def load_model_and_tokenizer(model_name: str):
    """
    Load the model from the model name.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
    print(type(model))
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
    return model, tokenizer


def query_batch(model_name: str, prompts: list[str] = [], temperature=0, n=1):
    """
    Query the model in batch.
    """
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Generate the responses
    outputs = model.generate(prompts, max_new_tokens=100, temperature=temperature, num_return_sequences=n)

    # Decode the responses
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return responses

    # # Tokenize the prompts
    # inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    # # Generate the responses
    # outputs = model.

    # # Decode the responses
    # responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # return responses

if __name__ == "__main__":
    query_batch("distilgpt2")
