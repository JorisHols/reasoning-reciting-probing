import json
import os
import logging
import random
import re
import torch
from typing import Literal
from datasets import Dataset
from probe_llama import ProbeLlamaModel
from .experiment_base import ExperimentBase


class GSMSymbolicExperiment(ExperimentBase):
    def __init__(self, input_path: str, output_path: str, chunk_size: int, chunk_id: int, model_name: str = "meta-llama/Llama-3.1-8B", sample_size: int = 1000):
        super().__init__(input_path, output_path, chunk_size, chunk_id)
        self.logger = logging.getLogger("chess")
        self.model_name = model_name
        self.sample_size = sample_size
        self.prober = self.setup_probe(model_name=model_name)
        
        # Create the (chunk) output folder if it doesn't exist
        # It will be stored under output/run_id/chunk_id/data
        if self.chunk_id >= 0:
            os.makedirs(self.output_path, exist_ok=True)

        # PROMPT SETTINGS
        self.TEMPLATE  = """As an expert problem solver, solve step by step the following mathematical questions.
        
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

    def run_experiment(self, intervention_vectors: list[torch.Tensor] = None, alpha: float = 0.0, collect_activations: bool = False, seed: int = 8888):
        """Collect the activations of the probe model for the chess data."""
        data = self.load_data(ds_name='GSM-symbolic', sample_size=self.sample_size, seed=seed)
        # Check if the probe is already set up

        if self.prober is None:
            self.setup_probe(model_name=self.model_name)

        prompts = self.format_prompts(data)
        dataset = self.prober.process_statements(
                prompts, 
                self.output_path, 
                intervention_vectors, 
                alpha,
                collect_activations=collect_activations
        )

        for key in data[0].keys():
            dataset = dataset.add_column(key, [data[i][key] for i in range(len(data))])

        try:
            # If we're collecting activations we want to save them for further analysis
            if collect_activations:
                dataset.save_to_disk(self.output_path)
                self.logger.info(f"Results saved to {self.output_path}")
            return dataset
        except Exception as e:
            self.logger.error(f"Error saving results to {self.output_path}: {e}")
            raise e


    def setup_probe(
        self, 
        model_name: str = "meta-llama/Llama-3.1-8B",
        batch_size: int = 4,
        max_new_tokens: int = 200,
        generate_response: bool = True,
    ):
        """Set up the probe model with the specified parameters."""
        self.logger.info(f"Setting up probe with model {model_name}")
        self.prober = ProbeLlamaModel(
            model_name=model_name,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            generate_response=generate_response,
        )
        return self.prober
    
    def parse_llm_response(self, dataset: Dataset):
        """
        Parse the LLM responses for evaluation.
        """
        pattern = "Let's think step by step"

        llm_responses = dataset['llm_response']
        parsed_responses = []
        for model_resp in llm_responses:
            # Find all occurrences of the pattern
            occurrences = [m.start() for m in re.finditer(pattern, model_resp)]

            # Get the ninth occurrence if it exists
            ninth_idx = occurrences[8] if len(occurrences) >= 9 else -1
            if ninth_idx != -1:
                response_after_prompt = model_resp[ninth_idx:]
                match = re.search(r"The final answer is (\d+\.?\d*)", response_after_prompt)
                if match:
                    parsed_responses.append(float(match.group(1)))
                else:
                    self.logger.warning(f"No final answer found in {response_after_prompt}")
                    parsed_responses.append(None)
            else:
                self.logger.warning(f"No ninth occurrence of {pattern} found in {model_resp}")
                parsed_responses.append(model_resp)
        return parsed_responses
         

    def evaluate_llm_responses(self, dataset: Dataset, seed: int = 8888):
        """Evaluate the LLM responses on the chess data.
        
        Returns:
            accuracy: float
            correct_prediction_indices: list[int]
            incorrect_prediction_indices: list[int]
        """
        predictions = self.parse_llm_response(dataset)
        data = self.load_data(ds_name='GSM-symbolic', sample_size=self.sample_size, seed=seed)
        correct_prediction_indices = []
        incorrect_prediction_indices = []
        total_predictions = len(predictions)
        for i, (entry, prediction) in enumerate(zip(data, predictions)):
            if entry["final_answer"] == prediction:
                entry['model_predict_correctness'] = True
                correct_prediction_indices.append(i)
            else:
                entry['model_predict_correctness'] = False
                incorrect_prediction_indices.append(i)

        accuracy = len(correct_prediction_indices) / total_predictions
        return accuracy, correct_prediction_indices, incorrect_prediction_indices


    
    def load_data(self, ds_name, sample_size: int, seed: int) -> list[dict]:
        """Load GSM data from a jsonl file at the input path and return a list of dictionaries."""
        def extract_final_answer(model_resp: str) -> float:
            # Remove commas so for example 5,000 becomes 5000
            model_resp = model_resp.replace(",", "")
            # Find the last number
            extracted_num = re.findall(r"-?\d+\.?\d*", model_resp)[-1]
            # Use float to ensure 3.0 and 3 are the same.
            return float(extracted_num)
        
        ds_data = []
        if ds_name == 'GSM-symbolic':
            with open(os.path.join(self.input_path, f'GSM_symbolic.jsonl'), 'r') as file:
                for line in file:
                    ds_data.append(json.loads(line))  # 解析每一行的JSON对象

            for entry in ds_data:
                entry['final_answer'] = extract_final_answer(model_resp=entry['answer'])

        if sample_size is not None:
            # Randomly sample the data
            random.seed(8888)
            ds_data = random.sample(ds_data, sample_size)

        return ds_data


    def format_prompts(self, data: list):
        """Format the prompts for the probe model."""
        def format_prompt(question: str):
            prompt = self.TEMPLATE.format(TARGET_QUESTION=question)
            return prompt
        
        prompts = []

        for line in data:
            prompt = format_prompt(line["question"])
            prompts.append(prompt)
    
        return prompts

    

        
