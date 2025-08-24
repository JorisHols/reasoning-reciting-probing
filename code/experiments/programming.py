"""Based on the Reasoning or Reciting paper https://github.com/ZhaofengWu/counterfactual-evaluation/"""
import logging
import os
import json
import traceback
from typing import Literal
import torch
from datasets import Dataset
from .experiment_base import ExperimentBase
from utils.programming.programming_utils import one_based_indexing_unit_tests, remove_type_hints, eval_program_with_calls
from utils.programming.human_eval.evaluation import evaluate_functional_correctness


class ProgrammingExperiment(ExperimentBase):
    def __init__(self, input_path: str, output_path: str, chunk_size: int, chunk_id: int, mode: Literal['counter_factual', 'real_world'], model_name: str = "meta-llama/Llama-3.1-8B"):
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        super().__init__(input_path, output_path, chunk_size, chunk_id, model_name)

    def run_experiment(self, alpha: float = 0.0, intervention_vectors: list[torch.Tensor] = None, collect_activations: bool = False):
        """Run the experiment."""
        if self.prober is None:
            self.setup_probe(
                model_name=self.model_name,
                max_new_tokens=512
            )

        data = self.load_data()
        prompts = self.format_prompts(data, mode=self.mode)
        
        dataset = self.prober.process_statements(
            prompts=prompts, 
            intervention_vectors=intervention_vectors, 
            alpha=alpha,
            collect_activations=collect_activations
        )
        # Rename prompt to human_eval_prompt to avoid duplicate column error
        for i, d in enumerate(data):
            d["human_eval_prompt"] = d['prompt']
        for d in data:
            if "prompt" in d:
                d.pop("prompt", None)
       
        dataset = self._combine_results_with_input(dataset, data)

        try:
            if collect_activations:
                dataset.save_to_disk(self.output_path)
                self.logger.info(f"Results saved to {self.output_path}")
                return dataset
            else:
                return dataset
        except Exception as e:
            self.logger.error(f"Error saving results to {self.output_path}: {e}")
            raise e

    def load_data(self):
        """Load the data from the input path."""
        # Check if the data.jsonl file exists at the input path
        data_file = os.path.join(self.input_path, 'data.jsonl')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file does not exist: {data_file}")
        
        # Load the data
        with open(data_file, 'r') as f:
            data = [json.loads(line) for line in f]

        # Only keep instances that would fail with 1-based indexing 
        print(f"original # instances {len(data)}")
        filtered_data = []
        for obj in data:
            program = remove_type_hints(
                obj["prompt"]
                + obj["canonical_solution"]
                + obj["test"]
                + f'\n\ncheck({obj["entry_point"]})'
            )
            try:
                eval_program_with_calls(program, perturbation="one_based_indexing", return_output=False)
            except ZeroDivisionError:
                filtered_data.append(obj)
            except AssertionError:
                trace = traceback.format_exc()
                assert trace.strip().split("\n")[-2].endswith("in check")
                filtered_data.append(obj)
        # TODO: Uncomment this to run the filtered data
        #data = filtered_data
        print(f"filtered # instances {len(data)}")

        # Return based on chunk_id
        return data[self.chunk_id * self.chunk_size:(self.chunk_id + 1) * self.chunk_size]
        
    def format_prompts(self, data, mode: str):
        """Format the prompts."""        
        def format_prompt(orig_prompt: str, mode: str):
            if mode == 'counter_factual':
                prompt = f"""You are an expert programmer who can readily adapt to new programming languages. \
            There is a new programming language, ThonPy, which is identical to Python 3.7 except all variables \
            of the `list`, `tuple`, and `str` types use 1-based indexing, like in the MATLAB and R languages, where sequence indices start from 1. \
            That is, index `n` represents the `n`-th element in a sequence, NOT the `n+1`-th as in 0-based indexing. \
            This change only affects when the index is non-negative. \
            When the index is negative, the behavior is the same as Python 3.7. \
            This also affects methods of these classes such as `index` and `pop`. \
            The built-in functions `enumerate` and `range` also use 1-based indexing: by default, the index of \
            `enumerate` starts from 1, and so does the lower bound of `range` when not supplied (the higher bound is unchanged).

            For example,
            ```thonpy
            {one_based_indexing_unit_tests()}
            ```
            Complete the following function in ThonPy. Please only output the code for the completed function.

            {orig_prompt}"""
            elif mode == 'real_world':
                prompt = f"""You are an expert programmer. Complete the following function in Python 3.7. Please only output the code for the completed function.

            {orig_prompt}"""
            return prompt
        
        prompts = [format_prompt(item['prompt'], mode) for item in data]
        return prompts
                

    def evaluate_llm_responses(self, dataset: Dataset):
        predictions = []
        for example in dataset:
            # Get the number of times """ occours in the human_eval_prompt
            double_quotes = True
            num_quotes = example["human_eval_prompt"].count('"""')
            if num_quotes == 0:
                double_quotes = False
                num_quotes = example["human_eval_prompt"].count("'''")
                if num_quotes == 0:
                    raise("ERROR!!!: No \"\"\" or ''' in the human_eval_prompt")
            # Split the llm response after the num_quotes-th """
            if double_quotes:
                completion = example["llm_response"].split('"""')[num_quotes]
            else:
                completion = example["llm_response"].split("'''")[num_quotes]
            # idx = example["llm_response"].rfind(example["human_eval_prompt"])
            # completion = example["llm_response"][idx + len(example["human_eval_prompt"]):]
            # if idx != -1:
            prediction = {
                    "task_id": example["task_id"],
                    "prompt": example["human_eval_prompt"],
                    "completion": completion
            }
            predictions.append(prediction)

        data = self.load_data()

        pass_at_k_result, results = evaluate_functional_correctness(
                predictions,
                index_from=1 if self.mode == 'counter_factual' else 0,
                k=[1],
                problems={example["task_id"]: example for example in data},
                return_results=True
            )
        
        # Map the task_id to the index in data
        task_id_to_index = {d["task_id"]: i for i, d in enumerate(data)}

        correct_prediction_indices = []
        incorrect_prediction_indices = []

        for task_id, result_list in results.items():
            data_index = task_id_to_index[task_id]
            
            # Check if any result passed for this task_id
            passed = any(r[1]["passed"] for r in result_list)
            
            if passed:
                correct_prediction_indices.append(data_index)
            else:
                incorrect_prediction_indices.append(data_index)
                
        return pass_at_k_result, correct_prediction_indices, incorrect_prediction_indices
