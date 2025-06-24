"""Class for collecting the dat (activations and intervention responses) 
for the arithmetic experiment. Further analysis is done in a notebook"""

import json
import os
import logging
import torch
from typing import Literal

from datasets import Dataset
from probe_llama import ProbeLlamaModel
from .experiment_base import ExperimentBase
from utils.arithmetic_utils import get_label, parse_output


class ArithmeticExperiment(ExperimentBase):
    def __init__(self, input_path: str, output_path: str, chunk_size: int, chunk_id: int, base: int = 10):
        super().__init__(input_path, output_path, chunk_size, chunk_id)
        self.logger = logging.getLogger("arithmetic")
        self.prober = self.setup_probe()
        self.base = base
        
        # Create the (chunk) output folder if it doesn't exist
        # It will be stored under output/run_id/chunk_id/data
        if self.chunk_id >= 0:
            os.makedirs(self.output_path, exist_ok=True)

        # PROMPT SETTINGS
        self.TEMPLATE = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    def run_experiment(self, intervention_vectors: list[torch.Tensor] = None, alpha: float = 0.0, collect_activations: bool = False):
        """Collect the activations of the probe model for the chess data."""
        data = self.load_data()
        # Check if the probe is already set up

        if self.prober is None:
            self.setup_probe()

        # if self.chunk_id == 0:
        #     # Add the control prompts to the prompts for the first chunk
        #     #prompts = self._create_control_prompts()
        #     prompts.extend(self.format_prompts(data))
        #     dataset = self.prober.process_statements(prompts, self.output_path)

        # else:
        prompts = self.format_prompts(data)
        dataset = self.prober.process_statements(
            prompts, 
            self.output_path, 
            intervention_vectors, 
            alpha
        )
        
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


    def setup_probe(
        self, 
        model_name: str = "meta-llama/Llama-3.1-8B",
        batch_size: int = 4,
        max_new_tokens: int = 10,
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

    def load_data(self):
        """Load data from a jsonl file at the input path."""
        # Check if the input path exists
        if not os.path.exists(self.input_path):
            raise ValueError(f"Input file does not exist: {self.input_path}")
        
        self.logger.info(f"Reading data from: {self.input_path}")
        
        # Load the txt file
        try:
            with open(self.input_path, "r") as f:
                lines = f.read().strip().split('\n')
                data = [{"expr": line, "base": self.base} for line in lines if line.strip()]
            
            # Apply chunking if specified
            if self.chunk_size > 0 and self.chunk_id >= 0:
                start_idx = self.chunk_id * self.chunk_size
                end_idx = start_idx + self.chunk_size
                data = data[start_idx:end_idx]
                
            return data
        except json.JSONDecodeError:
            raise ValueError(f"Invalid TXT format in file: {self.input_path}")
        
    def _combine_results_with_input(self, dataset: Dataset, data: list):
        """Combine the results with the input data."""
        new_columns = {k: [] for k in data[0].keys()}
        for line in data:
            for k, v in line.items():
                new_columns[k].append(v)

        # Add the new columns to the dataset
        for k, v in new_columns.items():
            dataset = dataset.add_column(k, v)

        return dataset


    def format_prompts(self, data: list):
        """Format the prompts for the probe model."""
        def format_prompt(expr: str):
            user_prompt = f"What is {expr}? End the response with the result in \"\\boxed{{result}}\"."
            prompt = "### Question: \n" + user_prompt + "\n"
            prompt += "### Answer: \n"
            return prompt
        
        def format_prompt_llama_base(expr: str):
            # Prompt adapted from https://arxiv.org/pdf/2502.00873
            prompt = f"The following is a correct addition problem.\n{expr}="
            return prompt
        
        prompts = []

        for line in data:
            system_prompt = self._create_system_prompt(line["base"])
            if self.prober.model_name == "meta-llama/Llama-3.1-8B":
                user_prompt = format_prompt_llama_base(line["expr"])
                final_prompt = system_prompt + "\n" + user_prompt
                prompts.append(final_prompt)

            else:
                user_prompt = format_prompt(line["expr"])
                prompts.append(self.TEMPLATE.format(system_prompt=system_prompt, prompt=user_prompt))


        return prompts
    
    def _create_system_prompt(self, base):
            digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            system_prompt = f"You are a mathematician. Assuming that all numbers are in base-{base}, where the digits are \"{digits[:base]}\"."

            return system_prompt

    # def _create_control_prompts(self, expr: str):
    #     return f"What is the next number after {expr}? Do this by counting the few preceding numbers and completing the sequence. End the response with the result."
    

    def run_intervention_study(
        self,
        intervention_type: Literal["ablation", "addition"] = "addition",
        intervention_vectors: list[torch.Tensor] = None,
        intervention_vector_path: str = None,
        alpha: float = 0.0,
        exclude_indices: list[int] = None,
    ):
        """Run the intervention study."""
        # If input path is not provided, use the input path of the experiment
     
        
        if intervention_vectors is None:
            if intervention_vector_path is None:
                raise ValueError("Intervention vector path is required if intervention vectors are not provided")
            intervention_vectors = self._load_intervention_vectors(intervention_vector_path)

        data = self.load_data()
        if exclude_indices is not None:
            data = [data[i] for i in range(len(data)) if i not in exclude_indices]
        prompts = self.format_prompts(data)
        dataset = self.prober.process_intervention(
            prompts,
            intervention_vectors,
            intervention_type,
            alpha,
        )
        dataset.save_to_disk(self.output_path)
        self.logger.info(f"Intervention study results saved to {self.output_path}")
        return dataset

    def _load_intervention_vectors(self, input_path):
        # Check if the intervention vector path exists
        if not os.path.exists(input_path):
            raise ValueError(
                f"Intervention vector path does not exist: {input_path}"
            )
        
        # Load the intervention vectors
        with open(input_path, "r") as f:
            intervention_vectors = json.load(f)

        if 'liref' in input_path:
            intervention_vectors = [
                torch.tensor(intervention_vectors[k]) for k in intervention_vectors.keys()
            ]
        else:
            intervention_vectors = [
                torch.tensor(intervention_vectors[k]['weights']) for k in intervention_vectors.keys()
            ]
        return intervention_vectors
    
    def evaluate_llm_responses(self, dataset: Dataset):
        llm_responses = dataset['llm_response']
        expressions = dataset['expr']
        correct_answer_indices = []
        real_world_answer_indices = []
        unparsable_answer_indices = []
        for i in range(len(llm_responses)):
            try:
                correct_response = get_label(expressions[i], self.base)
                real_world_response = None
                if self.base < 10:
                    real_world_response = get_label(expressions[i], 10)
                
                llm_response = llm_responses[i].split("=")[1].strip()
                llm_response = llm_response.split("\n")[0].strip()
                pred = parse_output(llm_response).upper()


                if pred == correct_response:
                    correct_answer_indices.append(i)
                elif real_world_response is not None and pred == real_world_response:
                    real_world_answer_indices.append(i)
                else:
                    unparsable_answer_indices.append(i)
            except Exception as e:
                print(llm_responses[i])

        accuracy = len(correct_answer_indices) / len(llm_responses)
        return accuracy, correct_answer_indices, real_world_answer_indices, unparsable_answer_indices
