import json
import os
import logging

from datasets import Dataset
from probe_llama import ProbeLlamaModel


class ChessExperiment:
    def __init__(self, input_path: str, output_path: str, chunk_size: int, chunk_id: int):
        self.input_path = input_path
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.chunk_id = chunk_id
        self.logger = logging.getLogger("chess")
        self.prober = self.setup_probe()
        
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

    def run_experiment(self):
        """Collect the activations of the probe model for the chess data."""
        data = self.load_data()
        # Check if the probe is already set up

        if self.prober is None:
            self.setup_probe()

        if self.chunk_id == 0:
            # Add the control prompts to the prompts for the first chunk
            prompts = self._create_control_prompts()
            prompts.extend(self.format_prompts(data))
            dataset = self.prober.process_statements(prompts, self.output_path)
            
            # Add four rows to the start of data with none values (to combine with dataset)
            for _ in range(8):
                data.insert(0, {"mode": "control", "real_world_answer": None, "counter_factual_answer": None, "opening": None}
)

            dataset = self._combine_results_with_input(dataset, data)
        else:
            prompts = self.format_prompts(data)
            dataset = self.prober.process_statements(prompts, self.output_path)
        
        try:
            dataset.save_to_disk(self.output_path)
            self.logger.info(f"Results saved to {self.output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results to {self.output_path}: {e}")
            raise e


    def setup_probe(
        self, 
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        batch_size: int = 8,
        max_new_tokens: int = 2048,
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
            raise ValueError(f"Input path does not exist: {self.input_path}")
        
        # Find a jsonl file in the directory
        jsonl_files = [f for f in os.listdir(self.input_path) if f.endswith('.jsonl')]
        if not jsonl_files:
            self.logger.error(f"No JSONL files found in: {self.input_path}")
            raise FileNotFoundError(f"No JSONL files found in: {self.input_path}")
            
        # Use the first jsonl file found
        file_path = os.path.join(self.input_path, jsonl_files[0])
        self.logger.info(f"Using JSONL file: {file_path}")
            
        # Load the jsonl file
        try:
            with open(file_path, "r") as f:
                data = [json.loads(line) for line in f]
            
            # Apply chunking if specified
            if self.chunk_size > 0 and self.chunk_id >= 0:
                start_idx = self.chunk_id * self.chunk_size
                end_idx = start_idx + self.chunk_size
                data = data[start_idx:end_idx]
                
            return data
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file_path}")
        
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
        def format_prompt(opening: str, mode: str):
            if mode == "real_world":
                user_prompt = f"Is the new opening \"{opening}\" legal? "
            elif mode == "counter_factual":
                user_prompt = f"Under the custom variant, is the new opening \"{opening}\" legal? "
        
            prompt = "### Question: \n" + user_prompt + "\n"
            prompt += "### Answer: \n"
            return prompt
        
        prompts = []

        for line in data:
            user_prompt = format_prompt(line["opening"], line["mode"])
            system_prompt = self._create_system_prompt(line["mode"])
            prompts.append(self.TEMPLATE.format(system_prompt=system_prompt, prompt=user_prompt))


        return prompts
    
    def _create_system_prompt(self, mode, is_control: bool = False):
            if mode not in ["counter_factual", "real_world"]:
                raise ValueError(f"Invalid mode: {mode}")

            system_prompt = "You are a chess player."

            if mode == "counter_factual":
                system_prompt += " You are playing a chess variant where the starting positions for knights and bishops are swapped. For each color, the knights are at placed that where bishops used to be and the bishops are now placed at where knights used to be."

            if not is_control:
                system_prompt += " Given an opening, determine whether the opening is legal. The opening doesn't need to be a good opening. Answer \"\\boxed{yes}\" if all moves are legal. Answer \"\\boxed{no}\" if the opening violates any rules of chess.\n"

            return system_prompt

    def _create_control_prompts(self):
        control_prompts = []
        for mode in ["real_world", "counter_factual"]:
            openings = ["white bishop", "black bishop", "white knight", "black knight"]
            for opening in openings:
                control_prompt = "In this chess variant, t" if mode == "counter_factual" else "T"
                control_prompt += f"he two {opening}s on the board should be initially at which squares?"
                control_prompts.append(self.TEMPLATE.format(system_prompt=self._create_system_prompt(mode, is_control=True), prompt=control_prompt))
        return control_prompts
       