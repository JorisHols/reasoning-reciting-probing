import json
import os
import logging
import torch
import random
import string
from typing import Literal

from datasets import Dataset
from probe_llama import ProbeLlamaModel
from .experiment_base import ExperimentBase

class ChessExperiment(ExperimentBase):
    def __init__(self, input_path: str, output_path: str, chunk_size: int, chunk_id: int, model_name: str = "meta-llama/Llama-3.1-8B", seed: int = 8888):
        super().__init__(input_path, output_path, chunk_size, chunk_id)
        self.logger = logging.getLogger("chess")
        # self.prober = self.setup_probe(model_name=model_name)
        self.model_name = model_name
        self.seed = seed
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

    def run_experiment(self, intervention_vectors: list[torch.Tensor] = None, alpha: float = 0.0, attach_control_prompts: bool = True, collect_activations: bool = True, gibberish = False, save_activations: bool = False):
        """Collect the activations of the probe model for the chess data."""
        data = self.load_data()
        # Check if the probe is already set up

        if self.prober is None:
            self.setup_probe(model_name=self.model_name)

        if self.chunk_id == 0 and attach_control_prompts:
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
            if gibberish:
                # Use th rw data to create gibberish examples
                cf_data, rw_data = [d for d in data if d['mode'] == 'counter_factual'], [d for d in data if d['mode'] == 'real_world']
                prompts = self.format_prompts(rw_data, gibberish=True)
                prompts.extend(self.format_prompts(cf_data, gibberish=False))
            else:
                prompts = self.format_prompts(data)

            dataset = self.prober.process_statements(
                prompts, 
                self.output_path, 
                intervention_vectors, 
                alpha,
                collect_activations=collect_activations
            )
        
        try:
            # If we're collecting activations we want to save them for further analysis
            if save_activations:
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
    
    def parse_llm_response(self, dataset: Dataset):
        """
        Parse the LLM responses and evaluate the LLM responses on the chess data.
        Returns the indices of the yes, no, and invalid responses.
        """
        # Parse the LLM responses
        llm_responses = dataset['llm_response']
        yes_indices = []
        no_indices = []
        invalid_indices = []
        for i in range(len(llm_responses)):
            response = llm_responses[i].lower()
            # Extract the answer part
            answer_part = None
            sep = "answer: "
            if sep in response:
                answer_part = response.split(sep)[-1].strip()
            else:
                self.logger.warning("Warning: Could not identify answer part in LLM response")
                invalid_indices.append(i)
                continue
            # Look for boxed yes/no
            has_yes = "yes" in answer_part
            has_no = "no" in answer_part if "benoni" not in answer_part else False
            
            # Check if both yes and no are present
            answer_part = answer_part.replace(".", " ")
            answer_part = answer_part.strip()
            last_line = answer_part.split("\n")[-1]
            success = False
            if has_yes and has_no:
                self.logger.warning(f"Warning: Both \\boxed{{yes}} and \\boxed{{no}} found in answer: {answer_part[:100]}...")
                invalid_indices.append(i)
                print(answer_part)
                continue

                
            if has_yes:
                yes_indices.append(i)
                success = True
            elif answer_part.endswith('is legal'):
                yes_indices.append(i)
                success = True
            elif 'are legal' in last_line or 'is legal' in last_line:
                yes_indices.append(i)
                success = True

            if has_no:
                no_indices.append(i)
                success = True
            elif answer_part.endswith('illegal') or answer_part.endswith('not legal'):
                no_indices.append(i)
                success = True
            elif 'not legal' in last_line or 'illegal' in last_line or 'not valid' in last_line or 'not a valid' in last_line:
                no_indices.append(i)
                success = True

            if not success:
                self.logger.warning("Warning: Could not identify an answer")
                print(answer_part)
                invalid_indices.append(i)

        evaluated_answer_count = len(yes_indices) + len(no_indices) + len(invalid_indices)
        if evaluated_answer_count != len(llm_responses):
            self.logger.warning("Warning: some answers seem to be ambigous as total evaluated answer length" 
                f"{evaluated_answer_count} does not match the dataset length {len(dataset)}"
                )
        print(len(llm_responses), len(yes_indices), len(no_indices), len(invalid_indices))
        return yes_indices, no_indices, invalid_indices
         

    def evaluate_llm_responses(self, dataset: Dataset):
        """Evaluate the LLM responses on the chess data."""
        yes_indices, no_indices, invalid_indices = self.parse_llm_response(dataset)
        # Ensure no overlap between yes_indices, no_indices, and invalid_indices
        assert len(set(yes_indices) & set(no_indices)) == 0
        assert len(set(yes_indices) & set(invalid_indices)) == 0
        assert len(set(no_indices) & set(invalid_indices)) == 0

        data =  self.load_data()
        results = {}
        # Create results dict:
        modes = [
        ('real_world', 'real_world_answer'),
        ('counter_factual', 'counter_factual_answer')
        ]

        for mode, answer_key in modes:
            results[mode] = {
                'correct': {
                    'yes':[],
                    'no':[],
                },
                'incorrect': {
                    'yes':[],
                    'no':[],
                    'invalid':[],
                }
            }

        for i, sample in enumerate(data):
            for mode, answer_key in modes:
                if sample['mode'] == mode:
                    answer = sample[answer_key]
                    if answer:
                        if i in yes_indices:
                            results[mode]['correct']['yes'].append(i)
                        elif i in no_indices:
                            results[mode]['incorrect']['no'].append(i)
                        else:
                            results[mode]['incorrect']['invalid'].append(i)
                    else:
                        if i in yes_indices:
                            results[mode]['incorrect']['yes'].append(i)
                        elif i in no_indices:
                            results[mode]['correct']['no'].append(i)
                        else:
                            results[mode]['incorrect']['invalid'].append(i)

        correct_prediction_indices = []
        incorrect_prediction_indices = []
        for mode, _ in modes:
            correct_prediction_indices.extend(results[mode]['correct']['yes'])
            correct_prediction_indices.extend(results[mode]['correct']['no'])
            incorrect_prediction_indices.extend(results[mode]['incorrect']['yes'])
            incorrect_prediction_indices.extend(results[mode]['incorrect']['no'])
            incorrect_prediction_indices.extend(results[mode]['incorrect']['invalid'])

        return results, correct_prediction_indices, incorrect_prediction_indices
    
    def get_chess_accuracies(self, results: dict):
        """Get the accuracies of the chess data."""
        modes = [
        ('real_world', 'real_world_answer'),
        ('counter_factual', 'counter_factual_answer')
        ]
        num_rw_instances = sum(len(results['real_world'][key1][key2]) for key1 in results['real_world'].keys() for key2 in results['real_world'][key1].keys())
        num_counter_factual_instances = sum(len(results['counter_factual'][key1][key2]) for key1 in results['counter_factual'].keys() for key2 in results['counter_factual'][key1].keys())
        if num_rw_instances == 0:
            num_rw_instances = 400
        if num_counter_factual_instances == 0:
            num_counter_factual_instances = 400

        rw_accuracy = (len(results['real_world']['correct']['yes']) + len(results['real_world']['correct']['no'])) / num_rw_instances
        counter_factual_accuracy = (len(results['counter_factual']['correct']['yes']) + len(results['counter_factual']['correct']['no'])) / num_counter_factual_instances

        return rw_accuracy, counter_factual_accuracy, num_rw_instances, num_counter_factual_instances
        
    
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


    def format_prompts(self, data: list , gibberish: bool = False):
        """Format the prompts for the probe model."""
        def format_prompt(opening: str, mode: str):
            if mode == "real_world":
                user_prompt = f"Is the new opening \"{opening}\" legal? "
            elif mode == "counter_factual":
                user_prompt = f"Under the custom variant, is the new opening \"{opening}\" legal? "
        
            prompt = "### Question: \n" + user_prompt + "\n"
            prompt += "### Answer: \n"
            return prompt
        
        def format_gibberish_prompt(opening: str, mode: str):            
            def generate_random_gibberish_opening(seed_value):
                """Generate a random gibberish opening with 20 random characters"""
                random.seed(seed_value)
                # Mix of letters, numbers, and some chess-like symbols
                chars = string.ascii_letters + string.digits + "+-=()[]{}*"
                return ''.join(random.choice(chars) for _ in range(20))
            
            def generate_random_label(seed_value):
                """Generate a random Yes/No label"""
                random.seed(seed_value + 12345)  # Different seed for labels
                return random.choice([True, False])
            
            def create_gibberish_examples(num_examples=8):
                """Create random gibberish chess opening examples"""
                examples = []
                
                # Use the opening string as base for deterministic randomness
                base_seed = hash(opening) % 1000000
                
                for i in range(num_examples):
                    seed_val = base_seed + i * 7  # Different seed for each example
                    
                    gibberish_opening = generate_random_gibberish_opening(seed_val)
                    label = generate_random_label(seed_val)
                    
                    examples.append({
                        'opening': gibberish_opening,
                        'label': 'Yes' if label else 'No',
                    })
                
                return examples
            
            # Convert the actual opening to gibberish too
            gibberish_target_opening = generate_random_gibberish_opening(hash(opening) % 1000000 + 999)

            prompt = "You are a chess player."
            if mode == "counter_factual":
                prompt += (" You are playing a chess variant where the starting positions"
                          " for knights and bishops are swapped. For each color, the knights"
                          " are at placed that where bishops used to be and the bishops"
                          " are now placed at where knights used to be.")
            
            prompt += " Given an opening, determine whether the opening is legal. The opening doesn't need to be a good opening."
            prompt += "\n\nExamples:\n"

            # Generate gibberish examples
            examples = create_gibberish_examples(8)
            
            for example in examples:
                prompt += f"Opening: {example['opening']}\n"
                prompt += f"Answer: {example['label']}\n\n"

            prompt += f"Opening: {gibberish_target_opening}\n"
            prompt += "Answer: "
            return prompt
        
        def format_llama_base_prompt(opening: str, mode: str, examples: list):
            prompt = "You are a chess player."
            if mode == "counter_factual":
                prompt += (" You are playing a chess variant where the starting positions"
                          " for knights and bishops are swapped. For each color, the knights"
                          " are at placed that where bishops used to be and the bishops"
                          " are now placed at where knights used to be.")
            #prompt += "Only answer 'Yes' or 'No' when asked about move legality. "
            prompt += "Given an opening, determine whether the opening is legal. The opening doesn't need to be a good opening."
            prompt += "\n\nExamples:\n"
            for example in examples:
                prompt += f"Opening: {example['opening']}\n"
                if example['mode'] == "counter_factual":
                    prompt += f"Answer: {'Yes' if example['counter_factual_answer'] else 'No'}\n\n"
                else:
                    prompt += f"Answer: {'Yes' if example['real_world_answer'] else 'No'}\n\n"
            prompt += f"Opening: {opening}\n"
            prompt += "Answer: "
            return prompt
     
        def sample_examples(current_opening: str, data: list):
            """Sample 8 examples for incontext prompting"""
            # Sample 4 examples for incontext prompting (but none of the examples should match the current line opening)
            random.seed(self.seed)
            examples = random.sample(data, 8)
            # Make sure none of the examples match the current line opening
            while any(example["opening"] == current_opening for example in examples):
                examples = random.sample(data, 8)
            return examples
        
        prompts = []

        for line in data:
            if 'Instruct' in self.model_name:
                # Sample 4 examples for incontext prompting (but none of the examples should match the current line opening)
                user_prompt = format_prompt(line["opening"], line["mode"])
                system_prompt = self._create_system_prompt(line["mode"])
                prompts.append(self.TEMPLATE.format(system_prompt=system_prompt, prompt=user_prompt))
            else:
                if gibberish:
                    prompt = format_gibberish_prompt(line["opening"], line["mode"])
                else:
                    examples = sample_examples(line["opening"], data)
                    prompt = format_llama_base_prompt(line["opening"], line["mode"], examples)
                prompts.append(prompt)


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
    

    def run_intervention_study(
        self,
        intervention_type: Literal["ablation", "addition"] = "addition",
        invervention_vector_path: str = None,
        alpha: float = 0.0,
    ):
        """Run the intervention study."""
        # If input path is not provided, use the input path of the experiment
        if invervention_vector_path is None:
            raise ValueError("Intervention vector path is required")
        
        intervention_vectors = self._load_intervention_vectors(invervention_vector_path)
        data = self.load_data()
        prompts = self.format_prompts(data)
        dataset = self.prober.process_intervention(
            prompts,
            intervention_vectors,
            intervention_type,
            alpha,
        )
        dataset.save_to_disk(self.output_path)
        self.logger.info(f"Intervention study results saved to {self.output_path}")

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