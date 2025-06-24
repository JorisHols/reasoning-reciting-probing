import json
import os
import logging
from probe_llama import ProbeLlamaModel
from datasets import Dataset
from .experiment_base import ExperimentBase


class KKExperiment(ExperimentBase):
    '''
    A class to collect the activations and responses for the K and K experiment.
    '''

    def __init__(self, input_path: str, output_path: str, 
                 chunk_size: int = None, chunk_id: int = None):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.logger = logging.getLogger("k_and_k")
        
        # Chunking parameters
        self.chunk_id = chunk_id
        self.chunk_size = chunk_size  # Number of examples per chunk

        self.prober = self.setup_probe()
        
        # The perturbed versions of the problems to use for the experiment (must match the folder names)
        self.perturbed_versions = [
            "perturbed_leaf",
            "perturbed_statement"
            # "flip_role",
            # "random_pair",
            # "reorder_statement",
            # "uncommon_name"
        ]

        # MODEL AND PROBING SETTINGS


        # PROMPT SETTINGS
        self.TEMPLATE = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

        self.SYSTEM_PROMPT = '''Your task is to solve a logical reasoning problem. You are given set of statements from which you must logically deduce the identity of a set of characters.

        You must infer the identity of each character. First, explain your reasoning. At the end of your answer, you must clearly state the identity of each character by following the format:

        CONCLUSION:
        (1) ...
        (2) ...
        (3) ...
        '''

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


    def run_experiment(self, intervention_vectors=None, alpha: float = 0.0, collect_activations: bool = True):
        """Run the K&K experiment."""
        # Load the data
        self.logger.info(f"Loading data from {self.input_path}")
        data = self.load_data()

        if not hasattr(self, 'prober') or self.prober is None:
            # Log that the probe model is not initialized and that you're using default settings
            self.logger.warning("Probe model not initialized. Using default settings.")
            self.prober = self.setup_probe()

        # Run the probe on the clean problems only
        
        self.logger.info(f"Processing {len(data)} problems")
        prompts = self.format_prompts(data)
                
        # Create a chunk-specific output path if we're in chunk mode
        self.logger.info(f"Chunk output path: {self.chunk_id}")
        if self.chunk_id is not None:
            os.makedirs(self.output_path, exist_ok=True)
                
            # Process statements with the chunk-specific output path
            results = self.prober.process_statements(
                    prompts=prompts,
                    output_file_path=self.output_path
            )
            self.logger.debug(f"LLM responses: {results['llm_response']}")
            
            return results
        else:
            results = self.prober.process_statements(
                prompts=prompts,
                output_file_path=self.output_path
            )
            self.logger.debug(f"LLM responses: {results['llm_response']}")
            
            return results

    def _load_jsonl_file(self, file_dir):
        """
        Helper function to load a JSONL file.
        
        Args:
            file_path (str): Path to the JSONL file
            dataset_name (str): Name of the dataset for logging purposes
            
        Returns:
            list: List of JSON objects from the file
        """
        items = []
        
        # Check if file_dir exists
        if not os.path.exists(file_dir):
            self.logger.error(f"Directory not found: {file_dir}")
            raise FileNotFoundError(f"Directory not found: {file_dir}")
            
        # Find a jsonl file in the directory
        jsonl_files = [f for f in os.listdir(file_dir) if f.endswith('.jsonl')]
        if not jsonl_files:
            self.logger.error(f"No JSONL files found in: {file_dir}")
            raise FileNotFoundError(f"No JSONL files found in: {file_dir}")
            
        # Use the first jsonl file found
        file_path = os.path.join(file_dir, jsonl_files[0])
        self.logger.info(f"Using JSONL file: {file_path}")
        
        try:
            with open(file_path, "r") as f:
                # Parse JSONL format - each line is a separate JSON object
                all_items = []
                for line_num, line in enumerate(f, 1):
                    if line.strip():  # Skip empty lines
                        try:
                            item = json.loads(line)  # Use json.loads for individual lines
                            all_items.append(item)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")
                
                # Apply chunking if specified
                if self.chunk_id is not None and self.chunk_size is not None:
                    start_idx = self.chunk_id * self.chunk_size
                    end_idx = min(start_idx + self.chunk_size, len(all_items))
                    
                    if start_idx >= len(all_items):
                        self.logger.warning(f"Chunk {self.chunk_id} starts beyond available data")
                        items = []
                    else:
                        items = all_items[start_idx:end_idx]
                        self.logger.info(f"Loaded chunk {self.chunk_id}: items {start_idx} to {end_idx-1} " +
                                       f"({len(items)} items out of {len(all_items)} total)")
                else:
                    items = all_items
        
        except Exception as e:
            raise e
        
        return items

    def load_data(self):
        '''
        Load the data from the given input path.
        
        The function accepts a direct path to the input directory and loads the data,
        applying chunking if specified.
        
        Expected structure:
            - input_path/
                - clean/
                    - clean_people3_num5000.jsonl
        
        The file contains JSONL (JSON Lines) where each line is a valid JSON object.
        '''
        if not os.path.exists(self.input_path):
            self.logger.error(f"Path not found: {self.input_path}")
            raise FileNotFoundError(f"Directory {self.input_path} not found")
        
        
        self.logger.info(f"Loading data from: {self.input_path}")
        data = self._load_jsonl_file(self.input_path)
        return data


    def format_prompts(self, data: list):
        def format_prompt(user_prompt: str):
            prompt = "### Question: \n" + user_prompt + "\n"
            prompt += "### Answer: \n"
            return self.TEMPLATE.format(system_prompt=self.SYSTEM_PROMPT, prompt=prompt)
        
        prompts = []
        for problem in data:
            prompts.append(format_prompt(problem['quiz']))
        return prompts
    
    # def evaluate_results(self, result_dataset: Dataset, original_dataset: Dataset = None):
    #     """
    #     Evaluate the results of the probe by parsing the LLM responses and comparing 
    #     with the ground truth solutions.
    #     """
    #     # Set up tracking variables
    #     correct_count = 0
    #     total_count = 0
    #     results = []
        
    #     # Define patterns for parsing responses
    #     conclusion_patterns = ['CONCLUSION:', 'conclusion:']
    #     finish_patterns = ["### Reason", "Let's think step by step again", 
    #                     "let's go back and check", "###"]
        
    #     # Validate input
    #     if not original_dataset and 'solution' not in result_dataset.features:
    #         self.logger.error("No solution data available for evaluation")
    #         raise ValueError(
    #             "Either original_dataset must be provided or "
    #             "result_dataset must contain solution field"
    #         )
        
    #     # Iterate through responses
    #     for idx, response in enumerate(result_dataset['llm_response']):
    #         if not response:
    #             # Skip empty responses
    #             results.append({
    #                 'correct': False,
    #                 'parsed_prediction': "",
    #                 'reason': "empty_response",
    #                 'correct_ratio': 0.0
    #             })
    #             continue
            
    #         # Get ground truth from either the result_dataset or original_dataset
    #         if original_dataset:
    #             solution = (
    #                 original_dataset[idx]['solution_text_format'] 
    #                 if 'solution_text_format' in original_dataset.features 
    #                 else original_dataset[idx]['solution']
    #             )
    #         else:
    #             solution = (
    #                 result_dataset[idx]['solution'] 
    #                 if 'solution' in result_dataset.features 
    #                 else None
    #             )
            
    #         if not solution:
    #             self.logger.warning(f"No solution found for example {idx}")
    #             continue
            
    #         # Parse and evaluate the response
    #         is_correct, parsed_pred, wrong_reason, correct_ratio, reformat_gold = (
    #             self.parse_cot_eval(
    #                 response, solution,
    #                 conclusion_patterns=conclusion_patterns,
    #                 finish_patterns=finish_patterns
    #             )
    #         )
            
    #         # Update results
    #         if is_correct:
    #             correct_count += 1
    #         total_count += 1
            
    #         # Store detailed results
    #         results.append({
    #             'correct': is_correct,
    #             'parsed_prediction': parsed_pred,
    #             'reason': wrong_reason if not is_correct else "correct",
    #             'correct_ratio': correct_ratio,
    #             'gold_conditions': reformat_gold
    #         })
            
    #         # Log progress occasionally
    #         if idx % 10 == 0:
    #             self.logger.info(
    #                 f"Processed {idx}/{len(result_dataset['llm_response'])} responses, "
    #                 f"accuracy so far: {correct_count/(idx+1):.3f}"
    #             )
        
    #     # Calculate and report overall accuracy
    #     accuracy = correct_count / total_count if total_count > 0 else 0
    #     self.logger.info(
    #         f"Overall accuracy: {accuracy:.3f}, {correct_count}/{total_count} correct"
    #     )
        
    #     # Return the evaluation results
    #     return {
    #         'accuracy': accuracy,
    #         'correct_count': correct_count,
    #         'total_count': total_count,
    #         'detailed_results': results
    #     }

    # def parse_cot_eval(self, pred_str, ans,
    #                    conclusion_patterns=['CONCLUSION:'],
    #                    verbose=False,
    #                    finish_patterns=["### Reason", "Let's think step by step again", "let's go back and check", "###"],
    #                    reformat_gold_conditions=None):
    #     """
    #     Parse and evaluate a chain-of-thought response from the LLM.
        
    #     Args:
    #         pred_str: The prediction string from the LLM
    #         ans: The ground truth answer string
    #         conclusion_patterns: Patterns to identify the conclusion section
    #         verbose: Whether to print verbose output
    #         finish_patterns: Patterns indicating the end of the relevant response
    #         reformat_gold_conditions: Pre-formatted gold conditions (if None, will be derived from ans)
            
    #     Returns:
    #         tuple: (is_correct, parsed_prediction, wrong_reason, correct_ratio, reformat_gold_conditions)
    #     """
        
    #     def judge_string(input_str, reformat_gold_conditions, wrong_reason, finish_patterns):
    #         """
    #         Parse the COT response and evaluate the correctness of the answer.
    #         """
    #         correct_count = 0
    #         is_correct = False
    #         beyond_id = len(reformat_gold_conditions)+1
    #         beyond_id_pattern = f"({beyond_id})"

    #         for finish_pattern in finish_patterns:
    #             if finish_pattern in input_str:
    #                 input_str = input_str.split(finish_pattern)[0]

    #         if beyond_id_pattern in input_str:
    #             is_correct = False
    #             wrong_reason = "beyond_list"
    #         elif "if" in input_str:
    #             is_correct = False
    #             wrong_reason = "contain_if"
    #         else:
    #             is_correct = True
    #             for gold_condition in reformat_gold_conditions:
    #                 if gold_condition not in input_str:
    #                     is_correct = False
    #                     wrong_reason = "wrong_identity"
    #                 else:
    #                     correct_count += 1
    #         correct_ratio = correct_count/len(reformat_gold_conditions)

    #         return is_correct, wrong_reason, correct_ratio

    #     def check_numbers_in_string(s, N):
    #         """
    #         Check if the numbers 1 to N are present in the string.
    #         """
    #         for i in range(1, N + 1):
    #             if f"({i})" not in s:
    #                 return False
    #         return True
        
    #     original_str = pred_str
    #     pred_str = pred_str.split("### Question")[0] if "### Question" in pred_str else pred_str
    #     pred_answer = pred_str
    #     is_correct = False
    #     correct_ratio = 0
        
    #     # Process gold conditions if not provided
    #     if reformat_gold_conditions is None:
    #         gold = ans.replace(" and ", "").replace(".", "")
    #         gold_conditions = gold.split(",")
    #         reformat_gold_conditions = []
    #         for condition in gold_conditions:
    #             gold_condition = condition.strip()    # Remove leading and trailing spaces
    #             reformat_gold_conditions.append(gold_condition)

    #     wrong_reason = "no_conclusion_matched"
    #     for pattern in conclusion_patterns:
    #         if pattern in pred_str:
    #             pred = pred_str.split(pattern)
    #             if len(pred) > 1:
    #                 if len(pred[1]) > 0:  # if the matched answer is not empty
    #                     pred_answer = pred[1]
    #                     is_correct, wrong_reason, correct_ratio = judge_string(
    #                         pred_answer, reformat_gold_conditions, wrong_reason, finish_patterns)
    #                     break
        
    #     # Fallback if no conclusion section found
    #     if is_correct == False and wrong_reason == "no_conclusion_matched": 
    #         if check_numbers_in_string(pred_str, len(reformat_gold_conditions)): # the answer contains (1)..(2)..
    #             is_correct, wrong_reason, correct_ratio = judge_string(
    #                 pred_str, reformat_gold_conditions, wrong_reason, finish_patterns)
        
    #     if is_correct == False and verbose == True:
    #         self.logger.debug(f"wrong_reason: {wrong_reason}")
    #         self.logger.debug(f"prediction before parse: {original_str}")
    #         self.logger.debug(f"prediction after parse: {pred_answer}")

    #     return is_correct, pred_answer, wrong_reason, correct_ratio, reformat_gold_conditions

    def _combine_results_with_input(self, dataset: Dataset, data: list):
        # Not implemented for KKExperiment, but provided for interface consistency
        raise NotImplementedError("_combine_results_with_input is not implemented for KKExperiment.")

    def _load_intervention_vectors(self, *args, **kwargs):
        # Not implemented for KKExperiment, but provided for interface consistency
        raise NotImplementedError("_load_intervention_vectors is not implemented for KKExperiment.")

    
    
