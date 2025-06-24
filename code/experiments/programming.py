import os
import json
import torch

from .experiment_base import ExperimentBase
from utils.programming_utils import one_based_indexing_unit_tests
class ProgrammingExperiment(ExperimentBase):
    def __init__(self, input_path: str, output_path: str, chunk_size: int, chunk_id: int):
        super().__init__(input_path, output_path, chunk_size, chunk_id)

    def load_data(self, *args, **kwargs):
        """Load the data from the input path."""
        # Check if the data.jsonl file exists at the input path
        data_file = os.path.join(self.input_path, 'data.jsonl')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file does not exist: {data_file}")
        
        # Load the data
        with open(data_file, 'r') as f:
            data = [json.loads(line) for line in f]
        
        return data
        
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
                

    def evaluate_llm_responses(self, *args, **kwargs):
        pass
