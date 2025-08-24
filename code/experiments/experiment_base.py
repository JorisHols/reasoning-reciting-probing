from typing import Any
from datasets import Dataset
import torch

from probe_llama import ProbeLlamaModel
class ExperimentBase:
    def __init__(self, input_path: str, output_path: str, chunk_size: int, chunk_id: int, model_name: str = "meta-llama/Llama-3.1-8B"):
        self.input_path = input_path
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.chunk_id = chunk_id
        self.model_name = model_name
        self.prober = None


        
    def run_experiment(self, intervention_vectors: list[torch.Tensor] = None, alpha: float = 0.0, collect_activations: bool = False) -> Dataset:
        data = self.load_data()
        # Check if the probe is already set up
        if self.prober is None:
            self.setup_probe()

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
        max_new_tokens: int,
        model_name: str = "meta-llama/Llama-3.1-8B",
        batch_size: int = 4,
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

    def load_data(self, *args, **kwargs) -> list[Any]:
        """Load the data."""
        raise NotImplementedError("load_data must be implemented by subclasses.")

    def format_prompts(self, *args, **kwargs):
        """Format the prompts."""
        raise NotImplementedError("format_prompts must be implemented by subclasses.")
    
    def evaluate_llm_responses(self, dataset: Dataset):
        """Evaluate the LLM responses."""
        raise NotImplementedError("evaluate_llm_responses must be implemented by subclasses.")

    def _combine_results_with_input(self, dataset: Dataset, data: list[Any]):
            """Combine the results with the input data."""
            new_columns = {k: [] for k in data[0].keys()}
            for line in data:
                for k, v in line.items():
                    new_columns[k].append(v)

            # Add the new columns to the dataset
            for k, v in new_columns.items():
                dataset = dataset.add_column(k, v)

            return dataset