from datasets import Dataset
from .experiment_base import ExperimentBase
from typing import Any, Literal

class LogicExperiment(ExperimentBase):
    def __init__(self, input_path: str, output_path: str, chunk_size: int, chunk_id: int, mode: Literal['counter_factual', 'real_world'], model_name: str = "meta-llama/Llama-3.1-8B"):
        self.mode = mode
        super().__init__(input_path, output_path, chunk_size, chunk_id, model_name)

    def run_experiment(self):
        pass
    
    def evaluate_llm_responses(self, dataset: Dataset):
        pass

    def load_data(self) -> list[Any]:
        pass

    def format_prompts(self, data: list[Any]) -> list[str]:
        def format_prompt(obj: Any) -> str:
            if self.mode == 'real_world':
                premises = obj["orig_premises"].strip()
                conclusion = obj["orig_conclusion"].strip()
            else:
                premises = obj["premises"].strip()
                conclusion = obj["conclusion"].strip()
            
            new_premises = []
            for line in premises.split("\n"):
                line = line.strip()
                if line[-1] in {".", '"'}:
                    new_premises.append(line)
                elif "a" <= line[-1] <= "z" or "A" <= line[-1] <= "Z":
                    new_premises.append(line + ".")
                else:
                    assert False
            premises = " ".join(new_premises)

            return f"Consider the following premises: \"{premises}\" Assuming no other commonsense or world knowledge, is the sentence \"{conclusion}\" necessarily true, necessarily false, or neither? End the response with either \"necessarily true\", \"necessarily false\", or \"neither\"."
        
        return [format_prompt(obj) for obj in data]
