import shutil
from typing import Literal
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer, 
    AutoModelForCausalLM, 
    AutoTokenizer
)
import torch
from tqdm import tqdm
from datasets import Dataset
from dotenv import load_dotenv
import os
import logging

import probe_utils
from probe_utils import ModelInterventionManager


class ProbeLlamaModel:
    """
    A class that probes a model for activations.
    """

    def __init__(
        self, 
        model_name=None, 
        batch_size=2, 
        max_new_tokens=2048, 
        generate_response=True, 
        checkpoint_every=0,
        ablation_layer=None,  # Layer to ablate (0-indexed)
        ablation_type=None,   # Type of activation to ablate 
                              # ('mlp', 'attention', 'residual')
    ):
        self.model_name = model_name or 'meta-llama/Llama-3.1-8B-Instruct'
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.generate_response = generate_response
        self.checkpoint_every = checkpoint_every
        self.ablation_layer = ablation_layer
        self.ablation_type = ablation_type
        self.logger = logging.getLogger("probe_llama")
        
        self.logger.info(
            f"Initializing ProbeLlamaModel with {self.model_name}"
        )
        if ablation_layer is not None and ablation_type is not None:
            self.logger.info(
                f"Will ablate {ablation_type} activations at "
                f"layer {ablation_layer}"
            )
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        # self._set_device()  # Commented out as device is handled by 
        # device_map="auto" in model loading

    def _load_model_and_tokenizer(
        self
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load the model from the model name.
        """
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if token is None:
            self.logger.error("HF_TOKEN environment variable not set")
            raise ValueError(
                "HF_TOKEN is not set in the environment variables"
            )
            
        self.logger.info(f"Loading model: {self.model_name}")
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_name, token=token, device_map="auto"
        )

        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=token, padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token
        self.logger.info("Model and tokenizer loaded successfully")

        return model, tokenizer
    
    # def _set_device(self, device: torch.device = None):
    #     """
    #     Set the device to use for the model.
    #     """
    #     self.device = device or torch.device(
    #         "cuda" if torch.cuda.is_available() 
    #         else "mps" if torch.backends.mps.is_available() 
    #         else "cpu"
    #     )
    #     self.model.to(self.device)
    #     self.logger.info(f"Model set to device: {self.device}")
    #     return self.device
    
    def _tokenize_batch(self, batch_prompts):
        """
        Tokenize a batch of prompts.
        """
        # Get the correct device
        self.device = self.model.get_input_embeddings().weight.device
        return self.tokenizer(
            batch_prompts, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)
    
    def _run_batch_with_hooks(self, batch_prompts, reset_hooks=True):
        """
        Process a batch with hooks, capturing activations.
        
        Args:
            batch_prompts: List of text prompts to process
            reset_hooks: Whether to reset activation storage before capture
            
        Returns:
            tuple: (processed_activations, tokenized_inputs)
        """
        # Initialize hook environment
        hooks, activation_dicts, num_layers = (
            probe_utils.setup_hooks_for_probing(self.model)
        )
        
        try:
            # Tokenize input
            batch_inputs = self._tokenize_batch(batch_prompts)
            
            # Reset activation storage if needed
            if reset_hooks:
                probe_utils.reset_activation_dicts(self.model)
                
            # Capture activations
            with torch.no_grad():
                _ = self.model(**batch_inputs)
            
            # Process captured activations
            activations = probe_utils.process_activations(
                activation_dicts, 
                num_layers, 
                batch_prompts
            )
            
            return activations, batch_inputs
        finally:
            # Always remove hooks
            probe_utils.remove_hooks(hooks)
            self.logger.debug("Hooks removed after batch processing")

    def _run_generation_without_hooks(self, batch_inputs, max_new_tokens=250):
        """
        Run generation without any hooks active.
        
        Args:
            batch_inputs: Tokenized inputs from the tokenizer
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            list: Generated responses
        """
        with torch.no_grad():
            generation_output = self.model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )

        # Decode outputs
        batch_responses = self.tokenizer.batch_decode(
            generation_output,
            skip_special_tokens=True
        )
        
        return batch_responses

    def process_statements(
        self, 
        prompts: list[str], 
        output_file_path: str = None
    ) -> Dataset:
        """
        Process statements and capture activations using clean hook management.
        First capture activations, then optionally generate responses.
        Batches are accumulated and saved periodically to improve efficiency.
        """
        self.logger.info(
            f"Starting batch processing of {len(prompts)} prompts "
            f"with batch size {self.batch_size}"
        )
        
        # Initialize accumulated data
        accumulated_data = {
            "prompt": [],
            "llm_response": [],
            "mlp_activations": [],
            "attention_activations": [],
            "residual_activations": [],
        }
        
        checkpoint_count = 0
        checkpoint_paths = []
        
        for batch_start in tqdm(
            range(0, len(prompts), self.batch_size),
            position=0,
            leave=False
        ):
            batch_prompts = prompts[batch_start:batch_start + self.batch_size]
            
            # Initialize output structure for this batch
            batch_output_data = {
                "prompt": batch_prompts,
                "llm_response": [],
            }

            # PHASE 1: Capture activations using hooks
            processed_activations, batch_inputs = self._run_batch_with_hooks(
                batch_prompts=batch_prompts
            )
            
            # Add processed activations to batch output
            batch_output_data.update(processed_activations)

            # PHASE 2: Generate responses if requested
            if self.generate_response:
                # Generate responses without any ablation
                batch_responses = self._run_generation_without_hooks(
                    batch_inputs=batch_inputs,
                    max_new_tokens=self.max_new_tokens
                )
                batch_output_data["llm_response"] = batch_responses

            # Accumulate batch data
            for key in accumulated_data:
                if key in batch_output_data:
                    accumulated_data[key].extend(batch_output_data[key])
                elif key == "llm_response" and not self.generate_response:
                    # Add empty placeholders if not generating responses
                    accumulated_data[key].extend([""] * len(batch_prompts))
            
            # Handle checkpointing if enabled
            if self.checkpoint_every > 0 and output_file_path:
                checkpoint_count += 1
                is_last_batch = (
                    batch_start + self.batch_size >= len(prompts)
                )
                
                if checkpoint_count >= self.checkpoint_every or is_last_batch:
                    checkpoint_path = self._save_checkpoint(
                        accumulated_data, 
                        output_file_path, 
                        batch_start // self.batch_size // self.checkpoint_every
                    )
                    checkpoint_paths.append(checkpoint_path)
                    
                    # Reset accumulated data after saving
                    accumulated_data = {
                        "prompt": [],
                        "llm_response": [],
                        "mlp_activations": [],
                        "attention_activations": [],
                        "residual_activations": [],
                    }
                    checkpoint_count = 0
        
        self.logger.info("Batch processing complete")
        
        # Create final dataset
        if self.checkpoint_every > 0 and output_file_path and checkpoint_paths:
            # Combine checkpoints if checkpointing was used
            final_dataset = self._combine_checkpoints(
                checkpoint_paths, 
                output_file_path
            )
        else:
            # Create dataset directly from accumulated data
            final_dataset = Dataset.from_dict(accumulated_data)
        
        return final_dataset
    
    def _save_checkpoint(self, data, output_file_path, checkpoint_id):
        """
        Save a checkpoint of the current accumulated data.
        
        Args:
            data: Dictionary of accumulated data
            output_file_path: Base path for saving checkpoints
            checkpoint_id: Identifier for this checkpoint
            
        Returns:
            str: Path where checkpoint was saved
        """
        dataset = Dataset.from_dict(data)
        checkpoint_path = f"{output_file_path}/checkpoint_{checkpoint_id}"
        dataset.save_to_disk(checkpoint_path)
        
        # Log checkpoint details
        checkpoint_size = len(data["prompt"])
        self.logger.info(
            f"Saved checkpoint {checkpoint_id} with {checkpoint_size} "
            f"examples at {checkpoint_path}"
        )
        
        return checkpoint_path
    
    def _combine_checkpoints(
        self, checkpoint_paths: list[str], output_file_path: str
    ) -> Dataset:
        """
        Combine multiple checkpoint datasets into a single dataset.
        
        Args:
            checkpoint_paths: List of paths to checkpoint datasets
            output_file_path: Path to save the final combined dataset
            
        Returns:
            Dataset: Combined dataset with all examples
        """
        self.logger.info(f"Combining {len(checkpoint_paths)} checkpoints")
        
        # Initialize empty combined data
        combined_data = {
            "prompt": [],
            "llm_response": [],
            "mlp_activations": [],
            "attention_activations": [],
            "residual_activations": [],
        }
        
        # Load and combine each checkpoint
        for path in tqdm(checkpoint_paths):
            checkpoint_dataset = Dataset.load_from_disk(path)
            
            # Add data from this checkpoint to combined data
            for key in combined_data:
                if key in checkpoint_dataset.features:
                    combined_data[key].extend(checkpoint_dataset[key])
        
        # Create the final dataset
        final_dataset = Dataset.from_dict(combined_data)
        self.logger.info(
            f"Created combined dataset with {len(final_dataset)} examples"
        )

        # Save the final dataset
        final_dataset.save_to_disk(output_file_path)
        self.logger.info(f"Saved final dataset to {output_file_path}")
        
        # Clean up checkpoint files
        for path in checkpoint_paths:
            if os.path.exists(path):
                shutil.rmtree(path)
                self.logger.info(f"Removed checkpoint: {path}")
        
        return final_dataset

    def process_intervention(
        self,
        prompts: list[str],
        intervention_vectors: list[torch.Tensor],
        intervention_type: Literal["ablation", "addition"] = "addition"
    ) -> Dataset:
        """
        Run an ablation study by applying custom vectors to each layer's 
        residual stream.
        
        Args:
            prompts: List of text prompts to process
            ablation_vectors: List of tensors to apply to each layer's 
                            residual stream. Length must match number of 
                            layers.
            output_file_path: Optional path to save results
            
        Returns:
            Dataset: Results containing original and ablated responses
        """
            

        # Initialize accumulated data
        accumulated_data = {
            "prompt": [],
            "intervention_response": [],
        }

        # Perform addition intervention to increase reasoning ability
        hook_manager = ModelInterventionManager(
            self.model,
            intervention_type=intervention_type,
            alpha=0.1
        )
        hook_manager.setup_hooks(intervention_vectors)

        # Process each batch
        for batch_start in tqdm(
            range(0, len(prompts), self.batch_size),
            position=0,
            leave=False
        ):
            batch_prompts = prompts[batch_start:batch_start + self.batch_size]
            
            # Initialize output structure for this batch
            batch_output_data = {
                "prompt": batch_prompts,
                "intervention_response": [],
            }
            
            batch_inputs = self._tokenize_batch(batch_prompts=batch_prompts)
            

            # Run generation with ablation
            with torch.no_grad():
                generation_output = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True
                )
                
            # Decode outputs
            intervention_responses = self.tokenizer.batch_decode(
                generation_output,
                skip_special_tokens=True
            )
            batch_output_data["intervention_response"] = intervention_responses
                        
            # Accumulate batch data
            for key in accumulated_data:
                if key in batch_output_data:
                    accumulated_data[key].extend(batch_output_data[key])
        
        # Create final dataset
        final_dataset = Dataset.from_dict(accumulated_data)
        hook_manager.remove_hooks()
        
        return final_dataset
