"""Utilities for setting up hooks to capture activations from the model."""
import logging
from typing import Literal
import numpy as np

from transformers import PreTrainedModel

# Configure logger
logger = logging.getLogger(__name__)


def get_activation_hook(layer_idx, activation_dict):
    """Create a hook that captures activations for a specific layer."""
    def hook(module, input, output):
        logger.debug(
            f"Saving activations for module {type(module)} of layer {layer_idx}"
        )

        # The attention layer returns a tuple of (attn_output, attn_weights)
        # We only want the attn_output
        if isinstance(output, tuple):
            output = output[0]
        # Detach to CPU to free up GPU memory and prepare for numpy array
        activation_dict[layer_idx].append(output.detach().cpu().numpy())

    return hook


def register_probing_hooks(
    model: PreTrainedModel, 
    mlp_activations={}, 
    attention_activations={}, 
    residual_activations={}
):
    """Register forward hooks on model components and return hook handles."""
    hook_handles = []
    for i, layer in enumerate(model.model.layers):
        if mlp_activations:
            hook_handle = layer.mlp.register_forward_hook(
                get_activation_hook(i, mlp_activations)
            )
            hook_handles.append(hook_handle)
        if attention_activations:
            hook_handle = layer.self_attn.register_forward_hook(
                get_activation_hook(i, attention_activations)
            )
            hook_handles.append(hook_handle)
        if residual_activations:
            hook_handle = layer.register_forward_hook(
                get_activation_hook(i, residual_activations)
            )
            hook_handles.append(hook_handle)

    return hook_handles


def reset_activation_dicts(model):
    """Reset all activation dictionaries to prepare for a new batch."""
    global activation_dicts

    # Initialize if not already defined
    if 'activation_dicts' not in globals():
        activation_dicts = {
            'mlp': {},
            'attention': {},
            'residual': {}
        }

    # Reset for all layers
    for i in range(len(model.model.layers)):
        for module_name in activation_dicts:
            activation_dicts[module_name][i] = []


def initialize_activation_dicts(num_layers):
    """Initialize activation dictionaries for all model layers."""
    mlp_activations = {i: [] for i in range(num_layers)}
    attention_activations = {i: [] for i in range(num_layers)}
    residual_activations = {i: [] for i in range(num_layers)}

    # Map the activation dictionaries to their names
    activation_dicts = {
        'mlp': mlp_activations,
        'attention': attention_activations,
        'residual': residual_activations
    }
    
    return activation_dicts


def remove_hooks(hooks):
    """Remove all hooks from the model."""
    for hook in hooks:
        hook.remove()
    logger.debug(f"Removed {len(hooks)} hooks")


def process_activations(activation_dicts, num_layers, batch_prompts):
    """Process captured activations into a structured format."""
    processed_data = {
        "mlp_activations": [],
        "attention_activations": [],
        "residual_activations": [],
    }
    
    for module_name, module_activations in activation_dicts.items():
        all_layer_activations = []
        for i in range(num_layers):
            if not module_activations[i]:
                logger.debug(
                    f"No activations captured for {module_name} layer {i}"
                )
                continue

            # Get the activations (first element is the current batch)
            layer_activations = module_activations[i][0]

            # For Llama (left padding), last token is always at the end
            last_token_activations = layer_activations[:, -1, :]
            logger.debug(
                f"Shape of {module_name} activations for layer {i}: "
                f"{last_token_activations.shape}"
            )
            all_layer_activations.append(last_token_activations)

        # Stack activations from all layers
        layer_stacked_batch = np.stack(all_layer_activations)
        logger.debug(
            f"Shape of batch of {module_name} stacked activations: "
            f"{layer_stacked_batch.shape}"
        )
        
        # Extract activations for each example in the batch
        for batch_idx in range(len(batch_prompts)):
            processed_data[f"{module_name}_activations"].append(
                layer_stacked_batch[:, batch_idx, :]
            )
    
    return processed_data


def setup_hooks_for_probing(model):
    """Set up the complete hook environment in a single function call."""
    num_layers = len(model.model.layers)
    activation_dicts = initialize_activation_dicts(num_layers)
    hooks = register_probing_hooks(
        model=model,
        mlp_activations=activation_dicts['mlp'],
        attention_activations=activation_dicts['attention'],
        residual_activations=activation_dicts['residual']
    )
    logger.debug(f"Registered {len(hooks)} probing hooks")
    
    return hooks, activation_dicts, num_layers


class ModelInterventionManager:
    """Class to manage model interventions through hooks."""
    
    def __init__(self, model,
                 intervention_type: Literal["ablation", "addition"],
                 alpha: float):
        """
        Initialize the intervention manager.
        
        Args:
            model: The model to apply interventions to
            intervention_vectors: Optional vectors to apply to each layer
            intervention_type: Type of intervention ("addition" or "ablation")
            alpha: Scaling factor for the intervention vector
        """
        self.model = model
        self.intervention_type = intervention_type
        self.alpha = alpha
        self.hooks = []
        self.device = model.get_input_embeddings().weight.device
        
    def setup_hooks(self, intervention_vectors=None):
        """
        Set up intervention hooks on the model.
        
        Args:
            intervention_vectors: Optional vectors to override the ones set at init
            
        Returns:
            List of registered hooks
        """
        vectors = intervention_vectors
        if vectors is None:
            raise ValueError("No intervention vectors provided")
            
        if len(vectors) != len(self.model.model.layers):
            raise ValueError(
                f"Number of intervention vectors ({len(vectors)}) "
                f"must match number of layers ({len(self.model.model.layers)})"
            )
            
        self.hooks = self._register_hooks(vectors)
        logger.debug(f"Registered {len(self.hooks)} intervention hooks")
        return self.hooks
    
    def _register_hooks(self, intervention_vectors):
        """Register intervention hooks on each layer."""
        hooks = []
        hidden_size = self.model.config.hidden_size
        logger.info(f"Model hidden size: {hidden_size}")
        
        for i, layer in enumerate(self.model.model.layers):
            # Verify intervention vector dimension matches hidden size
            if intervention_vectors[i].shape[0] != hidden_size:
                raise ValueError(
                    f"Intervention vector dimension ({intervention_vectors[i].shape[0]}) "
                    f"does not match model hidden size ({hidden_size}) for layer {i}"
                )
            
            # Register hook on the layer's input (before attention)
            # This way we modify the residual stream before it goes through attention
            hook_handle = layer.register_forward_pre_hook(
                self._create_hook(intervention_vectors[i], layer_idx=i)
            )
            hooks.append(hook_handle)
        return hooks
    
    def _create_hook(self, intervention_vec, layer_idx):
        """Create a hook function that applies the intervention vector."""
        def hook(module, input):
            # input is a tuple, we want the first element which is the hidden states
            # input[0] shape: [batch_size, seq_len, hidden_size]
            #hidden_states = input[0]            
            # Create a copy of the intervention vector and reshape it to (1, 1, hidden_size)
            # This shape will broadcast correctly to any sequence length
            intervention_vec_copy = intervention_vec.clone().view(1, 1, -1).to(self.device)
            
            # Create a modified input tensor
            #modified_hidden_states = hidden_states.clone()
            
            #if self.intervention_type == "ablation":
                # Normalize the intervention vector to get a unit vector
                # The normalization will broadcast correctly
            unit_vec = intervention_vec_copy / (intervention_vec_copy.norm(dim=-1, keepdim=True) + 1e-8)
            unit_vec.to(self.device)
            unit_vec = unit_vec.to(input[0].dtype)
                
                # The multiplication and sum will broadcast correctly to any sequence length
                # because intervention_vec_copy is (1, 1, hidden_size)
            projection = (input[0] * unit_vec).sum(dim=-1, keepdim=True) * unit_vec
                
                # Subtract the projection (scaled by alpha) from all tokens
            modified_hidden_states = input[0] + self.alpha * projection
            # elif self.intervention_type == "addition":
            #     # The addition will broadcast correctly to any sequence length
            #     # because intervention_vec_copy is (1, 1, hidden_size)
            #     modified_hidden_states = modified_hidden_states + self.alpha * intervention_vec_copy
            # else:
            #     raise ValueError(
            #         f"Invalid intervention type: {self.intervention_type}"
            #     )
            
            # Return the modified input tuple
            if isinstance(input, tuple):
                input[0][:,:,:] = modified_hidden_states[:,:,:]
            else:
                input[:,:,:] = modified_hidden_states[:,:,:]
        
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        logger.debug(f"Removed {len(self.hooks)} intervention hooks")
        self.hooks = []