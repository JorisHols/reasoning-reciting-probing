"""Utilities for setting up hooks to capture activations from the model."""
import logging
from transformers import PreTrainedModel
import numpy as np


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


def register_hooks(
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


def setup_hook_environment(model):
    """Set up the complete hook environment in a single function call."""
    num_layers = len(model.model.layers)
    activation_dicts = initialize_activation_dicts(num_layers)
    hooks = register_hooks(
        model=model,
        mlp_activations=activation_dicts['mlp'],
        attention_activations=activation_dicts['attention'],
        residual_activations=activation_dicts['residual']
    )
    logger.debug(f"Registered {len(hooks)} hooks")
    
    return hooks, activation_dicts, num_layers