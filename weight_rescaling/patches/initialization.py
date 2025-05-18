from math import sqrt
from torch import nn
from weight_rescaling.utils.utils import get_layers

def re_initialize_model(
    model,
    model_layer_path: str | None = None,
    initializer_range: float = 0.02,
    scale_with_depth: bool = False,
    scale_factor: float = 1.0,
    special_init_residual_module_names: list[str] | None = None,
    nlayer_config_name: str = "num_hidden_layers",
):
    """
    Reinitialize model weights with modularized options for different ranges for attention
    and MLP blocks.

    When scale_with_depth is True, the function searches for a ModuleList (via get_layers)
    that groups layers (e.g. LlamaDecoderLayer). For each layer in that container, the weight
    initialization std is scaled by the layer index.

    Args:
        model (nn.Module): The model whose weights will be reinitialized.
        model_layer_path (str, optional): Path to the layer container in the model.
        initializer_range (float): Base std for weight initialization.
        scale_with_depth (bool): If True, apply per-layer scaling.
        scale_factor (float): Factor to scale the depth-based scaling.
        special_init_residual_module_names (list[str], optional): Module name substrings for special residual initialization as per GPT-2 paper.
        nlayer_config_name (str): Name of the attribute in model.config that specifies the number of layers.
    """
    assert not (
        scale_with_depth and special_init_residual_module_names is not None
    ), "Special initialization of residual modules is not supported when scaling with depth is enabled."

    layer_container = get_layers(model, model_layer_path)
    n_layer = (
        len(layer_container)
        if layer_container is not None
        else getattr(model.config, nlayer_config_name, None)
    )

    if scale_with_depth and layer_container is not None:
        initialized_modules = set()
        for layer_index, layer in enumerate(layer_container, start=1):
            for name, module in layer.named_modules():
                std_to_use = _get_std_for_module(
                    initializer_range,
                    layer_index,
                    scale_factor,
                )
                if isinstance(module, nn.Linear):
                    _init_weights(module, std_to_use)
                initialized_modules.add(module)

        # Initialize modules not in the detected layer container using the non-scaled std.
        for name, module in model.named_modules():
            if module not in initialized_modules:
                std_to_use = _get_std_for_module(
                    initializer_range
                )
                _init_weights(module, std_to_use)
    else:
        for name, module in model.named_modules():
            if special_init_residual_module_names and any(
                s.lower() in name.lower() for s in special_init_residual_module_names
            ):
                _init_weights(module, initializer_range / sqrt(2 * n_layer), is_special=True)
            else:
                std_to_use = _get_std_for_module(
                    initializer_range
                )
                _init_weights(module, std_to_use)


def _init_weights(module, std=0.02, is_special=False):
    """
    Initialize weights for a given module.

    For nn.Linear and nn.Embedding layers, weights are initialized from a normal distribution
    with mean 0 and the given std. Biases (and padding embeddings) are zeroed.

    Args:
        module (nn.Module): The module to initialize.
        std (float): The standard deviation for weight initialization.
    """
    if is_special:
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.normal_(mean=0.0, std=std)
        else:
            module.data.normal_(mean=0.0, std=std)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

def _get_std_for_module(
    base_std: float,
    layer_index: int = None,
    scale_factor: float = 1.0,
):
    """
    Return the appropriate standard deviation for initializing a module.

    If a layer_index is provided (for depth scaling), the returned std is scaled by:
        std = provided_std / sqrt(layer_index * scale_factor)
    Otherwise, it returns the provided std value.
    """
    if layer_index is not None:
        return base_std / sqrt(layer_index * scale_factor)
    else:
        return base_std
