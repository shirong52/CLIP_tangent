from typing import Iterable

from transformers import CLIPModel


def _freeze_module(module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_module(module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def _maybe_unfreeze_last_layers(layer_stack: Iterable, n: int) -> None:
    if n <= 0:
        return
    layers = list(layer_stack)
    if not layers:
        return
    for layer in layers[-n:]:
        _unfreeze_module(layer)


def set_trainable_strategy(model: CLIPModel, strategy: str, unfreeze_last_n: int) -> None:
    """
    strategy:
      - "head_only": train projection layers + logit_scale
      - "last_n": train projection + last N text/vision encoder layers
      - "full": train all parameters
    """
    _freeze_module(model)

    # Always allow optimization of projection heads and logit scale.
    _unfreeze_module(model.visual_projection)
    _unfreeze_module(model.text_projection)
    model.logit_scale.requires_grad = True

    if strategy == "head_only":
        return

    if strategy == "last_n":
        _maybe_unfreeze_last_layers(model.vision_model.encoder.layers, unfreeze_last_n)
        _maybe_unfreeze_last_layers(model.text_model.encoder.layers, unfreeze_last_n)
        _unfreeze_module(model.vision_model.post_layernorm)
        _unfreeze_module(model.text_model.final_layer_norm)
        return

    if strategy == "full":
        _unfreeze_module(model)
        return

    raise ValueError(f"Unknown trainable strategy: {strategy}")


def build_clip_model(model_path: str) -> CLIPModel:
    return CLIPModel.from_pretrained(model_path)
