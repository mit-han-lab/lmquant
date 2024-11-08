import typing as tp

import torch
import torch.nn as nn

from ..nn.struct import DiffusionAttentionStruct, DiffusionFeedForwardStruct, DiffusionModelStruct
from .config import DiffusionQuantConfig

__all__ = ["get_needs_inputs_fn", "get_needs_outputs_fn", "wrap_joint_attn"]


def wrap_joint_attn(attn: nn.Module, /, *, indexes: int | tuple[int, ...] = 1) -> tp.Callable:
    if isinstance(indexes, int):

        def eval(*args, **kwargs) -> torch.Tensor:
            return attn(*args, **kwargs)[indexes]

    else:

        def eval(*args, **kwargs) -> tuple[torch.Tensor, ...]:
            tensors = attn(*args, **kwargs)
            result = torch.concat([tensors[i] for i in indexes], dim=-2)
            return result

    return eval


def get_needs_inputs_fn(
    model: DiffusionModelStruct, config: DiffusionQuantConfig
) -> tp.Callable[[str, nn.Module], bool]:
    """Get function that checks whether the module needs to cache inputs.

    Args:
        model (`DiffusionModelStruct`):
            The diffused model.
        config (`DiffusionQuantConfig`):
            The quantization configuration.

    Returns:
        `Callable[[str, nn.Module], bool]`:
            The function that checks whether the module needs to cache inputs.
    """

    needs_inputs_names = set()
    for module_key, module_name, _, parent, field_name in model.named_key_modules():
        if (config.enabled_wgts and config.wgts.is_enabled_for(module_key)) or (
            config.enabled_ipts and config.ipts.is_enabled_for(module_key)
        ):
            if isinstance(parent, DiffusionAttentionStruct):
                if field_name.endswith("o_proj"):
                    needs_inputs_names.add(module_name)
                elif field_name in ("q_proj", "k_proj", "v_proj"):
                    needs_inputs_names.add(parent.q_proj_name)
                    if parent.parent.parallel and parent.idx == 0:
                        needs_inputs_names.add(parent.parent.name)
                    else:
                        needs_inputs_names.add(parent.name)
                elif field_name in ("add_q_proj", "add_k_proj", "add_v_proj"):
                    needs_inputs_names.add(parent.add_k_proj_name)
                    if parent.parent.parallel and parent.idx == 0:
                        needs_inputs_names.add(parent.parent.name)
                    else:
                        needs_inputs_names.add(parent.name)
                else:
                    raise RuntimeError(f"Unknown field name: {field_name}")
            elif isinstance(parent, DiffusionFeedForwardStruct):
                if field_name == "up_proj":
                    needs_inputs_names.update(parent.up_proj_names[: parent.config.num_experts])
                elif field_name == "down_proj":
                    needs_inputs_names.update(parent.down_proj_names[: parent.config.num_experts])
                else:
                    raise RuntimeError(f"Unknown field name: {field_name}")
            else:
                needs_inputs_names.add(module_name)

    def needs_inputs(name: str, module: nn.Module) -> bool:
        return name in needs_inputs_names

    return needs_inputs


def get_needs_outputs_fn(
    model: DiffusionModelStruct, config: DiffusionQuantConfig
) -> tp.Callable[[str, nn.Module], bool]:
    """Get function that checks whether the module needs to cache outputs.

    Args:
        model (`DiffusionModelStruct`):
            The diffused model.
        config (`DiffusionQuantConfig`):
            The quantization configuration.

    Returns:
        `Callable[[str, nn.Module], bool]`:
            The function that checks whether the module needs to cache outputs.
    """

    # TODO: Implement the function that checks whether the module needs to cache outputs.

    def needs_outputs(name: str, module: nn.Module) -> bool:
        return False

    return needs_outputs
