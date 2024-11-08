# -*- coding: utf-8 -*-
"""QServe state dict converter module."""

import argparse
import os

import torch
import tqdm

from .utils import convert_to_qserve_w4x8y16_linear_weight, convert_to_qserve_w8x8y16_linear_weight

__all__ = ["convert_to_qserve_state_dict"]


def convert_to_qserve_w4x8y16_linear_state_dict(
    param_name: str,
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int = -1,
    subscale: torch.Tensor | None = None,
    zero_pre_scaled: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert a weight tensor to QServe W4-X8-Y16 linear state dictionary.

    Args:
        param_name (`str`):
            parameter name.
        weight (`torch.Tensor`):
            weight tensor to be converted.
        scale (`torch.Tensor`):
            scale tensor for the weight tensor.
        zero (`torch.Tensor`):
            zero point tensor for the weight tensor.
        group_size (`int`, *optional*, defaults to `-1`):
            quantization group size.
        subscale (`torch.Tensor` or `None`, *optional*, defaults to `None`):
            subscale tensor for the weight tensor.
        zero_pre_scaled (`bool`, *optional*, defaults to `False`):
            whether zero point tensor is pre-scaled.

    Returns:
        `dict[str, torch.Tensor]`:
            state dictionary for the quantized weight tensor.
    """
    module_name = param_name[:-7]
    weight, scale, zero, subscale = convert_to_qserve_w4x8y16_linear_weight(
        weight, scale=scale, zero=zero, group_size=group_size, subscale=subscale, zero_pre_scaled=zero_pre_scaled
    )
    state_dict: dict[str, torch.Tensor] = {}
    state_dict[f"{module_name}.qweight"] = weight.cpu()
    state_dict[f"{module_name}.s1_scales"] = scale.cpu()
    if subscale is None:
        state_dict[f"{module_name}.s1_szeros"] = zero.cpu()
    else:
        state_dict[f"{module_name}.s2_scales"] = subscale.cpu()
        state_dict[f"{module_name}.s2_zeros"] = zero.cpu()
    return state_dict


def convert_to_qserve_w8x8y16_linear_state_dict(
    param_name: str, weight: torch.Tensor, scale: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Convert a weight tensor to QServe W8-X8-Y16 linear state dictionary.

    Args:
        param_name (`str`):
            parameter name.
        weight (`torch.Tensor`):
            weight tensor to be converted.
        scale (`torch.Tensor`):
            scale tensor for the weight tensor.

    Returns:
        `dict[str, torch.Tensor]`:
            state dictionary for the quantized weight tensor.
    """
    module_name = param_name[:-7]
    weight, scale = convert_to_qserve_w8x8y16_linear_weight(weight, scale=scale)
    state_dict: dict[str, torch.Tensor] = {}
    state_dict[f"{module_name}.weight"] = weight.cpu()
    state_dict[f"{module_name}.dequant_scale"] = scale.cpu()
    return state_dict


def convert_to_qserve_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    weight_bits: int,
    group_size: int = -1,
) -> dict[str, torch.Tensor]:
    assert weight_bits in [4, 8], "weight bits should be 4 or 8."
    scales: dict[str, dict[tuple[int, ...], torch.Tensor]] = {}
    zeros: dict[str, tuple[torch.Tensor | None, bool]] = {}
    print("Loading scale tensors...")
    for name, tensor in tqdm.tqdm(scale_dict.items(), desc="Loading scale tensors", leave=False, dynamic_ncols=True):
        print(f"  - Loading tensor {name} (dtype: {tensor.dtype}, shape: {tensor.shape}, device: {tensor.device})")
        if name.endswith("zero"):
            # this is a zero point tensor
            zero = None if tensor is None or all(t.item() == 0 for t in tensor.flatten()) else tensor
            if name.endswith(".scaled_zero"):
                zeros[name[:-12]] = (zero, False)  # zero point tensor is post-scaled
            else:
                zeros[name[:-5]] = (zero, True)  # zero point tensor is pre-scaled
        else:
            assert ".weight.scale" in name
            # this is a scale tensor
            idx = name.index(".weight.scale")
            param_name = name[: idx + 7]
            scale_level = tuple(map(int, name[idx + 14 :].split(".")))
            scales.setdefault(param_name, {})[scale_level] = tensor
    for param_name in zeros.keys():
        assert param_name in state_dict, f"zero point tensor {param_name} not found in state dict."
        assert param_name in scales, f"scale tensor {param_name} not found in scale dict."
    converted: dict[str, torch.Tensor] = {}
    print("Converting state dict...")
    for param_name, param in tqdm.tqdm(state_dict.items(), desc="Converting state dict", dynamic_ncols=True):
        if param_name in scales:
            print(f"  - Converting {param_name} (dtype: {param.dtype}, shape: {param.shape}, device: {param.device})")
            weight = param.data.clone()
            if param_name in zeros:
                zero, zero_pre_scaled = zeros[param_name]
                zero = zero.clone() if zero is not None else None
            else:
                zero, zero_pre_scaled = None, False
            level_scales = sorted(scales[param_name].items(), key=lambda x: x[0])
            assert len(level_scales) <= 2, "more than two scale levels are not supported."
            scale = level_scales[0][1].clone()
            subscale = level_scales[1][1].clone() if len(level_scales) > 1 else None
            if weight_bits == 4:
                converted.update(
                    convert_to_qserve_w4x8y16_linear_state_dict(
                        param_name,
                        weight,
                        scale=scale,
                        zero=zero,
                        group_size=group_size,
                        subscale=subscale,
                        zero_pre_scaled=zero_pre_scaled,
                    )
                )
            else:
                assert zero is None, "zero point tensor is not supported for W8 quantization."
                assert subscale is None, "subscale tensor is not supported for W8 quantization."
                assert group_size == -1, "group size should be -1 for W8 quantization."
                converted.update(convert_to_qserve_w8x8y16_linear_state_dict(param_name, weight, scale=scale))
        else:
            if isinstance(param, torch.Tensor):
                print(f"  - Copying {param_name} (dtype: {param.dtype}, shape: {param.shape}, device: {param.device})")
                converted[param_name] = param.clone().cpu()
            else:
                print(f"  - Copying {param_name} (type: {type(param)}, value: {param})")
                converted[param_name] = param
    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-path", type=str, required=True, help="path to the quantization checkpoint directory.")
    parser.add_argument("--weight-bits", type=int, required=True, help="quantized weight bits.")
    parser.add_argument("--group-size", type=int, default=-1, help="quantization group size.")
    parser.add_argument("--output-root", type=str, default="", help="root to the output checkpoint directory.")
    parser.add_argument("--model-name", type=str, default=None, help="name of the model.")
    parser.add_argument("--model-path", type=str, default=None, help="path to the huggingface model directory.")
    parser.add_argument(
        "--copy-on-save",
        action="store_true",
        help="copy the original tokenizer and configuration files to the output directory.",
    )
    args = parser.parse_args()
    if not args.output_root:
        args.output_root = args.quant_path
    if args.model_name is None:
        assert args.model_path is not None, "model name or path is required."
        model_name = args.model_path.rstrip(os.sep).split(os.sep)[-1]
        print(f"Model name not provided. Using model name {model_name}.")
    else:
        model_name = args.model_name
    assert model_name, "model name is required."
    model_name = f"{model_name}-w{args.weight_bits}a8"
    model_name += f"-g{args.group_size}" if args.group_size > 0 else "-gchn"
    output_dirpath = os.path.join(args.output_root, model_name)
    output_path = os.path.join(output_dirpath, "quant_model.pt")
    state_dict = torch.load(
        os.path.join(args.quant_path, "model.pt"),
        map_location="cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu",
    )
    scale_dict = torch.load(os.path.join(args.quant_path, "scale.pt"), map_location="cpu")
    converted = convert_to_qserve_state_dict(
        state_dict, scale_dict, weight_bits=args.weight_bits, group_size=args.group_size
    )
    os.makedirs(output_dirpath, exist_ok=True)
    torch.save(converted, output_path)
    if args.model_path and os.path.exists(args.model_path):
        for filename in os.listdir(args.model_path):
            if filename == "tokenizer.model" or (
                filename.endswith(".json") and filename != "pytorch_model.bin.index.json"
            ):
                filepath = os.path.abspath(os.path.join(args.model_path, filename))
                if args.copy_on_save:
                    os.system(f"cp {filepath} {output_dirpath}/")
                else:
                    os.system(f"ln -s {filepath} {output_dirpath}/{filename}")
    print(f"Quantized model checkpoint saved to {output_path}.")
    print(f"Quantized model saved to {output_dirpath}.")
    print(f"Quantized model checkpoint saved to {output_path}.")
    print(f"Quantized model saved to {output_dirpath}.")
