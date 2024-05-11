# -*- coding: utf-8 -*-
"""LLM Selective Quantization config."""

from dataclasses import dataclass, field

import torch
from omniconfig import configclass

from lmquant.quant.config import QuantizerKernelConfig, TensorQuantizerConfig

__all__ = ["LlmSelectQuantConfig", "LlmSelectQuantizerConfig", "LlmProjQuantConfig", "LlmAttnQuantConfig"]


@configclass
@dataclass
class LlmSelectQuantizerConfig(TensorQuantizerConfig):
    """Selective quantizer configuration.

    Args:
        static (bool): Whether to use static quantization. Defaults to ``False``.
        dtype (QuantDataType): The quantization data type. Defaults to ``None``.
        group_shapes (list[list[int]] | list[int]): The shapes for per-group quantization.
            Defaults to ``((-1, -1, -1),)``.
        group_scale_dtypes (list[torch.dtype | QuantDataType | None] | torch.dtype | QuantDataType | None): The
            quantization scale data type for per-group quantization. Defaults to ``(None,)``.
        compute_dtype (QuantDataType | None): The quantization data type for compute. Defaults to ``None``.
        compute_group_level (int): The group level for compute. Defaults to ``-1``.
        saturate_compute_dtype (bool): Whether to saturate the compute dtype. Defaults to ``False``.
        calib_range (DynamicRangeCalibConfig | None): The quantizatizer dynamic range calibration configuration.
            Defaults to ``None``.
        update_calib_range (bool): Whether to update the dynamic range calibration configuration. Defaults to ``False``.
        all_layers (bool): Whether to quantize all layers. Defaults to ``True``.
        num_first_layers (int): The number of first layers to quantize. Defaults to ``0``.
        num_last_layers (int): The number of last layers to quantize. Defaults to ``0``.
        layer_interval (int): The layer interval to quantize. Defaults to ``1``.
    """

    skips: list[str] = field(init=False, default_factory=list)
    calib_kernel: QuantizerKernelConfig | None = field(init=False, default=None)
    update_calib_range: bool = False
    all_layers: bool = True
    num_first_layers: int = 0
    num_last_layers: int = 0
    layer_interval: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.all_layers:
            self.num_first_layers = 0
            self.num_last_layers = 0
            self.layer_interval = 0
        else:
            assert self.num_first_layers > 0 or self.num_last_layers > 0 or self.layer_interval > 0

    def is_selected(self, layer_idx: int, num_layers: int) -> bool:
        """Check if the decoder layer is selected."""
        if self.all_layers or layer_idx < 0 or num_layers < 0:
            return True
        if self.num_first_layers > 0 and layer_idx < self.num_first_layers:
            return True
        if self.num_last_layers > 0 and layer_idx >= (num_layers - self.num_last_layers):
            return True
        if self.layer_interval > 0 and layer_idx % self.layer_interval == 0:
            return True
        return False

    def __str__(self) -> str:
        s = super().__str__()
        if self.all_layers:
            return s
        return (
            s[:-1] + f", num_first_layers={self.num_first_layers}, num_last_layers={self.num_last_layers}, "
            f"layer_interval={self.layer_interval})"
        )

    def generate_dirnames(
        self,
        shape: torch.Size = torch.Size((4096, 4096, 16, 16)),
        dtype: torch.dtype = torch.float16,
        prefix: str = "",
    ) -> list[str]:
        names = super().generate_dirnames(shape, dtype)[:-1]
        if self.all_layers:
            names.append(f"select.all")
        else:
            names.append(f"select.[{self.num_first_layers}.{self.num_last_layers}.{self.layer_interval}]")
        return [f"{prefix}.{name}" for name in names] if prefix else names


@configclass
@dataclass
class LlmProjQuantConfig:
    """Large Language Model Projection Modules quantization configuration.

    Args:
        proj_qkv (LlmSelectQuantConfig): The quantization configuration for the projection of the qkv.
        proj_out (LlmSelectQuantConfig): The quantization configuration for the output projection.
        proj_1st (LlmSelectQuantConfig): The quantization configuration for the first layer of feed-forward network.
        proj_2nd (LlmSelectQuantConfig): The quantization configuration for the second layer of feed-forward network.
        router (LlmSelectQuantConfig): The quantization configuration for the router.
    """

    proj_qkv: LlmSelectQuantizerConfig | None = None
    proj_out: LlmSelectQuantizerConfig | None = None
    proj_1st: LlmSelectQuantizerConfig | None = None
    proj_2nd: LlmSelectQuantizerConfig | None = None
    router: LlmSelectQuantizerConfig | None = None

    def generate_dirnames(
        self,
        shape: torch.Size = torch.Size((4096, 4096, 16, 16)),
        dtype: torch.dtype = torch.float16,
        prefix: str = "",
    ) -> list[str]:
        proj_qkv = [] if self.proj_qkv is None else self.proj_qkv.generate_dirnames(shape, dtype)
        proj_out = [] if self.proj_out is None else self.proj_out.generate_dirnames(shape, dtype)
        proj_1st = [] if self.proj_1st is None else self.proj_1st.generate_dirnames(shape, dtype)
        proj_2nd = [] if self.proj_2nd is None else self.proj_2nd.generate_dirnames(shape, dtype)
        router = [] if self.router is None else self.router.generate_dirnames(shape, dtype)
        num_level = max(len(proj_qkv), len(proj_out), len(proj_1st), len(proj_2nd), len(router))
        names = []
        if num_level == 0:
            return names
        for level in range(num_level):
            name = f"-proj_qkv.[{proj_qkv[level]}]" if level < len(proj_qkv) else ""
            name += f"-proj_out.[{proj_out[level]}]" if level < len(proj_out) else ""
            name += f"-proj_1st.[{proj_1st[level]}]" if level < len(proj_1st) else ""
            name += f"-proj_2nd.[{proj_2nd[level]}]" if level < len(proj_2nd) else ""
            name += f"-router.[{router[level]}]" if level < len(router) else ""
            names.append(name[1:])
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names

    def generate_calib_range_dirnames(self, prefix: str = "") -> str:
        proj_qkv = [] if self.proj_qkv is None else self.proj_qkv.generate_calib_range_dirnames()
        proj_out = [] if self.proj_out is None else self.proj_out.generate_calib_range_dirnames()
        proj_1st = [] if self.proj_1st is None else self.proj_1st.generate_calib_range_dirnames()
        proj_2nd = [] if self.proj_2nd is None else self.proj_2nd.generate_calib_range_dirnames()
        router = [] if self.router is None else self.router.generate_calib_range_dirnames()
        num_level = max(len(proj_qkv), len(proj_out), len(proj_1st), len(proj_2nd), len(router))
        names = []
        if num_level == 0:
            return names
        for level in range(num_level):
            name = f"-proj_qkv.[{proj_qkv[level]}]" if level < len(proj_qkv) else ""
            name += f"-proj_out.[{proj_out[level]}]" if level < len(proj_out) else ""
            name += f"-proj_1st.[{proj_1st[level]}]" if level < len(proj_1st) else ""
            name += f"-proj_2nd.[{proj_2nd[level]}]" if level < len(proj_2nd) else ""
            name += f"-router.[{router[level]}]" if level < len(router) else ""
            names.append(name[1:])
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names


@configclass
@dataclass
class LlmAttnQuantConfig:
    """Large Language Model Attention Modules quantization configuration.

    Args:
        attn_q (LlmSelectQuantConfig): The quantization configuration for the query projection.
        attn_k (LlmSelectQuantConfig): The quantization configuration for the key projection.
        attn_v (LlmSelectQuantConfig): The quantization configuration for the value projection.
    """

    attn_q: LlmSelectQuantizerConfig | None = None
    attn_k: LlmSelectQuantizerConfig | None = None
    attn_v: LlmSelectQuantizerConfig | None = None

    def generate_dirnames(
        self,
        shape: torch.Size = torch.Size((4096, 4096, 16, 16)),
        dtype: torch.dtype = torch.float16,
        prefix: str = "",
    ) -> list[str]:
        attn_q = [] if self.attn_q is None else self.attn_q.generate_dirnames(shape, dtype)
        attn_k = [] if self.attn_k is None else self.attn_k.generate_dirnames(shape, dtype)
        attn_v = [] if self.attn_v is None else self.attn_v.generate_dirnames(shape, dtype)
        num_level = max(len(attn_q), len(attn_k), len(attn_v))
        if num_level == 0:
            return []
        names = []
        for level in range(num_level):
            name = f"-attn_q.[{attn_q[level]}]" if level < len(attn_q) else ""
            name += f"-attn_k.[{attn_k[level]}]" if level < len(attn_k) else ""
            name += f"-attn_v.[{attn_v[level]}]" if level < len(attn_v) else ""
            names.append(name[1:])
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names

    def generate_calib_range_dirnames(self, prefix: str = "") -> str:
        attn_q = [] if self.attn_q is None else self.attn_q.generate_calib_range_dirnames()
        attn_k = [] if self.attn_k is None else self.attn_k.generate_calib_range_dirnames()
        attn_v = [] if self.attn_v is None else self.attn_v.generate_calib_range_dirnames()
        num_level = max(len(attn_q), len(attn_k), len(attn_v))
        if num_level == 0:
            return []
        names = []
        for level in range(num_level):
            name = f"-attn_q.[{attn_q[level]}]" if level < len(attn_q) else ""
            name += f"-attn_k.[{attn_k[level]}]" if level < len(attn_k) else ""
            name += f"-attn_v.[{attn_v[level]}]" if level < len(attn_v) else ""
            names.append(name[1:])
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names
