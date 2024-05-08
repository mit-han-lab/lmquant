# -*- coding: utf-8 -*-
"""Quantization config."""

import copy
import os
import typing as tp
from dataclasses import dataclass, field
from enum import StrEnum

import omniconfig
import torch
import torch.nn as nn
from omniconfig import configclass
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from lmquant.quant.config import (
    BaseQuantCalibConfig,
    ModelQuantConfig,
    ModuleQuantizerConfig,
    QuantCachePath,
    QuantizerKernelConfig,
    TensorQuantizerConfig,
)

from ..nn import RotaryEmbedding

__all__ = ["LlmQuantConfig", "LlmModuleKey"]


class LlmModuleKey(StrEnum):
    """Large Language Model Module keys."""

    PROJ_QKV = "proj_qkv"
    PROJ_OUT = "proj_out"
    PROJ_1ST = "proj_1st"
    PROJ_2ND = "proj_2nd"
    ROUTER = "router"
    ATTN_Q = "attn_q"
    ATTN_K = "attn_k"
    ATTN_V = "attn_v"
    ATTN_QK = "attn_qk"


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


@configclass
@dataclass
class LlmQuantConfig(ModelQuantConfig):
    """Large Language Model Module quantization configuration.

    Args:
        wgts (WeightQuantizerConfig): The weight quantization configuration.
        ipts (ActivationQuantizerConfig): The input activation quantization configuration.
        opts (ActivationQuantizerConfig): The output activation quantization configuration.
        rotation (QuantRotationConfig): The rotation configuration. Defaults to ``None``.
        smooth (SmoothQuantConfig): The smooth quantization configuration. Defaults to ``None``.
        reorder (ChannelReorderConfig): The channel reorder configuration. Defaults to ``None``.
        bias_correction (bool): Whether to correct the bias. Defaults to ``False``.
        keywords_i (dict[str, list[str]]): The module name keywords for the input quantization.
            Defaults to ``{}``.
        keywords_w (dict[str, list[str]]): The param name keywords for the weight quantization.
            Defaults to ``{}``.
        keywords_o (dict[str, list[str]]): The module name keywords for the output quantization.
            Defaults to ``{}``.
        module_types_i (list[type[nn.Module]] | type[nn.Module]): The module types for the input quantization.
            Defaults to ``[]``.
        module_types_w (list[type[nn.Module]] | type[nn.Module]): The module types for the weight quantization.
            Defaults to ``[]``.
        module_types_o (list[type[nn.Module]] | type[nn.Module]): The module types for the output quantization.
            Defaults to ``[]``.
        channels_dims (dict[type[nn.Module], tuple[int, int]]): The channel dimensions of the inputs and outputs
            of the modules. Defaults to ``{}``.
        post_rotary (bool): Whether to apply quantization after the rotary embedding. Defaults to ``True``.
        select_wgts (LlmProjQuantConfig): The extra weight quantization configuration. Defaults to ``None``.
        select_ipts (LlmProjQuantConfig): The extra input quantization configuration. Defaults to ``None``.
        select_opts (LlmAttnQuantConfig): The extra output quantization configuration. Defaults to ``None``.
    """

    module_types_i: list[type[nn.Module]] = field(init=False, default=(nn.Linear, MixtralSparseMoeBlock))
    module_types_w: list[type[nn.Module]] = field(init=False, default=(nn.Linear,))
    module_types_o: list[type[nn.Module]] = field(init=False, default=(nn.Linear, RotaryEmbedding))
    channels_dims: dict[type[nn.Module], tuple[int, int]] = field(default_factory=dict)
    post_rotary: bool = True
    develop_dtype: torch.dtype = field(default_factory=lambda s=None: eval(s) if isinstance(s, str) else s)
    select_wgts: LlmProjQuantConfig | None = None
    select_ipts: LlmProjQuantConfig | None = None
    select_opts: LlmAttnQuantConfig | None = None
    num_hidden_layers: int = field(init=False, default=-1)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.channels_dims.update({nn.Linear: (-1, -1), RotaryEmbedding: (-1, -1), MixtralSparseMoeBlock: (-1, -1)})
        self.set_keywords(
            **self._generate_keywords(
                post_rotary=self.post_rotary, skip_router_ipts=self.ipts is None or not self.ipts.enabled_for("router")
            )
        )
        if self.enabled_reorder:
            if not self.reorder.dynamic:
                if LlmModuleKey.PROJ_QKV in self.reorder.skips:
                    self.reorder.skips.append("residual")
                if LlmModuleKey.PROJ_OUT in self.reorder.skips:
                    self.reorder.skips.append("residual")
            if "residual" in self.reorder.skips:
                self.reorder.skips.append(LlmModuleKey.PROJ_QKV.value)
                self.reorder.skips.append(LlmModuleKey.PROJ_OUT.value)
            self.reorder.skips = sorted(set(self.reorder.skips))
        # region update the quantization configuration for selective quantization
        keys: dict[str, tuple[str, ...]] = {
            "wgts": ("proj_qkv", "proj_out", "proj_1st", "proj_2nd", "router"),
            "opts": ("attn_q", "attn_k", "attn_v"),
        }
        keys["ipts"] = keys["wgts"]
        for tensor, tensor_keys in keys.items():
            select_config = getattr(self, f"select_{tensor}", None)
            if select_config is not None:
                default_config = getattr(self, tensor, None)
                assert default_config is not None, f"The default {tensor} quantization configuration is required."
                assert isinstance(default_config, TensorQuantizerConfig)
                for key in tensor_keys:
                    key_config = getattr(select_config, key, None)
                    if key_config is not None:
                        assert isinstance(key_config, LlmSelectQuantizerConfig)
                        key_config.skips = default_config.skips
                        if not default_config.calib_kernel.enabled_for(key):
                            key_config.calib_kernel = None
                        else:
                            key_config.calib_kernel = default_config.calib_kernel
                        for field in key_config.__dict__:
                            if field.startswith("calib_") and field != "calib_kernel":
                                default_field_config = getattr(default_config, field)
                                assert isinstance(default_field_config, BaseQuantCalibConfig)
                                if default_field_config.enabled_for(key):
                                    setattr(key_config, field, None)
                                elif not getattr(key_config, f"update_{field}", False):
                                    setattr(key_config, field, default_field_config)
        # endregion

    def dump(self, path: str = "") -> dict[str, tp.Any]:
        """Dump configurations.

        Args:
            path (str): Path to dump the configurations.

        Returns:
            dict[str, tp.Any]: Dumped configurations.
        """
        result = omniconfig.dump(self)
        result.pop("channels_dims", None)
        result.pop("keywords_i", None)
        result.pop("keywords_w", None)
        result.pop("keywords_o", None)
        result.pop("module_types_i", None)
        result.pop("module_types_w", None)
        result.pop("module_types_o", None)
        if path:
            if path.endswith(("yaml", "yml")):
                omniconfig.dump_yaml(result, path)
            elif path.endswith("toml"):
                omniconfig.dump_toml(result, path)
            else:
                raise ValueError(f"Unsupported file format: {path}")
        return result

    def generate_cache_dirpath(self) -> QuantCachePath:  # noqa: C901
        """Generate the cache paths for the module quantization configuration."""
        paths = super().generate_cache_dirpath()
        extra_names, extra_wgts_range_names, extra_acts_range_names = [], [], []
        if self.select_wgts is not None:
            extra_names.extend(self.select_wgts.generate_dirnames("w"))
            extra_wgts_range_names = self.select_wgts.generate_calib_range_dirnames("w")
        if self.select_ipts is not None:
            extra_names.extend(self.select_ipts.generate_dirnames("x"))
            extra_acts_range_names.extend(self.select_ipts.generate_calib_range_dirnames("x"))
        if self.select_opts is not None:
            extra_names.extend(self.select_opts.generate_dirnames("y"))
            extra_acts_range_names.extend(self.select_opts.generate_calib_range_dirnames("y"))
        if extra_names:
            extra_path = os.path.join(*extra_names)
            paths.add_chidren(extra_path)
        if extra_wgts_range_names:
            extra_path = os.path.join(*extra_wgts_range_names)
            if paths.wgts:
                paths.wgts = os.path.join(paths.wgts, extra_path)
            if paths.acts:
                paths.acts = os.path.join(paths.acts, extra_path)
        if extra_acts_range_names:
            extra_path = os.path.join(*extra_acts_range_names)
            if paths.acts:
                paths.acts = os.path.join(paths.acts, extra_path)
        return paths

    # region get quantization configuration

    def get_attn_q(self, *, layer_idx: int = -1) -> ModuleQuantizerConfig:
        """Get the projection key quantization configuration."""
        config = ModuleQuantizerConfig(wgts=None, ipts=None, opts=self.opts)
        if self.select_opts is not None and self.select_opts.attn_q is not None:
            if self.select_opts.attn_q.is_selected(layer_idx, self.num_hidden_layers):
                config.opts = self.select_opts.attn_q
        return config

    def get_attn_k(self, *, layer_idx: int = -1) -> ModuleQuantizerConfig:
        """Get the projection key quantization configuration."""
        config = ModuleQuantizerConfig(wgts=None, ipts=None, opts=self.opts)
        if self.select_opts is not None and self.select_opts.attn_k is not None:
            if self.select_opts.attn_k.is_selected(layer_idx, self.num_hidden_layers):
                config.opts = self.select_opts.attn_k
        return config

    def get_attn_v(self, *, layer_idx: int = -1) -> ModuleQuantizerConfig:
        """Get the projection value quantization configuration."""
        config = ModuleQuantizerConfig(wgts=None, ipts=None, opts=self.opts)
        if self.select_opts is not None and self.select_opts.attn_v is not None:
            if self.select_opts.attn_v.is_selected(layer_idx, self.num_hidden_layers):
                config.opts = self.select_opts.attn_v
        return config

    def get_attn_qk(self, *, layer_idx: int = -1) -> ModuleQuantizerConfig:
        """Get the projection value quantization configuration."""
        config = ModuleQuantizerConfig(wgts=None, ipts=self.opts, opts=self.opts)
        if self.select_opts is not None and self.select_opts.attn_q is not None:
            if self.select_opts.attn_q.is_selected(layer_idx, self.num_hidden_layers):
                config.ipts = self.select_opts.attn_q
        if self.select_opts is not None and self.select_opts.attn_k is not None:
            if self.select_opts.attn_k.is_selected(layer_idx, self.num_hidden_layers):
                config.opts = self.select_opts.attn_k
        return config

    def get_proj_qkv(self, *, layer_idx: int = -1) -> ModuleQuantizerConfig:
        """Get the projection value quantization configuration."""
        config = ModuleQuantizerConfig(wgts=self.wgts, ipts=self.ipts, opts=None)
        if self.select_wgts is not None and self.select_wgts.proj_qkv is not None:
            if self.select_wgts.proj_qkv.is_selected(layer_idx, self.num_hidden_layers):
                config.wgts = self.select_wgts.proj_qkv
        if self.select_ipts is not None and self.select_ipts.proj_qkv is not None:
            if self.select_ipts.proj_qkv.is_selected(layer_idx, self.num_hidden_layers):
                config.ipts = self.select_ipts.proj_qkv
        return config

    def get_proj_out(self, *, layer_idx: int = -1) -> ModuleQuantizerConfig:
        """Get the projection output quantization configuration."""
        config = ModuleQuantizerConfig(wgts=self.wgts, ipts=self.ipts, opts=None)
        if self.select_wgts is not None and self.select_wgts.proj_out is not None:
            if self.select_wgts.proj_out.is_selected(layer_idx, self.num_hidden_layers):
                config.wgts = self.select_wgts.proj_out
        if self.select_ipts is not None and self.select_ipts.proj_out is not None:
            if self.select_ipts.proj_out.is_selected(layer_idx, self.num_hidden_layers):
                config.ipts = self.select_ipts.proj_out
        return config

    def get_proj_1st(self, *, layer_idx: int = -1) -> ModuleQuantizerConfig:
        """Get the first feed-forward network quantization configuration."""
        config = ModuleQuantizerConfig(wgts=self.wgts, ipts=self.ipts, opts=None)
        if self.select_wgts is not None and self.select_wgts.proj_1st is not None:
            if self.select_wgts.proj_1st.is_selected(layer_idx, self.num_hidden_layers):
                config.wgts = self.select_wgts.proj_1st
        if self.select_ipts is not None and self.select_ipts.proj_1st is not None:
            if self.select_ipts.proj_1st.is_selected(layer_idx, self.num_hidden_layers):
                config.ipts = self.select_ipts.proj_1st
        return config

    def get_proj_2nd(self, *, layer_idx: int = -1) -> ModuleQuantizerConfig:
        """Get the second feed-forward network quantization configuration."""
        config = ModuleQuantizerConfig(wgts=self.wgts, ipts=self.ipts, opts=None)
        if self.select_wgts is not None and self.select_wgts.proj_2nd is not None:
            if self.select_wgts.proj_2nd.is_selected(layer_idx, self.num_hidden_layers):
                config.wgts = self.select_wgts.proj_2nd
        if self.select_ipts is not None and self.select_ipts.proj_2nd is not None:
            if self.select_ipts.proj_2nd.is_selected(layer_idx, self.num_hidden_layers):
                config.ipts = self.select_ipts.proj_2nd
        return config

    def get_router(self, *, layer_idx: int = -1) -> ModuleQuantizerConfig:
        """Get the router quantization configuration."""
        config = ModuleQuantizerConfig(wgts=self.wgts, ipts=self.ipts, opts=None)
        if self.select_wgts is not None and self.select_wgts.router is not None:
            if self.select_wgts.router.is_selected(layer_idx, self.num_hidden_layers):
                config.wgts = self.select_wgts.router
        if self.select_ipts is not None and self.select_ipts.router is not None:
            if self.select_ipts.router.is_selected(layer_idx, self.num_hidden_layers):
                config.ipts = self.select_ipts.router
        return config

    def specialize_for(self, key: str, *, layer_idx: int = -1) -> ModuleQuantizerConfig:
        """Get the quantization configuration for the specified key."""
        return getattr(self, f"get_{key}")(layer_idx=layer_idx)

    # endregion

    # region Large Language Model Quantization Configuration Arguments

    @staticmethod
    def _generate_keywords(post_rotary: bool = True, skip_router_ipts: bool = False) -> dict[str, dict[str, list[str]]]:
        """Get the keywords settings for the language model quantization configuration.

        Args:
            llama (bool): Whether to use the llama model.

        Returns:
            dict[str, dict[str, list[str]]]: The keywords settings.
        """
        keywords_i = {
            "proj_qkv": ["q_proj", "k_proj", "v_proj"],
            "proj_out": ["out_proj", "o_proj"],
            "proj_1st": ["fc1", "up_proj", "gate_proj", "w1", "w3"],
            "proj_2nd": ["fc2", "down_proj", "w2"],
            "head": ["output", "score", "qa_outputs"],
            "embed": ["embed", "lm_head", "embed_out"],
        }
        keywords_o = {
            "attn_q": ["q_rotary_emb" if post_rotary else "q_proj"],
            "attn_k": ["k_rotary_emb" if post_rotary else "k_proj"],
            "attn_v": ["v_proj"],
        }
        keywords_w = copy.deepcopy(keywords_i)
        keywords_i["router"] = ["block_sparse_moe"]
        keywords_w["router"] = ["block_sparse_moe.gate"]
        if not skip_router_ipts:
            keywords_i["proj_1st"].remove("w1")
            keywords_i["proj_1st"].remove("w3")
        return dict(keywords_i=keywords_i, keywords_w=keywords_w, keywords_o=keywords_o)

    @staticmethod
    def _generate_smooth_skip_flags(prefix: str = "") -> dict[str, bool]:
        prefix = f"{prefix.replace('-', '_')}_" if prefix else ""
        return {
            f"{prefix}xw_skip_proj_qkv": False,
            f"{prefix}xw_skip_proj_out": False,
            f"{prefix}xw_skip_proj_1st": False,
            f"{prefix}xw_skip_proj_2nd": False,
            f"{prefix}yx_skip_attn_qk": False,
        }

    @staticmethod
    def _generate_calib_xw_flags(prefix: str = "", action: str = "skip") -> dict[str, bool]:
        prefix = f"{prefix.replace('-', '_')}_" if prefix else ""
        return {
            f"{prefix}{action}_proj_qkv": False,
            f"{prefix}{action}_proj_out": False,
            f"{prefix}{action}_proj_1st": False,
            f"{prefix}{action}_proj_2nd": False,
            f"{prefix}{action}_router": False,
        }

    @staticmethod
    def _generate_calib_yx_skip_flags(prefix: str = "") -> dict[str, bool]:
        prefix = f"{prefix.replace('-', '_')}_" if prefix else ""
        return {
            f"{prefix}skip_attn_q": False,
            f"{prefix}skip_attn_k": False,
            f"{prefix}skip_attn_v": False,
        }

    @staticmethod
    def _generate_reorder_skip_flags(prefix: str = "") -> dict[str, bool]:
        prefix = f"{prefix.replace('-', '_')}_" if prefix else ""
        return {
            f"{prefix}skip_residual": False,
            f"{prefix}skip_proj_out": False,
            f"{prefix}skip_proj_2nd": False,
        }

    @staticmethod
    def _generate_rotation_transform_flags(prefix: str = "") -> dict[str, bool]:
        prefix = f"{prefix.replace('-', '_')}_" if prefix else ""
        return {
            f"{prefix}transform_proj_out": False,
            f"{prefix}transform_proj_2nd": False,
        }

    @staticmethod
    def _generate_skip_flags() -> dict[str, bool]:
        """Get the skip flags for the language model quantization configuration.

        Args:
            include_smooth (bool): Whether to include the skip flags for the smooth quantization configuration.

        Returns:
            dict[str, bool]: The skip flags.
        """
        flags = {
            # skip flags for outputs
            "opts_skip_attn_q": False,
            "opts_skip_attn_k": False,
            "opts_skip_attn_v": False,
            # skip flags for inputs
            "ipts_skip_head": False,
            "ipts_skip_embed": False,
            "ipts_skip_proj_qkv": False,
            "ipts_skip_proj_out": False,
            "ipts_skip_proj_1st": False,
            "ipts_skip_proj_2nd": False,
            "ipts_skip_router": False,
            # skip flags for weights
            "wgts_skip_head": False,
            "wgts_skip_embed": False,
            "wgts_skip_proj_qkv": False,
            "wgts_skip_proj_out": False,
            "wgts_skip_proj_1st": False,
            "wgts_skip_proj_2nd": False,
            "wgts_skip_router": False,
        }
        return flags

    @staticmethod
    def generate_flags() -> dict[str, bool]:
        """Get the flags for the language model quantization configuration.

        Returns:
            dict[str, bool]: The flags.
        """
        flags = LlmQuantConfig._generate_skip_flags()
        flags.update(LlmQuantConfig._generate_calib_xw_flags(prefix="wgts_calib_kernel_gptq", action="include"))
        flags.update(LlmQuantConfig._generate_calib_xw_flags(prefix="wgts_calib_range", action="skip"))
        flags.update(LlmQuantConfig._generate_calib_xw_flags(prefix="ipts_calib_range", action="skip"))
        flags.update(LlmQuantConfig._generate_calib_yx_skip_flags(prefix="opts_calib_range"))
        flags.update(LlmQuantConfig._generate_smooth_skip_flags(prefix="smooth"))
        flags.update(LlmQuantConfig._generate_rotation_transform_flags(prefix="rotation"))
        flags.update(LlmQuantConfig._generate_reorder_skip_flags(prefix="reorder"))
        return flags

    # endregion
