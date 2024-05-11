# -*- coding: utf-8 -*-
"""Quantization config."""

import copy
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from omniconfig import configclass
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from lmquant.quant.config import (
    BaseQuantCalibConfig,
    ModuleQuantizerConfig,
    QuantReorderConfig,
    QuantRotationConfig,
    QuantSmoothConfig,
    SearchBasedCalibGranularity,
    TensorQuantizerConfig,
)

from ...nn import RotaryEmbedding
from .key import LlmModuleKey
from .select import LlmAttnQuantConfig, LlmProjQuantConfig, LlmSelectQuantizerConfig

__all__ = ["LlmQuantConfig", "LlmQuantConfig"]


@dataclass
class LlmQuantCachePath:
    """LLM quantization cache path."""

    rotation: str = ""
    reorder: str = ""
    smooth: str = ""
    wgts: str = ""
    acts: str = ""

    def clone(self) -> "LlmQuantCachePath":
        """Clone the cache paths.

        Returns:
            ModuleQuantCachePath: The cloned cache paths.
        """
        return LlmQuantCachePath(
            rotation=self.rotation,
            reorder=self.reorder,
            smooth=self.smooth,
            wgts=self.wgts,
            acts=self.acts,
        )

    def add_parent_dirs(self, *parent_dirs: str) -> "LlmQuantCachePath":
        """Add the parent directories to the cache paths.

        Args:
            parent_dirs (str): The parent directories.
        """
        if self.rotation:
            self.rotation = os.path.join(*parent_dirs, self.rotation)
        if self.reorder:
            self.reorder = os.path.join(*parent_dirs, self.reorder)
        if self.smooth:
            self.smooth = os.path.join(*parent_dirs, self.smooth)
        if self.wgts:
            self.wgts = os.path.join(*parent_dirs, self.wgts)
        if self.acts:
            self.acts = os.path.join(*parent_dirs, self.acts)
        return self

    def add_chidren(self, *children: str) -> "LlmQuantCachePath":
        """Add the children to the cache paths.

        Args:
            children (str): The children.
        """
        if self.rotation:
            self.rotation = os.path.join(self.rotation, *children)
        if self.reorder:
            self.reorder = os.path.join(self.reorder, *children)
        if self.smooth:
            self.smooth = os.path.join(self.smooth, *children)
        if self.wgts:
            self.wgts = os.path.join(self.wgts, *children)
        if self.acts:
            self.acts = os.path.join(self.acts, *children)
        return self


@configclass
@dataclass
class LlmQuantConfig(ModuleQuantizerConfig):
    """Large Language Model Module quantization configuration.

    Args:
        wgts (WeightQuantizerConfig): The weight quantization configuration.
        ipts (ActivationQuantizerConfig): The input activation quantization configuration.
        opts (ActivationQuantizerConfig): The output activation quantization configuration.
        rotation (QuantRotationConfig): The rotation configuration. Defaults to ``None``.
        smooth (SmoothQuantConfig): The smooth quantization configuration. Defaults to ``None``.
        reorder (ChannelReorderConfig): The channel reorder configuration. Defaults to ``None``.
        bias_correction (bool): Whether to correct the bias. Defaults to ``False``.
        post_rotary (bool): Whether to apply quantization after the rotary embedding. Defaults to ``True``.
        select_wgts (LlmProjQuantConfig): The extra weight quantization configuration. Defaults to ``None``.
        select_ipts (LlmProjQuantConfig): The extra input quantization configuration. Defaults to ``None``.
        select_opts (LlmAttnQuantConfig): The extra output quantization configuration. Defaults to ``None``.
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
    """

    rotation: QuantRotationConfig | None = None
    reorder: QuantReorderConfig | None = None
    smooth: QuantSmoothConfig | None = None
    bias_correction: bool = False
    post_rotary: bool = True
    develop_dtype: torch.dtype = field(default_factory=lambda s=None: eval(s) if isinstance(s, str) else s)
    select_wgts: LlmProjQuantConfig | None = None
    select_ipts: LlmProjQuantConfig | None = None
    select_opts: LlmAttnQuantConfig | None = None
    keywords_i: dict[str, list[str]] = field(init=False, default_factory=dict)
    keywords_w: dict[str, list[str]] = field(init=False, default_factory=dict)
    keywords_o: dict[str, list[str]] = field(init=False, default_factory=dict)
    module_types_i: list[type[nn.Module]] = field(init=False, default=(nn.Linear, MixtralSparseMoeBlock))
    module_types_w: list[type[nn.Module]] = field(init=False, default=(nn.Linear,))
    module_types_o: list[type[nn.Module]] = field(init=False, default=(nn.Linear, RotaryEmbedding))
    num_hidden_layers: int = field(init=False, default=-1)

    @property
    def enabled_smooth(self) -> bool:
        """Whether to enable smooth quantization."""
        return self.smooth is not None

    @property
    def enabled_smooth_xw(self) -> bool:
        """Whether to enable xw smooth quantization."""
        return self.enabled_smooth and self.smooth.enabled_smooth_xw

    @property
    def enabled_smooth_yx(self) -> bool:
        """Whether to enable yy smooth quantization."""
        return self.enabled_smooth and self.smooth.enabled_smooth_yx

    @property
    def enabled_reorder(self) -> bool:
        """Whether to enable channel reorder."""
        return self.reorder is not None

    @property
    def enabled_rotation(self) -> bool:
        """Whether to enable rotation."""
        return self.rotation is not None

    @property
    def enabled_bias_correction(self) -> bool:
        """Whether to correct the bias."""
        return self.bias_correction

    @property
    def needs_orig_wgts(self) -> bool:
        """Whether to keep the original weights."""
        if self.enabled_ipts and self.ipts.enabled_calib_range:
            return True
        if self.enabled_opts and self.opts.enabled_calib_range:
            return True
        return self.enabled_wgts and self.enabled_bias_correction

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

    def init_keywords(self):
        """Initialize the keywords settings for the language model quantization configuration.

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
            "attn_q": ["q_rotary_emb" if self.post_rotary else "q_proj"],
            "attn_k": ["k_rotary_emb" if self.post_rotary else "k_proj"],
            "attn_v": ["v_proj"],
        }
        keywords_w = copy.deepcopy(keywords_i)
        keywords_i["router"] = ["block_sparse_moe"]
        keywords_w["router"] = ["block_sparse_moe.gate"]
        if self.ipts is not None and self.ipts.enabled_for("router"):
            keywords_i["proj_1st"].remove("w1")
            keywords_i["proj_1st"].remove("w3")
        self.keywords_i = keywords_i
        self.keywords_w = keywords_w
        self.keywords_o = keywords_o

    def __post_init__(self) -> None:  # noqa: C901
        self.init_keywords()
        if self.smooth is not None:
            if not self.smooth.enabled_smooth_xw and not self.smooth.enabled_smooth_yx:
                self.smooth = None
        if self.rotation is not None and self.reorder is not None:
            self.reorder.skips.append("residual")
            if self.rotation.with_hadamard_transform:
                self.reorder.skips.extend(self.rotation.transforms)
                self.reorder.skips = sorted(set(self.reorder.skips))
        if self.enabled_ipts:
            if self.ipts.enabled_calib_range and self.ipts.calib_range.granularity == SearchBasedCalibGranularity.Group:
                self.ipts.calib_range.granularity = SearchBasedCalibGranularity.ChannelGroup
            if self.ipts.static:
                assert self.ipts.smallest_group_shape[0] == -1, "static quantization requires batch group size to be -1"
        if self.enabled_opts:
            if self.opts.enabled_calib_range and self.opts.calib_range.granularity == SearchBasedCalibGranularity.Group:
                self.opts.calib_range.granularity = SearchBasedCalibGranularity.ChannelGroup
            if self.opts.static:
                assert self.opts.smallest_group_shape[0] == -1, "static quantization requires batch group size to be -1"
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

    def needs_quant_weights(self, param_name: str, module: nn.Module = None) -> bool:
        """Whether to quantize the weight of a module.

        Args:
            param_name (str): The name of the parameter.

        Returns:
            bool: Whether to quantize the weight of the module.
        """
        if not self.enabled_wgts:
            return False
        if module is None or isinstance(module, self.module_types_w):
            needs_quant = False
            for key, keywords in self.keywords_w.items():
                for k in keywords:
                    if k in param_name:
                        needs_quant = self.wgts.enabled_for(key)
                        break
            return needs_quant
        return False

    def needs_quant_inputs(self, module_name: str, module: nn.Module) -> bool:
        """Whether to quantize the input of a module.

        Args:
            module_name (str): The name of the module.
            module (nn.Module): The module.

        Returns:
            bool: Whether to quantize the input of the module.
        """
        if not self.enabled_ipts:
            return False
        if isinstance(module, self.module_types_i):
            needs_quant = False
            for key, keywords in self.keywords_i.items():
                for k in keywords:
                    if module_name.endswith(k):
                        needs_quant = self.ipts.enabled_for(key)
                        break
            return needs_quant
        return False

    def needs_quant_outputs(self, module_name: str, module: nn.Module) -> bool:
        """Whether to quantize the output of a module.

        Args:
            module_name (str): The name of the module.
            module (nn.Module): The module.

        Returns:
            bool: Whether to quantize the output of the module.
        """
        if not self.enabled_opts:
            return False
        if isinstance(module, self.module_types_o):
            needs_quant = False
            for key, keywords in self.keywords_o.items():
                for k in keywords:
                    if module_name.endswith(k):
                        needs_quant = self.opts.enabled_for(key)
                        break
            return needs_quant
        return False

    def generate_calib_name(self) -> str:
        name = ""
        if self.enabled_rotation:
            name += "-rot"
            if self.rotation.random:
                name += ".rnd"
        if self.enabled_reorder:
            name += "-reorder"
            if self.reorder.dynamic:
                name += ".dyn"
        if self.enabled_smooth:
            name += f"-smooth"
            if self.enabled_smooth_xw:
                name += f".xw"
            if self.enabled_smooth_yx:
                name += f".yx"
        if self.enabled_bias_correction:
            name += "-bias"
        calib_name = super().generate_calib_name()
        if calib_name:
            name += f"-{calib_name}"
        return name[1:] if name else name

    def generate_cache_dirpath(self) -> LlmQuantCachePath:  # noqa: C901
        """Generate the cache paths for the module quantization configuration."""
        quant_names = self.generate_dirnames()
        w_kernel_names = self.wgts.generate_calib_kernel_dirnames(prefix="w.kernel")
        if self.enabled_rotation:
            quant_names.extend(self.rotation.generate_dirnames(prefix="rotate"))
        reorder_dirpath = ""
        if self.enabled_reorder:
            reorder_names = self.reorder.generate_dirnames(prefix="reorder")
            if self.reorder.allow_kernel_calib:
                reorder_names.extend(w_kernel_names)
                w_kernel_names = []
            quant_names.extend(reorder_names)
            reorder_dirpath = os.path.join("reorder", *quant_names)
        smooth_dirpath = ""
        if self.enabled_smooth:
            smooth_names = self.smooth.generate_dirnames(prefix="smooth")
            if (self.smooth.enabled_smooth_xw and self.smooth.xw.allow_kernel_calib) or (
                self.smooth.enabled_smooth_yx and self.smooth.yx.allow_kernel_calib
            ):
                smooth_names.extend(w_kernel_names)
                w_kernel_names = []
            quant_names.extend(smooth_names)
            smooth_dirpath = os.path.join("smooth", *quant_names)
        quant_names.extend(w_kernel_names)
        wgts_dirpath = ""
        if self.enabled_wgts and self.wgts.enabled_calib_range:
            quant_names.extend(self.wgts.generate_calib_range_dirnames(prefix="w.range"))
            wgts_dirpath = os.path.join("wgts", *quant_names)
        needs_acts_cache, acts_dirpath = False, ""
        if self.enabled_ipts and self.ipts.enabled_calib_range:
            quant_names.extend(self.ipts.generate_calib_range_dirnames(prefix="x.range"))
            needs_acts_cache = True
        if self.enabled_opts and self.opts.enabled_calib_range:
            quant_names.extend(self.opts.generate_calib_range_dirnames(prefix="y.range"))
            needs_acts_cache = True
        if needs_acts_cache:
            acts_dirpath = os.path.join("acts", *quant_names)
        paths = LlmQuantCachePath(
            reorder=reorder_dirpath,
            smooth=smooth_dirpath,
            wgts=wgts_dirpath,
            acts=acts_dirpath,
        )
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
