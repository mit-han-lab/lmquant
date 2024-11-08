# -*- coding: utf-8 -*-
"""Transformer and attention module struct."""

import typing as tp
from abc import abstractmethod
from dataclasses import dataclass, field

import torch.nn as nn

from ...utils.common import join_name
from .base import BaseModuleStruct

__all__ = [
    "AttentionStruct",
    "SelfAttentionStruct",
    "CrossAttentionStruct",
    "JointAttentionStruct",
    "FeedForwardStruct",
    "FeedForwardStruct",
    "TransformerBlockStruct",
    "BaseTransformerStruct",
    "AttentionConfigStruct",
    "FeedForwardConfigStruct",
]


@dataclass(kw_only=True)
class AttentionConfigStruct:
    """Attention module configuration.

    Args:
        hidden_size (`int`):
            The size of the input/output activations, i.e., the number of input channels.
        add_hidden_size (`int`):
            The size of the additional input activations, i.e., the number of additional input channels.
        inner_size (`int`):
            The size of the inner activations, i.e., the number of **query** channels.
        num_query_heads (`int`):
            Number of query heads.
        num_key_value_heads (`int`):
            Number of key and value heads.
        with_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply normalization to queries and keys.
        with_rope (`bool`, *optional*, defaults to `True`):
            Whether to use Rotary Positional Encoding (RoPE).
        do_norm_before (`bool`, *optional*, defaults to `True`):
            Whether to apply normalization before the projection.
    """

    hidden_size: int
    add_hidden_size: int = 0
    inner_size: int
    num_query_heads: int
    num_key_value_heads: int
    with_qk_norm: bool = False
    with_rope: bool = True
    do_norm_before: bool = True

    @property
    def head_size(self) -> int:
        """Get the head size."""
        return self.num_query_channels // self.num_query_heads

    @property
    def num_key_value_groups(self) -> int:
        """Get the number of key-value groups."""
        return self.num_query_heads // self.num_key_value_heads

    @property
    def num_channels(self) -> int:
        """Get the number of channels for the input and output."""
        return self.hidden_size

    @property
    def num_add_channels(self) -> int:
        """Get the number of channels for additional inputs."""
        return self.add_hidden_size

    @property
    def num_query_channels(self) -> int:
        """Get the number of query channels."""
        return self.inner_size

    @property
    def num_key_value_channels(self) -> int:
        """Get the number of key-value channels."""
        return self.num_head_channels * self.num_key_value_heads

    @property
    def num_head_channels(self) -> int:
        """Get the head dimension."""
        return self.head_size

    @property
    def num_head_repeats(self) -> int:
        """Get the number of head repeats."""
        return self.num_key_value_groups


@dataclass(kw_only=True)
class FeedForwardConfigStruct:
    """Feed-forward module configuration.

    Args:
        hidden_size (`int`):
            The size of the input/output activations, i.e., the number of **input** channels.
        intermediate_size (`int`):
            The number of intermediate channels in the feedforward network.
        intermediate_act_type (`str`):
            The activation function for the intermediate activations in the feedforward network.
        num_experts (`int`, *optional*, defaults to `1`):
            Number of experts.
        do_norm_before (`bool`, *optional*, defaults to `True`):
            Whether to apply normalization before the projection.

    Attributes:
        intermediate_lowerbound (`float` or `None`):
            The lowerbound of the intermediate activations.
    """

    hidden_size: int
    intermediate_size: int
    intermediate_act_type: str
    num_experts: int = 1
    do_norm_before: bool = True

    @property
    def num_channels(self) -> int:
        """Get the model size."""
        return self.hidden_size

    @property
    def num_intermediate_channels(self) -> int:
        """Get the intermediate size."""
        return self.intermediate_size

    @property
    def intermediate_lowerbound(self) -> float | None:
        """The lowerbound of the intermediate activations."""
        return self.infer_lowerbound(self.intermediate_act_type)

    @staticmethod
    def infer_lowerbound(act_type: str) -> float | None:
        if act_type.endswith("_glu"):
            return None
        elif act_type.endswith("_shifted"):
            return 0
        elif act_type.startswith("relu"):
            return 0
        elif act_type == "gelu":
            return -0.171875  # -0.17
        elif act_type == "silu" or act_type == "swish":
            return -0.2734375  # -0.27
        elif act_type == "mish":
            return -0.31640625  # -0.31
        else:
            raise NotImplementedError(f"Unsupported activation type: {act_type}")


@dataclass(kw_only=True)
class AttentionStruct(BaseModuleStruct):
    """Attention module struct."""

    # region relative keys
    qkv_proj_rkey: tp.ClassVar[str] = "qkv_proj"
    add_qkv_proj_rkey: tp.ClassVar[str] = "add_qkv_proj"
    out_proj_rkey: tp.ClassVar[str] = "out_proj"
    add_out_proj_rkey: tp.ClassVar[str] = "add_out_proj"
    q_rkey: tp.ClassVar[str] = "q"
    k_rkey: tp.ClassVar[str] = "k"
    v_rkey: tp.ClassVar[str] = "v"
    # endregion

    config: AttentionConfigStruct

    # region child modules
    q_proj: nn.Linear
    """Query projection."""
    k_proj: nn.Linear | None
    """Key projection layer for self or joint attention."""
    v_proj: nn.Linear | None
    """Value projection layer for self or joint attention."""
    o_proj: nn.Linear
    """Output projection."""
    add_q_proj: nn.Linear | None
    """Additional query projection layer for joint attention."""
    add_k_proj: nn.Linear | None
    """Additional key projection layer for cross or joint attention."""
    add_v_proj: nn.Linear | None
    """Additional value projection layer for cross or joint attention."""
    add_o_proj: nn.Linear | None
    """Additional output projection."""
    q: nn.Module
    """Module that generates queries for the attention mechanism."""
    k: nn.Module
    """Module that generates keys for the attention mechanism."""
    v: nn.Module
    """Module that generates values for the attention mechanism."""
    # endregion
    # region relative names
    q_proj_rname: str
    k_proj_rname: str
    v_proj_rname: str
    o_proj_rname: str
    add_q_proj_rname: str
    add_k_proj_rname: str
    add_v_proj_rname: str
    add_o_proj_rname: str
    q_rname: str
    k_rname: str
    v_rname: str
    # endregion
    # region absolute names
    q_proj_name: str = field(init=False, repr=False)
    k_proj_name: str = field(init=False, repr=False)
    v_proj_name: str = field(init=False, repr=False)
    o_proj_name: str = field(init=False, repr=False)
    add_q_proj_name: str = field(init=False, repr=False)
    add_k_proj_name: str = field(init=False, repr=False)
    add_v_proj_name: str = field(init=False, repr=False)
    add_o_proj_name: str = field(init=False, repr=False)
    q_name: str = field(init=False, repr=False)
    k_name: str = field(init=False, repr=False)
    v_name: str = field(init=False, repr=False)
    # endregion
    # region absolute keys
    qkv_proj_key: str = field(init=False, repr=False)
    add_qkv_proj_key: str = field(init=False, repr=False)
    out_proj_key: str = field(init=False, repr=False)
    add_out_proj_key: str = field(init=False, repr=False)
    q_key: str = field(init=False, repr=False)
    k_key: str = field(init=False, repr=False)
    v_key: str = field(init=False, repr=False)
    # endregion

    # region aliases

    @property
    def qkv_proj(self) -> list[nn.Linear]:
        return [self.q_proj] if self.is_cross_attn() else [self.q_proj, self.k_proj, self.v_proj]

    @property
    def add_qkv_proj(self) -> list[nn.Linear]:
        if self.is_self_attn():
            return []
        elif self.is_cross_attn():
            return [self.add_k_proj, self.add_v_proj]
        else:
            return [self.add_q_proj, self.add_k_proj, self.add_v_proj]

    @property
    def out_proj(self) -> nn.Linear:
        return self.o_proj

    @property
    def add_out_proj(self) -> nn.Linear:
        return self.add_o_proj

    @property
    def qkv_proj_rnames(self) -> list[str]:
        return (
            [self.q_proj_rname] if self.is_cross_attn() else [self.q_proj_rname, self.k_proj_rname, self.v_proj_rname]
        )

    @property
    def add_qkv_proj_rnames(self) -> list[str]:
        if self.is_self_attn():
            return []
        elif self.is_cross_attn():
            return [self.add_k_proj_rname, self.add_v_proj_rname]
        else:
            return [self.add_q_proj_rname, self.add_k_proj_rname, self.add_v_proj_rname]

    @property
    def out_proj_rname(self) -> str:
        return self.o_proj_rname

    @property
    def add_out_proj_rname(self) -> str:
        return self.add_o_proj_rname

    @property
    def qkv_proj_names(self) -> list[str]:
        return [self.q_proj_name] if self.is_cross_attn() else [self.q_proj_name, self.k_proj_name, self.v_proj_name]

    @property
    def add_qkv_proj_names(self) -> list[str]:
        if self.is_self_attn():
            return []
        elif self.is_cross_attn():
            return [self.add_k_proj_name, self.add_v_proj_name]
        else:
            return [self.add_q_proj_name, self.add_k_proj_name, self.add_v_proj_name]

    @property
    def out_proj_name(self) -> str:
        return self.o_proj_name

    @property
    def add_out_proj_name(self) -> str:
        return self.add_o_proj_name

    # endregion

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.o_proj is not None
        if self.add_k_proj is None:  # self attention
            assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None
            assert self.add_q_proj is None and self.add_v_proj is None
            assert self.add_o_proj is None
        elif self.k_proj is None:  # cross attention
            assert self.q_proj is not None and self.add_v_proj is not None
            assert self.add_q_proj is None and self.v_proj is None
            assert self.add_o_proj is None
        else:  # joint attention
            assert self.q_proj is not None and self.add_q_proj is not None
            assert self.v_proj is not None and self.add_v_proj is not None
            # self.add_o_proj can be None or not
        for field_name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "add_o_proj",
            "q",
            "k",
            "v",
        ):
            rname = getattr(self, f"{field_name}_rname")
            if getattr(self, field_name) is not None or rname:
                assert rname, f"`{field_name}_rname` must not be empty if `{field_name}` is not None"
                setattr(self, f"{field_name}_name", join_name(self.name, rname))
            else:
                setattr(self, f"{field_name}_name", "")
        self.qkv_proj_key = join_name(self.key, self.qkv_proj_rkey, sep="_")
        self.add_qkv_proj_key = join_name(self.key, self.add_qkv_proj_rkey, sep="_")
        self.out_proj_key = join_name(self.key, self.out_proj_rkey, sep="_")
        self.add_out_proj_key = join_name(self.key, self.add_out_proj_rkey, sep="_")
        self.q_key = join_name(self.key, self.q_rkey, sep="_")
        self.k_key = join_name(self.key, self.k_rkey, sep="_")
        self.v_key = join_name(self.key, self.v_rkey, sep="_")
        # region assertions
        if self.q_proj is not None:
            assert self.q_proj.weight.shape[1] == self.config.num_channels
            assert self.q_proj.weight.shape[0] == self.config.num_query_channels
        if self.add_q_proj is not None:
            assert self.add_q_proj.weight.shape[1] == self.config.num_add_channels
            assert self.add_q_proj.weight.shape[0] == self.config.num_query_channels
        if self.k_proj is not None:
            assert self.k_proj.weight.shape[1] == self.config.num_channels
            assert self.k_proj.weight.shape[0] == self.config.num_key_value_channels
        if self.add_k_proj is not None:
            assert self.add_k_proj.weight.shape[0] == self.config.num_key_value_channels
            assert self.add_k_proj.weight.shape[1] == self.config.num_add_channels
        if self.v_proj is not None:
            assert self.v_proj.weight.shape[1] == self.config.num_channels
            assert self.v_proj.weight.shape[0] == self.config.num_key_value_channels
        if self.add_v_proj is not None:
            assert self.add_v_proj.weight.shape[0] == self.config.num_key_value_channels
            assert self.add_v_proj.weight.shape[1] == self.config.num_add_channels
        if self.o_proj is not None:
            assert self.o_proj.weight.shape[1] == self.config.num_query_channels
            assert self.o_proj.weight.shape[0] == self.config.num_channels
        if self.add_o_proj is not None:
            assert self.add_o_proj.weight.shape[1] == self.config.num_query_channels
            assert self.add_o_proj.weight.shape[0] == self.config.num_add_channels
        # endregion

    def is_self_attn(self) -> bool:
        return self.add_k_proj is None

    def is_cross_attn(self) -> bool:
        return self.k_proj is None

    def is_joint_attn(self) -> bool:
        return self.add_k_proj is not None and self.k_proj is not None

    def filter_kwargs(self, kwargs: dict) -> dict:
        """Extract the keyword arguments that are relevant to the attention module."""
        return kwargs

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        yield self.qkv_proj_key, self.q_proj_name, self.q_proj, self, "q_proj"
        if self.k_proj is not None:
            yield self.qkv_proj_key, self.k_proj_name, self.k_proj, self, "k_proj"
        if self.v_proj is not None:
            yield self.qkv_proj_key, self.v_proj_name, self.v_proj, self, "v_proj"
        if self.add_q_proj is not None:
            yield self.add_qkv_proj_key, self.add_q_proj_name, self.add_q_proj, self, "add_q_proj"
        if self.add_k_proj is not None:
            yield self.add_qkv_proj_key, self.add_k_proj_name, self.add_k_proj, self, "add_k_proj"
        if self.add_v_proj is not None:
            yield self.add_qkv_proj_key, self.add_v_proj_name, self.add_v_proj, self, "add_v_proj"
        yield self.out_proj_key, self.o_proj_name, self.o_proj, self, "o_proj"
        if self.add_o_proj is not None:
            yield self.add_out_proj_key, self.add_o_proj_name, self.add_o_proj, self, "add_o_proj"

    def iter_attention_structs(self) -> tp.Generator[tp.Self, None, None]:
        yield self

    @classmethod
    def get_default_keys(cls) -> list[str]:
        """Get the default keys."""
        return [cls.qkv_proj_rkey, cls.add_qkv_proj_rkey, cls.out_proj_rkey, cls.add_out_proj_rkey]


@dataclass(kw_only=True)
class SelfAttentionStruct(AttentionStruct):
    """Self-attention module struct."""

    # region child modules
    k_proj: nn.Linear
    """Key projection."""
    v_proj: nn.Linear
    """Value projection."""
    add_q_proj: None = field(init=False, repr=False, default=None)
    add_k_proj: None = field(init=False, repr=False, default=None)
    add_v_proj: None = field(init=False, repr=False, default=None)
    add_o_proj: None = field(init=False, repr=False, default=None)
    # endregion
    # region relative names
    add_q_proj_rname: str = field(init=False, repr=False, default="")
    add_k_proj_rname: str = field(init=False, repr=False, default="")
    add_v_proj_rname: str = field(init=False, repr=False, default="")
    add_o_proj_rname: str = field(init=False, repr=False, default="")
    # endregion

    @classmethod
    def get_default_keys(cls) -> list[str]:
        """Get the default keys."""
        return [cls.qkv_proj_rkey, cls.out_proj_rkey]


@dataclass(kw_only=True)
class CrossAttentionStruct(AttentionStruct):
    """Cross-attention module struct."""

    # region child modules
    k_proj: None = field(init=False, repr=False, default=None)
    v_proj: None = field(init=False, repr=False, default=None)
    add_q_proj: None = field(init=False, repr=False, default=None)
    add_k_proj: nn.Linear
    """Additional key projection."""
    add_v_proj: nn.Linear
    """Additional value projection."""
    add_o_proj: None = field(init=False, repr=False, default=None)
    # endregion
    # region relative names
    k_proj_rname: str = field(init=False, repr=False, default="")
    v_proj_rname: str = field(init=False, repr=False, default="")
    add_q_proj_rname: str = field(init=False, repr=False, default="")
    add_o_proj_rname: str = field(init=False, repr=False, default="")
    # endregion

    @classmethod
    def get_default_keys(cls) -> list[str]:
        """Get the default keys."""
        return [cls.qkv_proj_rkey, cls.add_qkv_proj_rkey, cls.out_proj_rkey]


@dataclass(kw_only=True)
class JointAttentionStruct(AttentionStruct):
    """Joint-attention module struct."""

    # region child modules
    k_proj: nn.Linear
    """Key projection."""
    v_proj: nn.Linear
    """Value projection."""
    add_q_proj: nn.Linear
    """Additional query projection."""
    add_k_proj: nn.Linear
    """Additional key projection."""
    add_v_proj: nn.Linear
    """Additional value projection."""
    # endregion


@dataclass(kw_only=True)
class FeedForwardStruct(BaseModuleStruct):
    """Feed-forward module struct."""

    # region relative keys
    up_proj_rkey: tp.ClassVar[str] = "up_proj"
    down_proj_rkey: tp.ClassVar[str] = "down_proj"
    moe_gate_rkey: tp.ClassVar[str] = "moe_gate"
    # endregion

    config: FeedForwardConfigStruct

    # region child modules
    up_projs: list[nn.Linear]
    """Up projections."""
    down_projs: list[nn.Linear]
    """Down projections."""
    moe_gate: nn.Linear | None
    """Mixture of experts gate."""
    experts: list[nn.Module]
    """Expert modules."""
    # endregion
    # region relative names
    up_proj_rnames: list[str]
    down_proj_rnames: list[str]
    moe_gate_rname: str
    experts_rname: str
    # endregion
    # region absolute names
    up_proj_names: list[str] = field(init=False, repr=False)
    down_proj_names: list[str] = field(init=False, repr=False)
    moe_gate_name: str = field(init=False, repr=False)
    experts_name: str = field(init=False, repr=False)
    expert_names: list[str] = field(init=False, repr=False)
    # endregion
    # region absolute keys
    up_proj_key: str = field(init=False, repr=False)
    down_proj_key: str = field(init=False, repr=False)
    moe_gate_key: str = field(init=False, repr=False)
    # endregion

    def __post_init__(self) -> None:
        super().__post_init__()
        num_experts = len(self.experts)
        assert len(self.up_projs) == num_experts * len(self.up_proj_rnames)
        assert len(self.down_projs) == num_experts * len(self.down_proj_rnames)
        if num_experts > 1:
            assert self.experts_rname, "experts name must be provided for MoE"
            assert self.moe_gate_rname, "moe gate name must be provided for MoE"
            assert self.moe_gate is not None, "moe gate must be provided for MoE"
            self.moe_gate_name = join_name(self.name, self.moe_gate_rname)
            self.experts_name = join_name(self.name, self.experts_rname)
            self.expert_names = [join_name(self.experts_name, str(e)) for e in range(num_experts)]
        else:
            assert self.moe_gate is None, "moe gate must be empty for non-MoE"
            self.experts_rname = self.experts_name = self.moe_gate_rname = self.moe_gate_name = ""
            self.expert_names = [self.name]
        self.up_proj_names = [
            join_name(expert_name, rname) for rname in self.up_proj_rnames for expert_name in self.expert_names
        ]
        self.down_proj_names = [
            join_name(expert_name, rname) for rname in self.down_proj_rnames for expert_name in self.expert_names
        ]
        self.up_proj_key = join_name(self.key, self.up_proj_rkey, sep="_")
        self.down_proj_key = join_name(self.key, self.down_proj_rkey, sep="_")
        self.moe_gate_key = join_name(self.key, self.moe_gate_rkey, sep="_")
        # region assertions
        assert num_experts == self.config.num_experts
        if self.moe_gate is not None:
            assert self.moe_gate.weight.shape[1] == self.config.num_channels
        for up_proj in self.up_projs:
            assert up_proj.weight.shape[1] == self.config.num_channels
            assert up_proj.weight.shape[0] in (
                self.config.num_intermediate_channels,
                self.config.num_intermediate_channels * 2,  # for fused GLU
            )
        for down_proj in self.down_projs:
            assert down_proj.weight.shape[1] == self.config.num_intermediate_channels
            assert down_proj.weight.shape[0] == self.config.num_channels
        # endregion

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        if self.moe_gate is not None:
            yield self.moe_gate_key, self.moe_gate_name, self.moe_gate, self, "moe_gate"
        num_experts = self.config.num_experts
        for expert_idx in range(num_experts):
            for name, module in zip(
                self.up_proj_names[expert_idx::num_experts], self.up_projs[expert_idx::num_experts], strict=True
            ):
                yield self.up_proj_key, name, module, self, "up_proj"
            for name, module in zip(
                self.down_proj_names[expert_idx::num_experts], self.down_projs[expert_idx::num_experts], strict=True
            ):
                yield self.down_proj_key, name, module, self, "down_proj"


@dataclass(kw_only=True)
class TransformerBlockStruct(BaseModuleStruct):
    """Transformer block struct."""

    # region relative keys
    attn_rkey: tp.ClassVar[str] = "attn"
    ffn_rkey: tp.ClassVar[str] = "ffn"
    add_ffn_rkey: tp.ClassVar[str] = "ffn_add"
    attn_struct_cls: tp.ClassVar[type[AttentionStruct]] = AttentionStruct
    ffn_struct_cls: tp.ClassVar[type[FeedForwardStruct]] = FeedForwardStruct
    # endregion

    parallel: bool
    """Whether the feed-forward modules are parallel to the attention modules."""

    # region child modules
    attn_norms: list[nn.Module] = field(repr=False)
    """Pre-attention normalization layers."""
    add_attn_norms: list[nn.Module] = field(repr=False)
    """Additional pre-attention normalization layers."""
    attns: list[nn.Module] = field(repr=False)
    """Attention modules."""
    ffn_norm: nn.Module | None = field(repr=False)
    """Pre-feed-forward normalization."""
    ffn: nn.Module | None = field(repr=False)
    """Feed-forward module."""
    add_ffn_norm: nn.Module | None = field(repr=False)
    """Additional pre-feed-forward normalization."""
    add_ffn: nn.Module | None = field(repr=False)
    """Additional feed-forward module."""
    # endregion
    # region relative names
    attn_norm_rnames: list[str]
    add_attn_norm_rnames: list[str]
    attn_rnames: list[str]
    ffn_norm_rname: str
    ffn_rname: str
    add_ffn_norm_rname: str
    add_ffn_rname: str
    # endregion
    # region absolute names
    attn_norm_names: list[str] = field(init=False, repr=False)
    add_attn_norm_names: list[str] = field(init=False, repr=False)
    attn_names: list[str] = field(init=False, repr=False)
    ffn_norm_name: str = field(init=False, repr=False)
    ffn_name: str = field(init=False, repr=False)
    add_ffn_norm_name: str = field(init=False, repr=False)
    add_ffn_name: str = field(init=False, repr=False)
    # endregion
    # region child structs
    attn_structs: list[AttentionStruct] = field(init=False, repr=False)
    ffn_struct: FeedForwardStruct | None = field(init=False, repr=False)
    add_ffn_struct: FeedForwardStruct | None = field(init=False, repr=False)
    # endregion

    def __post_init__(self) -> None:
        super().__post_init__()
        assert issubclass(self.attn_struct_cls, AttentionStruct)
        assert issubclass(self.ffn_struct_cls, FeedForwardStruct)
        assert len(self.attn_norms) == len(self.attns) == len(self.attn_norm_rnames) == len(self.attn_rnames)
        assert len(self.add_attn_norms) == len(self.add_attn_norm_rnames)
        self.attn_norm_names = [join_name(self.name, rname) for rname in self.attn_norm_rnames]
        self.add_attn_norm_names = [join_name(self.name, rname) for rname in self.add_attn_norm_rnames]
        self.attn_names = [join_name(self.name, rname) for rname in self.attn_rnames]
        self.ffn_norm_name = join_name(self.name, self.ffn_norm_rname)
        self.ffn_name = join_name(self.name, self.ffn_rname)
        self.add_ffn_norm_name = join_name(self.name, self.add_ffn_norm_rname)
        self.add_ffn_name = join_name(self.name, self.add_ffn_rname)
        self.attn_structs = [
            self.attn_struct_cls.construct(
                attn, parent=self, fname="attn", rname=self.attn_rnames[idx], rkey=self.attn_rkey, idx=idx
            )
            for idx, attn in enumerate(self.attns)
        ]
        if self.ffn is not None:
            self.ffn_struct = self.ffn_struct_cls.construct(
                self.ffn, parent=self, fname="ffn", rname=self.ffn_rname, rkey=self.ffn_rkey
            )
            self.ffn = self.ffn_struct.module
        else:
            self.ffn_struct = None
        if self.add_ffn is not None:
            self.add_ffn_struct = self.ffn_struct_cls.construct(
                self.add_ffn, parent=self, fname="add_ffn", rname=self.add_ffn_rname, rkey=self.add_ffn_rkey
            )
            self.add_ffn = self.add_ffn_struct.module
        else:
            self.add_ffn_struct = None
        if self.add_attn_norms:
            num_add_attn_norms = len(self.add_attn_norms)
            assert len(self.attns) >= num_add_attn_norms
            for i, attn in enumerate(self.attn_structs):
                if i < num_add_attn_norms:
                    if attn.is_self_attn():
                        assert self.add_attn_norms[i] is None, "self attention cannot have additional norm"
                else:
                    assert attn.is_self_attn(), "cross or joint attention must have additional norm"
        else:
            assert all(attn.is_self_attn() for attn in self.attn_structs)

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        for attn_struct in self.attn_structs:
            yield from attn_struct.named_key_modules()
        if self.ffn_struct is not None:
            yield from self.ffn_struct.named_key_modules()
        if self.add_ffn_struct is not None:
            yield from self.add_ffn_struct.named_key_modules()

    def iter_attention_structs(self) -> tp.Generator[AttentionStruct, None, None]:
        for attn_struct in self.attn_structs:
            yield from attn_struct.iter_attention_structs()

    def iter_transformer_block_structs(self) -> tp.Generator[tp.Self, None, None]:
        yield self


@dataclass(kw_only=True)
class BaseTransformerStruct(BaseModuleStruct):
    """Base Transformer struct."""

    # region relative keys
    proj_in_rkey: tp.ClassVar[str] = "proj_in"
    proj_out_rkey: tp.ClassVar[str] = "proj_out"
    # endregion

    # region child modules
    norm_in: nn.Module | None
    """Input normalization."""
    proj_in: nn.Linear | None
    """Input projection."""
    norm_out: nn.Module | None
    """Output normalization."""
    proj_out: nn.Linear | None
    """Output projection."""
    # endregion
    # region relative names
    norm_in_rname: str
    proj_in_rname: str
    norm_out_rname: str
    proj_out_rname: str
    # endregion
    # region absolute names
    norm_in_name: str = field(init=False, repr=False)
    proj_in_name: str = field(init=False, repr=False)
    norm_out_name: str = field(init=False, repr=False)
    proj_out_name: str = field(init=False, repr=False)
    # endregion
    # region absolute keys
    proj_in_key: str = field(init=False, repr=False)
    proj_out_key: str = field(init=False, repr=False)
    # endregion

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        """Get the number of transformer blocks."""
        ...

    @property
    @abstractmethod
    def block_structs(self) -> list[TransformerBlockStruct]:
        """Get the list of transformer block structs."""
        ...

    @property
    @abstractmethod
    def block_names(self) -> list[str]:
        """Get the list of transformer block names."""
        ...

    def __post_init__(self) -> None:
        super().__post_init__()
        for field_name in ("norm_in", "proj_in", "norm_out", "proj_out"):
            rname = getattr(self, f"{field_name}_rname")
            if getattr(self, field_name) is not None or rname:
                assert rname, f"{field_name} relative name must not be empty"
                setattr(self, f"{field_name}_name", join_name(self.name, rname))
            else:
                setattr(self, f"{field_name}_name", "")
        self.proj_in_key = join_name(self.key, self.proj_in_rkey, sep="_")
        self.proj_out_key = join_name(self.key, self.proj_out_rkey, sep="_")

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        if self.proj_in is not None:
            yield self.proj_in_key, self.proj_in_name, self.proj_in, self, "proj_in"
        for block in self.block_structs:
            yield from block.named_key_modules()
        if self.proj_out is not None:
            yield self.proj_out_key, self.proj_out_name, self.proj_out, self, "proj_out"

    def iter_attention_structs(self) -> tp.Generator[AttentionStruct, None, None]:
        for block in self.block_structs:
            yield from block.iter_attention_structs()

    def iter_transformer_block_structs(self) -> tp.Generator[TransformerBlockStruct, None, None]:
        for block in self.block_structs:
            yield from block.iter_transformer_block_structs()
