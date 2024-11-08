# -*- coding: utf-8 -*-
"""Utility functions for Large Language Models."""

import typing as tp
from dataclasses import dataclass, field

import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaFlashAttention2,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaMLP,
    LlamaModel,
    LlamaSdpaAttention,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralConfig,
    MistralDecoderLayer,
    MistralFlashAttention2,
    MistralForCausalLM,
    MistralForSequenceClassification,
    MistralMLP,
    MistralModel,
    MistralSdpaAttention,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralConfig,
    MixtralDecoderLayer,
    MixtralFlashAttention2,
    MixtralForCausalLM,
    MixtralForSequenceClassification,
    MixtralModel,
    MixtralSdpaAttention,
    MixtralSparseMoeBlock,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2Config,
    Qwen2DecoderLayer,
    Qwen2FlashAttention2,
    Qwen2ForCausalLM,
    Qwen2ForSequenceClassification,
    Qwen2MLP,
    Qwen2Model,
    Qwen2SdpaAttention,
)
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5Block,
    T5Config,
    T5DenseActDense,
    T5DenseGatedActDense,
    T5EncoderModel,
    T5LayerFF,
    T5LayerSelfAttention,
    T5Stack,
)

from deepcompressor.nn.struct.attn import (
    AttentionConfigStruct,
    BaseTransformerStruct,
    FeedForwardConfigStruct,
    FeedForwardStruct,
    SelfAttentionStruct,
    TransformerBlockStruct,
)
from deepcompressor.nn.struct.base import BaseModuleStruct
from deepcompressor.utils.common import join_name

from .rope import RotaryEmbedding

__all__ = [
    "LlmConfigStruct",
    "LlmModelStruct",
    "LlmTransformerStruct",
    "LlmTransformerBlockStruct",
    "LlmSelfAttentionStruct",
    "LlmFeedForwardStruct",
]

ATTENTION_CLS = (
    LlamaAttention
    | MistralAttention
    | MixtralAttention
    | Qwen2Attention
    | T5Attention
    | LlamaFlashAttention2
    | MistralFlashAttention2
    | MixtralFlashAttention2
    | Qwen2FlashAttention2
    | LlamaSdpaAttention
    | MistralSdpaAttention
    | MixtralSdpaAttention
    | Qwen2SdpaAttention
)
FEEDFORWARD_CLS = LlamaMLP | MistralMLP | MixtralSparseMoeBlock | Qwen2MLP | T5DenseActDense | T5DenseGatedActDense
TRANSFORMER_BLOCK_CLS = LlamaDecoderLayer | MistralDecoderLayer | MixtralDecoderLayer | Qwen2DecoderLayer | T5Block
TRANSFORMER_CLS = tp.Union[LlamaModel, MistralModel, MixtralModel, Qwen2Model, T5Stack]
CASUALLM_CLS = tp.Union[LlamaForCausalLM, MistralForCausalLM, MixtralForCausalLM, Qwen2ForCausalLM]
SEQCLSLM_CLS = tp.Union[
    LlamaForSequenceClassification,
    MistralForSequenceClassification,
    MixtralForSequenceClassification,
    Qwen2ForSequenceClassification,
]


@dataclass(kw_only=True)
class LlmTransformerBlockConfigStruct(FeedForwardConfigStruct, AttentionConfigStruct):
    """Large Language Model Transformer Block Configuration.

    Args:
        hidden_size (`int`):
            The size of the input/output activations, i.e., the number of input channels.
        inner_size (`int`):
            The size of the inner activations, i.e., the number of **query** channels in the attention block.
        intermediate_size (`int`):
            The number of intermediate channels in the feedforward network.
        intermediate_act_type (`str`):
            The activation function for the intermediate activations in the feedforward network.
        num_query_heads (`int`):
            The number of query heads.
        num_key_value_heads (`int`):
            The number of key-value heads.
        num_experts (`int`):
            The number of experts (for the feedforward network).
        with_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply normalization to queries and keys.
        with_rope (`bool`):
            Whether to use Rotary Positional Encoding (RoPE).
        do_norm_before (`bool`):
            Whether to apply layer normalization before the projection.

    Attributes:
        head_size (`int`):
            The size of the head, equal to `num_query_channels // num_query_heads`.
        num_key_value_groups (`int`):
            The number of key-value groups, equal to `num_query_heads // num_key_value_heads`.
        intermediate_lowerbound (`float` or `None`):
            The lowerbound of the intermediate activations in feedforward network.
    """

    pass


@dataclass(kw_only=True)
class LlmTransformerConfigStruct(LlmTransformerBlockConfigStruct):
    """Large Language Model Transformer Configuration.

    Args:
        hidden_size (`int`):
            The size of the input/output activations, i.e., the number of input channels.
        inner_size (`int`):
            The size of the inner activations, i.e., the number of **query** channels in the attention block.
        intermediate_size (`int`):
            The number of intermediate channels in the feedforward network.
        intermediate_act_type (`str`):
            The activation function for the intermediate activations in the feedforward network.
        num_query_heads (`int`):
            The number of query heads.
        num_key_value_heads (`int`):
            The number of key-value heads.
        num_experts (`int`):
            The number of experts (for the feedforward network).
        with_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply normalization to queries and keys.
        with_rope (`bool`):
            Whether to use Rotary Positional Encoding (RoPE).
        do_norm_before (`bool`):
            Whether to apply layer normalization before the projection.
        vocab_size (`int`):
            The size of the vocabulary.
        num_hidden_layers (`int`):
            The number of hidden layers.

    Attributes:
        head_size (`int`):
            The size of the head, equal to `num_query_channels // num_query_heads`.
        num_key_value_groups (`int`):
            The number of key-value groups, equal to `num_query_heads // num_key_value_heads`.
        intermediate_lowerbound (`float` or `None`):
            The lowerbound of the intermediate activations in feedforward network.
    """

    vocab_size: int
    num_hidden_layers: int


@dataclass(kw_only=True)
class LlmConfigStruct(LlmTransformerConfigStruct):
    """Large Language Model Configuration.

    Args:
        hidden_size (`int`):
            The size of the input/output activations, i.e., the number of input channels.
        inner_size (`int`):
            The size of the inner activations, i.e., the number of **query** channels in the attention block.
        intermediate_size (`int`):
            The number of intermediate channels in the feedforward network.
        intermediate_act_type (`str`):
            The activation function for the intermediate activations in the feedforward network.
        num_query_heads (`int`):
            The number of query heads.
        num_key_value_heads (`int`):
            The number of key-value heads.
        num_experts (`int`):
            The number of experts (for the feedforward network).
        with_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply normalization to queries and keys.
        with_rope (`bool`):
            Whether to use Rotary Positional Encoding (RoPE).
        do_norm_before (`bool`):
            Whether to apply layer normalization before the projection.
        vocab_size (`int`):
            The size of the vocabulary.
        num_hidden_layers (`int`):
            The number of hidden layers.
        tie_word_embeddings (`bool`):
            Whether to tie the word embeddings with the head weights.

    Attributes:
        head_size (`int`):
            The size of the head, equal to `num_query_channels // num_query_heads`.
        num_key_value_groups (`int`):
            The number of key-value groups, equal to `num_query_heads // num_key_value_heads`.
        intermediate_lowerbound (`float` or `None`):
            The lowerbound of the intermediate activations in feedforward network.
    """

    tie_word_embeddings: bool = False


@dataclass(kw_only=True)
class LlmSelfAttentionStruct(SelfAttentionStruct):
    """Large Language Model Attention Block."""

    # region relative keys
    q_rkey: tp.ClassVar[str] = "attn_q"
    k_rkey: tp.ClassVar[str] = "attn_k"
    v_rkey: tp.ClassVar[str] = "attn_v"
    # endregion

    parent: tp.Optional["LlmTransformerBlockStruct"] = field(repr=False)
    kwargs: tuple[str, ...]

    def filter_kwargs(self, kwargs: dict) -> dict:
        """Filter layer kwargs to attn kwargs."""
        return {k: v for k, v in kwargs.items() if k in self.kwargs}

    @staticmethod
    def _default_construct(
        module: ATTENTION_CLS,
        /,
        parent: tp.Optional["LlmTransformerBlockStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "LlmSelfAttentionStruct":
        if isinstance(module, T5Attention):
            with_rope, num_query_heads, num_key_value_heads = False, module.n_heads, module.n_heads
            q_proj, k_proj, v_proj, o_proj = module.q, module.k, module.v, module.o
            q_proj_rname, k_proj_rname, v_proj_rname, o_proj_rname = "q", "k", "v", "o"
            q, k, v = module.q, module.k, module.v
            q_rname, k_rname, v_rname = "q", "k", "v"
            kwargs = (
                "mask",
                "key_value_states",
                "position_bias",
                "past_key_value",
                "layer_head_mask",
                "query_length",
                "use_cache",
                "output_attentions",
            )
        elif isinstance(module, (LlamaAttention, MistralAttention, MixtralAttention, Qwen2Attention)):
            with_rope, num_query_heads, num_key_value_heads = True, module.num_heads, module.num_key_value_heads
            q_proj, k_proj, v_proj, o_proj = module.q_proj, module.k_proj, module.v_proj, module.o_proj
            q_proj_rname, k_proj_rname, v_proj_rname, o_proj_rname = "q_proj", "k_proj", "v_proj", "o_proj"
            if hasattr(module, "q_rotary_emb"):
                q, k = module.q_rotary_emb, module.k_rotary_emb
                q_rname, k_rname = "q_rotary_emb", "k_rotary_emb"
                assert isinstance(q, RotaryEmbedding)
                assert isinstance(k, RotaryEmbedding)
            else:
                q, k = module.q_proj, module.k_proj
                q_rname, k_rname = "q_proj", "k_proj"
            v, v_rname = module.v_proj, "v_proj"
            kwargs = (
                "attention_mask",
                "position_ids",
                "past_key_value",
                "output_attentions",
                "use_cache",
                "cache_position",
            )
            if not isinstance(module, LlamaAttention):
                kwargs = kwargs[:-1]
        else:
            raise ValueError(f"Unsupported attention type: {type(module)}")
        config = AttentionConfigStruct(
            hidden_size=q_proj.weight.shape[1],
            inner_size=q_proj.weight.shape[0],
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            with_qk_norm=False,
            with_rope=with_rope,
            do_norm_before=True,
        )
        if parent is not None and parent.config is not None:
            assert parent.config.hidden_size == config.hidden_size
            assert parent.config.inner_size == config.inner_size
            assert parent.config.num_query_heads == config.num_query_heads
            assert parent.config.num_key_value_heads == config.num_key_value_heads
            assert parent.config.with_qk_norm == config.with_qk_norm
            assert parent.config.with_rope == config.with_rope
            assert parent.config.do_norm_before == config.do_norm_before
        return LlmSelfAttentionStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            config=config,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
            q=q,
            k=k,
            v=v,
            q_proj_rname=q_proj_rname,
            k_proj_rname=k_proj_rname,
            v_proj_rname=v_proj_rname,
            o_proj_rname=o_proj_rname,
            q_rname=q_rname,
            k_rname=k_rname,
            v_rname=v_rname,
            kwargs=kwargs,
        )


@dataclass(kw_only=True)
class LlmFeedForwardStruct(FeedForwardStruct):
    """Large Language Model Feedforward Network."""

    parent: tp.Optional["LlmTransformerBlockStruct"] = field(repr=False)

    @staticmethod
    def _default_construct(
        module: FEEDFORWARD_CLS,
        /,
        parent: tp.Optional["LlmTransformerBlockStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "LlmFeedForwardStruct":
        if isinstance(module, (LlamaMLP, MistralMLP, Qwen2MLP)):
            if parent is not None:
                assert parent.config.intermediate_act_type.endswith("_glu")
                act_type = parent.config.intermediate_act_type
            else:
                act_type = str(module.act_fn.__class__.__name__).removesuffix("activation").lower() + "_glu"
            up_projs, down_projs = [module.up_proj, module.gate_proj], [module.down_proj]
            experts = [module]
            moe_gate = None
            up_proj_rnames = ["up_proj", "gate_proj"]
            down_proj_rnames = ["down_proj"]
            experts_rname = ""
            moe_gate_rname = ""
        elif isinstance(module, MixtralSparseMoeBlock):
            if parent is not None:
                assert parent.config.intermediate_act_type.endswith("_glu")
                act_type = parent.config.intermediate_act_type
            else:
                act_type = str(module.experts[0].act_fn.__class__.__name__).removesuffix("activation").lower() + "_glu"
            up_projs = [expert.w3 for expert in module.experts] + [expert.w1 for expert in module.experts]
            down_projs = [expert.w2 for expert in module.experts]
            experts = list(module.experts)
            moe_gate = module.gate
            up_proj_rnames = ["w3", "w1"]
            down_proj_rnames = ["w2"]
            experts_rname = "experts"
            moe_gate_rname = "gate"
        elif isinstance(module, T5DenseActDense):
            if parent is not None:
                assert not parent.config.intermediate_act_type.endswith("_glu")
                act_type = parent.config.intermediate_act_type
            else:
                act_type = str(module.act.__class__.__name__).removesuffix("activation").lower()
            up_projs = [module.wi]
            down_projs = [module.wo]
            experts = [module]
            moe_gate = None
            up_proj_rnames = ["wi"]
            down_proj_rnames = ["wo"]
            experts_rname = ""
            moe_gate_rname = ""
        elif isinstance(module, T5DenseGatedActDense):
            if parent is not None:
                assert parent.config.intermediate_act_type.endswith("_glu")
                act_type = parent.config.intermediate_act_type
            else:
                act_type = str(module.act.__class__.__name__).removesuffix("activation").lower() + "_glu"
            up_projs = [module.wi_1, module.wi_0]
            down_projs = [module.wo]
            experts = [module]
            moe_gate = None
            up_proj_rnames = ["wi_1", "wi_0"]
            down_proj_rnames = ["wo"]
            experts_rname = ""
            moe_gate_rname = ""
        else:
            raise ValueError(f"Unsupported feed forward network type: {type(module)}")
        config = FeedForwardConfigStruct(
            hidden_size=up_projs[0].weight.shape[1],
            intermediate_size=up_projs[0].weight.shape[0],
            intermediate_act_type=act_type,
            num_experts=len(experts),
            do_norm_before=True,
        )
        if parent is not None and parent.config is not None:
            assert parent.config.hidden_size == config.hidden_size
            assert parent.config.intermediate_size == config.intermediate_size
            assert parent.config.intermediate_act_type == config.intermediate_act_type
            assert parent.config.num_experts == config.num_experts
            assert parent.config.do_norm_before == config.do_norm_before
        return LlmFeedForwardStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            config=config,
            up_projs=up_projs,
            down_projs=down_projs,
            moe_gate=moe_gate,
            experts=experts,
            up_proj_rnames=up_proj_rnames,
            down_proj_rnames=down_proj_rnames,
            moe_gate_rname=moe_gate_rname,
            experts_rname=experts_rname,
        )


@dataclass(kw_only=True)
class LlmTransformerBlockStruct(TransformerBlockStruct):
    """Large Language Model Transformer Block."""

    # region relative keys
    attn_rkey: tp.ClassVar[str] = ""
    ffn_rkey: tp.ClassVar[str] = ""
    add_ffn_rkey: tp.ClassVar[str] = "add"
    attn_struct_cls: tp.ClassVar[tp.Type[LlmSelfAttentionStruct]] = LlmSelfAttentionStruct
    ffn_struct_cls: tp.ClassVar[tp.Type[LlmFeedForwardStruct]] = LlmFeedForwardStruct
    # endregion

    parent: tp.Optional["LlmTransformerStruct"] = field(repr=False)
    parallel: bool = field(init=False, repr=False, default=False)
    config: LlmTransformerBlockConfigStruct = field(default=None)

    # region child modules
    add_attn_norms: list[nn.LayerNorm] = field(init=False, repr=False, default_factory=list)
    add_ffn_norm: None = field(init=False, repr=False, default=None)
    add_ffn: None = field(init=False, repr=False, default=None)
    # endregion
    # region relative names
    add_attn_norm_rnames: list[str] = field(init=False, repr=False, default_factory=list)
    add_ffn_norm_rname: str = field(init=False, repr=False, default="")
    add_ffn_rname: str = field(init=False, repr=False, default="")
    # endregion
    # region child structs
    attn_structs: list[LlmSelfAttentionStruct] = field(init=False, repr=False)
    ffn_struct: LlmFeedForwardStruct = field(init=False, repr=False)
    add_ffn_struct: None = field(init=False, repr=False, default=None)
    # endregion

    # region aliases

    @property
    def attn_norm(self) -> nn.LayerNorm:
        return self.attn_norms[0]

    @property
    def attn(self) -> nn.Module:
        return self.attns[0]

    @property
    def attn_norm_rname(self) -> str:
        return self.attn_norm_rnames[0]

    @property
    def attn_rname(self) -> str:
        return self.attn_rnames[0]

    @property
    def attn_norm_name(self) -> str:
        return self.attn_norm_names[0]

    @property
    def attn_name(self) -> str:
        return self.attn_names[0]

    @property
    def attn_struct(self) -> LlmSelfAttentionStruct:
        return self.attn_structs[0]

    # endregion

    def __post_init__(self):
        super().__post_init__()
        assert len(self.attn_structs) == 1
        if self.config is None:
            assert self.attn_struct.config.do_norm_before == self.ffn_struct.config.do_norm_before
            self.config = LlmTransformerBlockConfigStruct(
                hidden_size=self.attn_struct.config.hidden_size,
                inner_size=self.attn_struct.config.inner_size,
                num_query_heads=self.attn_struct.config.num_query_heads,
                num_key_value_heads=self.attn_struct.config.num_key_value_heads,
                with_qk_norm=self.attn_struct.config.with_qk_norm,
                with_rope=self.attn_struct.config.with_rope,
                do_norm_before=self.attn_struct.config.do_norm_before,
                intermediate_size=self.ffn_struct.config.intermediate_size,
                intermediate_act_type=self.ffn_struct.config.intermediate_act_type,
                num_experts=self.ffn_struct.config.num_experts,
            )

    @staticmethod
    def _default_construct(
        module: TRANSFORMER_BLOCK_CLS,
        /,
        parent: tp.Optional["LlmTransformerStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "LlmTransformerBlockStruct":
        if isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer, Qwen2DecoderLayer, MixtralDecoderLayer)):
            attn_norm, attn = module.input_layernorm, module.self_attn
            attn_norm_rname, attn_rname = "input_layernorm", "self_attn"
            ffn_norm = module.post_attention_layernorm
            ffn_norm_rname = "post_attention_layernorm"
            if isinstance(module, MixtralDecoderLayer):
                ffn, ffn_rname = module.block_sparse_moe, "block_sparse_moe"
            else:
                ffn, ffn_rname = module.mlp, "mlp"
            attn_norms, attns = [attn_norm], [attn]
            attn_norm_rnames, attn_rnames = [attn_norm_rname], [attn_rname]
        elif isinstance(module, T5Block):
            attn_norms, attns, attn_norm_rnames, attn_rnames = [], [], [], []
            for i, layer in enumerate(module.layer):
                if isinstance(layer, T5LayerSelfAttention):
                    attn_norms.append(layer.layer_norm)
                    attns.append(layer.SelfAttention)
                    attn_norm_rnames.append(f"layer.{i}.layer_norm")
                    attn_rnames.append(f"layer.{i}.SelfAttention")
                else:
                    assert isinstance(layer, T5LayerFF)
                    ffn_norm, ffn = layer.layer_norm, layer.DenseReluDense
                    ffn_norm_rname, ffn_rname = f"layer.{i}.layer_norm", f"layer.{i}.DenseReluDense"
        else:
            raise ValueError(f"Unsupported layer type: {type(module)}")
        config = parent.config if parent is not None and parent.config is not None else None
        return LlmTransformerBlockStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            config=config,
            attn_norms=attn_norms,
            attns=attns,
            ffn_norm=ffn_norm,
            ffn=ffn,
            attn_norm_rnames=attn_norm_rnames,
            attn_rnames=attn_rnames,
            ffn_norm_rname=ffn_norm_rname,
            ffn_rname=ffn_rname,
        )


@dataclass(kw_only=True)
class LlmTransformerStruct(BaseTransformerStruct):
    """Large Language Model Structure."""

    # region relative keys
    layer_rkey: tp.ClassVar[str] = ""
    layer_struct_cls: tp.ClassVar[tp.Type[LlmTransformerBlockStruct]] = LlmTransformerBlockStruct
    # endregion

    parent: tp.Optional["LlmModelStruct"] = field(repr=False)
    config: LlmTransformerConfigStruct = field(default=None)

    # region child modules
    # embeddings: list[nn.Embedding]
    # """list of embeddings [embed_tokens, embed_positions]"""
    embed_tokens: nn.Embedding
    """Token embedding module."""
    embed_positions: nn.Embedding | None
    """Position embedding module."""
    layers: nn.ModuleList
    # endregion
    # region relative names
    embed_tokens_rname: str
    embed_positions_rname: str
    layers_rname: str
    # endregion
    # region absolute names
    embed_tokens_name: str = field(init=False, repr=False)
    embed_positions_name: str = field(init=False, repr=False)
    layers_name: str = field(init=False, repr=False)
    layer_names: list[str] = field(init=False, repr=False)
    # endregion
    # region child structs
    layer_structs: list[LlmTransformerBlockStruct] = field(init=False, repr=False)
    # endregion

    # region abstractmethod implementations

    @property
    def num_blocks(self) -> int:
        """Get the number of transformer blocks."""
        return len(self.layers)

    @property
    def block_structs(self) -> list[LlmTransformerBlockStruct]:
        return self.layer_structs

    @property
    def block_names(self) -> list[str]:
        """Get the list of transformer block names."""
        return self.layer_names

    # endregion

    def __post_init__(self) -> None:
        super().__post_init__()
        self.embed_tokens_name = join_name(self.name, self.embed_tokens_rname)
        if self.embed_positions is not None:
            self.embed_positions_name = join_name(self.name, self.embed_positions_rname)
        else:
            self.embed_positions_name = ""
        self.layers_name = join_name(self.name, self.layers_rname)
        layer_rnames = [f"{self.layers_rname}.{idx}" for idx in range(len(self.layers))]
        self.layer_names = [join_name(self.name, rname) for rname in layer_rnames]
        self.layer_structs = [
            self.layer_struct_cls.construct(
                layer, parent=self, fname="layer", rname=rname, rkey=self.layer_rkey, idx=idx
            )
            for idx, (layer, rname) in enumerate(zip(self.layers, layer_rnames, strict=True))
        ]
        if self.config is None:
            assert all(block.config == self.block_structs[0].config for block in self.block_structs)
            ref_config = self.block_structs[0].config
            self.config = LlmTransformerConfigStruct(
                hidden_size=ref_config.hidden_size,
                inner_size=ref_config.inner_size,
                num_query_heads=ref_config.num_query_heads,
                num_key_value_heads=ref_config.num_key_value_heads,
                with_qk_norm=ref_config.with_qk_norm,
                with_rope=ref_config.with_rope,
                do_norm_before=ref_config.do_norm_before,
                intermediate_size=ref_config.intermediate_size,
                intermediate_act_type=ref_config.intermediate_act_type,
                num_experts=ref_config.num_experts,
                vocab_size=self.embed_tokens.num_embeddings,
                num_hidden_layers=self.num_blocks,
            )
        else:
            assert self.config.vocab_size == self.embed_tokens.num_embeddings
            assert self.config.num_hidden_layers == self.num_blocks

    def get_iter_layer_activations_args(
        self, **kwargs
    ) -> tuple[list[nn.Module], list[LlmTransformerBlockStruct], list[bool], list[bool]]:
        """
        Get the arguments for iterating over the layers and their activations.

        Args:
            skip_pre_modules (`bool`):
                Whether to skip the pre-modules
            skip_post_modules (`bool`):
                Whether to skip the post-modules

        Returns:
            `tuple[list[nn.Module], list[LlmTransformerBlockStruct], list[bool], list[bool]]`:
                the layers, the layer structs, the recomputes, and the use_prev_layer_outputs
        """
        return self.layers, self.layer_structs, [False] * len(self.layers), [True] * len(self.layers)

    @staticmethod
    def _default_construct(
        module: TRANSFORMER_CLS,
        /,
        parent: tp.Optional["LlmModelStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "LlmTransformerStruct":
        if isinstance(module, (LlamaModel, MistralModel, MixtralModel, Qwen2Model)):
            embed_tokens, embed_positions = module.embed_tokens, None
            layers = module.layers
            norm_in, norm_out = None, module.norm
            proj_in, proj_out = None, None
            embed_tokens_rname, embed_positions_rname = "embed_tokens", ""
            layers_rname = "layers"
            norm_in_rname, norm_out_rname = "", "norm"
            proj_in_rname, proj_out_rname = "", ""
        elif isinstance(module, T5Stack):
            embed_tokens, embed_positions = module.embed_tokens, None
            layers = module.block
            norm_in, norm_out = None, module.final_layer_norm
            proj_in, proj_out = None, None
            embed_tokens_rname, embed_positions_rname = "embed_tokens", ""
            layers_rname = "block"
            norm_in_rname, norm_out_rname = "", "final_layer_norm"
            proj_in_rname, proj_out_rname = "", ""
        else:
            raise ValueError(f"Unsupported backbone type: {type(module)}")
        config = parent.config if parent is not None and parent.config is not None else None
        return LlmTransformerStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            config=config,
            embed_tokens=embed_tokens,
            embed_positions=embed_positions,
            norm_in=norm_in,
            proj_in=proj_in,
            layers=layers,
            norm_out=norm_out,
            proj_out=proj_out,
            embed_tokens_rname=embed_tokens_rname,
            embed_positions_rname=embed_positions_rname,
            norm_in_rname=norm_in_rname,
            proj_in_rname=proj_in_rname,
            layers_rname=layers_rname,
            norm_out_rname=norm_out_rname,
            proj_out_rname=proj_out_rname,
        )


@dataclass(kw_only=True)
class LlmModelStruct(BaseModuleStruct):
    """Large Language Model Structure."""

    # region relative keys
    backbone_rkey: tp.ClassVar[str] = ""
    head_rkey: tp.ClassVar[str] = "head"
    backbone_struct_cls: tp.ClassVar[tp.Type[LlmTransformerStruct]] = LlmTransformerStruct
    # endregion

    module: PreTrainedModel = field(repr=False, kw_only=False)
    config: LlmConfigStruct
    # region child modules
    backbone: nn.Module
    head: nn.Linear | None
    # endregion
    # region relative names
    backbone_rname: str
    head_rname: str
    # endregion
    # region absolute names
    backbone_name: str = field(init=False, repr=False)
    head_name: str = field(init=False, repr=False)
    # endregion
    # region absolute keys
    head_key: str = field(init=False, repr=False)
    # endregion
    # region child structs
    backbone_struct: LlmTransformerStruct = field(init=False, repr=False)
    # endregion

    def __post_init__(self) -> None:
        super().__post_init__()
        self.backbone_name = join_name(self.name, self.backbone_rname)
        if self.head is not None or self.head_rname:
            self.head_name = join_name(self.name, self.head_rname)
        else:
            self.head_name = self.head_rname = ""
        self.head_key = join_name(self.key, self.head_rkey, sep="_")
        self.backbone_struct = self.backbone_struct_cls.construct(
            self.backbone, parent=self, fname="backbone", rname=self.backbone_rname, rkey=self.backbone_rkey
        )

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        yield from self.backbone_struct.named_key_modules()
        if self.head is not None:
            yield self.head_key, self.head_name, self.head, self, "head"

    def iter_attention_structs(self) -> tp.Generator[LlmSelfAttentionStruct, None, None]:
        yield from self.backbone_struct.iter_attention_structs()

    def iter_transformer_block_structs(self) -> tp.Generator[LlmTransformerBlockStruct, None, None]:
        yield from self.backbone_struct.iter_transformer_block_structs()

    def get_iter_layer_activations_args(
        self, **kwargs
    ) -> tuple[list[nn.Module], list[LlmTransformerBlockStruct], list[bool], list[bool]]:
        """
        Get the arguments for iterating over the layers and their activations.

        Args:
            skip_pre_modules (`bool`):
                Whether to skip the pre-modules
            skip_post_modules (`bool`):
                Whether to skip the post-modules

        Returns:
            `tuple[list[nn.Module], list[LlmTransformerBlockStruct], list[bool], list[bool]]`:
                the layers, the layer structs, the recomputes, and the use_prev_layer_outputs
        """
        return self.backbone_struct.get_iter_layer_activations_args(**kwargs)

    @staticmethod
    def _default_construct(
        model: nn.Module,
        /,
        parent: tp.Optional[BaseModuleStruct] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "LlmModelStruct":
        """Build the Large Language Model Structure."""
        if isinstance(model, CASUALLM_CLS) or isinstance(model, SEQCLSLM_CLS):
            backbone = model.model
            backbone_rname = "model"
        elif isinstance(model, T5EncoderModel):
            backbone = model.encoder
            backbone_rname = "encoder"
        elif isinstance(model, TRANSFORMER_CLS):
            backbone = model
            backbone_rname = ""
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        if isinstance(model, CASUALLM_CLS):
            head = model.lm_head
            head_rname = "lm_head"
        elif isinstance(model, SEQCLSLM_CLS):
            head = model.score
            head_rname = "score"
        elif isinstance(model, T5EncoderModel):
            head = None
            head_rname = ""
        elif isinstance(model, TRANSFORMER_CLS):
            head = None
            head_rname = ""
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        config = backbone.config
        if isinstance(config, (LlamaConfig, MistralConfig, MixtralConfig, Qwen2Config)):
            config_struct = LlmConfigStruct(
                hidden_size=config.hidden_size,
                inner_size=config.hidden_size,
                num_query_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                with_qk_norm=False,
                with_rope=True,
                do_norm_before=True,
                intermediate_size=config.intermediate_size,
                intermediate_act_type=f"{config.hidden_act}_glu".lower(),
                num_experts=getattr(config, "num_local_experts", 1),
                vocab_size=config.vocab_size,
                num_hidden_layers=config.num_hidden_layers,
                tie_word_embeddings=config.tie_word_embeddings,
            )
        elif isinstance(config, T5Config):
            config_struct = LlmConfigStruct(
                hidden_size=config.d_model,
                inner_size=config.d_kv * config.num_heads,
                num_query_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                with_rope=False,
                do_norm_before=True,
                intermediate_size=config.d_ff,
                intermediate_act_type=config.dense_act_fn.lower(),
                num_experts=1,
                vocab_size=config.vocab_size,
                num_hidden_layers=config.num_layers,
                tie_word_embeddings=False,
            )
            if config.is_gated_act:
                config_struct.intermediate_act_type += "_glu"
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
        return LlmModelStruct(
            module=model,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            config=config_struct,
            backbone=backbone,
            head=head,
            backbone_rname=backbone_rname,
            head_rname=head_rname,
        )


LlmSelfAttentionStruct.register_factory(ATTENTION_CLS, LlmSelfAttentionStruct._default_construct)

LlmFeedForwardStruct.register_factory(FEEDFORWARD_CLS, LlmFeedForwardStruct._default_construct)

LlmTransformerBlockStruct.register_factory(TRANSFORMER_BLOCK_CLS, LlmTransformerBlockStruct._default_construct)

LlmTransformerStruct.register_factory(TRANSFORMER_CLS, LlmTransformerStruct._default_construct)

LlmModelStruct.register_factory(
    tp.Union[TRANSFORMER_CLS, CASUALLM_CLS, SEQCLSLM_CLS, T5EncoderModel], LlmModelStruct._default_construct
)
