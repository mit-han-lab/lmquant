# -*- coding: utf-8 -*-
"""Utility functions for Large Language Models."""

import typing as tp
from dataclasses import dataclass, field

import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaMLP,
    LlamaModel,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralConfig,
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralForSequenceClassification,
    MistralMLP,
    MistralModel,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralConfig,
    MixtralDecoderLayer,
    MixtralForCausalLM,
    MixtralForSequenceClassification,
    MixtralModel,
    MixtralSparseMoeBlock,
)
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTConfig,
    OPTDecoder,
    OPTDecoderLayer,
    OPTForCausalLM,
    OPTForQuestionAnswering,
    OPTForSequenceClassification,
    OPTModel,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2Config,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2ForSequenceClassification,
    Qwen2MLP,
    Qwen2Model,
)

__all__ = ["LlmModelStruct", "LlmDecoderLayerStruct", "LlmBackboneStruct"]


@dataclass
class LlmConfigStruct:
    vocab_size: int
    head_size: int = field(init=False)
    hidden_size: int
    intermediate_size: int
    intermediate_act: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_key_value_groups: int = field(init=False)
    num_experts: int = 1
    with_rope: bool = True
    do_norm_before: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        self.head_size = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

    @property
    def num_hidden_channels(self) -> int:
        """Get the hidden size."""
        return self.hidden_size

    @property
    def num_query_channels(self) -> int:
        """Get the hidden size."""
        return self.hidden_size

    @property
    def num_key_value_channels(self) -> int:
        """Get the intermediate size."""
        return self.num_head_channels * self.num_key_value_heads

    @property
    def num_head_channels(self) -> int:
        """Get the head dimension."""
        return self.head_size

    @property
    def num_intermediate_channels(self) -> int:
        """Get the intermediate size."""
        return self.intermediate_size

    @property
    def hidden_act(self) -> str:
        """Get the hidden activation function."""
        return self.intermediate_act

    @property
    def num_query_heads(self) -> int:
        """Get the number of query heads."""
        return self.num_attention_heads

    @property
    def num_head_repeats(self) -> int:
        """Get the number of head repeats."""
        return self.num_key_value_groups


@dataclass
class LlmModelStruct:
    """Large Language Model Structure."""

    module: nn.Module
    """the nn.Module of the model"""
    backbone: nn.Module
    fc: nn.Linear | None
    backbone_name: str
    fc_name: str
    config: LlmConfigStruct

    _backbone_struct: tp.Optional["LlmBackboneStruct"] = None

    @property
    def backbone_full_name(self) -> str:
        """Get the backbone full name."""
        return self.backbone_name

    @property
    def fc_full_name(self) -> str:
        """Get the fc full name."""
        return self.fc_name

    @property
    def backbone_struct(self) -> "LlmBackboneStruct":
        """Extract backbone."""
        if self._backbone_struct is None:
            self._backbone_struct = extract_llm_backbone(self.backbone, self)
        return self._backbone_struct

    @staticmethod
    def build(model: nn.Module) -> tp.Optional["LlmModelStruct"]:
        """Build the Large Language Model Structure."""
        return extract_llm(model)


@dataclass
class LlmBackboneStruct:
    """Large Language Model Backbone Structure."""

    module: nn.Module
    """the nn.Module of the backbone"""
    parent: LlmModelStruct
    # region modules inside backbone
    embeddings: list[nn.Embedding]
    """list of embeddings [embed_tokens, embed_positions]"""
    proj_in: nn.Linear | None
    first_ln: nn.LayerNorm | None
    layers: nn.ModuleList
    final_ln: nn.LayerNorm | None
    proj_out: nn.Linear | None
    # endregion
    # region name inside the backbone
    embedding_names: list[str]
    proj_in_name: str
    first_ln_name: str
    layers_name: str
    final_ln_name: str
    proj_out_name: str
    full_name: str = field(init=False)
    embedding_full_names: list[str] = field(init=False)
    proj_in_full_name: str = field(init=False)
    first_ln_full_name: str = field(init=False)
    layers_full_name: str = field(init=False)
    final_ln_full_name: str = field(init=False)
    proj_out_full_name: str = field(init=False)
    # endregion
    # endregion
    _layer_structs: list["LlmDecoderLayerStruct"] | None = None

    def __post_init__(self) -> None:
        self.full_name = self.parent.backbone_full_name
        self.embedding_full_names = [f"{self.full_name}.{name}" for name in self.embedding_names]
        self.proj_in_full_name = f"{self.full_name}.{self.proj_in_name}"
        self.first_ln_full_name = f"{self.full_name}.{self.first_ln_name}"
        self.layers_full_name = f"{self.full_name}.{self.layers_name}"
        self.final_ln_full_name = f"{self.full_name}.{self.final_ln_name}"
        self.proj_out_full_name = f"{self.full_name}.{self.proj_out_name}"

    @property
    def config(self) -> LlmConfigStruct:
        """Get the config."""
        return self.parent.config

    @property
    def embed_tokens(self) -> nn.Embedding:
        """Get the token embedding module."""
        return self.embeddings[0]

    @property
    def embed_positions(self) -> nn.Embedding | None:
        """Get the position embedding module."""
        return self.embeddings[1] if len(self.embeddings) > 1 else None

    @property
    def embed_tokens_name(self) -> str:
        """Get the token embedding module name."""
        return self.embedding_names[0]

    @property
    def embed_positions_name(self) -> str:
        """Get the position embedding module name."""
        return self.embedding_names[1] if len(self.embedding_names) > 1 else ""

    @property
    def embed_tokens_full_name(self) -> str:
        """Get the token embedding module full name."""
        return self.embedding_full_names[0]

    @property
    def embed_positions_full_name(self) -> str:
        """Get the position embedding module full name."""
        return self.embedding_full_names[1] if len(self.embedding_full_names) > 1 else ""

    @property
    def layer_structs(self) -> list["LlmDecoderLayerStruct"]:
        """Extract decoder layers."""
        if self._layer_structs is None:
            self._layer_structs = [
                extract_llm_layer(layer, layer_idx, self) for layer_idx, layer in enumerate(self.layers)
            ]
        return self._layer_structs


@dataclass
class LlmDecoderLayerStruct:
    """Large Language Model Decoder Layer."""

    module: nn.Module
    """the nn.Module of the block."""
    parent: LlmBackboneStruct
    idx: int
    # region modules inside decoder layer
    attn_ln: nn.LayerNorm
    attn_block: nn.Module
    ffn_ln: nn.LayerNorm
    ffn_block: nn.Module
    proj_qkv: list[nn.Linear]
    """list of query, key, value projections."""
    proj_out: nn.Linear
    proj_1st: list[nn.Linear]
    """list of 1st layer projections.
        ``[expert_idx::num_experts]`` is the 1st layer projection for expert ``expert_idx``."""
    proj_2nd: list[nn.Linear]
    """list of 2nd layer projections.
        ``[expert_idx]`` is the 2nd layer projection for expert ``expert_idx``."""
    experts: list[nn.Module]
    router: nn.Linear | None
    proj_2nd_lowerbound: float | None
    # endregion
    # region evaluation settings
    attn_block_kwargs: tuple[str, ...]
    # endregion
    # region names inside decoder layer
    attn_ln_name: str
    attn_block_name: str
    ffn_ln_name: str
    ffn_block_name: str
    proj_qkv_names: list[str]
    proj_out_name: str
    proj_1st_names: list[str]
    proj_2nd_name: str
    experts_name: str
    router_name: str
    full_name: str = field(init=False)
    attn_ln_full_name: str = field(init=False)
    attn_block_full_name: str = field(init=False)
    ffn_ln_full_name: str = field(init=False)
    ffn_block_full_name: str = field(init=False)
    proj_qkv_full_names: list[str] = field(init=False)
    proj_out_full_name: str = field(init=False)
    proj_1st_full_names: list[str] = field(init=False)
    proj_2nd_full_names: list[str] = field(init=False)
    experts_full_name: str = field(init=False)
    expert_full_names: list[str] = field(init=False)
    router_full_name: str = field(init=False)
    # endregion

    def __post_init__(self):
        assert len(self.proj_qkv) == 3
        assert len(self.proj_2nd) == self.config.num_experts
        assert len(self.proj_1st) == self.config.num_experts * len(self.proj_1st_names)

        self.full_name = f"{self.parent.full_name}.{self.parent.layers_name}.{self.idx}"
        self.attn_ln_full_name = f"{self.full_name}.{self.attn_ln_name}"
        self.attn_block_full_name = (
            f"{self.full_name}.{self.attn_block_name}" if self.attn_block_name else self.full_name
        )
        self.ffn_ln_full_name = f"{self.full_name}.{self.ffn_ln_name}"
        self.ffn_block_full_name = f"{self.full_name}.{self.ffn_block_name}" if self.ffn_block_name else self.full_name
        self.proj_qkv_full_names = [f"{self.attn_block_full_name}.{name}" for name in self.proj_qkv_names]
        self.proj_out_full_name = f"{self.attn_block_full_name}.{self.proj_out_name}"
        if self.config.num_experts > 1:
            assert self.experts_name, "Experts name must be provided when num_experts > 1."
            assert len(self.experts) == self.config.num_experts
            self.experts_full_name = f"{self.ffn_block_full_name}.{self.experts_name}"
            self.expert_full_names = [
                f"{self.experts_full_name}.{expert_idx}" for expert_idx in range(self.num_experts)
            ]
            self.router_full_name = f"{self.ffn_block_full_name}.{self.router_name}"
        else:
            assert len(self.experts) == 1
            self.experts_full_name = f"{self.ffn_block_full_name}"
            self.expert_full_names = [self.experts_full_name]
            self.router_full_name = ""
        self.proj_1st_full_names = [
            f"{self.expert_full_names[expert_idx]}.{proj_1st_name}"
            for proj_1st_name in self.proj_1st_names
            for expert_idx in range(self.config.num_experts)
        ]
        self.proj_2nd_full_names = [f"{expert_name}.{self.proj_2nd_name}" for expert_name in self.expert_full_names]

    @property
    def config(self) -> LlmConfigStruct:
        """Get the config."""
        return self.parent.config

    @property
    def proj_q(self) -> nn.Linear:
        """Get the query projection module."""
        return self.proj_qkv[0]

    @property
    def proj_k(self) -> nn.Linear:
        """Get the key projection module."""
        return self.proj_qkv[1]

    @property
    def proj_v(self) -> nn.Linear:
        """Get the value projection module."""
        return self.proj_qkv[2]

    @property
    def proj_o(self) -> nn.Linear:
        """Get the output projection module."""
        return self.proj_out

    @property
    def proj_q_name(self) -> str:
        """Get the query projection module name."""
        return self.proj_qkv_names[0]

    @property
    def proj_k_name(self) -> str:
        """Get the key projection module name."""
        return self.proj_qkv_names[1]

    @property
    def proj_v_name(self) -> str:
        """Get the value projection module name."""
        return self.proj_qkv_names[2]

    @property
    def proj_o_name(self) -> str:
        """Get the output projection module name."""
        return self.proj_out_name

    @property
    def proj_q_full_name(self) -> str:
        """Get the query projection module full name."""
        return self.proj_qkv_full_names[0]

    @property
    def proj_k_full_name(self) -> str:
        """Get the key projection module full name."""
        return self.proj_qkv_full_names[1]

    @property
    def proj_v_full_name(self) -> str:
        """Get the value projection module full name."""
        return self.proj_qkv_full_names[2]

    @property
    def proj_o_full_name(self) -> str:
        """Get the output projection module full name."""
        return self.proj_out_full_name

    @property
    def num_experts(self) -> int:
        """Get the number of experts."""
        return self.config.num_experts

    def filter_layer_kwargs_to_attn_kwargs(self, kwargs: dict) -> dict:
        """Filter layer kwargs to attn kwargs."""
        return {k: v for k, v in kwargs.items() if k in self.attn_block_kwargs}


def extract_llm(model: nn.Module) -> LlmModelStruct | None:
    """Extract llm into components."""
    if isinstance(model, (OPTForCausalLM, OPTForSequenceClassification, OPTForQuestionAnswering)):
        backbone = model.model.decoder
        backbone_name = "model.decoder"
    elif isinstance(
        model,
        (
            LlamaForCausalLM,
            LlamaForSequenceClassification,
            MistralForCausalLM,
            MistralForSequenceClassification,
            MixtralForCausalLM,
            MixtralForSequenceClassification,
            Qwen2ForCausalLM,
            Qwen2ForSequenceClassification,
        ),
    ):
        backbone = model.model
        backbone_name = "model"
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    if isinstance(model, (OPTForCausalLM, LlamaForCausalLM, MistralForCausalLM, MixtralForCausalLM, Qwen2ForCausalLM)):
        fc = model.lm_head
        fc_name = "lm_head"
    elif isinstance(model, (OPTForQuestionAnswering)):
        fc = model.qa_outputs
        fc_name = "qa_outputs"
    elif isinstance(
        model,
        (
            OPTForSequenceClassification,
            LlamaForSequenceClassification,
            MistralForSequenceClassification,
            MixtralForSequenceClassification,
            Qwen2ForSequenceClassification,
        ),
    ):
        fc = model.score
        fc_name = "score"
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    config = model.config
    if isinstance(config, OPTConfig):
        config_struct = LlmConfigStruct(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.ffn_dim,
            intermediate_act=config.activation_function,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            num_experts=1,
            with_rope=False,
            do_norm_before=config.do_layer_norm_before,
            tie_word_embeddings=config.tie_word_embeddings,
        )
    elif isinstance(config, (LlamaConfig, MistralConfig, MixtralConfig, Qwen2Config)):
        config_struct = LlmConfigStruct(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            intermediate_act=config.hidden_act,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            num_experts=getattr(config, "num_local_experts", 1),
            with_rope=True,
            do_norm_before=True,
            tie_word_embeddings=config.tie_word_embeddings,
        )
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")
    return LlmModelStruct(
        module=model,
        backbone=backbone,
        fc=fc,
        backbone_name=backbone_name,
        fc_name=fc_name,
        config=config_struct,
    )


def extract_llm_backbone(backbone: nn.Module, parent: LlmModelStruct) -> LlmBackboneStruct | None:
    """Extract llm backbone into components."""
    if isinstance(backbone, OPTModel):
        backbone = backbone.decoder
    if isinstance(backbone, OPTDecoder):
        embeddings = [backbone.embed_tokens, backbone.embed_positions]
        layers = backbone.layers
        first_ln, final_ln = None, backbone.final_layer_norm
        proj_in, proj_out = backbone.project_in, backbone.project_out
        embedding_names = ["embed_tokens", "embed_positions"]
        layers_name = "layers"
        first_ln_name, final_ln_name = "", "final_layer_norm"
        proj_in_name, proj_out_name = "project_in", "project_out"
    elif isinstance(backbone, (LlamaModel, MistralModel, MixtralModel, Qwen2Model)):
        embeddings = [backbone.embed_tokens]
        layers = backbone.layers
        first_ln, final_ln = None, backbone.norm
        proj_in, proj_out = None, None
        embedding_names = ["embed_tokens"]
        layers_name = "layers"
        first_ln_name, final_ln_name = "", "norm"
        proj_in_name, proj_out_name = "", ""
    else:
        raise ValueError(f"Unsupported backbone type: {type(backbone)}")
    return LlmBackboneStruct(
        module=backbone,
        parent=parent,
        embeddings=embeddings,
        proj_in=proj_in,
        first_ln=first_ln,
        layers=layers,
        final_ln=final_ln,
        proj_out=proj_out,
        embedding_names=embedding_names,
        proj_in_name=proj_in_name,
        first_ln_name=first_ln_name,
        layers_name=layers_name,
        final_ln_name=final_ln_name,
        proj_out_name=proj_out_name,
    )


def extract_llm_layer(layer: nn.Module, layer_idx: int, parent: LlmBackboneStruct) -> LlmDecoderLayerStruct | None:
    """Extract llm block.

    Args:
        block (nn.Module): Block module.

    Returns:
        LlmBlockStruct: Block.
    """
    if isinstance(layer, OPTDecoderLayer):
        attn_ln = layer.self_attn_layer_norm
        attn_block = layer.self_attn
        assert isinstance(attn_block, OPTAttention)
        ffn_ln = layer.final_layer_norm
        ffn_block = nn.Sequential(layer.fc1, layer.activation_fn, layer.fc2)
        proj_qkv = [attn_block.q_proj, attn_block.k_proj, attn_block.v_proj]
        proj_out = attn_block.out_proj
        proj_1st = [layer.fc1]
        proj_2nd = [layer.fc2]
        experts = [ffn_block]
        router = None
        proj_2nd_lowerbound = 0  # ReLU
        attn_block_kwargs = (
            "key_value_states",
            "past_key_value",
            "attention_mask",
            "layer_head_mask",
            "output_attentions",
        )
        attn_ln_name = "self_attn_layer_norm"
        attn_block_name = "self_attn"
        proj_qkv_names = ["q_proj", "k_proj", "v_proj"]
        proj_out_name = "out_proj"
        ffn_ln_name = "final_layer_norm"
        ffn_block_name = ""
        proj_1st_names = ["fc1"]
        proj_2nd_name = "fc2"
        experts_name = ""
        router_name = ""
    elif isinstance(layer, (LlamaDecoderLayer, MistralDecoderLayer, Qwen2DecoderLayer)):
        attn_ln = layer.input_layernorm
        attn_block = layer.self_attn
        assert isinstance(attn_block, (LlamaAttention, MistralAttention, Qwen2Attention))
        ffn_ln = layer.post_attention_layernorm
        ffn_block = layer.mlp
        assert isinstance(ffn_block, (LlamaMLP, MistralMLP, Qwen2MLP))
        proj_qkv = [attn_block.q_proj, attn_block.k_proj, attn_block.v_proj]
        proj_out = attn_block.o_proj
        proj_1st = [ffn_block.up_proj, ffn_block.gate_proj]
        proj_2nd = [ffn_block.down_proj]
        experts = [ffn_block]
        router = None
        proj_2nd_lowerbound = None
        attn_block_kwargs = (
            "attention_mask",
            "position_ids",
            "past_key_value",
            "output_attentions",
            "use_cache",
            "cache_position",
        )
        if not isinstance(layer, LlamaDecoderLayer):
            attn_block_kwargs = attn_block_kwargs[:-1]
        attn_ln_name = "input_layernorm"
        attn_block_name = "self_attn"
        proj_qkv_names = ["q_proj", "k_proj", "v_proj"]
        proj_out_name = "o_proj"
        ffn_ln_name = "post_attention_layernorm"
        ffn_block_name = "mlp"
        proj_1st_names = ["up_proj", "gate_proj"]
        proj_2nd_name = "down_proj"
        experts_name = ""
        router_name = ""
    elif isinstance(layer, MixtralDecoderLayer):
        attn_ln = layer.input_layernorm
        attn_block = layer.self_attn
        assert isinstance(attn_block, MixtralAttention)
        ffn_ln = layer.post_attention_layernorm
        ffn_block = layer.block_sparse_moe
        assert isinstance(ffn_block, MixtralSparseMoeBlock)
        proj_qkv = [attn_block.q_proj, attn_block.k_proj, attn_block.v_proj]
        proj_out = attn_block.o_proj
        proj_1st = [expert.w3 for expert in ffn_block.experts] + [expert.w1 for expert in ffn_block.experts]
        proj_2nd = [expert.w2 for expert in ffn_block.experts]
        experts = [expert for expert in ffn_block.experts]
        router = ffn_block.gate
        proj_2nd_lowerbound = None
        attn_block_kwargs = ("attention_mask", "position_ids", "past_key_value", "output_attentions", "use_cache")
        attn_ln_name = "input_layernorm"
        attn_block_name = "self_attn"
        proj_qkv_names = ["q_proj", "k_proj", "v_proj"]
        proj_out_name = "o_proj"
        ffn_ln_name = "post_attention_layernorm"
        ffn_block_name = "block_sparse_moe"
        proj_1st_names = ["w3", "w1"]
        proj_2nd_name = "w2"
        experts_name = "experts"
        router_name = "gate"
    else:
        raise ValueError(f"Unsupported layer type: {type(layer)}")
    return LlmDecoderLayerStruct(
        module=layer,
        parent=parent,
        idx=layer_idx,
        attn_ln=attn_ln,
        attn_block=attn_block,
        ffn_ln=ffn_ln,
        ffn_block=ffn_block,
        proj_qkv=proj_qkv,
        proj_out=proj_out,
        proj_1st=proj_1st,
        proj_2nd=proj_2nd,
        experts=experts,
        router=router,
        proj_2nd_lowerbound=proj_2nd_lowerbound,
        attn_block_kwargs=attn_block_kwargs,
        attn_ln_name=attn_ln_name,
        attn_block_name=attn_block_name,
        proj_qkv_names=proj_qkv_names,
        proj_out_name=proj_out_name,
        ffn_ln_name=ffn_ln_name,
        ffn_block_name=ffn_block_name,
        proj_1st_names=proj_1st_names,
        proj_2nd_name=proj_2nd_name,
        experts_name=experts_name,
        router_name=router_name,
    )
