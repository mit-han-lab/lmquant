# -*- coding: utf-8 -*-
"""Net configurations."""

from dataclasses import dataclass, field

import torch
from omniconfig import configclass
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lmquant.model.config import BaseModelConfig

from .nn.attention import patch_attention

__all__ = ["LlmModelConfig"]


@configclass
@dataclass
class LlmModelConfig(BaseModelConfig):
    """Arguments for creating a large language model.

    Args:
        name (str): Name of the model.
        path (str): Path of the model. Defaults to ``None``.
        root (str): Root directory path for models. Defaults to ``""``.
        local_path (str): Local path of the model. Defaults to ``None``.
        local_root (str): Local root directory path for models. Defaults to ``""``.
    """

    size: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        size = self.name.split("-")[-1]
        if size[-1] == "m":
            size = float(size[:-1]) / 1000
        else:
            assert size[-1] == "b"
            size = size[:-1]
            if "x" in size:
                num_experts, expert_gb = size.split("x")
                num_experts = int(num_experts)
                expert_size = float(expert_gb)
                size = num_experts * expert_size
            else:
                size = float(size)
        self.size = size

    def build(
        self, dtype: torch.dtype = torch.float16, cpu: bool = False
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Build model and tokenizer.

        Returns:
            tuple[AutoModelForCausalLM, AutoTokenizer]: Model and tokenizer.
        """
        config = AutoConfig.from_pretrained(self.path)
        tokenizer = AutoTokenizer.from_pretrained(self.path)
        kwargs = {} if cpu else {"device_map": "balanced"}
        kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(self.path, config=config, **kwargs)
        patch_attention(model)
        model.eval()
        return model, tokenizer
