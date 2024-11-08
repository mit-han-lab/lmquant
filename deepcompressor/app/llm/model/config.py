# -*- coding: utf-8 -*-
"""Net configurations."""

import typing as tp
from dataclasses import dataclass, field

import torch
from omniconfig import configclass
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from deepcompressor.data.utils.dtype import eval_dtype
from deepcompressor.utils.config.model import BaseModelConfig

from ..nn.rope import patch_attention

__all__ = ["LlmModelConfig"]


@configclass
@dataclass
class LlmModelConfig(BaseModelConfig):
    """Arguments for creating a large language model.

    Args:
        name (`str`):
            Name of the model.
        path (`str`, *optional*, defaults to `""`):
            Path of the model.
        root (`str`, *optional*, defaults to `""`):
            Root directory path for models.
        local_path (`str`, *optional*, defaults to `""`):
            Local path of the model.
        local_root (`str`, *optional*, defaults to `""`):
            Local root directory path for models.
        dtype (`torch.dtype`, *optional*, defaults to `None`):
            Data type of the model. If not specified, the original data type of the model will be used.

    Attributes:
        size (`float`):
            Size of the model.
        variant (`str`):
            Variant of the model.
    """

    _model_factories: tp.ClassVar[dict[str, tp.Callable[[str], tuple[PreTrainedModel, PreTrainedTokenizer]]]] = {}

    size: float = field(init=False)
    variant: str = field(init=False)
    dtype: torch.dtype = field(default_factory=lambda s=None: eval_dtype(s, with_quant_dtype=False))
    orig_dtype: torch.dtype = field(init=False)

    def __post_init__(self):
        parts = self.name.split("-")
        # we first infer the size, it should be a string matching "$\d+[mb]$"
        family, size, variant = "", "", ""
        for i, part in enumerate(parts):
            part = part.lower()
            if part[-1] == "m" or part[-1] == "b":
                _part = part[:-1].replace("x", "", 1)
                if _part.isdigit():
                    size = part
                    family = "-".join(parts[:i])
                    if len(parts) > i + 1:
                        variant = "-".join(parts[i + 1 :])
                    break
        assert size, f"Cannot infer size from {self.name}"
        assert family, f"Cannot infer family from {self.name}"
        if not self.family:
            self.family = family
        self.variant = variant
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
        super().__post_init__()
        self.name = self.name.lower()
        self.family = self.family.lower()
        self.variant = self.variant.lower()
        config = AutoConfig.from_pretrained(self.path)
        self.orig_dtype = config.torch_dtype
        if self.orig_dtype == torch.float32:
            self.dtype = self.dtype or torch.float16
        elif self.orig_dtype == torch.float16:
            self.dtype = self.dtype or torch.float16
        elif self.orig_dtype == torch.bfloat16:
            self.dtype = self.dtype or torch.bfloat16
        else:
            raise ValueError(f"Unsupported data type: {self.orig_dtype}")

    def build(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Build model and tokenizer.

        Args:
            dtype (`torch.dtype`, *optional*, defaults to `None`):
                Data type of the model.

        Returns:
            `tuple[PreTrainedModel, PreTrainedTokenizer]`:
                Model and tokenizer.
        """
        torch_dtype = self.dtype
        if self.name in self._model_factories:
            return self._model_factories[self.name](self.path, torch_dtype=torch_dtype)
        kwargs = {"torch_dtype": torch_dtype}
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            kwargs["device_map"] = "balanced"
        return self._default_build(self.path, **kwargs)

    @staticmethod
    def _default_build(path: str, **kwargs) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Build model and tokenizer.

        Args:
            dtype (`torch.dtype`, *optional*, defaults to `None`):
                Data type of the model.

        Returns:
            `tuple[PreTrainedModel, PreTrainedTokenizer]`:
                Model and tokenizer.
        """
        config = AutoConfig.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, config=config, **kwargs)
        patch_attention(model)
        model.eval()
        return model, tokenizer

    @classmethod
    def register_model_factory(
        cls,
        names: str | tuple[str, ...],
        /,
        factory: tp.Callable[[str, torch.dtype], tuple[PreTrainedModel, PreTrainedTokenizer]],
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a model factory.

        Args:
            names (`str` or `tuple[str, ...]`):
                Names of the model.
            factory (`Callable[[str, torch.dtype], tuple[PreTrainedModel, PreTrainedTokenizer]]`):
                Factory function.
            overwrite (`bool`, *optional*, defaults to `False`):
                Whether to overwrite the existing factory for the model.
        """
        if isinstance(names, str):
            names = (names,)
        for name in names:
            if not overwrite and name in cls._model_factories:
                raise ValueError(f"Factory for {name} already exists")
            cls._model_factories[name] = factory
