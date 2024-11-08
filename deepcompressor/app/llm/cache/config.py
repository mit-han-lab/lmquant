# -*- coding: utf-8 -*-
"""LLM quantization cache configuration."""

from dataclasses import dataclass, field

from omniconfig import configclass

from deepcompressor.utils.config.path import BasePathConfig

__all__ = ["LlmQuantCacheConfig", "LlmCacheConfig"]


@configclass
@dataclass
class LlmQuantCacheConfig(BasePathConfig):
    """Large language model quantization cache path.

    Args:
        rotation (`str`, *optional*, default=`""`):
            The rotation matrix cache path.
        reorder (`str`, *optional*, default=`""`):
            The reorder channel indexes cache path.
        smooth (`str`, *optional*, default=`""`):
            The smoothing scales cache path.
        wgts (`str`, *optional*, default=`""`):
            The weight quantizers state dict cache path.
        acts (`str`, *optional*, default=`""`):
            The activation quantizers state dict cache path.
    """

    rotation: str = ""
    reorder: str = ""
    smooth: str = ""
    wgts: str = ""
    acts: str = ""


@configclass
@dataclass
class LlmCacheConfig:
    """LLM quantization cache configuration.

    Attributes:
        root (`str`, *optional*, default=`""`):
            The root directory path for the cache.
        dirpath (`LlmQuantCacheConfig`, *optional*, default=`LlmQuantCacheConfig()`):
            The directory paths for the cache.
        path (`LlmQuantCacheConfig`, *optional*, default=`LlmQuantCacheConfig()`):
            The file paths for the cache.
    """

    root: str = field(default="")
    dirpath: LlmQuantCacheConfig = field(init=False, default_factory=LlmQuantCacheConfig)
    path: LlmQuantCacheConfig = field(default_factory=LlmQuantCacheConfig)
