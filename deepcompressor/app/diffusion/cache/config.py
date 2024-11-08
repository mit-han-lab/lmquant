# -*- coding: utf-8 -*-
"""LLM quantization cache configuration."""

import functools
import re
import typing as tp
from dataclasses import dataclass, field

from omniconfig import configclass

from deepcompressor.utils.config.path import BasePathConfig

from ..nn.struct import DiffusionModelStruct

__all__ = ["DiffusionQuantCacheConfig", "DiffusionPtqCacheConfig"]


@dataclass
class DiffusionQuantCacheConfig(BasePathConfig):
    """Denoising diffusion model quantization cache path.

    Args:
        smooth (`str`, *optional*, default=`""`):
            The smoothing scales cache path.
        branch (`str`, *optional*, default=`""`):
            The low-rank branches cache path.
        wgts (`str`, *optional*, default=`""`):
            The weight quantizers state dict cache path.
        acts (`str`, *optional*, default=`""`):
            The activation quantizers state dict cache path
    """

    smooth: str = ""
    branch: str = ""
    wgts: str = ""
    acts: str = ""

    @staticmethod
    def simplify_path(path: str, key_map: dict[str, set[str]]) -> str:
        """Simplify the cache path."""
        to_replace = {}
        # we first extract all the parts matching the pattern "(skip|include).\[[a-zA-Z0-9_\+]+\]"
        for part in re.finditer(r"(skip|include)\.\[[a-zA-Z0-9_\+]+\]", path):
            # remove the "skip." or "include." prefix
            part = part.group(0)
            if part[0] == "s":
                prefix, keys = part[:4], part[6:-1]
            else:
                prefix, keys = part[:7], part[9:-1]
            # simplify the keys
            keys = "+".join(
                (
                    "".join((s[0] for s in x.split("_")))
                    for x in DiffusionModelStruct._simplify_keys(keys.split("+"), key_map=key_map)
                )
            )
            to_replace[part] = f"{prefix}.[{keys}]"
        # we then replace the parts
        for key, value in to_replace.items():
            path = path.replace(key, value)
        return path

    def simplify(self, key_map: dict[str, set[str]]) -> tp.Self:
        """Simplify the cache paths."""
        return self.apply(functools.partial(self.simplify_path, key_map=key_map))


@configclass
@dataclass
class DiffusionPtqCacheConfig:
    root: str
    dirpath: DiffusionQuantCacheConfig = field(init=False)
    path: DiffusionQuantCacheConfig = field(init=False)
