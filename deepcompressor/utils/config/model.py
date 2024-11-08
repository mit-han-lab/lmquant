# -*- coding: utf-8 -*-
"""Net configurations."""

import os
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass

from omniconfig import configclass

__all__ = ["BaseModelConfig"]


@configclass
@dataclass
class BaseModelConfig(ABC):
    """Base class for all model configs.

    Args:
        name (`str`):
            Name of the model.
        family (`str`, *optional*, defaults to `""`):
            Family of the model. If not specified, it will be inferred from the name.
        path (`str`, *optional*, defaults to `""`):
            Path of the model.
        root (`str`, *optional*, defaults to `""`):
            Root directory path for models.
        local_path (`str`, *optional*, defaults to `""`):
            Local path of the model.
        local_root (`str`, *optional*, defaults to `""`):
            Local root directory path for models.
    """

    name: str
    family: str = ""
    path: str = ""
    root: str = ""
    local_path: str = ""
    local_root: str = ""

    def __post_init__(self):
        if not self.family:
            self.family = self.name.split("-")[0]
        self.local_root = os.path.expanduser(self.local_root)
        if not self.local_path:
            self.local_path = os.path.join(self.local_root, self.family, self.name)
        if not self.path:
            self.path = os.path.join(self.root, self.family, self.name)
        if os.path.exists(self.local_path):
            self.path = self.local_path

    @abstractmethod
    def build(self, *args, **kwargs) -> tp.Any:
        """Build model from config."""
        ...
