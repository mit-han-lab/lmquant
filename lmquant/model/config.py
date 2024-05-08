# -*- coding: utf-8 -*-
"""Net configurations."""

import os
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from omniconfig import configclass

__all__ = ["BaseModelConfig"]


@configclass
@dataclass
class BaseModelConfig(ABC):
    """Base class for all model configs.

    Args:
        name (str): Name of the model.
        path (str): Path of the model. Defaults to ``None``.
        root (str): Root directory path for models. Defaults to ``""``.
        local_path (str): Local path of the model. Defaults to ``None``.
        local_root (str): Local root directory path for models. Defaults to ``""``.
    """

    name: str
    path: str = None
    root: str = ""
    local_path: str = None
    local_root: str = ""
    family: str = field(init=False)

    def __post_init__(self):
        self.family = self.name.split("-")[0]
        self.local_root = os.path.expanduser(self.local_root)
        if self.local_path is None:
            self.local_path = os.path.join(self.local_root, self.family, self.name)
        if self.path is None:
            self.path = os.path.join(self.root, self.family, self.name)
        if os.path.exists(self.local_path):
            self.path = self.local_path

    @abstractmethod
    def build(self, *args, **kwargs) -> tp.Any:
        """Build model from config."""
        ...
