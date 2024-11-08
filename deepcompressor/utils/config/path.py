# -*- coding: utf-8 -*-
"""Path configuration."""

import os
import typing as tp

from ..dataclass import get_fields

__all__ = ["BasePathConfig"]


class BasePathConfig:
    """Base path configuration."""

    def is_all_set(self) -> bool:
        """Check if the path configuration is all set.

        Returns:
            `bool`:
                Whether the path configuration is all set.
        """
        fields = get_fields(self)
        for f in fields:
            if not getattr(self, f.name):
                return False
        return True

    def is_all_empty(self) -> bool:
        """Check if the path configuration is all empty.

        Returns:
            `bool`:
                Whether the path configuration is all empty.
        """
        fields = get_fields(self)
        for f in fields:
            if getattr(self, f.name):
                return False
        return True

    def clone(self) -> tp.Self:
        """Clone the path configuration.

        Returns:
            `Self`:
                The cloned path configuration.
        """
        fields = get_fields(self)
        return self.__class__(**{f.name: getattr(self, f.name) for f in fields})

    def add_parent_dirs(self, *parent_dirs: str) -> tp.Self:
        """Add the parent directories to the paths.

        Args:
            parent_dirs (`str`):
                The parent directories.
        """
        fields = get_fields(self)
        for f in fields:
            path = getattr(self, f.name)
            if path:
                setattr(self, f.name, os.path.join(*parent_dirs, path))
        return self

    def add_children(self, *children: str) -> tp.Self:
        """Add the children to the paths.

        Args:
            children (`str`):
                The children paths.
        """
        fields = get_fields(self)
        for f in fields:
            path = getattr(self, f.name)
            if path:
                setattr(self, f.name, os.path.join(path, *children))
        return self

    def to_dirpath(self) -> tp.Self:
        """Convert the paths to directory paths."""
        fields = get_fields(self)
        for f in fields:
            path = getattr(self, f.name)
            if path:
                setattr(self, f.name, os.path.dirname(path))
        return self

    def apply(self, fn: tp.Callable) -> tp.Self:
        """Apply the function to the paths.

        Args:
            fn (`Callable`):
                The function to apply.
        """
        fields = get_fields(self)
        for f in fields:
            path = getattr(self, f.name)
            if path:
                setattr(self, f.name, fn(path))
        return self
