# -*- coding: utf-8 -*-
"""Dataclass utilities."""

from dataclasses import _FIELD, _FIELD_CLASSVAR, _FIELD_INITVAR, _FIELDS, Field

__all__ = ["get_fields"]


def get_fields(class_or_instance, *, init_vars: bool = False, class_vars: bool = False) -> tuple[Field, ...]:
    """Get the fields of the dataclass.

    Args:
        class_or_instance:
            The dataclass type or instance.
        init_vars (`bool`, *optional*, defaults to `False`):
            Whether to include the init vars.
        class_vars (`bool`, *optional*, defaults to `False`):
            Whether to include the class vars.

    Returns:
        tuple[Field, ...]: The fields.
    """
    try:
        fields = getattr(class_or_instance, _FIELDS)
    except AttributeError:
        raise TypeError("must be called with a dataclass type or instance") from None
    return tuple(
        v
        for v in fields.values()
        if v._field_type is _FIELD
        or (init_vars and v._field_type is _FIELD_INITVAR)
        or (class_vars and v._field_type is _FIELD_CLASSVAR)
    )
