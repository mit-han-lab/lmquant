# -*- coding: utf-8 -*-
"""Common utilities."""

import typing as tp

__all__ = ["num2str"]


def num2str(num: tp.Union[int, float]) -> str:
    """Convert a number to a string.

    Args:
        num (int | float): Number to convert.

    Returns:
        str: Converted string.
    """
    s = str(num).replace("-", "n")
    us = s.split(".")
    if len(us) == 1 or int(us[1]) == 0:
        return us[0]
    else:
        return us[0] + "p" + us[1]
