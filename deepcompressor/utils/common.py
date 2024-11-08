# -*- coding: utf-8 -*-
"""Common utilities."""

import typing as tp

import numpy as np
import torch

__all__ = [
    "join_name",
    "join_names",
    "num2str",
    "split_sequence",
    "tree_map",
    "tree_copy_with_ref",
    "tree_split",
    "tree_collate",
    "hash_str_to_int",
]


def join_name(prefix: str, name: str, sep: str = ".", relative: bool = True) -> str:
    """Join a prefix and a name with a separator.

    Args:
        prefix (`str`): Prefix.
        name (`str`): Name.
        sep (`str`, *optional*, defaults to `.`): Separator.
        relative (`bool`, *optional*, defaults to `True`):
            Whether to resolve relative name.

    Returns:
        `str`: Joined name.
    """
    if prefix:
        assert not prefix.endswith(sep), f"prefix={prefix} ends with sep={sep}"
        if name:
            if name.startswith(sep) and relative:
                # Remove leading separator
                prefix_names = prefix.split(sep)
                unsep_name = name.lstrip(sep)
                num_leading_seps = len(name) - len(unsep_name)
                if num_leading_seps > len(prefix_names):
                    prefix = sep * (num_leading_seps - len(prefix_names) - 1)
                else:
                    prefix = sep.join(prefix_names[:-num_leading_seps])
                return f"{prefix}{sep}{unsep_name}"
            else:
                return f"{prefix}{sep}{name}"
        else:
            return prefix
    else:
        return name


def join_names(*names: str, sep: str = ".", relative: bool = True) -> str:
    """Join multiple names with a separator.

    Args:
        names (`str`): Names.
        sep (`str`, *optional*, defaults to `.`): Separator.
        relative (`bool`, *optional*, defaults to `True`):
            Whether to resolve relative name.

    Returns:
        `str`: Joined name.
    """
    if not names:
        return ""
    prefix = ""
    for name in names:
        prefix = join_name(prefix, name, sep=sep, relative=relative)
    return prefix


def num2str(num: int | float) -> str:
    """Convert a number to a string.

    Args:
        num (`int` or `float`): Number to convert.

    Returns:
        str: Converted string.
    """
    s = str(num).replace("-", "n")
    us = s.split(".")
    if len(us) == 1 or int(us[1]) == 0:
        return us[0]
    else:
        return us[0] + "p" + us[1]


def split_sequence(lst: tp.Sequence[tp.Any], splits: tp.Sequence[int]) -> list[list[tp.Any]]:
    """Split a sequence into multiple sequences.

    Args:
        lst (`Sequence`):
            Sequence to split.
        splits (`Sequence`):
            Split indices.

    Returns:
        `list[list]`:
            Splitted sequences.
    """
    ret = []
    start = 0
    for end in splits:
        ret.append(lst[start:end])
        start = end
    ret.append(lst[start:])
    return ret


def tree_map(func: tp.Callable[[tp.Any], tp.Any], tree: tp.Any) -> tp.Any:
    """Apply a function to tree-structured data."""
    if isinstance(tree, dict):
        return {k: tree_map(func, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(func, v) for v in tree)
    elif isinstance(tree, (torch.Tensor, np.ndarray)):
        return func(tree)
    else:
        return tree


def tree_copy_with_ref(
    tree: tp.Any, /, ref: tp.Any, copy_func: tp.Callable[[tp.Any, tp.Any], tp.Any] | None = None
) -> tp.Any:
    """Copy tree-structured data with reference."""
    if isinstance(tree, dict):
        return {k: tree_copy_with_ref(v, ref[k]) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_copy_with_ref(v, ref[i]) for i, v in enumerate(tree))
    elif isinstance(tree, torch.Tensor):
        assert isinstance(ref, torch.Tensor), f"source is a tensor but reference is not: {type(ref)}"
        assert tree.shape == ref.shape, f"source.shape={tree.shape} != reference.shape={ref.shape}"
        if tree.data_ptr() == ref.data_ptr() or tree.allclose(ref):
            return ref
        else:
            return tree
    elif copy_func is not None:
        return copy_func(tree, ref)
    else:
        return tree


def tree_split(tree: tp.Any) -> list[tp.Any]:
    """Split tree-structured data into a list of data samples."""

    def get_batch_size(tree: tp.Any) -> int | None:
        if isinstance(tree, dict):
            for v in tree.values():
                b = get_batch_size(v)
                if b is not None:
                    return b
        elif isinstance(tree, (list, tuple)):
            for samples in tree:
                b = get_batch_size(samples)
                if b is not None:
                    return b
        elif isinstance(tree, torch.Tensor) and tree.ndim > 0:
            return tree.shape[0]
        return None

    def get_batch(tree: tp.Any, batch_id: int) -> tp.Any:
        if isinstance(tree, dict):
            return {k: get_batch(v, batch_id) for k, v in tree.items()}
        elif isinstance(tree, (list, tuple)):
            return [get_batch(samples, batch_id) for samples in tree]
        elif isinstance(tree, torch.Tensor) and tree.ndim > 0:
            return tree[batch_id : batch_id + 1]
        else:
            return tree

    ret = []
    batch_size = get_batch_size(tree)
    assert batch_size is not None, "Cannot determine batch size"
    for i in range(batch_size):
        ret.append(get_batch(tree, i))
    return ret


def tree_collate(batch: list[tp.Any] | tuple[tp.Any, ...]) -> tp.Any:
    """Collate function for tree-structured data."""
    if isinstance(batch[0], dict):
        return {k: tree_collate([d[k] for d in batch]) for k in batch[0]}
    elif isinstance(batch[0], (list, tuple)):
        return [tree_collate(samples) for samples in zip(*batch, strict=True)]
    elif isinstance(batch[0], torch.Tensor):
        return torch.cat(batch)
    else:
        return batch[0]


def hash_str_to_int(s: str) -> int:
    """Hash a string to an integer."""
    modulus = 10**9 + 7  # Large prime modulus
    hash_int = 0
    for char in s:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int
