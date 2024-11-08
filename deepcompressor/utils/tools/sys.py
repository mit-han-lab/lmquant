# -*- coding: utf-8 -*-
"""System tools."""

import psutil
import torch

__all__ = ["get_max_memory_map"]


def _get_visible_gpu_capacity_list() -> list[int]:
    """Get visible GPU capacity list.

    Returns:
        `list[int]`: Visible GPU capacity list.
    """
    return [torch.cuda.get_device_properties(i).total_memory // 1024**3 for i in range(torch.cuda.device_count())]


def _get_ram_capacity() -> int:
    """Get RAM capacity.

    Returns:
        `int`: RAM capacity in GiB.
    """
    return psutil.virtual_memory().total // 1024**3  # in GiB


def get_max_memory_map(ratio: float = 0.9) -> dict[str, str]:
    """Get maximum memory map.

    Args:
        ratio (`float`, *optional*, defaults to `0.9`): The ratio of the maximum memory to use.

    Returns:
        `dict[str, str]`: Maximum memory map.
    """
    gpu_capacity_list = _get_visible_gpu_capacity_list()
    ram_capacity = _get_ram_capacity()
    gpu_capacity_list = [str(int(c * ratio)) + "GiB" for c in gpu_capacity_list]
    ram_capacity = str(int(ram_capacity * ratio)) + "GiB"
    ret_dict = {str(idx): gpu_capacity_list[idx] for idx in range(len(gpu_capacity_list))}
    ret_dict["cpu"] = ram_capacity
    return ret_dict
