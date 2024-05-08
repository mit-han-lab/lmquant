# -*- coding: utf-8 -*-
"""System tools."""

import psutil
import torch

__all__ = ["get_max_memory_map"]


def _get_visible_gpu_capacity_list():
    capacity_list = []
    for i in range(torch.cuda.device_count()):
        capacity_list.append(torch.cuda.get_device_properties(i).total_memory // 1024**3)  # in GB
    return capacity_list


def _get_ram_capacity():
    return psutil.virtual_memory().total // 1024**3  # in GB


def get_max_memory_map(ratio=0.9):
    """Get maximum memory map.

    Args:
        ratio (float): Ratio of maximum memory to use.

    Returns:
        dict[int, str]: Maximum memory map.
    """
    gpu_capacity_list = _get_visible_gpu_capacity_list()
    ram_capacity = _get_ram_capacity()
    gpu_capacity_list = [str(int(c * ratio)) + "GiB" for c in gpu_capacity_list]
    ram_capacity = str(int(ram_capacity * ratio)) + "GiB"
    ret_dict = {idx: gpu_capacity_list[idx] for idx in range(len(gpu_capacity_list))}
    ret_dict["cpu"] = ram_capacity
    return ret_dict
