# -*- coding: utf-8 -*-
"""Dataset for diffusion models."""

import os
import random
import typing as tp

import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F

from deepcompressor.utils.common import tree_collate, tree_map

__all__ = ["DiffusionDataset"]


class DiffusionDataset(torch.utils.data.Dataset):
    path: str
    filenames: list[str]
    filepaths: list[str]

    def __init__(self, path: str, num_samples: int = -1, seed: int = 0, ext: str = ".npy") -> None:
        if os.path.exists(path):
            self.path = path
            if "caches" in os.listdir(path):
                path = os.path.join(path, "caches")
            filenames = [f for f in sorted(os.listdir(path)) if f.endswith(ext)]
            if num_samples > 0 and num_samples < len(filenames):
                random.Random(seed).shuffle(filenames)
                filenames = filenames[:num_samples]
                filenames = sorted(filenames)
            self.filenames = filenames
            self.filepaths = [os.path.join(path, f) for f in filenames]
        else:
            raise ValueError(f"Invalid data path: {path}")

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx) -> dict[str, tp.Any]:
        data = np.load(self.filepaths[idx], allow_pickle=True).item()
        if isinstance(data["input_args"][0], str):
            name = data["input_args"][0]
            latent = np.load(os.path.join(self.path, "latents", name))
            data["input_args"][0] = latent
        if isinstance(data["input_kwargs"]["encoder_hidden_states"], str):
            name = data["input_kwargs"]["encoder_hidden_states"]
            text_emb = np.load(os.path.join(self.path, "text_embs", name))
            data["input_kwargs"]["encoder_hidden_states"] = text_emb
        data = tree_map(lambda x: torch.from_numpy(x), data)

        # Pad encoder_hidden_states to 300 for pixart
        if "encoder_attention_mask" in data["input_kwargs"]:
            encoder_attention_mask = data["input_kwargs"]["encoder_attention_mask"]
            encoder_hidden_states = data["input_kwargs"]["encoder_hidden_states"]
            encoder_hidden_states = F.pad(
                encoder_hidden_states,
                (0, 0, 0, encoder_attention_mask.shape[1] - encoder_hidden_states.shape[1]),
            )
            data["input_kwargs"]["encoder_hidden_states"] = encoder_hidden_states

        return data

    def build_loader(self, **kwargs) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self, collate_fn=tree_collate, **kwargs)
