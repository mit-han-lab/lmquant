# -*- coding: utf-8 -*-
"""LLM Quantization key."""

from enum import StrEnum

__all__ = ["LlmModuleKey"]


class LlmModuleKey(StrEnum):
    """Large Language Model Module keys."""

    PROJ_QKV = "proj_qkv"
    PROJ_OUT = "proj_out"
    PROJ_1ST = "proj_1st"
    PROJ_2ND = "proj_2nd"
    ROUTER = "router"
    ATTN_Q = "attn_q"
    ATTN_K = "attn_k"
    ATTN_V = "attn_v"
    ATTN_QK = "attn_qk"
