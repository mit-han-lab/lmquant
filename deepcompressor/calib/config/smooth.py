# -*- coding: utf-8 -*-
"""Smooth quantization configuration."""

import enum
from dataclasses import dataclass, field

import omniconfig
from omniconfig import configclass

from ...utils.common import num2str
from ...utils.config import SkipBasedConfig
from .search import (
    SearchBasedCalibConfig,
    SearchBasedCalibGranularity,
    SearchBasedCalibObjective,
    SearchBasedCalibStrategy,
)

__all__ = [
    "SmoothSpanMode",
    "SmoothCalibConfig",
    "SmoothAttentionCalibConfig",
    "SkipBasedSmoothCalibConfig",
    "SmoothTransfomerConfig",
]


class SmoothSpanMode(enum.Enum):
    """The mode for computing the span used in smoothing scale calculation."""

    AbsMax = enum.auto()
    RootMeanSquare = enum.auto()


@configclass
@dataclass
class SmoothCalibConfig(SearchBasedCalibConfig):
    """Configuration for smooth quantization.

    Args:
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        objective (`SearchBasedCalibObjective`, *optional*, default=`SearchBasedCalibObjective.OutputsError`):
            The objective for quantization calibration.
        strategy (`SearchBasedCalibStrategy`, *optional*, default=`SearchBasedCalibStrategy.Manual`):
            The strategy for quantization calibration.
        granularity (`SearchBasedCalibGranularity`, *optional*, default=`SearchBasedCalibGranularity.Layer`):
            The granularity for quantization calibration.
        element_batch_size (`int`, *optional*, default=`-1`):
            The element batch size for calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        element_size (`int`, *optional*, default=`-1`):
            The calibration element size.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        pre_reshape (`bool`, *optional*, default=`True`):
            Whether to enable reshaping the tensor before calibration.
        outputs_device (`str`, *optional*, default=`"cpu"`):
            The device to store the precomputed outputs of the module.
        allow_a_quant (`bool`, *optional*, default=`True`):
            Whether to allow the quantization for alpha tensor.
        allow_b_quant (`bool`, *optional*, default=`True`):
            Whether to allow the quantization for beta tensor.
        spans (`list[tuple[SmoothSpanMode, SmoothSpanMode]]`, *optional*, default=`[]`):
            The span combinations. The first element is for the alpha and the second element is for the beta.
        alpha (`float`, *optional*, default=`0.5`):
            The smoothing alpha.
        beta (`float`, *optional*, default=`-1`):
            The smoothing beta.
        num_grids (`int`, *optional*, default=`20`):
            The number of grids for grid search.
        allow_low_rank (`bool`, *optional*, default=`False`):
            Whether to allow quantization low-rank branch during calibration.
    """

    allow_a_quant: bool = True
    allow_b_quant: bool = True
    spans: list[tuple[SmoothSpanMode, SmoothSpanMode]] = field(
        default_factory=list,
        metadata={
            omniconfig.ARGPARSE_KWARGS: {
                "nargs": "+",
                "type": lambda s: tuple(SmoothSpanMode[x.split(".")[-1]] for x in s.split(",")),
            }
        },
    )
    a_spans: list[SmoothSpanMode] = field(default_factory=list, init=False)
    b_spans: list[SmoothSpanMode] = field(default_factory=list, init=False)
    alpha: float = 0.5
    beta: float = -1
    num_grids: int = 20
    allow_low_rank: bool = False

    def __post_init__(self) -> None:  # noqa: C901
        # region remove duplicates of ranges
        _spans, _spanset, _a_spanset, _b_spanset = [], set(), set(), set()
        self.a_spans, self.b_spans = [], []
        for a_span, b_span in self.spans:
            if isinstance(a_span, str):
                a_span = SmoothSpanMode[a_span]
            if isinstance(b_span, str):
                b_span = SmoothSpanMode[b_span]
            assert isinstance(a_span, SmoothSpanMode), f"Invalid span mode used for alpha: {a_span}"
            assert isinstance(b_span, SmoothSpanMode), f"Invalid span mode used for beta: {b_span}"
            _span = (a_span, b_span)
            if _span in _spanset:
                continue
            _spans.append(_span)
            _spanset.add(_span)
            if a_span not in _a_spanset:
                _a_spanset.add(a_span)
                self.a_spans.append(a_span)
            if b_span not in _b_spanset:
                _b_spanset.add(b_span)
                self.b_spans.append(b_span)
        self.spans = _spans
        # endregion
        if self.strategy == SearchBasedCalibStrategy.Manual:
            assert len(self.spans) == 1, "Only one span combination is allowed in manual mode"
            assert self.alpha != 0 or self.beta != 0, "alpha and beta cannot be both zero"
            self.alpha, self.beta = self.get_alpha_beta_pairs()[0]
        if self.granularity == SearchBasedCalibGranularity.Group:
            self.granularity = SearchBasedCalibGranularity.ChannelGroup
        if self.allow_low_rank:
            self.granularity = SearchBasedCalibGranularity.Layer
        assert -3 <= self.alpha <= 1, "alpha must be less than or equal to 1"
        assert -3 <= self.beta <= 1, "beta must be less than or equal to 1"
        super().__post_init__()

    def get_alpha_beta_pairs(self) -> list[tuple[float, float]]:  # noqa: C901
        """Get the alpha and beta pairs for smooth quantization.

        Returns:
            `list[tuple[float, float]]`:
                The alpha and beta pair candidates.
        """
        if self.strategy == SearchBasedCalibStrategy.Manual:
            if self.beta < 0:
                assert 0 <= self.alpha <= 1, "alpha must be in [0, 1]"
                return [(self.alpha, 1 - self.alpha)]
            elif self.alpha < 0:
                assert 0 <= self.beta <= 1, "beta must be in [0, 1]"
                return [(1 - self.beta, self.beta)]
            else:
                assert 0 <= self.alpha <= 1, "alpha must be in [0, 1]"
                assert 0 <= self.beta <= 1, "beta must be in [0, 1]"
                return [(self.alpha, self.beta)]
        choices = [i / self.num_grids for i in range(1, self.num_grids)]
        if self.alpha > 0:
            if self.beta > 0:
                return [(0, 0)] + [(alpha, alpha) for alpha in choices]
            if self.beta == 0:
                return [(0, 0)] + [(alpha, 0) for alpha in choices]
            if self.beta == -1:
                return [(0, 0)] + [(alpha, 1 - alpha) for alpha in choices]
            if self.beta == -2:
                return [(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, 1 - alpha) for alpha in choices]
            return (
                [(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, beta) for alpha in choices for beta in choices]
            )
        if self.alpha == 0:
            if self.beta > 0:
                return [(0, 0)] + [(0, beta) for beta in choices]
            if self.beta == 0:
                return [(0, 0)] + [(alpha, 0) for alpha in choices] + [(0, beta) for beta in choices]
            if self.beta == -1:
                return [(0, 0)] + [(0, beta) for beta in choices] + [(alpha, 1 - alpha) for alpha in choices]
            if self.beta == -2:
                return (
                    [(0, 0)]
                    + [(alpha, 0) for alpha in choices]
                    + [(0, beta) for beta in choices]
                    + [(alpha, 1 - alpha) for alpha in choices]
                )
            return (
                [(0, 0)]
                + [(alpha, 0) for alpha in choices]
                + [(0, beta) for beta in choices]
                + [(alpha, beta) for alpha in choices for beta in choices]
            )
        if self.alpha == -1:
            if self.beta > 0 or self.beta == -1:
                return [(0, 0)] + [(alpha, 1 - alpha) for alpha in choices]
            if self.beta == 0 or self.beta == -2:
                return [(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, 1 - alpha) for alpha in choices]
            return (
                [(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, beta) for alpha in choices for beta in choices]
            )
        if self.alpha == -2:
            if self.beta > 0 or self.beta == -1:
                return [(0, 0)] + [(0, beta) for beta in choices] + [(alpha, 1 - alpha) for alpha in choices]
            if self.beta == 0 or self.beta == -2:
                return (
                    [(0, 0)]
                    + [(alpha, 0) for alpha in choices]
                    + [(0, beta) for beta in choices]
                    + [(alpha, 1 - alpha) for alpha in choices]
                )
            return (
                [(0, 0)]
                + [(alpha, 0) for alpha in choices]
                + [(0, beta) for beta in choices]
                + [(alpha, beta) for alpha in choices for beta in choices]
            )
        if self.alpha == -3:
            if self.beta > 0:
                return (
                    [(0, 0)]
                    + [(0, beta) for beta in choices]
                    + [(alpha, beta) for alpha in choices for beta in choices]
                )
            return (
                [(0, 0)]
                + [(0, beta) for beta in choices]
                + [(alpha, 0) for alpha in choices]
                + [(alpha, beta) for alpha in choices for beta in choices]
            )
        raise ValueError("Invalid alpha and beta values")

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Get the directory names of the smooth quantization configuration.

        Args:
            prefix (`str`, *optional*, default=`""`):
                The prefix of the directory.

        Returns:
            `list[str]`:
                The directory names of the configuration.
        """
        names = super().generate_dirnames(**kwargs)
        names.append("[{}]".format("+".join(f"a.{a_span.name}.b.{b_span.name}" for a_span, b_span in self.spans)))
        alpha, beta = num2str(self.alpha), num2str(self.beta)
        if self.strategy == SearchBasedCalibStrategy.Manual:
            names.append(f"a{alpha}.b{beta}")
        elif self.alpha > 0:
            names.append(f"g{self.num_grids}.b{beta}")
        elif self.beta > 0:
            names.append(f"g{self.num_grids}.a{alpha}")
        else:
            names.append(f"g{self.num_grids}.a{alpha}.b{beta}")
        if self.allow_low_rank:
            names[-1] += ".lr"
        disallows = []
        if not self.allow_a_quant:
            disallows.append("a")
        if not self.allow_b_quant:
            disallows.append("b")
        if disallows:
            names.append(f"disallow.[{'+'.join(disallows)}]")
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names


@configclass
@dataclass
class SkipBasedSmoothCalibConfig(SkipBasedConfig, SmoothCalibConfig):
    """Configuration for smooth quantization.

    Args:
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        objective (`SearchBasedCalibObjective`, *optional*, default=`SearchBasedCalibObjective.OutputsError`):
            The objective for quantization calibration.
        strategy (`SearchBasedCalibStrategy`, *optional*, default=`SearchBasedCalibStrategy.Manual`):
            The strategy for quantization calibration.
        granularity (`SearchBasedCalibGranularity`, *optional*, default=`SearchBasedCalibGranularity.Layer`):
            The granularity for quantization calibration.
        element_batch_size (`int`, *optional*, default=`-1`):
            The element batch size for calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        element_size (`int`, *optional*, default=`-1`):
            The calibration element size.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        pre_reshape (`bool`, *optional*, default=`True`):
            Whether to enable reshaping the tensor before calibration.
        outputs_device (`str`, *optional*, default=`"cpu"`):
            The device to store the precomputed outputs of the module.
        allow_a_quant (`bool`, *optional*, default=`True`):
            Whether to allow the quantization for alpha tensor.
        allow_b_quant (`bool`, *optional*, default=`True`):
            Whether to allow the quantization for beta tensor.
        spans (`list[tuple[SmoothSpanMode, SmoothSpanMode]]`, *optional*, default=`[]`):
            The span combinations. The first element is for the alpha and the second element is for the beta.
        alpha (`float`, *optional*, default=`0.5`):
            The smoothing alpha.
        beta (`float`, *optional*, default=`-1`):
            The smoothing beta.
        num_grids (`int`, *optional*, default=`20`):
            The number of grids for grid search.
        allow_low_rank (`bool`, *optional*, default=`False`):
            Whether to allow quantization SVD during calibration.
        skips (`list[str]`, *optional*, default=`[]`):
            The keys of the modules to skip.
    """

    pass


@configclass
@dataclass
class SmoothAttentionCalibConfig(SmoothCalibConfig):
    """Configuration for smooth quantization.

    Args:
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        strategy (`SearchBasedCalibStrategy`, *optional*, default=`SearchBasedCalibStrategy.Manual`):
            The strategy for quantization calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        outputs_device (`str`, *optional*, default=`"cpu"`):
            The device to store the precomputed outputs of the module.
        allow_a_quant (`bool`, *optional*, default=`True`):
            Whether to allow the quantization for alpha tensor.
        allow_b_quant (`bool`, *optional*, default=`True`):
            Whether to allow the quantization for beta tensor.
        spans (`list[tuple[SmoothSpanMode, SmoothSpanMode]]`, *optional*, default=`[]`):
            The span combinations. The first element is for the alpha and the second element is for the beta.
        alpha (`float`, *optional*, default=`0.5`):
            The smoothing alpha.
        beta (`float`, *optional*, default=`-1`):
            The smoothing beta.
        num_grids (`int`, *optional*, default=`20`):
            The number of grids for grid search.
    """

    objective: SearchBasedCalibObjective = field(init=False, default=SearchBasedCalibObjective.OutputsError)
    granularity: SearchBasedCalibGranularity = field(init=False, default=SearchBasedCalibGranularity.Layer)
    element_batch_size: int = field(init=False, default=-1)
    element_size: int = field(init=False, default=-1)
    pre_reshape: bool = field(init=False, default=True)
    allow_low_rank: bool = field(init=False, default=False)


@configclass
@dataclass
class SmoothTransfomerConfig:
    """Configuration for smooth quantization of transformer-based models.

    Args:
        proj (`SkipBasedSmoothCalibConfig` or `None`, *optional*, default=`None`):
            The smooth configuration for projections.
        attn (`SmoothAttentionCalibConfig` or `None`, *optional*, default=`None`):
            The smooth configuration for attentions.
    """

    proj: SkipBasedSmoothCalibConfig | None = None
    attn: SmoothAttentionCalibConfig | None = None

    @property
    def enabled_proj(self) -> bool:
        """Whether the smooth quantization is enabled for projections."""
        return self.proj is not None

    @property
    def enabled_attn(self) -> bool:
        """Whether the smooth quantization is enabled for attentions."""
        return self.attn is not None

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Get the names of the smooth quantization configuration.

        Args:
            prefix (`str`, *optional*, default=`""`):
                The prefix of the directory.

        Returns:
            `list[str]`:
                The names of the smooth quantization configuration
        """
        proj_names = self.proj.generate_dirnames(prefix="proj") if self.proj is not None else []
        attn_names = self.attn.generate_dirnames(prefix="attn") if self.attn is not None else []
        num_names = max(len(proj_names), len(attn_names))
        names = []
        for index in range(num_names):
            name = []
            if index < len(proj_names):
                name.append(proj_names[index])
            if index < len(attn_names):
                name.append(attn_names[index])
            names.append("-".join(name))
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names
