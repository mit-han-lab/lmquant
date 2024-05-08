# -*- coding: utf-8 -*-
"""Smooth quantization configuration."""

import enum
import typing as tp
from dataclasses import dataclass, field

from omniconfig import Arguments, configclass

from ....utils import num2str
from .base.search import SearchBasedCalibConfig, SearchBasedCalibGranularity, SearchBasedCalibStrategy

__all__ = ["QuantSmoothCalibConfig", "QuantSmoothConfig"]


@configclass
@dataclass
class QuantSmoothCalibConfig(SearchBasedCalibConfig):
    """Configuration for smooth quantization.

    Args:
        objective (SearchBasedCalibObjective): The objective for quantization calibration.
            Defaults to ``SearchBasedCalibObjective.OutputsError``.
        strategy (SearchBasedCalibStrategy): The strategy for quantization calibration.
            Defaults to ``SearchBasedCalibStrategy.Manual``.
        granularity (SearchBasedCalibGranularity): The granularity for quantization calibration.
            Defaults to ``SearchBasedCalibGranularity.Layer``.
        degree (int): The power degree for the quantization error. Defaults to ``2``.
        element_batch_size (int): The element batch size for calibration. Defaults to ``-1``.
        sample_batch_size (int): The samples batch size for calibration. Defaults to ``-1``.
        element_size (int): The calibration element size. Defaults to ``-1``.
        sample_size (int): The calibration sample size. Defaults to ``-1``.
        pre_reshape (bool): Whether to enable reshaping the tensor before calibration.
            Defaults to ``True``.
        outputs_device (str): The device to store the precomputed outputs of the module.
            Defaults to ``"cpu"``.
        allow_kernel_calib (bool): Whether to allow kernel calibration (e.g., GPTQ). Defaults to ``False``.
        skips (list[str]): The keys of the modules to skip. Defaults to ``[]``.
        ranges (list[tuple[SmoothRangeMode, SmoothRangeMode]]): The range combinations. The first element is for
            the inputs and the second element is for the weights.
        alpha (float): The smoothing alpha. Defaults to ``0.5``.
        beta (float): The smoothing beta. Defaults to ``-1``. If ``beta < 0``, use ``1 - alpha``.
        num_grids (int): The number of grids for grid search. Defaults to ``20``.
    """

    class RangeMode(enum.Enum):
        """The mode for computing the range."""

        AbsMax = enum.auto()
        RootMeanSquare = enum.auto()

    ranges: list[tuple[RangeMode, RangeMode]] = field(default_factory=list)
    x_ranges: list[RangeMode] = field(default_factory=list, init=False)
    w_ranges: list[RangeMode] = field(default_factory=list, init=False)
    alpha: float = 0.5
    beta: float = -1
    num_grids: int = 20

    @property
    def y_ranges(self) -> list[RangeMode]:
        """The y ranges."""
        return self.w_ranges

    def __post_init__(self) -> None:  # noqa: C901
        self.skips = sorted(set(self.skips or []))
        # region remove duplicates of ranges
        _ranges, _range_set, _w_range_set, _x_range_set = [], set(), set(), set()
        self.w_ranges, self.x_ranges = [], []
        for x_range, w_range in self.ranges:
            if isinstance(w_range, str):
                w_range = QuantSmoothCalibConfig.RangeMode[w_range]
            if isinstance(x_range, str):
                x_range = QuantSmoothCalibConfig.RangeMode[x_range]
            xw_range = (x_range, w_range)
            if xw_range in _range_set:
                continue
            _ranges.append(xw_range)
            _range_set.add(xw_range)
            if w_range not in _w_range_set:
                self.w_ranges.append(w_range)
                _w_range_set.add(w_range)
            if x_range not in _x_range_set:
                self.x_ranges.append(x_range)
                _x_range_set.add(x_range)
        self.ranges = _ranges
        # endregion
        if self.strategy == SearchBasedCalibStrategy.Manual:
            assert len(self.ranges) == 1, "Only one range combination is allowed in manual mode."
            assert self.alpha != 0 or self.beta != 0, "alpha and beta cannot be both zero"
            self.alpha, self.beta = self.get_alpha_beta_pairs()[0]
        if self.granularity == SearchBasedCalibGranularity.Group:
            self.granularity = SearchBasedCalibGranularity.ChannelGroup
        assert -3 <= self.alpha <= 1, "alpha must be less than or equal to 1"
        assert -3 <= self.beta <= 1, "beta must be less than or equal to 1"
        super().__post_init__()

    def get_alpha_beta_pairs(self) -> list[tuple[float, float]]:  # noqa: C901
        """Get the alpha and beta pairs for smooth quantization.

        Returns:
            list[tuple[float, float]]: The alpha and beta pairs.
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

    def __str__(self) -> str:
        s = f"(ranges=[{', '.join(f'(w={w_range.name}, x={x_range.name})' for w_range, x_range in self.ranges)}]"
        if self.strategy == SearchBasedCalibStrategy.Manual:
            s += f", alpha={self.alpha}"
        elif self.strategy == SearchBasedCalibStrategy.GridSearch:
            s += f", objective={self.objective.name}, granularity={self.granularity.name}"
            s += f", num_grids={self.num_grids}"
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        if self.skips:
            s += f", skips=[{', '.join(self.skips)}]"
        return s + ")"

    def _generate_dirnames(self) -> list[str]:
        """Get the directory names of the smooth quantization configuration.

        Returns:
            list[str]: The directory names of the smooth quantization.
        """
        names = ["[{}]".format("+".join(f"x.{x_range.name}.w.{w_range.name}" for x_range, w_range in self.ranges))]
        alpha, beta = num2str(self.alpha), num2str(self.beta)
        if self.alpha < 0:
            alpha = f"_{alpha}"
        if self.beta < 0:
            beta = f"_{beta}"
        if self.strategy == SearchBasedCalibStrategy.Manual:
            names.append(f"a{alpha}.b{beta}")
        elif self.alpha > 0:
            names.append(f"g{self.num_grids}.b{beta}")
        elif self.beta > 0:
            names.append(f"g{self.num_grids}.a{alpha}")
        else:
            names.append(f"g{self.num_grids}.a{alpha}.b{beta}")
        return names

    @classmethod
    def update_get_arguments(
        cls: type["QuantSmoothCalibConfig"],
        *,
        overwrites: dict[str, tp.Callable[[Arguments], None] | None] | None = None,
        defaults: dict[str, tp.Any] | None = None,
    ) -> tuple[dict[str, tp.Callable[[Arguments], None] | None], dict[str, tp.Any]]:
        """Get the arguments for the smooth quantization configuration."""
        overwrites, defaults = SearchBasedCalibConfig.update_get_arguments(overwrites=overwrites, defaults=defaults)
        overwrites.setdefault(
            "ranges",
            lambda parser: parser.add_argument(
                "--ranges",
                nargs="+",
                type=lambda s: tuple(QuantSmoothCalibConfig.RangeMode[x.split(".")[-1]] for x in s.split(",")),
                default=defaults.get("ranges", []),
                help="Range combinations, the first is for the inputs and the second is for the weights.",
            ),
        )
        return overwrites, defaults


@configclass
@dataclass
class QuantSmoothConfig:
    """Configuration for smooth quantization.

    Args:
        xw (QuantSmoothCalibConfig): The smooth quantization configuration for weights.
            Defaults to ``None``.
        yx (QuantSmoothCalibConfig): The smooth quantization configuration for outputs.
            Defaults to ``None``.
    """

    xw: QuantSmoothCalibConfig | None = None
    yx: QuantSmoothCalibConfig | None = None

    @property
    def enabled_smooth_xw(self) -> bool:
        """Whether the xw smooth quantization is enabled."""
        return self.xw is not None

    @property
    def enabled_smooth_yx(self) -> bool:
        """Whether the yy smooth quantization is enabled."""
        return self.yx is not None

    def __post_init__(self) -> None:
        if self.yx is not None:
            self.yx.allow_kernel_calib = False

    def __str__(self) -> str:
        return f"(xw={self.xw}, yx={self.yx})"

    def generate_dirnames(self, prefix: str = "") -> list[str]:
        """Get the names of the smooth quantization configuration."""
        xw_names = self.xw.generate_dirnames(prefix="xw") if self.xw is not None else []
        yx_names = self.yx.generate_dirnames(prefix="yx") if self.yx is not None else []
        num_levels = max(len(xw_names), len(yx_names))
        names = []
        for level in range(num_levels):
            name = []
            if level < len(xw_names):
                name.append(xw_names[level])
            if level < len(yx_names):
                name.append(yx_names[level])
            names.append("-".join(name))
        if prefix and names:
            names = [f"{prefix}.{name}" for name in names]
        return names
