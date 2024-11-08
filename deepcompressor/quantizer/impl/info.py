# -*- coding: utf-8 -*-
"""Quantization information class."""

from dataclasses import dataclass, field

import torch

from ...data.dtype import QuantDataType
from ...data.range import ProtectiveQuantRange, QuantRange, RangeBound
from ...data.utils import ShapeUtils
from ...data.zero import ZeroPointDomain
from ..config.base import DecomposedQuantizerConfig, QuantizerConfig
from .scale import QuantScaleInfo

__all__ = ["QuantScaleInfo", "QuantStepInfo", "QuantInfo"]


@dataclass
class QuantStepInfo:
    # region config
    quant_dtype: QuantDataType
    zero_domain: ZeroPointDomain | None
    group_shapes: tuple[tuple[int, ...], ...]
    scale_dtypes: tuple[torch.dtype | QuantDataType | None, ...]
    quant_range: QuantRange | None
    range_bound: RangeBound | None
    default_dtype: torch.dtype
    # endregion
    # region information
    tensor_shape: torch.Size
    """the shape is a torch.Size (s0, s1, ...)"""
    tensor_group_shapes: list[torch.Size]
    """each group shape is a torch.Size (gs0, gs1, ...)"""
    tensor_view_shape: torch.Size
    """the view shape is a torch.Size (#g0, gs0, #g1, gs1, ...)"""
    # endregion
    scale: QuantScaleInfo = field(init=False)

    def __post_init__(self):
        self.scale = QuantScaleInfo(
            tensor_view_shape=self.tensor_view_shape,
            tensor_quant_dtype=self.quant_dtype,
            tensor_zero_domain=self.zero_domain,
            tensor_quant_range=self.quant_range,
            tensor_range_bound=self.range_bound,
            scale_view_shapes=ShapeUtils.infer_scale_view_shapes(self.tensor_group_shapes, shape=self.tensor_shape),
            scale_quant_dtypes=self.scale_dtypes,
            default_quant_dtype=self.default_dtype,
        )

    @property
    def tensor_zero_domain(self) -> ZeroPointDomain | None:
        return self.scale.tensor_zero_domain

    @property
    def tensor_quant_range(self) -> QuantRange:
        """The intersection of the quant_range and quant_dtype."""
        return self.scale.tensor_quant_range

    @property
    def tensor_range_bound(self) -> RangeBound | None:
        return self.scale.tensor_range_bound

    def to_config(self) -> QuantizerConfig:
        return QuantizerConfig(
            dtype=self.quant_dtype,
            zero_point=self.zero_domain,
            group_shapes=self.tensor_group_shapes,
            scale_dtypes=self.scale.scale_quant_dtypes,
        )

    @staticmethod
    def construct(
        config: QuantizerConfig,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype,
        quant_range: QuantRange | None = None,
        range_bound: RangeBound | None = None,
    ) -> "QuantStepInfo":
        tensor_group_shapes = ShapeUtils.infer_group_shapes(config.group_shapes, shape=tensor_shape)
        tensor_view_shape = ShapeUtils.infer_view_shape(tensor_shape, group_shape=tensor_group_shapes[-1])
        return QuantStepInfo(
            quant_dtype=config.dtype,
            zero_domain=config.zero_point,
            group_shapes=config.group_shapes,
            scale_dtypes=config.scale_dtypes,
            quant_range=quant_range,
            range_bound=range_bound,
            default_dtype=default_dtype,
            tensor_shape=tensor_shape,
            tensor_group_shapes=tensor_group_shapes,
            tensor_view_shape=tensor_view_shape,
        )


@dataclass
class QuantInfo:
    steps: tuple[QuantStepInfo, ...]
    needs_dequant_saturation: bool = False

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def get_child(self, idx: int) -> QuantStepInfo:
        return self.steps[idx]

    def is_outdated(
        self,
        config: DecomposedQuantizerConfig,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype,
        quant_range: QuantRange | None = None,
        range_bound: RangeBound | None = None,
    ) -> bool:
        """Check if the current quantization information is outdated."""
        if self.num_steps != config.num_steps:
            return True
        for step_info, step_config in zip(self.steps, config.steps, strict=True):
            if step_info.quant_dtype != step_config.quant_dtype:
                return True
            if step_info.group_shapes != step_config.group_shapes:
                return True
            if step_info.scale_dtypes != step_config.scale_dtypes:
                return True
        if self.num_steps > 0:
            first_step = self.steps[0]
            if first_step.tensor_shape != tensor_shape:
                return True
            if first_step.default_dtype != default_dtype:
                return True
            if first_step.range_bound != range_bound:
                return True
            if self.steps[-1].quant_range != quant_range:
                return True
            if self.num_steps > 1 and self.needs_dequant_saturation != config.needs_dequant_saturation:
                return True
        return False

    @staticmethod
    def construct(
        config: DecomposedQuantizerConfig,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype,
        quant_range: QuantRange | None = None,
        range_bound: RangeBound | None = None,
    ) -> "QuantInfo":
        steps: list[QuantStepInfo] = []
        num_steps = config.num_steps
        step_default_dtype = default_dtype
        step_range_bound = range_bound
        for step, step_config in enumerate(config.steps):
            assert step_config.quant_dtype is not None, f"quant_dtype is required for step {step}"
            if step == num_steps - 1:
                step_quant_range = quant_range
            elif step < num_steps - 2 or config.needs_dequant_saturation:
                step_quant_range = None
            else:  # ! only second last step quantization can be protected without saturation in the computation
                step_quant_range = ProtectiveQuantRange.construct(
                    outer_dtype=step_config.quant_dtype,
                    inner_dtype=config.steps[-1].quant_dtype,
                    zero_domain=config.steps[-1].zero_domain,
                    inner_quant_range=quant_range,
                )
            steps.append(
                QuantStepInfo.construct(
                    step_config,
                    tensor_shape=tensor_shape,
                    default_dtype=step_default_dtype,
                    quant_range=step_quant_range,
                    range_bound=step_range_bound,
                )
            )
            step_default_dtype = step_config.quant_dtype
            step_range_bound = steps[-1].scale.tensor_quant_range
        return QuantInfo(steps=tuple(steps), needs_dequant_saturation=config.needs_dequant_saturation)
