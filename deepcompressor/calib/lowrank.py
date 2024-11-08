# -*- coding: utf-8 -*-
"""Quantization SVD calibration module."""

from dataclasses import _MISSING_TYPE, MISSING

import torch
import torch.nn as nn

from ..data.common import TensorType
from ..nn.patch.lowrank import LowRankBranch
from ..quantizer.processor import Quantizer
from ..utils import math, tools
from ..utils.config import KeyEnableConfig
from .config import QuantLowRankCalibConfig, SearchBasedCalibObjective
from .search import SearchBasedCalibrator

__all__ = ["QuantLowRankCalibrator"]


class QuantLowRankCalibrator(SearchBasedCalibrator[QuantLowRankCalibConfig, LowRankBranch]):
    """The quantization low-rank branch calibrator."""

    def __init__(
        self,
        config: QuantLowRankCalibConfig,
        w_quantizer: Quantizer,
        x_quantizer: Quantizer | None,
        develop_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the calibrator.

        Args:
            config (`QuantLowRankCalibConfig`):
                The configuration of the quantization low-rank branch calibrator.
            w_quantizer (`Quantizer`):
                The quantizer for weights.
            x_quantizer (`Quantizer` or `None`):
                The quantizer for inputs.
            develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                The development data type.
        """
        if isinstance(config, KeyEnableConfig):
            assert config.is_enabled_for(w_quantizer.key), "The calibrator should be enabled for the quantizer."
        else:
            assert config.is_enabled(), "The calibrator should be enabled."
        super().__init__(
            tensor_type=TensorType.Weights,
            config=config,
            w_quantizer=w_quantizer,
            x_quantizer=x_quantizer,
            y_quantizer=None,
            develop_dtype=develop_dtype,
        )
        assert self.needs_quant, "The tensor should be quantized."
        self.num_iters = config.num_iters

    @property
    def population_size(self) -> int:
        """Return the population size of the current iteration."""
        return 1

    @property
    def allows_x_quant_for_wgts(self) -> bool:
        """Whether the calibrator allows input quantization when tensor_type is Weights."""
        return True

    @property
    def allows_w_quant_for_wgts(self) -> bool:
        """Whether the calibrator needs weight quantization when tensor_type is Weights."""
        return True

    def is_done(self) -> bool:
        """Check if the calibration is done."""
        return self.iter >= self.num_iters or self.early_stopped

    def is_last_iter(self) -> bool:
        """Check if the current iteration is the last one."""
        return self.iter == self.num_iters - 1

    def _reset(self, x_wgts: list[torch.Tensor | nn.Parameter], **kwargs) -> None:  # noqa: C901
        """Reset the calibrator.

        Args:
            x_wgts (`list[torch.Tensor | nn.Parameter]`):
                The weights in x-w computation.
        """
        self.best_branch: LowRankBranch = None
        self.best_error: torch.Tensor = None
        self.error_history: list[tuple[float, float]] = []
        self.early_stopped = False
        if len(x_wgts) > 1 and not self.config.exclusive:
            self.w = torch.cat([wgt.data for wgt in x_wgts], dim=0)
        else:
            assert len(x_wgts) == 1
            self.w = x_wgts[0].data
        if self.config.compensate:
            self.qw = torch.cat(
                [
                    self.w_quantizer.quantize(wgt.data, kernel=None, develop_dtype=self.develop_dtype).data
                    for wgt in x_wgts
                ],
                dim=0,
            )
        else:
            self.qw = 0
        self.hat_ws: list[torch.Tensor] = [None] * len(x_wgts)
        self.ocs: list[int] = [wgt.shape[0] for wgt in x_wgts]

    def get_best(self) -> LowRankBranch:
        """Get the best candidate.

        Returns:
            `LowRankBranch`:
                The best candidate.
        """
        return self.best_branch

    def _ask(self) -> LowRankBranch:
        """Ask for the next candidate.

        Returns:
            `LowRankBranch`:
                The next candidate.
        """
        branch = LowRankBranch(
            self.w.shape[1],
            self.w.shape[0],
            rank=self.config.rank,
            weight=self.w - self.qw,
        )
        self.wgt_idx = 0
        if len(self.hat_ws) > 1:
            lw = branch.get_effective_weight()
            rw = self.w - lw
            oc_idx = 0
            for idx, oc in enumerate(self.ocs):
                self.hat_ws[idx] = self.w_quantizer.quantize(
                    rw[oc_idx : oc_idx + oc], kernel=None, develop_dtype=self.develop_dtype
                ).data
                oc_idx += oc
            self.qw = torch.cat(self.hat_ws, dim=0)
            if self.objective != SearchBasedCalibObjective.OutputsError:
                oc_idx = 0
                for idx, oc in enumerate(self.ocs):
                    self.hat_ws[idx].add_(lw[oc_idx : oc_idx + oc])
                    oc_idx += oc
        else:
            lw = branch.get_effective_weight()
            self.qw = self.w_quantizer.quantize(self.w - lw, kernel=None, develop_dtype=self.develop_dtype).data
            if self.objective != SearchBasedCalibObjective.OutputsError:
                self.hat_ws = [self.qw + lw]
            else:
                self.hat_ws = [self.qw]
        return branch

    def _tell(self, error: list[torch.Tensor]) -> None:  # noqa: C901
        """Tell the error of the last candidate and update the best candidate.

        Args:
            errors (list[torch.Tensor]): The error of the last candidate.
        """
        if len(error) > 1:
            error = [sum(error)]
        error = error[0]
        assert isinstance(error, torch.Tensor)
        assert error.numel() == 1, "The error should only have one value."
        if self.best_error is None or error <= self.best_error:
            self.best_error = error
            self.best_branch = self.candidate
        elif self.config.early_stop:
            self.early_stopped = True
        if self.logger.level <= tools.logging.DEBUG:
            self.error_history.append(
                (
                    math.root_(error.to(torch.float64), self.config.degree).item(),
                    math.root_(self.best_error.to(torch.float64), self.config.degree).item(),
                )
            )
            if self.iter % 10 == 9 or self.is_last_iter() or self.early_stopped:
                iter_end = ((self.iter + 10) // 10) * 10
                iter_start = iter_end - 10
                iter_end = min(iter_end, self.iter + 1)
                history = self.error_history[iter_start:iter_end]
                self.logger.debug("  -      iter  = [%s]", ", ".join(f"{i:10d}" for i in range(iter_start, iter_end)))
                self.logger.debug("  -      error = [%s]", ", ".join(f"{e[0]:10.4f}" for e in history))
                self.logger.debug("  - best error = [%s]", ", ".join(f"{e[1]:10.4f}" for e in history))

    def _process_x_in_xw(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        if not self.needs_x_quant_for_wgts:
            return x
        return self.x_quantizer.quantize(x, channels_dim=channels_dim, develop_dtype=self.develop_dtype).data

    def _process_w_in_xw(self, w: torch.Tensor) -> torch.Tensor:
        hat_w = self.hat_ws[self.wgt_idx]
        self.hat_ws[self.wgt_idx] = None
        self.wgt_idx += 1
        return hat_w if self.needs_w_quant_for_wgts else w

    def _process_y_in_yx(self, y: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        raise RuntimeError("_process_y_in_yx should not be called in QuantSVDCalibrator.")

    def _process_x_in_yx(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        raise RuntimeError("_process_x_in_yx should not be called in QuantSVDCalibrator.")

    def _process_xw_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_process_xw_in_yx should not be called in QuantSVDCalibrator.")

    def _process_yw_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_process_yw_in_yx should not be called in QuantSVDCalibrator.")

    def _process_wgts_centric_mod(
        self, wgts: list[nn.Parameter], mods: list[nn.Module], update_state_dict: bool = True, **kwargs
    ) -> None:
        assert len(self.hat_ws) == len(wgts) == len(mods)
        shared = self.candidate
        if len(self.hat_ws) > 1:
            oc_idx = 0
            for mod, wgt, hat_w in zip(mods, wgts, self.hat_ws, strict=True):
                if update_state_dict:
                    self._state_dict.append((wgt, wgt.data))
                wgt.data = hat_w
                branch = LowRankBranch(wgt.shape[1], wgt.shape[0], rank=self.config.rank)
                branch.a = shared.a
                branch.b.to(dtype=wgt.dtype, device=wgt.device)
                branch.b.weight.copy_(shared.b.weight[oc_idx : oc_idx + wgt.data.shape[0]])
                oc_idx += wgt.data.shape[0]
                self._hooks.append(branch.as_hook().register(mod))
        else:
            if update_state_dict:
                self._state_dict.append((wgts[0], wgts[0].data))
            wgts[0].data = self.hat_ws[0]
            self._hooks.append(shared.as_hook().register(mods))
        if self.needs_x_quant_for_wgts:
            self._hooks.append(self.x_quantizer.as_hook().register(mods))
        self.hat_ws = [None] * len(self.hat_ws)
