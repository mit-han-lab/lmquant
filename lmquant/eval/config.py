# -*- coding: utf-8 -*-
"""Base class for all evaluation configs."""

import typing as tp
from dataclasses import dataclass, field
from datetime import datetime as DateTime

import omniconfig
from omniconfig import configclass

__all__ = ["BaseEvaluationConfig"]


@configclass
@dataclass
class BaseEvaluationConfig:
    """Base class for all evaluation configs.

    Args:
        num_gpus (int, optional): The number of GPUs to use. Defaults to ``1``.
        batch_size (int, optional): The batch size used for inference. Defaults to ``1``.
        output_dirname (str | None, optional): The output directory name. Defaults to ``None``.
        dirname_with_timestamp (bool, optional): Whether to append the current time to the output directory name.
            Defaults to ``True``.
    """

    # device settings
    num_gpus: int = field(default=1, metadata={omniconfig.ARGPARSE_ARGS: ("--num-gpus", "-n")})
    # evaluation settings
    batch_size: int = 1
    # output settings
    output_root: str = "runs"
    output_dirname: str = "default"
    attach_timestamp: bool = True

    timestamp: str = field(init=False)
    output_dirname_without_timestamp: str = field(init=False)

    def __post_init__(self):
        self.timestamp = self.generate_timestamp()
        self.output_dirname_without_timestamp = self.output_dirname or "default"
        if self.attach_timestamp:
            self.output_dirname = self.output_dirname_without_timestamp + "-" + self.timestamp

    @staticmethod
    def generate_timestamp() -> str:
        """Generate a timestamp."""
        return DateTime.now().strftime("%y%m%d.%H%M%S")

    @staticmethod
    def extract_timestamp(output_dirname: str) -> str:
        """Extract the timestamp from the output directory name."""
        splits = output_dirname.split("-")
        # if splits[-1] follows the timestamp format "%y%m%d.%H%M%S"
        possible_timestamp = splits[-1]
        if len(possible_timestamp) == 13 and possible_timestamp[6] == ".":
            if possible_timestamp[:6].isdigit() and possible_timestamp[7:].isdigit():
                return possible_timestamp
        return ""

    @staticmethod
    def remove_timestamp(output_dirname: str) -> str:
        """Remove the timestamp from the output directory name."""
        splits = output_dirname.split("-")
        # if splits[-1] follows the timestamp format "%y%m%d.%H%M%S"
        possible_timestamp = splits[-1]
        if len(possible_timestamp) == 13 and possible_timestamp[6] == ".":
            if possible_timestamp[:6].isdigit() and possible_timestamp[7:].isdigit():
                return "-".join(splits[:-1])
        return output_dirname

    def evaluate(self, *args, **kwargs) -> tp.Any:
        """Evaluate the model."""
        raise NotImplementedError
