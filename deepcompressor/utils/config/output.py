# -*- coding: utf-8 -*-
"""Output configuration."""

import os
from dataclasses import dataclass, field
from datetime import datetime as DateTime

from omniconfig import configclass

__all__ = ["OutputConfig"]


@configclass
@dataclass
class OutputConfig:
    """Output configuration.

    Args:
        root (`str`, *optional*, defaults to `"runs"`):
            The output root directory.
        dirname (`str`, *optional*, defaults to `"default"`):
            The output directory name.
        job (`str`, *optional*, defaults to `"run"`):
            The job name.

    Attributes:
        dirpath (`str`):
            The output directory path.
        timestamp (`str`):
            The timestamp.
    """

    root: str = "runs"
    dirname: str = "default"
    job: str = "run"
    dirpath: str = field(init=False)
    timestamp: str = field(init=False)

    def __post_init__(self):
        self.timestamp = self.generate_timestamp()
        self.dirpath = os.path.join(self.root, self.dirname)

    @property
    def running_dirpath(self) -> str:
        """Get the running directory path."""
        return f"{self.dirpath}.RUNNING"

    @property
    def error_dirpath(self) -> str:
        """Get the error directory path."""
        return f"{self.dirpath}.ERROR"

    @property
    def job_dirname(self) -> str:
        """Get the job directory name."""
        return f"{self.job}-{self.timestamp}"

    @property
    def job_dirpath(self) -> str:
        """Get the job directory path."""
        return os.path.join(self.dirpath, self.job_dirname)

    @property
    def running_job_dirname(self) -> str:
        """Get the running job directory name."""
        return f"{self.job_dirname}.RUNNING"

    @property
    def error_job_dirname(self) -> str:
        """Get the error job directory name."""
        return f"{self.job_dirname}.ERROR"

    @property
    def running_job_dirpath(self) -> str:
        """Get the running job directory path."""
        return os.path.join(self.running_dirpath, self.running_job_dirname)

    def lock(self) -> None:
        """Lock the running (job) directory."""
        try:
            if os.path.exists(self.dirpath):
                os.rename(self.dirpath, self.running_dirpath)
            elif os.path.exists(self.error_dirpath):
                os.rename(self.error_dirpath, self.running_dirpath)
        except Exception:
            pass
        os.makedirs(self.running_job_dirpath, exist_ok=True)

    def unlock(self, error: bool = False) -> None:
        """Unlock the running (job) directory."""
        job_dirpath = os.path.join(self.running_dirpath, self.error_job_dirname if error else self.job_dirname)
        os.rename(self.running_job_dirpath, job_dirpath)
        if not self.is_locked_by_others():
            os.rename(self.running_dirpath, self.error_dirpath if error else self.dirpath)

    def is_locked_by_others(self) -> bool:
        """Check if the running directory is locked by others."""
        running_job_dirname = self.running_job_dirname
        for dirname in os.listdir(self.running_dirpath):
            if dirname.endswith(".RUNNING") and dirname != running_job_dirname:
                return True
        return False

    def get_running_path(self, filename: str) -> str:
        """Get the file path in the running directory."""
        name, ext = os.path.splitext(filename)
        return os.path.join(self.running_dirpath, f"{name}-{self.timestamp}{ext}")

    def get_running_job_path(self, filename: str) -> str:
        """Get the file path in the running job directory."""
        name, ext = os.path.splitext(filename)
        return os.path.join(self.running_job_dirpath, f"{name}-{self.timestamp}{ext}")

    @staticmethod
    def generate_timestamp() -> str:
        """Generate a timestamp."""
        return DateTime.now().strftime("%y%m%d.%H%M%S")
