from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
import os
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Union
from lightning_utilities.core.imports import RequirementCache
from lightning_fabric.utilities.logger import (
    _add_prefix,
    _convert_json_serializable,
    _convert_params,
    _sanitize_callable_params,
)

_WANDB_AVAILABLE = RequirementCache("wandb>=0.12.10")


class CustomWandBLogger(WandbLogger):
    @rank_zero_only
    def log_metrics(
            self,
            metrics: Mapping[str, float],
            step: int,
    ) -> None:
        # This is exactly like the regular WandbLogger, except the step must be
        # specified, and images are logged correctly according to the step.
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        self.experiment.log(dict(metrics, **{"trainer/global_step": step}), step=step)
