from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import (
    RealTabPFNv25Model,
    TabPFNModel,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class TabPFNv2Model(TabPFNModel):
    """Original TabPFNv2 (NeurIPS 2024).

    Loaded via `create_default_for_version(ModelVersion.V2)` from the
    `tabpfn>=7.0.0` package, which ships v2 weights alongside v2.5/v2.6.
    """

    ag_key = "TA-TABPFN-V2"
    ag_name = "TA-TabPFN-v2"

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        num_cpus: int = 1,
        num_gpus: int = 0,
        time_limit: float | None = None,
        **kwargs,
    ):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        from torch.cuda import is_available

        is_classification = self.problem_type in ["binary", "multiclass"]

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        X = self.preprocess(X, y=y, is_train=True)

        hps = self._get_model_params()

        model_base = TabPFNClassifier if is_classification else TabPFNRegressor
        self.model = model_base.create_default_for_version(ModelVersion.V2)

        params_to_set = {
            "device": device,
            "n_jobs": num_cpus,
            "categorical_features_indices": self._cat_indices,
        }
        if self.fixed_random_state is not None:
            params_to_set[self.seed_name] = self.fixed_random_state
        elif self.seed_name in hps:
            params_to_set[self.seed_name] = hps[self.seed_name]

        self.model.set_params(**params_to_set)
        self.model = self.model.fit(X=X, y=y)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 10_000,
                "max_features": 500,
                "max_classes": 10,
            }
        )
        return default_auxiliary_params


class RealTabPFNv25Model_V2Limits(RealTabPFNv25Model):
    """RealTabPFN-v2.5 with v2's size constraints.

    Used to ensure apples-to-apples comparison: v2 and v2.5 are skipped on
    the exact same set of (dataset, fold) tasks by AutoGluon, so neither
    method can score on a task the other cannot run.
    """

    ag_key = "TA-REALTABPFN-V2.5-V2LIMITS"
    ag_name = "TA-RealTabPFN-v2.5-v2limits"

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 10_000,
                "max_features": 500,
                "max_classes": 10,
            }
        )
        return default_auxiliary_params
