from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tabarena.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabarena.benchmark.models.ag import (
    RealTabPFNv25Model_V2Limits,
    TabPFNv2Model,
)
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.website.website_format import format_leaderboard


if __name__ == "__main__":
    expname = str(Path(__file__).parent / "experiments" / "tabpfn_v2_vs_v25")
    eval_dir = Path(__file__).parent / "eval"
    ignore_cache = False

    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata

    # Full TabArena benchmark: 51 datasets x 3 folds.
    # For a quick smoke test, swap to: datasets = ["anneal", "credit-g", "diabetes"]; folds = [0]
    datasets = list(task_metadata["name"])
    folds = [0, 1, 2]

    methods = [
        AGModelBagExperiment(
            name="TabPFNv2_BAG_L1",
            model_cls=TabPFNv2Model,
            model_hyperparameters={},
            num_bag_folds=8,
            time_limit=3600,
        ),
        AGModelBagExperiment(
            name="TabPFNv25_V2Limits_BAG_L1",
            model_cls=RealTabPFNv25Model_V2Limits,
            model_hyperparameters={},
            num_bag_folds=8,
            time_limit=3600,
        ),
    ]

    runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)
    results_lst: list[dict[str, Any]] = runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    end_to_end = EndToEnd.from_raw(
        results_lst=results_lst,
        task_metadata=task_metadata,
        cache=False,
        cache_raw=False,
    )
    end_to_end_results = end_to_end.to_results()

    eval_dir.mkdir(parents=True, exist_ok=True)
    end_to_end_results.model_results.to_csv(eval_dir / "model_results.csv", index=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{end_to_end_results.model_results.head(100)}")

    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=eval_dir,
        only_valid_tasks=True,
        use_model_results=True,
        new_result_prefix="V2vs25_",
    )
    leaderboard.to_csv(eval_dir / "leaderboard_raw.csv", index=False)

    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    leaderboard_website.to_csv(eval_dir / "leaderboard_website.csv", index=False)
    leaderboard_md = leaderboard_website.to_markdown(index=False)
    (eval_dir / "leaderboard_website.md").write_text(leaderboard_md, encoding="utf-8")
    print(leaderboard_md)
