"""
Hyperparameter tuning for YOLO26 using Optuna (TPE Bayesian search).

Usage:
  Bayesian search (Optuna TPE), 20 trials, 30-epoch runs:
  python -m src.models.yolo26.tune \\
      --coco_dir  data/coco_export \\
      --output_dir runs/yolo26_tune \\
      --model l \\
      --epochs 30 \\
      --imgsz 640 \\
      --batch 16 \\
      --workers 4 \\
      --patience 10 \\
      --iterations 20 \\
      --search_alg optuna \\
      --search_space '{
        "lr0":           ["log_float", 1e-5, 1e-2],
        "lrf":           ["float",     0.01, 1.0],
        "momentum":      ["float",     0.7,  0.98],
        "weight_decay":  ["float",     0.0,  1e-3],
        "warmup_epochs": ["float",     0.0,  5.0],
        "box":           ["float",     1.0,  20.0],
        "cls":           ["float",     0.1,  4.0],
        "dfl":           ["float",     0.4,  12.0]
      }'

  With augmentation search (--tune_aug requires --aug_search_space):
  python -m src.models.yolo26.tune \\
      ... \\
      --tune_aug \\
      --aug_search_space '{
        "fliplr":     ["float", 0.0, 1.0],
        "flipud":     ["float", 0.0, 1.0],
        "degrees":    ["float", 0.0, 180.0],
        "hsv_h":      ["float", 0.0, 0.1],
        "hsv_s":      ["float", 0.0, 0.9],
        "hsv_v":      ["float", 0.0, 0.9],
        "translate":  ["float", 0.0, 0.3],
        "scale":      ["float", 0.0, 0.5],
        "mosaic":     ["float", 0.0, 1.0],
        "mixup":      ["float", 0.0, 0.3],
        "copy_paste": ["float", 0.0, 1.0]
      }'

  Resuming an interrupted search:
    Re-run the exact same command. Optuna loads the existing study from
    output_dir/optuna.db (SQLite) and continues from where it left off.
    Already-completed trials are not repeated.

Search-space JSON schema (for --search_space and --aug_search_space):
  Each flag accepts either an inline JSON string or a path to a JSON file.
  The JSON must be an object mapping parameter name to a spec list:

    ["log_float", low, high]   — log-uniform float sampling
    ["float",     low, high]   — uniform float sampling
    ["int",       low, high]   — uniform integer sampling (inclusive)
    ["categorical", [c1, c2]]  — categorical sampling
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import optuna
from ultralytics import YOLO

from src.models.yolo26.train import AUG_CONFIG
from src.models.yolo26.utils import (
    convert_coco_labels_to_yolo,
    get_coco_yaml_path,
    validate_coco_dir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_STUDY_NAME = "yolo26_hparam_search"
_TUNE_METRIC = "metrics/mAP50-95(B)"


def _load_search_space(value: str) -> dict[str, tuple]:
    """Parse a search-space definition from a JSON string or file.

    Args:
        value: Either a path to a JSON file or an inline JSON string.
            The JSON must be an object mapping hyperparameter name to a
            spec list (e.g. ["float", 0.7, 0.98]).

    Returns:
        Dictionary mapping each hyperparameter name to its spec tuple.

    Raises:
        ValueError: If value cannot be parsed as JSON (as a file path or
            as an inline string), or if the parsed JSON is not a top-level
            object.
    """
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        candidate_path = Path(value)
        if not candidate_path.is_file():
            raise ValueError(
                f"Search-space value is not valid JSON and is not a readable file path: {value!r}"
            )
        parsed = json.loads(candidate_path.read_text())

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Search-space must be a JSON object, got {type(parsed).__name__}."
        )

    return {name: tuple(spec) for name, spec in parsed.items()}


def _suggest(trial: optuna.Trial, name: str, spec: tuple) -> Any:
    """Sample a single hyperparameter value from an Optuna trial.

    Args:
        trial: Active Optuna trial used to sample the value.
        name: Hyperparameter name passed to trial.suggest_*.
        spec: A tuple whose first element is a kind string and whose
            remaining elements are the bounds or choices:

            - ("log_float", low, high) — log-uniform float.
            - ("float",     low, high) — uniform float.
            - ("int",       low, high) — uniform integer (inclusive).
            - ("categorical", choices) — categorical from a list.

    Returns:
        The sampled value (type depends on kind).

    Raises:
        ValueError: If spec[0] is not one of the recognised kind strings.
    """
    kind = spec[0]
    if kind == "log_float":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec[1]))
    raise ValueError(f"Unknown spec kind for {name!r}: {kind!r}")


def run_trial(
    trial_idx: int,
    params: dict[str, Any],
    args: argparse.Namespace,
    trials_dir: Path,
    yaml_path: str,
) -> float | None:
    """Run one YOLO26 training trial in-process and return its mAP50-95.

    Loads a fresh YOLO model, calls model.train() with the sampled hyperparameter
    values merged into the fixed training settings, and returns the best
    mAP50-95 score from the run.

    Args:
        trial_idx: Optuna trial number, used to name the output sub-directory.
        params: Sampled hyperparameter values for this trial. Keys come from
            --search_space and (when --tune_aug is set) --aug_search_space.
        args: Parsed argparse.Namespace from parse_args().
        trials_dir: Root directory where per-trial output directories are
            created. Ultralytics writes weights, results.csv, etc. here.
        yaml_path: Absolute path to the COCO dataset.yaml.

    Returns:
        Best mAP50-95 as a float, or None if the metric could not be read
        (caller should treat None as a pruned trial). Returns 0.0 if training
        raised an exception.
    """
    trial_name = f"trial_{trial_idx:04d}"
    trial_output = trials_dir / trial_name
    trial_output.mkdir(parents=True, exist_ok=True)

    with open(trial_output / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    log.info("Trial %d — params: %s", trial_idx, json.dumps(params))

    # Explicit optimizer so Ultralytics does not use optimizer='auto', which
    # ignores sampled lr0/momentum. Optuna may override via params, e.g.
    # "optimizer": ["categorical", ["AdamW", "SGD"]] in --search_space.
    overrides: dict[str, Any] = {
        "data":       yaml_path,
        "epochs":     args.epochs,
        "imgsz":      args.imgsz,
        "batch":      args.batch,
        "workers":    args.workers,
        "patience":   args.patience,
        "project":    str(trials_dir),
        "name":       trial_name,
        "exist_ok":   True,
        "optimizer":  "AdamW",
    }
    if args.device is not None:
        overrides["device"] = args.device
    if not args.tune_aug:
        overrides.update(AUG_CONFIG[args.aug_config])
    overrides.update(params)

    t0 = time.time()
    try:
        model = YOLO(f"yolo26{args.model}.pt")
        train_results = model.train(**overrides)
        elapsed = time.time() - t0

        score: float | None = None
        if isinstance(train_results, dict):
            val = train_results.get(_TUNE_METRIC)
            if val is not None:
                score = float(val)
        if score is None and hasattr(model, "trainer") and model.trainer is not None:
            fitness = getattr(model.trainer, "best_fitness", None)
            if fitness is not None:
                score = float(fitness)

        if score is None:
            log.warning(
                "Trial %d: training succeeded but mAP50-95 could not be read "
                "(check %s/results.csv)",
                trial_idx, trial_output,
            )
        else:
            log.info("Trial %d finished in %.0fs — mAP50-95 = %.4f", trial_idx, elapsed, score)

        return score

    except Exception as exc:
        elapsed = time.time() - t0
        log.error("Trial %d raised exception after %.0fs: %s", trial_idx, elapsed, exc, exc_info=True)
        return 0.0


def _tune(
    args: argparse.Namespace,
    output_dir: Path,
    trials_dir: Path,
    yaml_path: str,
) -> dict[str, Any]:
    """Run hyperparameter search using Optuna (TPE or random sampler).

    Creates (or restores) an Optuna study backed by a SQLite database at
    output_dir/optuna.db so that interrupted runs can be resumed by re-running
        the same command (to same output_dir). Runs up to args.iterations trials;
        already-completed trials in the database count toward this total.

    Writes a cumulative trials/results.json (list of {trial, score, params})
    after each trial so progress is visible even if the job is killed.

    Args:
        args: Parsed argparse.Namespace from parse_args().
        output_dir: Root output directory; optuna.db is written here.
        trials_dir: Directory where per-trial sub-directories are created.
        yaml_path: Absolute path to the COCO dataset.yaml.

    Returns:
        Dictionary of best hyperparameter values from the study.
        Returns an empty dict if no trials completed successfully.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    space = _load_search_space(args.search_space)
    if args.tune_aug:
        space.update(_load_search_space(args.aug_search_space))

    log.info("Search space: %s", {k: list(v) for k, v in space.items()})

    sampler = (
        optuna.samplers.TPESampler(seed=args.seed)
        if args.search_alg == "optuna"
        else optuna.samplers.RandomSampler(seed=args.seed)
    )

    storage = f"sqlite:///{output_dir}/optuna.db"
    log.info("Optuna study storage: %s (resume by re-running the same command)", storage)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=_STUDY_NAME,
        storage=storage,
        load_if_exists=True,
    )

    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_done > 0:
        log.info(
            "Resuming: %d/%d trials already completed in %s",
            n_done, args.iterations, storage,
        )

    results: list[dict] = []
    results_path = trials_dir / "results.json"

    # Reload any results already written from a previous run.
    if results_path.is_file():
        try:
            results = json.loads(results_path.read_text())
        except (json.JSONDecodeError, OSError):
            results = []

    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in space.items()}
        score = run_trial(trial.number, params, args, trials_dir, yaml_path)

        if score is None:
            raise optuna.TrialPruned()

        results.append({"trial": trial.number, "score": score, "params": params})
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        return score

    study.optimize(objective, n_trials=args.iterations, show_progress_bar=False)

    if not study.best_trials:
        log.error("No completed trials — cannot determine best params.")
        return {}

    best = study.best_trial
    log.info(
        "Best trial: %d — mAP50-95 = %.4f\n%s",
        best.number, best.value, json.dumps(best.params, indent=2),
    )
    return best.params


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for YOLO26 hyperparameter tuning.

    Args:
        None

    Returns:
        argparse.Namespace with the following attributes:

        - coco_dir (str): Path to the COCO export directory.
        - output_dir (str): Root directory for trial outputs and result JSON.
        - model (str): YOLO26 size variant ("n", "s", "m", "l", or "x").
        - epochs (int): Training epochs per trial.
        - imgsz (int): Square input resolution; must be divisible by 32.
        - batch (int): Batch size per trial.
        - workers (int): Dataloader worker processes per trial.
        - patience (int): Per-trial early-stopping patience.
        - iterations (int): Total number of Optuna trials to run (including
          any already completed in a resumed study).
        - search_alg (str): "optuna" for TPE Bayesian search (default),
          or "random" for random search.
        - seed (int): Random seed for the Optuna sampler.
        - aug_config (str): Fixed augmentation preset when --tune_aug is off.
        - tune_aug (bool): Whether to search augmentation parameters.
        - search_space (str): Inline JSON or file path for the HP search space.
        - aug_search_space (str | None): Inline JSON or file path for the
          augmentation search space; required when tune_aug is True.
        - device (str | None): Device string for Ultralytics (e.g. "0", "0,1").
          Auto-detected if not set.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter search for YOLO26 using Optuna TPE. "
            "Requires: pip install -U ultralytics optuna"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--coco_dir", required=True, help="Path to the COCO export directory.")
    parser.add_argument(
        "--output_dir",
        default="runs/yolo26_tune",
        help="Root directory for trial outputs and result JSON (default: runs/yolo26_tune).",
    )
    parser.add_argument(
        "--model",
        choices=["n", "s", "m", "l", "x"],
        required=True,
        help="YOLO26 size variant. Expanded to yolo26{variant}.pt (e.g. 'l' → 'yolo26l.pt').",
    )
    parser.add_argument("--epochs", type=int, required=True, help="Training epochs per trial.")
    parser.add_argument(
        "--imgsz",
        type=int,
        required=True,
        help="Input image resolution in pixels. Must be divisible by 32 (YOLO stride).",
    )
    parser.add_argument("--batch", type=int, required=True, help="Batch size per trial.")
    parser.add_argument("--workers", type=int, required=True, help="Dataloader worker processes per trial.")
    parser.add_argument(
        "--patience",
        type=int,
        required=True,
        help="Per-trial early-stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help=(
            "Total number of Optuna trials to run. When resuming from an existing study, "
            "already-completed trials count toward this total."
        ),
    )
    parser.add_argument(
        "--search_alg",
        default="optuna",
        choices=["optuna", "random"],
        help=(
            "Search algorithm: 'optuna' for Optuna TPE Bayesian search (default), "
            "or 'random' for Optuna random search."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67123,
        help="Random seed for the Optuna sampler (default: 67123).",
    )
    parser.add_argument(
        "--aug_config",
        choices=list(AUG_CONFIG.keys()),
        default="custom",
        help=(
            "Fixed augmentation preset applied to every trial when --tune_aug is not set "
            "(default: custom). Ignored when --tune_aug is set."
        ),
    )
    parser.add_argument(
        "--tune_aug",
        action="store_true",
        help=(
            "Also search augmentation parameters. "
            "Requires --aug_search_space. Non-sampled aug params fall back to "
            "Ultralytics defaults when this flag is set."
        ),
    )
    parser.add_argument(
        "--search_space",
        required=True,
        help=(
            "HP search space as an inline JSON string or a path to a JSON file. "
            "See the module docstring for the schema and a palynomorph-tuned example."
        ),
    )
    parser.add_argument(
        "--aug_search_space",
        default=None,
        help=(
            "Augmentation search space as an inline JSON string or a path to a JSON file. "
            "Same format as --search_space. Required when --tune_aug is set."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device string passed to Ultralytics (e.g. '0', '0,1', 'cpu'). "
            "Auto-detected if not set. Trials run sequentially; each trial uses "
            "all specified GPUs (multi-GPU via DDP is safe here)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for YOLO26 hyperparameter tuning.

    Workflow:
        1. Validates coco_dir and imgsz (must be divisible by 32).
        2. Validates that aug_search_space is provided when tune_aug is set.
        3. Converts COCO annotations to YOLO format once (idempotent).
        4. Creates (or restores) an Optuna study backed by output_dir/optuna.db.
        5. Runs Optuna trials sequentially; each trial calls YOLO.train() in-process.
        6. Writes a cumulative trials/results.json after every trial.
        7. Writes best_params.json to output_dir when all trials are done.

    Resume:
        Re-run the exact same command. The SQLite study is loaded and only the
        remaining (iterations - completed) trials are executed.

    Args:
        None

    Returns:
        None

    Raises:
        FileNotFoundError: If coco_dir does not exist or is missing dataset.yaml.
        ValueError: If imgsz is not divisible by 32, or if tune_aug is set
            without aug_search_space.
        SystemExit: If no trials completed successfully.
    """
    args = parse_args()

    validate_coco_dir(args.coco_dir)

    if args.imgsz % 32 != 0:
        raise ValueError(f"--imgsz {args.imgsz} must be divisible by 32 (YOLO stride).")

    if args.tune_aug and args.aug_search_space is None:
        raise ValueError("--tune_aug requires --aug_search_space to be provided.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    convert_coco_labels_to_yolo(args.coco_dir)
    yaml_path = get_coco_yaml_path(args.coco_dir)

    log.info(
        "Starting YOLO26 hyperparameter search: search_alg=%s, iterations=%d, tune_aug=%s",
        args.search_alg, args.iterations, args.tune_aug,
    )

    best_params = _tune(args, output_dir, trials_dir, yaml_path)

    if not best_params:
        log.error("No valid trials completed. Check trial logs in %s.", trials_dir)
        sys.exit(1)

    best_path = output_dir / "best_params.json"
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log.info("Best params saved to %s", best_path)


if __name__ == "__main__":
    main()
