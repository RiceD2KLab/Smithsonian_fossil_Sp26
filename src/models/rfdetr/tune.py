"""
Hyperparameter tuning for RF-DETR.

Usage:
  Bayesian optimisation, X trials, Y-epoch runs for search:
  python -m src.models.rfdetr.tune \\
      --coco_dir  data/coco_export \\
      --output_dir runs/rfdetr_tune \\
      --n_trials X \\
      --method bayesian \\
      --epochs Y \\
      --imgsz 1008 \\
      --workers 4 \\
      --nproc 2 \\
      --model 2xlarge \\
      --early_stopping_patience 10 \\
      --search_space '{"lr": ["log_float", 1e-6, 1e-3], "weight_decay": ["log_float", 1e-4, 0.1],
       "drop_path": ["float", 0.0, 0.3], "warmup_epochs": ["int", 1, 5], "lr_min_factor": ["float", 0.01, 0.3]}' \\

  With augmentation search (--tune_aug also requires --aug_search_space):
  python -m src.models.rfdetr.tune \\
      ... \\
      --tune_aug \\
      --aug_search_space '{"rotate_p": ["float", 0.3, 0.9], "brightness_p": ["float", 0.1, 0.5], "hsv_p": ["float", 0.1, 0.5]}'

Search-space JSON schema (for --search_space and --aug_search_space):
  Each flag accepts either an inline JSON string or a path to a JSON file.
  The JSON must be an object mapping parameter name to a spec list:

    ["log_float", low, high]  — log-uniform float sampling
    ["float",     low, high]  — uniform float sampling
    ["int",       low, high]  — uniform integer sampling
    ["categorical", [c1, c2]] — categorical sampling

  Example search space (previously the hard-coded defaults):
    {
      "lr":            ["log_float", 1e-6,  1e-3],
      "weight_decay":  ["log_float", 1e-4,  0.1],
      "drop_path":     ["float",     0.0,   0.3],
      "warmup_epochs": ["int",       1,     5],
      "lr_min_factor": ["float",     0.01,  0.3]
    }

  Example augmentation search space (previously the hard-coded defaults):
    {
      "rotate_p":     ["float", 0.3, 0.9],
      "brightness_p": ["float", 0.1, 0.5],
      "hsv_p":        ["float", 0.1, 0.5]
    }

Requires: pip install optuna
"""

from __future__ import annotations
from src.models.rfdetr.train import _MODEL_IMGSZ_DIVISOR

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import optuna

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# DDP coordinator address/port used when nproc > 1 (passed to torchrun).
_DDP_MASTER_ADDR = "localhost"
_DDP_MASTER_PORT = "29500"


def _load_search_space(value: str) -> dict[str, tuple]:
    """Parse a search-space definition.

    Accepts JSON whose top-level value is an object mapping hyperparameter name to a spec list.

    Args:
        value: Either a path to a JSON file or an inline JSON string.

    Returns:
        Dictionary mapping hyperparameter name to its spec tuple.

    Raises:
        ValueError: If `value` cannot be parsed as JSON (as a file path or
            as an inline string), or if the parsed JSON is not a top-level
            object.
    """
    candidate_path = Path(value)
    if candidate_path.is_file():
        raw_text = candidate_path.read_text()
    else:
        raw_text = value

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Search-space value is not valid JSON "
            f"(and is not a readable file path): {value!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Search-space must be a JSON object, got {type(parsed).__name__}."
        )

    return {name: tuple(spec) for name, spec in parsed.items()}


def _read_score(trial_dir: Path) -> float | None:
    """Extract the best validation mAP@0.5 from a completed trial directory.

    Reads results.json written by RF-DETR and returns the "map" value.

    Args:
        trial_dir: Path to the trial output directory that contains results.json.
            results.json must have a top-level "map" key.

    Returns:
        The mAP@0.5 as a float, or None if the file is absent or malformed.
    """
    results_file = trial_dir / "results.json"
    if results_file.is_file():
        try:
            data = json.loads(results_file.read_text())
            if isinstance(data, dict) and "map" in data:
                return float(data["map"])
        except (json.JSONDecodeError, ValueError, KeyError):
            log.warning("Failed to parse results.json: %s", results_file)

    return None


def _build_aug_json(params: dict[str, Any]) -> str | None:
    """Build the --aug_json override string from sampled augmentation probabilities.

    Reads rotate_p, brightness_p, and hsv_p from params (if present) and formats them
    as a JSON string compatible with train.py's --aug_json flag.

    Args:
        params: Dictionary of sampled hyperparameter values.
            Only keys rotate_p, brightness_p, and hsv_p are consumed; all other keys are ignored.

    Returns:
        A JSON-encoded override string (e.g. '{"Rotate": {"p": 0.7}}'), or None if none of the augmentation keys are present.
    """
    aug_overrides: dict[str, dict] = {}
    if "rotate_p" in params:
        aug_overrides["Rotate"] = {"p": params["rotate_p"]}
    if "brightness_p" in params:
        aug_overrides["RandomBrightnessContrast"] = {"p": params["brightness_p"]}
    if "hsv_p" in params:
        aug_overrides["HueSaturationValue"] = {"p": params["hsv_p"]}
    return json.dumps(aug_overrides) if aug_overrides else None


def _suggest(trial: optuna.Trial, name: str, spec: tuple) -> Any:
    """Sample a single hyperparameter value from an Optuna trial.

    Dispatches to the appropriate trial.suggest_* method based on the
    kind field of spec.

    Args:
        trial: Active Optuna trial used to sample the value.
        name: Hyperparameter name passed to trial.suggest_*.
        spec: A tuple whose first element is a kind string and whose
            remaining elements are the bounds or choices:

            - ("log_float", low, high) — log-uniform float.
            - ("float", low, high)     — uniform float.
            - ("int", low, high)        — uniform integer.
            - ("categorical", choices)  — categorical from a list.

    Returns:
        The sampled value (type depends on kind: float, int, or the element type of choices).

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
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"Unknown spec kind: {kind!r}")


def run_trial(
    trial_idx: int,
    params: dict[str, Any],
    base_args: argparse.Namespace,
    trials_dir: Path,
) -> float | None:
    """Run one RF-DETR training trial as a subprocess.

    Builds the training command from base_args and params, launches
    it via sys.executable (or torchrun for multi-GPU), and returns
    the best mAP@0.5 once the process exits.

    Stdout and stderr are merged into train.log inside the trial
    directory. Sampled hyperparameters are written to params.json
    in the same directory.

    Args:
        trial_idx: Trial index, used to name the output sub-directory.
        params: Sampled hyperparameter values for this trial. Keys are
            determined by the search spaces passed via --search_space and --aug_search_space.
        base_args: Parsed argparse.Namespace from parse_args():
            supplies fixed settings such as coco_dir, epochs,
            imgsz, workers, model, batch_size, grad_accum, nproc, and early_stopping_patience.
        trials_dir: Root directory where per-trial subdirectories are created.

    Returns:
        Best mAP@0.5 as a float if the training script succeeded and results.json is present and parseable;
        None if the script succeeded but no metric could be read; 0.0 if the script exited with a non-zero return code or raised an exception.
    """
    trial_output = trials_dir / f"trial_{trial_idx:04d}"
    trial_output.mkdir(parents=True, exist_ok=True)

    if base_args.nproc > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={base_args.nproc}",
            f"--master_addr={_DDP_MASTER_ADDR}",
            f"--master_port={_DDP_MASTER_PORT}",
            "-m", "src.models.rfdetr.train",
        ]
    else:
        cmd = [sys.executable, "-m", "src.models.rfdetr.train"]

    cmd += [
        "--coco_dir",                base_args.coco_dir,
        "--epochs",                  str(base_args.epochs),
        "--imgsz",                   str(base_args.imgsz),
        "--workers",                 str(base_args.workers),
        "--model",                   base_args.model,
        "--output_dir",              str(trial_output),
        "--early_stopping_patience", str(base_args.early_stopping_patience),
        "--lr_scheduler",            "cosine",
        "--lr",                      str(params["lr"]),
        "--weight_decay",            str(params["weight_decay"]),
        "--drop_path",               str(params["drop_path"]),
        "--warmup_epochs",           str(params["warmup_epochs"]),
        "--lr_min_factor",           str(params["lr_min_factor"]),
        "--batch_size",              str(base_args.batch_size),
        "--grad_accum",              str(base_args.grad_accum),
    ]

    aug_json = _build_aug_json(params)
    if aug_json:
        cmd += ["--aug_json", aug_json]

    log.info("Trial %d — params: %s", trial_idx, json.dumps(params))

    with open(trial_output / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    log_path = trial_output / "train.log"
    t0 = time.time()
    try:
        with open(log_path, "w") as flog:
            result = subprocess.run(cmd, stdout=flog, stderr=subprocess.STDOUT, check=False)
        elapsed = time.time() - t0

        if result.returncode != 0:
            log.warning(
                "Trial %d failed (exit %d) after %.0fs — see %s",
                trial_idx, result.returncode, elapsed, log_path,
            )
            return 0.0

        score = _read_score(trial_output)
        if score is None:
            log.warning(
                "Trial %d: training succeeded but mAP could not be read from %s/results.json",
                trial_idx, trial_output,
            )
        else:
            log.info("Trial %d finished in %.0fs — mAP@0.5 = %.4f", trial_idx, elapsed, score)
        return score

    except Exception as exc:
        log.error("Trial %d raised an exception: %s", trial_idx, exc)
        return 0.0


def _tune(args: argparse.Namespace, trials_dir: Path) -> dict[str, Any]:
    """Run hyperparameter search using Optuna (TPE or random sampler).

    Creates an in-memory Optuna study, runs args.n_trials trials via run_trial,
    and persists a cumulative results.json (list of {trial, score, params} dicts) after each trial completes.

    Args:
        args: Parsed argparse.Namespace from parse_args():
            supplies the relevant fields: method, seed, n_trials, tune_aug, search_space, aug_search_space.
        trials_dir: Directory where per-trial outputs and the cumulative results.json are written.

    Returns:
        Dictionary of best hyperparameter values from the completed study.
        Returns an empty dict if no trials completed successfully.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Build the combined search space from CLI-supplied definitions.
    space = _load_search_space(args.search_space)
    if args.tune_aug:
        space.update(_load_search_space(args.aug_search_space))

    log.info("Search space: %s", {k: list(v) for k, v in space.items()})

    sampler = (
        optuna.samplers.TPESampler(seed=args.seed)
        if args.method == "bayesian"
        else optuna.samplers.RandomSampler(seed=args.seed)
    )

    results: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in space.items()}

        score = run_trial(trial.number, params, args, trials_dir)

        if score is None:
            raise optuna.TrialPruned()

        results.append({"trial": trial.number, "score": score, "params": params})
        with open(trials_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="rfdetr_hparam_search",
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    if not study.best_trials:
        log.error("No completed trials — cannot determine best params.")
        return {}

    best = study.best_trial
    log.info("Best trial: %d — mAP@0.5 = %.4f\n%s", best.number, best.value, json.dumps(best.params, indent=2))
    return best.params


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RF-DETR hyperparameter tuning.

    Returns:
        argparse.Namespace with the following attributes:

        - coco_dir (str): Path to the COCO export directory.
        - output_dir (str): Root directory for trial outputs and results JSON.
        - model (str): RF-DETR variant to tune.
        - nproc (int): GPUs per trial (>1 uses torchrun DDP).
        - epochs (int): Training epochs per trial.
        - imgsz (int): Square input resolution.
        - workers (int): Dataloader worker processes per trial.
        - early_stopping_patience (int): Early-stopping patience.
        - method (str): "bayesian" or "random".
        - n_trials (int): Number of trials to run.
        - seed (int): Random seed for reproducibility.
        - tune_aug (bool): Whether to also search augmentation probabilities.
        - batch_size (int): Batch size per GPU.
        - grad_accum (int): Gradient accumulation steps.
        - search_space (str): Inline JSON or file path defining the
          hyperparameter search space (required).
        - aug_search_space (str | None): Inline JSON or file path defining
          the augmentation search space (required when --tune_aug is set).
    """
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for RF-DETR (Bayesian or random). Requires optuna.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--coco_dir", required=True, help="Path to the COCO export directory.")
    parser.add_argument(
        "--output_dir",
        default="runs/rfdetr_tune",
        help="Root directory for all trial outputs and results JSON (default: runs/rfdetr_tune).",
    )
    parser.add_argument(
        "--model",
        choices=["nano", "small", "base", "large", "xlarge", "2xlarge"],
        required=True,
        help="RF-DETR model variant to tune.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help=(
            "Number of GPUs per trial. >1 launches each trial via torchrun "
            "--nproc_per_node=N for DDP (default: 1)."
        ),
    )
    parser.add_argument("--epochs", type=int, required=True, help="Epochs per trial (e.g. 30 for fast search).")
    parser.add_argument(
        "--imgsz",
        type=int,
        required=True,
        help=(
            "Image resolution. Must be divisible by the model's block size: "
            "56 for nano/small/base/large, 40 for xlarge/2xlarge."
        ),
    )
    parser.add_argument("--workers", type=int, required=True, help="Dataloader worker processes per trial.")
    parser.add_argument("--early_stopping_patience", type=int, required=True, help="Early-stopping patience.")
    parser.add_argument(
        "--method",
        choices=["bayesian", "random"],
        default="bayesian",
        help="Search strategy: 'bayesian' (Optuna TPE) or 'random' (Optuna RandomSampler) (default: bayesian).",
    )
    parser.add_argument("--n_trials", type=int, required=True, help="Number of trials to evaluate.")
    parser.add_argument("--seed", type=int, default=67123, help="Random seed for reproducibility.")
    parser.add_argument(
        "--tune_aug",
        action="store_true",
        help=(
            "Also search over augmentation probabilities.  "
            "Requires --aug_search_space to be provided."
        ),
    )
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size per GPU.")
    parser.add_argument("--grad_accum", type=int, required=True, help="Gradient accumulation steps.")
    parser.add_argument(
        "--search_space",
        required=True,
        help=(
            "Hyperparameter search space as an inline JSON string or a path to a JSON file.  "
            "The JSON must be an object mapping parameter name to a spec list.  "
            "See the module docstring for the schema and an example."
        ),
    )
    parser.add_argument(
        "--aug_search_space",
        default=None,
        help=(
            "Augmentation search space as an inline JSON string or a path to a JSON file.  "
            "Same format as --search_space.  Required when --tune_aug is set."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for RF-DETR hyperparameter tuning.

    Workflow:
        1. Validates coco_dir exists and imgsz satisfies the
           model's block-size divisibility constraint.
        2. Validates that aug_search_space is provided when tune_aug is set.
        3. Creates the trials/ subdirectory inside output_dir.
        4. Runs _tune() to execute all Optuna trials.
        5. Saves best hyperparameters to best_params.json.

    Raises:
        FileNotFoundError: If coco_dir does not exist.
        ValueError: If imgsz is not divisible by the model's required block size, or if tune_aug is set without aug_search_space.
        SystemExit: If no trials completed successfully.
    """
    args = parse_args()

    if not Path(args.coco_dir).is_dir():
        raise FileNotFoundError(f"coco_dir not found: {args.coco_dir}")

    divisor = _MODEL_IMGSZ_DIVISOR[args.model]
    if args.imgsz % divisor != 0:
        raise ValueError(
            f"--imgsz {args.imgsz} is not divisible by {divisor} "
            f"(required for --model {args.model})."
        )

    if args.tune_aug and args.aug_search_space is None:
        raise ValueError("--tune_aug requires --aug_search_space to be provided.")

    trials_dir = Path(args.output_dir) / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Starting RF-DETR hyperparameter search: method=%s, n_trials=%d, nproc=%d, tune_aug=%s",
        args.method, args.n_trials, args.nproc, args.tune_aug,
    )

    best_params = _tune(args, trials_dir)

    if not best_params:
        log.error("No valid trials completed. Check trial logs in %s.", trials_dir)
        sys.exit(1)

    best_path = Path(args.output_dir) / "best_params.json"
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log.info("Best params saved to %s", best_path)


if __name__ == "__main__":
    main()
