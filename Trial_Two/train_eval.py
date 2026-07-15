"""
train_eval.py

Owns the model. This is the one function the LOO orchestration (loo_runner.py)
calls repeatedly -- it doesn't need to know anything about XGBoost internals,
just that this function trains on whatever rows it's given and returns a
single comparable metric.

Keep hyperparameters FIXED across every call in a given experiment run --
we're validating a data-valuation method, not tuning a model. Every run
should differ only in which rows are in train_df.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb

DEFAULT_CONFIG = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
}

# Set once, module-level, so every LOO call scores against the exact same
# held-out data without having to pass it through every function call.
_TEST_X = None
_TEST_Y = None


def set_test_set(test_df: pd.DataFrame) -> None:
    """Call this once at the start of an experiment run."""
    global _TEST_X, _TEST_Y
    feature_cols = [c for c in test_df.columns if c != "label"]
    _TEST_X = test_df[feature_cols]
    _TEST_Y = test_df["label"]


def train_and_eval(train_df: pd.DataFrame, config: dict = None,
                    seed: int = 42) -> float:
    """
    Fits XGBoost on train_df (all columns except 'label' are features),
    returns AUC on the fixed held-out test set.

    Parameters
    ----------
    train_df : DataFrame with feature columns + a 'label' column
    config   : hyperparameter dict; defaults to DEFAULT_CONFIG if None
    seed     : random seed for the model fit (NOT for data splitting --
               that's fixed once in data_prep.py)

    Returns
    -------
    float : AUC on the held-out test set. Higher = better.
    """
    if _TEST_X is None:
        raise RuntimeError("Call set_test_set(test_df) before train_and_eval().")

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    feature_cols = [c for c in train_df.columns if c != "label"]

    model = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        eval_metric=cfg["eval_metric"],
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_df[feature_cols], train_df["label"])

    preds = model.predict_proba(_TEST_X)[:, 1]
    return roc_auc_score(_TEST_Y, preds)


def train_and_eval_averaged(train_df: pd.DataFrame, config: dict = None,
                             seeds: tuple = (0, 1, 2)) -> float:
    """
    Same contract as train_and_eval, but averages AUC over multiple model
    seeds. This shrinks the noise floor (random model variance) without
    changing what's being measured -- useful when a single-seed LOO delta
    is too close to single-seed noise to trust (see loo_runner.py notes).
    Costs len(seeds)x the compute of a single train_and_eval call.
    """
    scores = [train_and_eval(train_df, config, seed=s) for s in seeds]
    return float(np.mean(scores))


def measure_noise_floor(train_df: pd.DataFrame, config: dict = None,
                         n_seeds: int = 5) -> dict:
    """
    Fits the SAME full data n_seeds times with different model random seeds.
    The spread of resulting AUCs is your noise floor -- any chunk's LOO delta
    smaller than this spread is indistinguishable from random model variance,
    not a real data-value signal.
    """
    scores = [train_and_eval(train_df, config, seed=s) for s in range(n_seeds)]
    result = {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "range": float(np.max(scores) - np.min(scores)),
    }
    print(f"[noise floor] mean={result['mean']:.5f} std={result['std']:.5f} "
          f"range={result['range']:.5f} over {n_seeds} seeds")
    return result


def measure_noise_floor_averaged(train_df: pd.DataFrame, config: dict = None,
                                  n_groups: int = 5, seeds_per_group: int = 3,
                                  seed_start: int = 0) -> dict:
    """
    Same idea as measure_noise_floor(), but for comparing against LOO deltas
    that were themselves computed with train_and_eval_averaged(). A single
    seed's AUC is noisier than a `seeds_per_group`-seed average, so comparing
    LOO deltas against single-seed noise (measure_noise_floor's default)
    understates how tight the true noise floor is once averaging is used --
    see the README warning about apples-to-apples noise floor comparisons.

    Fits n_groups independent `seeds_per_group`-seed averages (each group
    uses a disjoint block of seeds) on the SAME full data, and reports the
    spread across those group averages -- this is the actual noise floor
    that an averaged LOO delta needs to clear.
    """
    scores = []
    for g in range(n_groups):
        group_seeds = tuple(range(seed_start + g * seeds_per_group,
                                   seed_start + (g + 1) * seeds_per_group))
        scores.append(train_and_eval_averaged(train_df, config, seeds=group_seeds))

    result = {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "range": float(np.max(scores) - np.min(scores)),
        "seeds_per_group": seeds_per_group,
    }
    print(f"[noise floor] (averaged, {seeds_per_group} seeds/group) "
          f"mean={result['mean']:.5f} std={result['std']:.5f} "
          f"range={result['range']:.5f} over {n_groups} groups")
    return result
