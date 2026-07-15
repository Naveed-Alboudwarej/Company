"""
loo_runner.py

Owns the chunking / orchestration logic. Calls train_eval.train_and_eval()
once per chunk (leave-that-chunk-out) plus once on the full pool, and records
the performance delta -- this is your ground truth for validating the fast
estimator in week 2.

This file deliberately does NOT know anything about XGBoost internals --
it only calls train_and_eval(train_df, config, seed) -> float. That's the
contract that lets two people build this and train_eval.py in parallel.
"""

import time
import pandas as pd

from train_eval import (train_and_eval, train_and_eval_averaged,
                         set_test_set, measure_noise_floor,
                         measure_noise_floor_averaged, DEFAULT_CONFIG)


def run_loo_sweep(train_pool: pd.DataFrame, chunk_ids: pd.Series,
                   config: dict = None, seeds: tuple = (0, 1, 2)) -> pd.DataFrame:
    """
    Runs: 1 full-pool baseline fit + 1 fit per chunk with that chunk excluded.
    Each score is averaged over `seeds` model runs to shrink the noise floor --
    a single-seed LOO delta on this dataset was found to be too close to
    single-seed random variance to trust reliably (see notes in data_prep.py).

    Returns a DataFrame with columns:
      chunk_id, chunk_size, full_pool_score, loo_score, true_marginal_value

    true_marginal_value = full_pool_score - loo_score
      (positive => removing this chunk HURT performance => chunk had positive value
       negative => removing this chunk HELPED performance => chunk was net-harmful)
    """
    results = []

    t0 = time.time()
    baseline_score = train_and_eval_averaged(train_pool, config, seeds=seeds)
    print(f"[loo] full-pool baseline AUC (avg of {len(seeds)} seeds) = "
          f"{baseline_score:.5f} ({time.time() - t0:.1f}s)")

    unique_chunks = sorted(chunk_ids.unique())
    for i, cid in enumerate(unique_chunks):
        t0 = time.time()
        mask = chunk_ids != cid
        train_without_chunk = train_pool.loc[mask[mask].index]

        loo_score = train_and_eval_averaged(train_without_chunk, config, seeds=seeds)
        marginal_value = baseline_score - loo_score

        chunk_size = int((chunk_ids == cid).sum())
        results.append({
            "chunk_id": cid,
            "chunk_size": chunk_size,
            "full_pool_score": baseline_score,
            "loo_score": loo_score,
            "true_marginal_value": marginal_value,
        })
        print(f"[loo] chunk {cid:>3} ({i+1}/{len(unique_chunks)}, "
              f"{chunk_size} rows): loo_auc={loo_score:.5f} "
              f"delta={marginal_value:+.5f} ({time.time() - t0:.1f}s)")

    return pd.DataFrame(results)


if __name__ == "__main__":
    import datetime
    import os

    train_pool = pd.read_parquet("train_pool.parquet")
    test = pd.read_parquet("test.parquet")
    chunk_ids = pd.read_parquet("chunk_ids.parquet")["chunk_id"]

    set_test_set(test)

    # Step 1: noise floor -- run this BEFORE trusting any LOO delta below it.
    # The sweep below scores each chunk as a `len(seeds)`-seed average, so the
    # noise floor must be measured the SAME way (see train_eval.py docstring)
    # -- a single-seed noise floor is noisier than an averaged one and makes
    # real signal look weaker than it is. seed_start=100 keeps these seeds
    # disjoint from the sweep's own seeds below.
    seeds = (0, 1, 2)
    print("\n=== Noise floor check (single-seed, for reference) ===")
    measure_noise_floor(train_pool, n_seeds=5)

    print("\n=== Noise floor check (averaged, apples-to-apples with sweep) ===")
    noise_floor = measure_noise_floor_averaged(
        train_pool, n_groups=5, seeds_per_group=len(seeds), seed_start=100
    )

    # Step 2: the actual LOO sweep
    print("\n=== LOO sweep ===")
    results_df = run_loo_sweep(train_pool, chunk_ids, seeds=seeds)
    results_df.to_csv("loo_ground_truth.csv", index=False)
    print("\n[loo_runner] wrote loo_ground_truth.csv")
    print(results_df.sort_values("true_marginal_value", ascending=False))

    # Archive a timestamped, config-tagged copy so metric progression across
    # runs is visible AND comparable -- the tag encodes every knob that
    # changes what's being measured (pool size, chunk count, model config,
    # seed count), so mismatched runs are obvious from the filename alone.
    history_dir = "loo_history"
    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_tag = (
        f"pool{len(train_pool)}_chunks{chunk_ids.nunique()}_"
        f"est{DEFAULT_CONFIG['n_estimators']}_depth{DEFAULT_CONFIG['max_depth']}_"
        f"lr{DEFAULT_CONFIG['learning_rate']}_seeds{len(seeds)}"
    )
    archive_path = os.path.join(
        history_dir, f"loo_ground_truth_{timestamp}_{config_tag}.csv"
    )
    results_df.to_csv(archive_path, index=False)

    noise_floor_path = os.path.join(
        history_dir, f"noise_floor_{timestamp}_{config_tag}.txt"
    )
    with open(noise_floor_path, "w") as f:
        f.write(f"mean={noise_floor['mean']:.5f}\n")
        f.write(f"std={noise_floor['std']:.5f}\n")
        f.write(f"range={noise_floor['range']:.5f}\n")
        f.write(f"scores={noise_floor['scores']}\n")

    # Quality checklist -- clears-noise-floor and correct-sign are judged
    # against noise_floor['std'] since that's the per-observation spread from
    # random model variance; the range check compares the full spread of
    # true_marginal_value against the full spread of noise-floor scores.
    baseline_score = results_df["full_pool_score"].iloc[0]
    clears_floor = (results_df["true_marginal_value"].abs() > noise_floor["std"]).mean()
    correct_sign = (results_df["true_marginal_value"] > 0).mean()
    value_range = results_df["true_marginal_value"].max() - results_df["true_marginal_value"].min()
    range_ratio = value_range / noise_floor["range"] if noise_floor["range"] > 0 else float("inf")

    checklist_rows = [
        ("Check", "Your run", "Good target"),
        ("Baseline score not at ceiling (>0.99)", f"{baseline_score:.5f}",
         "leaves room" if baseline_score < 0.99 else "AT CEILING"),
        ("% chunks clearing noise floor", f"{clears_floor:.0%}",
         ">50%, ideally 80%+"),
        ("% chunks with correct (positive) sign", f"{correct_sign:.0%}", ">70%"),
        ("Range of true_marginal_value vs noise floor", f"{value_range:.5f}",
         f"notably > {noise_floor['range']:.5f} ({range_ratio:.1f}x)"),
    ]
    check_col = max(len(row[0]) for row in checklist_rows) + 2
    value_col = max(len(row[1]) for row in checklist_rows) + 2
    checklist_text = "\n".join(
        f"{label:<{check_col}}{value:<{value_col}}{target}"
        for label, value, target in checklist_rows
    )
    print("\n=== Quality checklist ===")
    print(checklist_text)

    checklist_path = os.path.join(
        history_dir, f"checklist_{timestamp}_{config_tag}.txt"
    )
    with open(checklist_path, "w") as f:
        f.write(checklist_text + "\n")

    print(f"\n[loo_runner] archived run to {archive_path}")
