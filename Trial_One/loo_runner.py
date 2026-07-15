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
                         set_test_set, measure_noise_floor)


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
    train_pool = pd.read_parquet("train_pool.parquet")
    test = pd.read_parquet("test.parquet")
    chunk_ids = pd.read_parquet("chunk_ids.parquet")["chunk_id"]

    set_test_set(test)

    # Step 1: noise floor -- run this BEFORE trusting any LOO delta below it
    print("\n=== Noise floor check ===")
    measure_noise_floor(train_pool, n_seeds=5)

    # Step 2: the actual LOO sweep
    print("\n=== LOO sweep ===")
    results_df = run_loo_sweep(train_pool, chunk_ids)
    results_df.to_csv("loo_ground_truth.csv", index=False)
    print("\n[loo_runner] wrote loo_ground_truth.csv")
    print(results_df.sort_values("true_marginal_value", ascending=False))
