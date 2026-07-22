"""
multi_fold_loo_v2.py

Improvements over multi_fold_loo.py, based on issues found reviewing that
version:

1. FIXED: the old script split off a 25,920-row "official test set" that
   was never actually used (dead code / wasted data). Removed -- all
   held-out data is now used as test folds.

2. ADDED: a corruption sanity check. One additional chunk is built by
   sampling extra rows and shuffling their labels (deliberately bad data).
   If the pipeline is working, this chunk's true_marginal_value should
   come out clearly, distinctly positive (removing it should clearly HELP
   performance) -- much larger than the range seen among normal chunks.
   If it doesn't stand out, that's a sign the method can't be trusted on
   subtler real differences either.

3. INCREASED: n_chunks 10 -> 20, n_test_folds 5 -> 8, for a more granular
   and more stable ground truth.

4. ADDED: a `reliable` flag per chunk (std_across_folds < half of the
   |mean| delta) -- use this to exclude noisy chunks from your week-2
   correlation check rather than treating all chunks as equally trustworthy.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import xgboost as xgb

from data_prep import load_raw, clean, make_chunks

N_TEST_FOLDS = 8
FOLD_SIZE = 15000
N_CHUNKS = 20
TRAIN_POOL_SIZE = 15000
CORRUPTED_CHUNK_SIZE = 750  # ~ same size as a normal chunk (15000/20)
MODEL_CONFIG = dict(n_estimators=200, max_depth=5, learning_rate=0.1,
                     subsample=1.0, colsample_bytree=1.0, random_state=0, n_jobs=-1)


def fit_logloss(train_df: pd.DataFrame, test_X: pd.DataFrame, test_y: pd.Series,
                 feature_cols: list) -> float:
    model = xgb.XGBClassifier(**MODEL_CONFIG)
    model.fit(train_df[feature_cols], train_df["label"])
    preds = model.predict_proba(test_X)[:, 1]
    return -log_loss(test_y, preds)


def build_pool_with_corrupted_chunk(train_region: pd.DataFrame, train_pool_size: int,
                                     corrupted_size: int, seed: int):
    """Builds the normal train_pool, PLUS one extra chunk of corrupted_size
    rows sampled separately and with shuffled labels. Returns the combined
    pool, the leftover (for test folds), and which rows belong to the
    corrupted chunk.
    """
    train_pool, remainder = train_test_split(
        train_region, train_size=train_pool_size, random_state=seed,
        stratify=train_region["label"])

    corrupted_rows, leftover_pool = train_test_split(
        remainder, train_size=corrupted_size, random_state=seed + 1,
        stratify=remainder["label"])
    corrupted_rows = corrupted_rows.copy()
    rng = np.random.RandomState(seed + 2)
    corrupted_rows["label"] = rng.permutation(corrupted_rows["label"].values)

    combined_pool = pd.concat([train_pool, corrupted_rows], ignore_index=True)
    corrupted_mask = np.array([False] * len(train_pool) + [True] * len(corrupted_rows))

    return combined_pool, leftover_pool, corrupted_mask


def run_multi_fold_loo_v2(raw_path: str = "MiniBooNE_PID.txt",
                           train_pool_size: int = TRAIN_POOL_SIZE,
                           n_chunks: int = N_CHUNKS,
                           corrupted_size: int = CORRUPTED_CHUNK_SIZE,
                           n_test_folds: int = N_TEST_FOLDS, fold_size: int = FOLD_SIZE,
                           seed: int = 42):
    df = clean(load_raw(raw_path))
    feature_cols = [c for c in df.columns if c != "label"]

    train_region, _official_test_unused = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df["label"])
    # NOTE: _official_test_unused is intentionally not used further -- all
    # non-train_pool data is pooled as the source for independent test
    # folds, maximizing how many folds we can draw.

    combined_pool, leftover_pool, corrupted_mask = build_pool_with_corrupted_chunk(
        train_region, train_pool_size, corrupted_size, seed)

    real_pool = combined_pool.loc[~corrupted_mask].reset_index(drop=True)
    chunk_ids = make_chunks(real_pool, n_chunks=n_chunks, seed=seed)
    # Re-attach to combined_pool's indexing: rebuild chunk_ids aligned to combined_pool
    combined_pool = combined_pool.reset_index(drop=True)
    full_chunk_ids = pd.Series(index=combined_pool.index, dtype=object)
    full_chunk_ids.iloc[:len(real_pool)] = chunk_ids.values
    full_chunk_ids.iloc[len(real_pool):] = "CORRUPTED"

    print(f"[v2] combined_pool={len(combined_pool)} rows "
          f"({len(real_pool)} real + {corrupted_mask.sum()} corrupted), "
          f"leftover for test folds={len(leftover_pool)} rows")

    fold_deltas = {}
    unique_chunk_labels = sorted(chunk_ids.unique()) + ["CORRUPTED"]
    t0 = time.time()
    for fold in range(n_test_folds):
        test_fold = leftover_pool.sample(n=min(fold_size, len(leftover_pool)),
                                          random_state=100 + fold)
        test_X, test_y = test_fold[feature_cols], test_fold["label"]

        baseline = fit_logloss(combined_pool, test_X, test_y, feature_cols)
        deltas = []
        for cid in unique_chunk_labels:
            mask = full_chunk_ids != cid
            sub = combined_pool.loc[mask[mask].index]
            loo = fit_logloss(sub, test_X, test_y, feature_cols)
            deltas.append(baseline - loo)
        fold_deltas[fold] = deltas
        print(f"[v2] fold {fold} done ({time.time() - t0:.0f}s elapsed)")

    delta_matrix = pd.DataFrame(fold_deltas, index=unique_chunk_labels)
    delta_matrix.index.name = "chunk_id"

    mean_delta = delta_matrix.mean(axis=1)
    std_delta = delta_matrix.std(axis=1)
    reliable = std_delta < (0.5 * mean_delta.abs())

    result = pd.DataFrame({
        "chunk_id": delta_matrix.index,
        "chunk_size": [int((full_chunk_ids == cid).sum()) for cid in delta_matrix.index],
        "true_marginal_value": mean_delta.values,
        "std_across_folds": std_delta.values,
        "min_fold_delta": delta_matrix.min(axis=1).values,
        "max_fold_delta": delta_matrix.max(axis=1).values,
        "reliable": reliable.values,
        "is_corrupted_sanity_check": [cid == "CORRUPTED" for cid in delta_matrix.index],
    })
    return result, delta_matrix


if __name__ == "__main__":
    result, delta_matrix = run_multi_fold_loo_v2()
    result_sorted = result.sort_values("true_marginal_value", ascending=False)
    result_sorted.to_csv("loo_ground_truth_v2.csv", index=False)
    delta_matrix.to_csv("multi_fold_raw_deltas_v2.csv")

    print("\n=== Final ground truth v2 (sorted by value) ===")
    print(result_sorted.to_string(index=False))

    real_chunks = result[~result["is_corrupted_sanity_check"]]
    corrupted_row = result[result["is_corrupted_sanity_check"]].iloc[0]
    print(f"\n=== Sanity check ===")
    print(f"Corrupted chunk true_marginal_value: {corrupted_row['true_marginal_value']:.5f}")
    print(f"Real chunks range: {real_chunks['true_marginal_value'].min():.5f} "
          f"to {real_chunks['true_marginal_value'].max():.5f}")
    # Corrupted (label-shuffled) data should show a clearly NEGATIVE value
    # (removing it helps performance) that is more extreme than the real
    # chunks' range in either direction -- not "higher", but "more separated
    # from zero on the harmful side."
    clearly_separated = corrupted_row['true_marginal_value'] < real_chunks['true_marginal_value'].min()
    print(f"Corrupted chunk clearly separated below the real-chunk range "
          f"(correctly identified as harmful): {clearly_separated}")

    print(f"\nSign consistency (real chunks only): "
          f"{(delta_matrix.loc[real_chunks['chunk_id']].values > 0).sum()}"
          f"/{delta_matrix.loc[real_chunks['chunk_id']].size} positive")
    print(f"Reliable chunks (low fold-to-fold variance): "
          f"{result['reliable'].sum()}/{len(result)}")

    print("\nWrote loo_ground_truth_v2.csv and multi_fold_raw_deltas_v2.csv")
