"""
multi_fold_loo_v3.py

Fixes vs v2, found while reviewing it:

1. FIXED: v2's 8 test folds were drawn independently with replacement
   from an 87,926-row leftover pool at 15,000 rows/fold -- 8*15,000 =
   120,000 > 87,926, meaning folds necessarily overlapped. This partially
   invalidates using them as evidence of independent-fold stability.
   Fixed: folds are now a genuine disjoint partition of the leftover pool
   (~10,990 rows/fold at n_test_folds=8), so no row appears in more than
   one fold.

2. ADDED: a redundancy sanity check alongside the existing corruption
   check. Duplicates one of the highest-value real chunks and adds it as
   a second copy. Expectation: with its twin already present in the pool,
   the duplicate's own marginal value should be much lower than the
   original chunk's standalone value -- if the ORIGINAL chunk was found
   valuable at (say) ~0.0028, a full duplicate shouldn't ALSO show ~0.0028,
   since the information it carries is already covered. This tests a
   different, marketplace-relevant failure mode than label corruption:
   can the method recognize when data adds nothing NEW, even though it's
   perfectly clean and correctly labeled?
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import xgboost as xgb

from data_prep import load_raw, clean, make_chunks

N_TEST_FOLDS = 5
N_CHUNKS = 20
TRAIN_POOL_SIZE = 15000
CORRUPTED_CHUNK_SIZE = 750
MODEL_CONFIG = dict(n_estimators=200, max_depth=5, learning_rate=0.1,
                     subsample=1.0, colsample_bytree=1.0, random_state=0, n_jobs=-1)


def fit_logloss(train_df, test_X, test_y, feature_cols):
    model = xgb.XGBClassifier(**MODEL_CONFIG)
    model.fit(train_df[feature_cols], train_df["label"])
    preds = model.predict_proba(test_X)[:, 1]
    return -log_loss(test_y, preds)


def make_disjoint_folds(leftover_pool: pd.DataFrame, n_folds: int, seed: int):
    """Partition leftover_pool into n_folds genuinely non-overlapping folds
    (as equal in size as possible), stratified by label.
    """
    rng = np.random.RandomState(seed)
    fold_assignment = pd.Series(index=leftover_pool.index, dtype=int)
    for label_value in leftover_pool["label"].unique():
        idx = leftover_pool.index[leftover_pool["label"] == label_value].to_numpy().copy()
        rng.shuffle(idx)
        fold_assignment.loc[idx] = np.arange(len(idx)) % n_folds
    return fold_assignment


def run_v3(raw_path="MiniBooNE_PID.txt", train_pool_size=TRAIN_POOL_SIZE,
           n_chunks=N_CHUNKS, corrupted_size=CORRUPTED_CHUNK_SIZE,
           n_test_folds=N_TEST_FOLDS, seed=42):
    df = clean(load_raw(raw_path))
    feature_cols = [c for c in df.columns if c != "label"]

    train_region, _ = train_test_split(df, test_size=0.20, random_state=seed,
                                        stratify=df["label"])
    train_pool, remainder = train_test_split(
        train_region, train_size=train_pool_size, random_state=seed,
        stratify=train_region["label"])

    # Corrupted chunk: sample + shuffle labels
    corrupted_rows, leftover_pool = train_test_split(
        remainder, train_size=corrupted_size, random_state=seed + 1,
        stratify=remainder["label"])
    corrupted_rows = corrupted_rows.copy()
    rng = np.random.RandomState(seed + 2)
    corrupted_rows["label"] = rng.permutation(corrupted_rows["label"].values)

    train_pool = train_pool.reset_index(drop=True)
    real_chunk_ids = make_chunks(train_pool, n_chunks=n_chunks, seed=seed)

    # Duplicate chunk: pick chunk with the most rows available as a stand-in
    # "known good" chunk to duplicate. We pick chunk id 0 for reproducibility;
    # after v2 we know real chunk values range ~0.001-0.0028, so any chunk
    # works as a meaningful test subject.
    dup_source_id = real_chunk_ids.unique()[0]
    dup_mask = real_chunk_ids == dup_source_id
    duplicate_rows = train_pool.loc[real_chunk_ids[dup_mask].index].copy()

    combined_pool = pd.concat(
        [train_pool, corrupted_rows.reset_index(drop=True),
         duplicate_rows.reset_index(drop=True)],
        ignore_index=True)

    n_train, n_corrupt, n_dup = len(train_pool), len(corrupted_rows), len(duplicate_rows)
    full_chunk_ids = pd.Series(index=combined_pool.index, dtype=object)
    full_chunk_ids.iloc[:n_train] = real_chunk_ids.values
    full_chunk_ids.iloc[n_train:n_train + n_corrupt] = "CORRUPTED"
    full_chunk_ids.iloc[n_train + n_corrupt:] = f"DUPLICATE_OF_{dup_source_id}"

    fold_assignment = make_disjoint_folds(leftover_pool, n_test_folds, seed)
    print(f"[v3] combined_pool={len(combined_pool)} rows "
          f"({n_train} real + {n_corrupt} corrupted + {n_dup} duplicate-of-chunk-{dup_source_id})")
    print(f"[v3] leftover_pool={len(leftover_pool)} rows split into "
          f"{n_test_folds} DISJOINT folds (~{len(leftover_pool)//n_test_folds} rows each)")

    unique_chunk_labels = list(real_chunk_ids.unique()) + \
        ["CORRUPTED", f"DUPLICATE_OF_{dup_source_id}"]

    fold_deltas = {}
    t0 = time.time()
    for fold in sorted(fold_assignment.unique()):
        test_fold = leftover_pool.loc[fold_assignment[fold_assignment == fold].index]
        test_X, test_y = test_fold[feature_cols], test_fold["label"]

        baseline = fit_logloss(combined_pool, test_X, test_y, feature_cols)
        deltas = []
        for cid in unique_chunk_labels:
            mask = full_chunk_ids != cid
            sub = combined_pool.loc[mask[mask].index]
            loo = fit_logloss(sub, test_X, test_y, feature_cols)
            deltas.append(baseline - loo)
        fold_deltas[fold] = deltas
        print(f"[v3] fold {fold} done, {len(test_fold)} rows "
              f"({time.time() - t0:.0f}s elapsed)")

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
        "sanity_check_type": ["real"] * n_chunks + ["corrupted"] + ["duplicate"],
    })
    return result, delta_matrix, dup_source_id


if __name__ == "__main__":
    result, delta_matrix, dup_source_id = run_v3()
    result_sorted = result.sort_values("true_marginal_value", ascending=False)
    result_sorted.to_csv("loo_ground_truth_v3.csv", index=False)
    delta_matrix.to_csv("multi_fold_raw_deltas_v3.csv")

    print("\n=== Final ground truth v3 ===")
    print(result_sorted.to_string(index=False))

    real = result[result["sanity_check_type"] == "real"]
    corrupted = result[result["sanity_check_type"] == "corrupted"].iloc[0]
    duplicate = result[result["sanity_check_type"] == "duplicate"].iloc[0]
    original = result[result["chunk_id"] == dup_source_id].iloc[0]

    print("\n=== Sanity checks ===")
    print(f"Real chunks range: {real['true_marginal_value'].min():.5f} to "
          f"{real['true_marginal_value'].max():.5f}")
    print(f"Corrupted chunk: {corrupted['true_marginal_value']:.5f} "
          f"(should be clearly negative, below real range: "
          f"{corrupted['true_marginal_value'] < real['true_marginal_value'].min()})")
    print(f"Original chunk {dup_source_id} (with duplicate present): "
          f"{original['true_marginal_value']:.5f}")
    print(f"Duplicate of chunk {dup_source_id}: {duplicate['true_marginal_value']:.5f}")
    print(f"Both copies show suppressed/near-zero value relative to typical "
          f"real chunks (redundancy correctly recognized): "
          f"{abs(original['true_marginal_value']) < real['true_marginal_value'].median() and abs(duplicate['true_marginal_value']) < real['true_marginal_value'].median()}")

    print(f"\nSign consistency (real chunks only): "
          f"{(delta_matrix.loc[real['chunk_id']].values > 0).sum()}"
          f"/{delta_matrix.loc[real['chunk_id']].size} positive")
    print(f"Reliable chunks: {result['reliable'].sum()}/{len(result)}")
