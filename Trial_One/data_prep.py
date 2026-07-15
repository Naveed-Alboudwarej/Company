"""
data_prep.py

Loads the raw MiniBooNE_PID.txt file, cleans it, and produces:
  1. A fixed held-out test set (never touched by LOO / chunking)
  2. A chunked "training pool" for the LOO ground-truth sweep

MiniBooNE-specific quirk: missing sensor readings are encoded as -999.0
rather than NaN. We convert those to real NaN and drop the (small number of)
affected rows, since only ~0.36% of rows are affected.
"""

import numpy as np
import pandas as pd

RAW_PATH_DEFAULT = "MiniBooNE_PID.txt"
MISSING_SENTINEL = -999.0
RANDOM_SEED = 42
TEST_FRACTION = 0.20
N_CHUNKS = 10
TRAIN_POOL_SIZE = 15000
# NOTE on TRAIN_POOL_SIZE: empirically, the full ~104k-row training pool
# saturates XGBoost (baseline AUC ~0.984) -- 18/25 chunk LOO deltas came out
# smaller than the noise floor from random seed variance alone, and 24/25
# were the WRONG SIGN (removing data appeared to *help*). Subsampling to
# 15k rows restores real signal (baseline AUC ~0.982, 7/10 deltas clear the
# noise floor, correct sign). Re-check this yourself if you change n_chunks,
# the model config, or the eval metric -- always run measure_noise_floor()
# and compare against your actual LOO deltas before trusting them.


def load_raw(path: str = RAW_PATH_DEFAULT) -> pd.DataFrame:
    """Load the raw MiniBooNE file into a DataFrame with a 'label' column.

    File format: first line is "n_signal n_background". Signal rows come
    first, followed by background rows. label=1 for signal, 0 for background.
    """
    with open(path) as f:
        n_signal, n_background = (int(x) for x in f.readline().split())

    df = pd.read_csv(path, skiprows=1, sep=r"\s+", header=None)
    df.columns = [f"f{i}" for i in range(df.shape[1])]
    df["label"] = np.array([1] * n_signal + [0] * n_background)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the -999 missing-value sentinel to NaN and drop affected rows."""
    feature_cols = [c for c in df.columns if c != "label"]
    df = df.copy()
    df[feature_cols] = df[feature_cols].replace(MISSING_SENTINEL, np.nan)

    before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    dropped = before - len(df)
    print(f"[clean] dropped {dropped} rows containing -999 sentinel values "
          f"({dropped / before * 100:.2f}% of data)")
    return df


def make_fixed_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION,
                      train_pool_size: int = TRAIN_POOL_SIZE,
                      seed: int = RANDOM_SEED):
    """Stratified train_pool / test split. test set is NEVER used for chunking
    or LOO -- it's the fixed yardstick every train_and_eval() call is scored against.

    train_pool is subsampled down to train_pool_size AFTER the split, so the
    test set still reflects the full data distribution. See the
    TRAIN_POOL_SIZE note above for why this subsampling matters -- at full
    size, this dataset saturates XGBoost and chunk-level LOO signal vanishes
    into noise.
    """
    from sklearn.model_selection import train_test_split

    feature_cols = [c for c in df.columns if c != "label"]
    train_pool, test = train_test_split(
        df, test_size=test_fraction, random_state=seed, stratify=df["label"]
    )

    if train_pool_size is not None and train_pool_size < len(train_pool):
        train_pool, _ = train_test_split(
            train_pool, train_size=train_pool_size, random_state=seed,
            stratify=train_pool["label"]
        )

    train_pool = train_pool.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print(f"[split] train_pool: {len(train_pool)} rows | test: {len(test)} rows")
    print(f"[split] train_pool label balance: "
          f"{train_pool['label'].mean() * 100:.1f}% signal")
    print(f"[split] test label balance: {test['label'].mean() * 100:.1f}% signal")
    return train_pool[feature_cols + ["label"]], test[feature_cols + ["label"]]


def make_chunks(train_pool: pd.DataFrame, n_chunks: int = N_CHUNKS,
                 seed: int = RANDOM_SEED) -> pd.Series:
    """Assign each row in train_pool to one of n_chunks, stratified by label
    so every chunk has roughly the same class balance as the whole pool.
    Returns a Series of chunk ids aligned to train_pool's index.
    """
    rng = np.random.RandomState(seed)
    chunk_ids = pd.Series(index=train_pool.index, dtype=int)

    for label_value in train_pool["label"].unique():
        idx = train_pool.index[train_pool["label"] == label_value].to_numpy().copy()
        rng.shuffle(idx)
        # round-robin assignment keeps chunk sizes balanced within a class
        assigned = np.arange(len(idx)) % n_chunks
        chunk_ids.loc[idx] = assigned

    sizes = chunk_ids.value_counts().sort_index()
    print(f"[chunks] {n_chunks} chunks created, sizes range "
          f"{sizes.min()}-{sizes.max()} rows (mean {sizes.mean():.0f})")
    return chunk_ids


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else RAW_PATH_DEFAULT
    df = load_raw(path)
    df = clean(df)
    train_pool, test = make_fixed_split(df)
    chunk_ids = make_chunks(train_pool)

    train_pool.to_parquet("train_pool.parquet")
    test.to_parquet("test.parquet")
    chunk_ids.to_frame("chunk_id").to_parquet("chunk_ids.parquet")
    print("[data_prep] wrote train_pool.parquet, test.parquet, chunk_ids.parquet")
