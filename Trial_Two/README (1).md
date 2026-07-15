# MiniBooNE Week-1 Valuation Pipeline

Built and tested against the actual MiniBooNE_PID.txt file. Everything
below reflects real numbers from running this code, not estimates.

## Files

- `data_prep.py` — loads the raw file, cleans the -999 missing-value
  sentinel, makes the fixed train_pool/test split, and assigns chunks.
- `train_eval.py` — the `train_and_eval(train_df, config, seed)` contract.
  Owns the XGBoost model. Also has `measure_noise_floor()` and
  `train_and_eval_averaged()`.
- `loo_runner.py` — orchestrates the leave-one-chunk-out sweep, calling
  `train_eval.py` once per chunk. This is what produces your ground truth.

## How to run

```bash
pip install pandas numpy scikit-learn xgboost pyarrow
python3 data_prep.py MiniBooNE_PID.txt   # writes train_pool/test/chunk_ids parquet files
python3 loo_runner.py                     # writes loo_ground_truth.csv
```

## Important finding from actually running this

**The full 103,676-row training pool saturates XGBoost** (baseline AUC
~0.984). At that size, 18 of 25 chunk-level LOO deltas were smaller than
noise from random model-seed variance alone, and 24 of 25 were the wrong
sign (removing a chunk appeared to *help*, not hurt). This means the
dataset at full size doesn't have enough real per-chunk signal left for
any valuation method to detect — not a bug, a property of the data at
that scale.

**Fix applied:** the pipeline now subsamples the training pool to 15,000
rows (`TRAIN_POOL_SIZE` in `data_prep.py`) and averages each score over 3
model seeds (`train_and_eval_averaged` in `train_eval.py`, used by
`loo_runner.py`). With both fixes: **6 of 10 chunks now clear the noise
floor, and 8 of 10 show the correct sign.** This is real, usable signal.

## Before you trust any new run

Always compare your LOO deltas against a noise floor measured the SAME
way (same averaging, same config) — `measure_noise_floor()` in
`train_eval.py` does single-seed by default; if you change
`loo_runner.py` to average over seeds, measure the noise floor with the
same averaging or the comparison isn't apples-to-apples (this bit us
once already during testing — see the numbers above).

## Next steps (week 2+)

- This CSV (`loo_ground_truth.csv`) is your ground truth. Build the fast
  estimator (TracIn / gradient-projection) next and correlate its scores
  against this file (Spearman correlation).
- If you want more chunks (finer granularity) or a bigger pool, re-check
  the noise floor every time you change `TRAIN_POOL_SIZE` or `N_CHUNKS` —
  the saturation problem can come back.
- The 468 rows dropped for `-999` values and the 20% test holdout are
  fixed once in `data_prep.py` — don't regenerate them with a different
  seed mid-project or your results won't be comparable across runs.
