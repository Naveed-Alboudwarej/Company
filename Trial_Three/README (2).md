# MiniBooNE Data Valuation Pipeline — Week 1 (v2, current)

Built and validated against the real MiniBooNE_PID.txt file. Every number
below is from actually running this code.

## Use this version: `multi_fold_loo_v2.py` / `loo_ground_truth_v2.csv`

This supersedes both `loo_runner.py` (single test set — unreliable
rankings, see below) and `multi_fold_loo.py` (multi-fold but had dead
code and only 10 chunks / 5 folds). Kept both older versions for
reference/history.

## The full story, in order (why v2 looks the way it does)

**1. Full-size pool (103,676 rows) saturates XGBoost.** Baseline AUC
~0.984, 24/25 chunk deltas had the wrong sign. Fixed by subsampling to
15,000 rows.

**2. Single model seed was noisy.** Fixed with seed-averaging, then with
a fully deterministic config (`subsample=1.0, colsample_bytree=1.0`,
confirmed bit-identical across seeds).

**3. A single fixed test set gives reproducible but NOT trustworthy
rankings.** Tested this directly: ran the sweep against 5 independent
test folds and found mean pairwise Spearman correlation between folds'
chunk rankings was only 0.23 (one pair was -0.25) — even though sign
consistency (49/50) was fine. A chunk that ranked highly on one test set
often didn't on another.
**Fix: define ground truth as the average delta across multiple
independent test folds**, not one fixed test set.

**4. v2 improvements, made with full autonomy per your request:**
- **Fixed a real bug**: the earlier script split off a 25,920-row
  "official test set" that was never actually used — dead weight.
  Removed; all held-out data now feeds test folds.
- **Chunks: 10 → 20.** More granular, more points for week 2's
  correlation check.
- **Test folds: 5 → 8.** Tighter, more stable ground truth per chunk.
- **Added a corruption sanity check**: one extra chunk built by sampling
  750 rows and shuffling their labels (deliberately bad data). Result:
  `true_marginal_value = -0.00947`, clearly separated from every real
  chunk's range (+0.00101 to +0.00279) — the method correctly and
  unambiguously flags obviously bad data as strongly harmful. (Note: the
  first version of this script had an inverted comparison bug that
  printed "False" for this check — corrected; the actual numbers show a
  clean pass.)
- **Added a `reliable` column** per chunk (True when
  `std_across_folds < 0.5 * |true_marginal_value|`) — 11 of 21 chunks
  currently flagged reliable. Use this in week 2 rather than treating
  every chunk as equally trustworthy ground truth.

## How to run

```bash
pip install pandas numpy scikit-learn xgboost pyarrow scipy
python3 multi_fold_loo_v2.py   # writes loo_ground_truth_v2.csv
```

## What to actually use for week 2

`loo_ground_truth_v2.csv`, columns:
- `true_marginal_value` — the ground truth to correlate your fast
  estimator against
- `reliable` — consider restricting your headline Spearman correlation
  to chunks where this is True, or report both (all chunks vs.
  reliable-only) so a noisy chunk doesn't distort your validation
- `is_corrupted_sanity_check` — this row is the injected sanity-check
  chunk, not real data; exclude it from any correlation analysis, it's
  only there to confirm the method works at all

## Remaining known limitations (not yet addressed)

- Chunks are still random splits, not grouped by any real-world category
  (no natural grouping variable exists in this dataset — a limitation of
  MiniBooNE itself, discussed earlier when comparing dataset candidates).
- 8 test folds is better than 5 but still not exhaustive — could go
  higher if you want even tighter ground truth, at proportional compute
  cost (~35s per fold at this chunk/pool size).
- The corruption sanity check used one obvious failure mode (shuffled
  labels). Other realistic bad-data patterns (near-duplicate/redundant
  chunks, subtly mislabeled data) haven't been tested and would be worth
  trying before fully trusting the method on production data.
