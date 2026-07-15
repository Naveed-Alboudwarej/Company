# Company (Data Valuation — Project Overview)

## The idea

We're building a way to estimate the **value of a chunk of training data** to a
model, given a dataset made up of many "chunks" (batches, sources,
time periods, sellers, etc.), can we quantify how much each chunk actually
contributes to model performance?

The bigger picture behind this: a "data economy" marketplace with automatic,
fair-value pricing for training data. Instead of AI companies scraping data
for free or negotiating one-off licensing deals, imagine infrastructure that
ingests a dataset, estimates its marginal contribution to a model's
performance, and auto-prices it so data owners get paid based on actual
usefulness, not just volume. As courts and legislators are
increasingly pushing AI companies to pay for training data (see NYT v.
OpenAI, Getty v. Stability, EU AI Act provisions), and there's currently no
principled way to price a dataset by what it's actually worth to a model.

This matters for a few reasons:
- **Data marketplaces**: a fair pricing mechanism based on marginal value,
  not just size.
- **Data curation**: knowing which chunks of your training data are helping,
  hurting, or redundant lets you clean, prioritize, or drop data
  intelligently.
- **Attribution**: if a model's performance depends heavily on a few
  sources, that's useful (and sometimes legally necessary) to be able to
  prove.
- **AI-generated data detection**: web scraped data increasingly contains
  AI generated text, which can degrade model quality and cause
  hallucinations. A working data valuation method should be able to flag
  this kind of low diversity/synthetic content as low value, which is both
  a curation tool on its own and a stress test of whether our valuation
  method actually works.

The "gold standard" way to measure a chunk's value is **Leave-One-Out (LOO)
valuation**: train the model on everything, then retrain with one chunk
removed, and see how much performance drops. The size of that drop is the
chunk's value. 

**Our end goal** is to design and validate a *fast estimator* of chunk value
(e.g. TracIn, influence functions, Data Shapley) that gets close to the LOO
ground truth without requiring a full retrain per chunk. But you can't
validate a fast estimator without first having a trustworthy LOO ground
truth to compare it against. This phase of the project is entirely about
building that ground-truth harness correctly, not about the fast estimator
itself.


## The plan (current phase: LOO ground-truth harness)

### Model: XGBoost
We're using XGBoost on structured/tabular data rather than starting with an
LLM or deep learning approach. It's fast, well-understood, and behaves
predictably — we want the *model* to be the boring, solved part of this
phase while we focus on getting the *valuation harness* right. A
smaller/faster model (or a real transformer) can come in later once this is
proven, especially if we extend the AI-generated-text angle to real text
data.

### Dataset (current phase): MiniBooNE Particle Identification
We chose this over datasets we're more familiar with (e.g. Medicare claims,
Weibo social data) for a specific reason: those datasets have **correlated,
non-independent rows** (patients/providers across years, social network
structure), which makes it hard to tell whether a weird LOO result is a real
data effect or a bug in our code. MiniBooNE is:
- ~130k rows, 50 numeric features, binary classification label
  (signal vs. background particle event)
- Fully numeric, no missing/messy schema decisions required
- Rows are effectively independent detector events — no entity overlap, no
  temporal drift across years
- Fast to fit (seconds per XGBoost run at this size)

This gives us close to the cleanest possible ground truth to build against.
**Caveat**: the 50 feature columns are anonymized (no public column-name
dictionary), so we can't currently explain results in physics terms. That's
fine for building/validating the harness itself, but it's a real limitation
if we ever want to explain *why* a chunk was valuable in domain language.

We plan to move to a more realistic/interpretable dataset (e.g. Medicare, or
a real text dataset for the AI-generated-content angle) once the harness is
proven correct on this cleaner case.

### Chunking
Chunks are the unit of valuation — LOO removes one chunk at a time and
retrains. For this phase:
- **Random, stratified by label** (not grouped by any real-world category
  yet) — this keeps chunks balanced and avoids confounding "chunk size" or
  "chunk composition" with the thing we're trying to measure.
- **~20-30 chunks total.** Fewer than ~15-20 and our validation signal
  (correlation between true and estimated value) is too thin to trust; more
  than ~50-100 and retraining cost balloons without adding much statistical
  power.
- Each chunk is sized so that removing it produces a real, detectable change
  in the metric — roughly 1-5% of the training set per chunk, not so small
  that its removal is noise. (Note: per-example chunking on a small dataset
  showed this directly in an earlier test — removing one example moved
  accuracy by ~0.0000, which is why per-chunk, not per-row, granularity
  matters.)

### Metric
A single, fixed, boring metric: **AUC** on a held-out test set. The test set
is carved out once, up front, stratified by label, and never touched again
until final reporting — no tuning against it, no re-splitting mid-week.

### The shared interface
Everything is built around one function signature, so work can be split
without constant re-syncing:

```python
def train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict, seed: int = 42) -> float:
    """Fits XGBoost on train_df (fixed feature set, fixed target column),
    returns AUC on the fixed held-out test_df."""
```

Anyone building on this project should treat this signature as fixed. If it
needs to change, that's a conversation, not a silent edit.

### The sanity check we build in from day 1
Before trusting any LOO result, we inject a chunk of **pure noise** (real
rows, shuffled labels) into the training set and confirm the harness
correctly flags it as low/zero/negative value. If it doesn't, nothing
downstream can be trusted yet this is our canary, not an afterthought.
This mirrors the corruption-injection test already validated on the toy
TracIn experiment above, just applied to the full-retrain LOO ground truth
instead of the fast estimator.

## What "done" looks like for this phase

- A working pipeline that: loads the data, creates a fixed train/test split,
  assigns chunks, fits a full-data baseline, fits N leave-one-chunk-out
  models, and outputs a table of (chunk_id, chunk_size, delta) per chunk.
- The negative control passes (noise chunk shows no benefit).
- Total runtime for the full sweep is minutes, not hours.
- A short write-up of what the LOO deltas look like across chunks — this
  becomes the ground truth we'll compare our fast estimator (TracIn, or
  similar) against.

## What we're explicitly *not* doing yet
- No marketplace, pricing API, or payment infrastructure, That's a later
  phase, only after the valuation method is validated end to end.
- No hyperparameter tuning. XGBoost params are fixed so every run is
  comparable.
- No feature engineering. Using the dataset's existing numeric features
  as-is.
- No real world grouping for chunks yet (that comes with the richer
  dataset phase).

## Reference points we still need to fix (open question)
Since a data point's value is relative, not absolute, we'll eventually need
to fix reference points to make scores meaningful and comparable across
datasets/sellers: a reference model architecture/size, a reference eval
suite representing the target task, and a reference training recipe.
Without fixing these, two data points can't be compared on a shared "value"
scale, which breaks the pricing story long-term. Worth a deliberate
conversation before the marketplace phase, not something to leave implicit.
