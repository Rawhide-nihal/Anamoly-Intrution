"""
Reconcile the zero-variance feature count: 10 (EDA notebook / per-file audit)
vs 8 (Phase 1 / Phase 2A, computed on the full approved TRAIN set).

Reproduces the approved split exactly (same code path as phase2a.py) and reports,
for every column that is constant in ANY of the candidate scopes, its non-zero
counts in each scope. Writes results/zero_variance_reconciliation.txt.
"""
import gc
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
(ROOT / "results").mkdir(exist_ok=True)
out = []


def log(s=""):
    print(s, flush=True)
    out.append(s)


def constant_cols(df):
    """Columns with a single distinct value (NaN ignored) — same test as the EDA notebook."""
    num = df.select_dtypes(include=np.number)
    return set(num.columns[num.nunique() <= 1])


csv_files = list(glob.glob("*.csv"))
dfs, eda_sample = [], []
for f in csv_files:
    d = pd.read_csv(f)
    d.columns = d.columns.str.strip()
    dfs.append(d)
    eda_sample.append(d.head(5000))           # EDA notebook: nrows=5000 per file
raw = pd.concat(dfs, ignore_index=True)
eda = pd.concat(eda_sample, ignore_index=True)
per_file = {f: constant_cols(d) for f, d in zip(csv_files, dfs)}
del dfs, eda_sample
gc.collect()

dedup = raw.drop_duplicates()
labels = dedup["Label"]
num_cols = dedup.select_dtypes(include=np.number).columns
X = dedup[num_cols].replace([np.inf, -np.inf], np.nan)
benign = labels == "BENIGN"
Xb = X[benign]
train_size = int(len(Xb) * 0.7)
np.random.seed(42)
idx = np.random.permutation(len(Xb))
X_train = Xb.iloc[idx[:train_size]]
X_test = pd.concat([Xb.iloc[idx[train_size:]], X[~benign]])
labels_test = pd.concat([labels[benign].iloc[idx[train_size:]], labels[~benign]])
assert len(X_train) == 1_467_538 and len(X_test) == 1_054_824

# The exact rule used by Phase 1 / Phase 2A
var = X_train.var(numeric_only=True)
phase_rule = set(var[var == 0].index)

scopes = {
    "EDA sample (5,000 rows x 8 files = 40,000)": eda,
    "Full raw data (2,830,743 rows)": raw,
    "Deduplicated data": dedup,
    "Approved TRAIN (1,467,538 BENIGN)": X_train,
    "TEST (1,054,824)": X_test,
}
const = {k: constant_cols(v) for k, v in scopes.items()}

log("ZERO-VARIANCE RECONCILIATION")
log("=" * 78)
for k, s in const.items():
    log(f"{k:<46} constant columns: {len(s)}")
log(f"{'Phase 1/2A rule: X_train.var() == 0':<46} constant columns: {len(phase_rule)}")
log("\nPer-file constant counts (audit.py / audit_results.json):")
for f, s in per_file.items():
    log(f"  {f:<62}{len(s)}")

union = set().union(*const.values(), *per_file.values(), phase_rule)
always = set.intersection(*const.values())
disputed = sorted(union - always)
log(f"\nConstant in EVERY scope ({len(always)}): {sorted(always)}")
log(f"Disputed columns: {disputed}")

for c in disputed:
    log(f"\n--- {c} ---")
    for k, v in scopes.items():
        col = v[c]
        log(f"  {k:<46} non-zero rows: {int((col.fillna(0) != 0).sum()):>6,}   "
            f"distinct values: {sorted(col.dropna().unique().tolist())[:5]}")
    for f, d in per_file.items():
        log(f"  constant in {f[:45]:<45}: {c in d}")
    nz_test = labels_test[(X_test[c].fillna(0) != 0)].value_counts()
    log(f"  TEST non-zero rows by Label: {nz_test.to_dict()}")
    log(f"  TRAIN variance: {var[c]:.3e}")

if set(disputed) >= {"Fwd URG Flags", "CWE Flag Count"}:
    same = bool((dedup["Fwd URG Flags"] == dedup["CWE Flag Count"]).all())
    log(f"\nFwd URG Flags identical to CWE Flag Count on every deduplicated row: {same}")

(ROOT / "results" / "zero_variance_reconciliation.txt").write_text("\n".join(out), encoding="utf-8")
