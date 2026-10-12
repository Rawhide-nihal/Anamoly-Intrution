"""
Builds CICIDS2017_Final_Experiment.ipynb so that it reproduces the EXACT approved
Phase 1 / Phase 2A split and verifies the FROZEN Phase 2A Isolation Forest results.

The notebook never overwrites frozen artifacts. It:
  * rebuilds the split with the same code path as phase2a.py (glob order ->
    drop_duplicates -> np.random.seed(42) permutation, first 70% of BENIGN = TRAIN);
  * proves the TRAIN rows are identical by matching the frozen imputer/scaler
    statistics bit-for-bit;
  * proves the TEST rows (and their order) are identical by matching the frozen
    model's anomaly scores on every test row;
  * refits the 200-tree baseline from scratch and checks it reproduces the frozen
    scores exactly.

Run:  python build_final_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "CICIDS2017_Final_Experiment.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [
md("""
# Anomaly-Based Intrusion Detection System
**Using DBSCAN and Isolation Forest Algorithms — CIC-IDS2017**

## Section 1: Project Overview
The problem is formulated as anomaly detection: models are trained **only on BENIGN traffic**
and evaluated on a mixed test set (unseen BENIGN + all attacks).

| Set | Rows |
|---|---|
| TRAIN (BENIGN only) | 1,467,538 |
| TEST — unseen BENIGN | 628,946 |
| TEST — attacks | 425,878 |
| TEST total | 1,054,824 |

Ground-truth labels are never used for fitting or for selecting hyperparameters.

**Phase 2A (Isolation Forest) is frozen.** This notebook rebuilds the approved split and
preprocessing, then *verifies* the frozen Phase 2A artifacts rather than overwriting them.
The canonical script is `phase2a.py`; artifact hashes are in `results/phase2a_frozen_manifest.json`.
"""),
code("""
import glob
import gc
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
EXPECTED_TRAIN = 1_467_538
EXPECTED_TEST_BENIGN = 628_946
EXPECTED_TEST_ATTACK = 425_878
EXPECTED_TEST = 1_054_824
EXPECTED_ZERO_VAR = [
    "Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]
ATTACK_CATEGORIES = [
    "Heartbleed", "Infiltration", "Bot", "PortScan", "DDoS", "DoS Hulk", "DoS GoldenEye",
    "DoS slowloris", "DoS Slowhttptest", "FTP-Patator", "SSH-Patator",
    "Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection",
]
MODEL_DIR, RESULT_DIR, PLOT_DIR = Path("models"), Path("results"), Path("plots/phase2a")
"""),
md("""
## Section 2: Dataset Loading
File order matters: row positions drive both `drop_duplicates` (keep-first) and the seeded
permutation split, so the files are discovered with the same `glob.glob("*.csv")` call as
`validate_phase1.py` and `phase2a.py`. Run the notebook from the project folder.
"""),
code("""
csv_files = list(glob.glob("*.csv"))
print(f"Discovered {len(csv_files)} CSV files (in this order):")
dfs, expected_cols = [], None
for f in csv_files:
    d = pd.read_csv(f)
    d.columns = d.columns.str.strip()
    if expected_cols is None:
        expected_cols = list(d.columns)
    assert list(d.columns) == expected_cols, f"Schema mismatch in {f}"
    print(f"  {f}: {len(d):,} rows")
    dfs.append(d)
combined_df = pd.concat(dfs, ignore_index=True)
del dfs
gc.collect()
print(f"Total rows: {len(combined_df):,} | columns: {combined_df.shape[1]}")
"""),
md("""
## Sections 3–5: Cleaning and Label Separation
* Exact duplicate rows are removed.
* `±inf` → `NaN` (imputed later with the TRAIN median).
* The raw Web Attack labels contain a mis-encoded dash (U+FFFD); it is normalised to
  `"Web Attack - XSS"` etc. This is applied **after** deduplication and does not affect the
  BENIGN mask or the split.
"""),
code("""
combined_df = combined_df.drop_duplicates()
print(f"Rows after duplicate removal: {len(combined_df):,}")

labels = combined_df["Label"].copy()
labels = labels.str.replace("\\ufffd", "-", regex=False).str.replace(r"\\s+", " ", regex=True).str.strip()
binary_labels = (labels != "BENIGN").astype(int)

numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
combined_df[numeric_cols] = combined_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
X = combined_df.drop(columns=["Label"])
del combined_df
gc.collect()
assert "Label" not in X.columns and "binary_label" not in X.columns
"""),
md("""
## Section 6: Train/Test Construction (approved split)
70% of BENIGN rows (seeded permutation, `np.random.seed(42)`) form TRAIN; the remaining 30% of
BENIGN plus **all** attack rows form TEST. This is the split validated in Phase 1 and used in
Phase 2A. (`sklearn.model_selection.train_test_split` would give the same counts but
**different rows**, so it is deliberately not used.)
"""),
code("""
benign_mask = labels == "BENIGN"
X_benign, labels_benign, y_benign = X[benign_mask], labels[benign_mask], binary_labels[benign_mask]
X_attack, labels_attack, y_attack = X[~benign_mask], labels[~benign_mask], binary_labels[~benign_mask]
del X
gc.collect()

train_size = int(len(X_benign) * 0.7)
np.random.seed(RANDOM_STATE)
indices = np.random.permutation(len(X_benign))
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train = X_benign.iloc[train_idx]
labels_train = labels_benign.iloc[train_idx]
X_test = pd.concat([X_benign.iloc[test_idx], X_attack])
labels_test = pd.concat([labels_benign.iloc[test_idx], labels_attack])
y_test_binary = pd.concat([y_benign.iloc[test_idx], y_attack]).to_numpy().astype(np.int8)
del X_benign, X_attack
gc.collect()

assert len(X_train) == EXPECTED_TRAIN
assert len(X_test) == EXPECTED_TEST
assert int((y_test_binary == 0).sum()) == EXPECTED_TEST_BENIGN
assert int((y_test_binary == 1).sum()) == EXPECTED_TEST_ATTACK
assert (labels_train == "BENIGN").all()
assert set(labels_test.unique()) == {"BENIGN", *ATTACK_CATEGORIES}
assert len(set(X_train.index) & set(X_test.index)) == 0

print(f"Training rows: {len(X_train):,} (BENIGN only)")
print(f"Test rows:     {len(X_test):,}")
print(labels_test.value_counts().to_string())
"""),
md("""
## Section 7: Feature Investigation — Destination Port (descriptive only)
Destination Port is **retained** in Phase 2A. This cell only describes it; the ablation is a
later phase.
"""),
code("""
port_col = "Destination Port"
print(f"Unique destination ports (test set): {X_test[port_col].nunique():,}")
top = pd.DataFrame({"Port": X_test[port_col], "Label": labels_test})
for cat in ATTACK_CATEGORIES:
    vc = top.loc[top["Label"] == cat, "Port"].value_counts().head(3)
    print(f"{cat:<28} top ports: {{{', '.join(f'{int(p)}: {int(n):,}' for p, n in vc.items())}}}")
"""),
md("""
## Section 8: Preprocessing (fit on TRAIN only)
Zero-variance rule: drop columns with `X_train.var() == 0` on the **full** approved TRAIN set →
**8** columns, 70 features remain.

*Why not 10?* The earlier EDA notebook found 10 constant columns on a 40,000-row sample (first
5,000 rows per file). `Fwd URG Flags` and `CWE Flag Count` are constant in that sample and in 7
of 8 files, but each has 56 non-zero TRAIN rows and 24 non-zero TEST rows (all BENIGN, all from
the Thursday-Afternoon file), and the two columns are identical. Full details:
`results/zero_variance_reconciliation.txt` (from `reconcile_zero_variance.py`).

The cell then checks that the imputer and scaler fitted here are **bit-identical** to the frozen
Phase 2A artifacts — which can only happen if TRAIN contains exactly the same rows.
"""),
code("""
variances = X_train.var(numeric_only=True)
zero_var_cols = variances[variances == 0].index.tolist()
assert zero_var_cols == EXPECTED_ZERO_VAR, zero_var_cols
X_train = X_train.drop(columns=zero_var_cols)
X_test = X_test.drop(columns=zero_var_cols)
feature_columns = list(X_train.columns)
print(f"Removed {len(zero_var_cols)} zero-variance features -> {len(feature_columns)} features")

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(imputer.fit_transform(X_train)).astype(np.float32)
X_test_s = scaler.transform(imputer.transform(X_test)).astype(np.float32)
del X_train, X_test
gc.collect()

assert not np.isnan(X_train_s).any() and not np.isnan(X_test_s).any()
assert np.isfinite(X_train_s).all() and np.isfinite(X_test_s).all()

frozen_imputer = joblib.load(MODEL_DIR / "imputer.pkl")
frozen_scaler = joblib.load(MODEL_DIR / "scaler.pkl")
frozen_meta = joblib.load(MODEL_DIR / "preprocessing_meta.pkl")
assert feature_columns == frozen_meta["feature_columns"]
assert np.array_equal(imputer.statistics_, frozen_imputer.statistics_)
assert np.array_equal(scaler.mean_, frozen_scaler.mean_)
assert np.array_equal(scaler.scale_, frozen_scaler.scale_)
print("TRAIN preprocessing is bit-identical to the frozen Phase 2A imputer/scaler.")
"""),
md("""
## Section 9: Isolation Forest — Phase 2A (FROZEN)
Primary, pre-specified configuration: `n_estimators=200, max_samples="auto",
contamination="auto", random_state=42`.

* Score convention: `anomaly_score = -decision_function(X)` → **higher = more anomalous**
  (threshold 0 for `contamination="auto"`).
* Prediction convention: sklearn `+1/-1` → project `0 = normal, 1 = anomaly`.
* The n_estimators = 100/300 and contamination results in the report are **sensitivity
  analyses only** and never changed the primary model.

### 9.1 Verify the frozen artifacts
"""),
code("""
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = json.load(open(RESULT_DIR / "phase2a_frozen_manifest.json"))
assert manifest["status"] == "FROZEN"
for rel, digest in manifest["sha256"].items():
    assert sha256(rel) == digest, f"Frozen artifact modified: {rel}"
print(f"All {len(manifest['sha256'])} frozen artifacts match their SHA-256 hashes "
      f"(frozen at {manifest['frozen_at']}).")
"""),
md("""
### 9.2 Score the test set with the frozen model
Matching the frozen per-row scores on all 1,054,824 test rows confirms the TEST rows **and their
order** are exactly those of Phase 2A.
"""),
code("""
model = joblib.load(MODEL_DIR / "isolation_forest.pkl")
assert model.n_estimators == 200 and len(model.estimators_) == 200
assert model.contamination == "auto" and model.random_state == 42

decision = model.decision_function(X_test_s)
anomaly_score = -decision                                   # higher = more anomalous
y_pred_if = (model.predict(X_test_s) == -1).astype(np.int8)  # 0 normal, 1 anomaly

frozen = np.load(RESULT_DIR / "if_test_outputs.npz")
assert np.array_equal(frozen["y_test_binary"], y_test_binary)
assert np.array_equal(frozen["anomaly_score"], anomaly_score.astype(np.float32))
assert np.array_equal(frozen["y_pred_if"], y_pred_if)
print("Frozen model reproduces the Phase 2A scores and predictions on every test row.")
"""),
md("""
### 9.3 Independent refit check
Refitting the 200-tree baseline from scratch on the rebuilt TRAIN matrix must give exactly the
same scores (takes a few seconds; nothing is saved).
"""),
code("""
refit = IsolationForest(n_estimators=200, max_samples="auto", contamination="auto",
                        random_state=RANDOM_STATE, n_jobs=-1).fit(X_train_s)
assert np.array_equal(refit.decision_function(X_test_s), decision)
del refit
print("Refit from scratch reproduces the frozen baseline exactly.")
"""),
md("""
### 9.4 Baseline evaluation (labels used only here, after prediction)
"""),
code("""
tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_if, labels=[0, 1]).ravel()
metrics = {
    "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    "precision": precision_score(y_test_binary, y_pred_if, zero_division=0),
    "recall": recall_score(y_test_binary, y_pred_if, zero_division=0),
    "f1": f1_score(y_test_binary, y_pred_if, zero_division=0),
    "fpr": fp / (fp + tn), "fnr": fn / (fn + tp),
    "roc_auc": roc_auc_score(y_test_binary, anomaly_score),
}
for k, v in manifest["headline"].items():
    assert np.isclose(metrics[k], v), (k, metrics[k], v)

print(f"Accuracy: {accuracy_score(y_test_binary, y_pred_if):.4f}")
for k, v in metrics.items():
    print(f"{k:>9}: {v:,}" if isinstance(v, int) else f"{k:>9}: {v:.4f}")
print()
print(classification_report(y_test_binary, y_pred_if, labels=[0, 1],
                            target_names=["Normal (0)", "Anomaly (1)"], digits=4, zero_division=0))
"""),
md("""
### 9.5 Attack-category flag rates
Share of each category's test traffic **flagged as anomalous** — not an attack classification.
Categories with n < 100 (Heartbleed, Infiltration, Web Attack - Sql Injection) have very wide
uncertainty.
"""),
code("""
lt = labels_test.to_numpy()
rows = []
for cat in ATTACK_CATEGORIES:
    m = lt == cat
    rows.append({"category": cat, "total": int(m.sum()), "flagged": int(y_pred_if[m].sum()),
                 "missed": int(m.sum() - y_pred_if[m].sum()),
                 "flag_rate": y_pred_if[m].mean(), "small_n": m.sum() < 100})
cat_table = pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)
print(cat_table.to_string(formatters={"flag_rate": "{:.2%}".format}))
print(f"\\nReference: unseen BENIGN flagged {y_pred_if[lt == 'BENIGN'].mean():.2%}")
"""),
md("""
### 9.6 Frozen report and figures
"""),
code("""
print((RESULT_DIR / "phase2a_report.txt").read_text(encoding="utf-8"))
try:
    from IPython.display import Image, display
    for png in sorted(PLOT_DIR.glob("*.png")):
        display(Image(filename=str(png)))
except ImportError:
    pass
"""),
md("""
## Next phases (not started)
DBSCAN, PCA, the Destination Port ablation and the final comparison are pending review of
Phase 2A.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(notebook["cells"]):
    c["id"] = f"cell-{i:02d}"
NB_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB_PATH.name} ({len(cells)} cells)")
