"""
PHASE 2A — ISOLATION FOREST (anomaly-detection formulation)  — FROZEN after first
successful run (see results/phase2a_frozen_manifest.json).

Primary, pre-specified configuration: n_estimators=200, max_samples="auto",
contamination="auto", random_state=42. The n_estimators (100/300) and contamination
experiments are sensitivity analyses only and never change the primary model.

TRAIN : BENIGN only (1,467,538 rows)
TEST  : 628,946 unseen BENIGN + 425,878 attacks = 1,054,824 rows

Ground-truth labels (Label / binary_label) are NEVER passed to any fit() call and
are NEVER used to select a hyperparameter. They are used only after prediction,
for evaluation tables and plots.

Score convention used throughout this script
--------------------------------------------
sklearn IsolationForest.decision_function(X): higher = more normal, < 0 => outlier.
We define
        anomaly_score = -decision_function(X)
so that HIGHER = MORE ANOMALOUS. ROC-AUC and score plots use anomaly_score.
The default ("auto") decision threshold therefore sits at anomaly_score = 0.

Prediction convention: sklearn predict() returns +1 normal / -1 anomaly.
Project convention:     0 = normal, 1 = anomaly  (y_pred_if).
"""
import ctypes
import hashlib
import gc
import glob
import json
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "plots" / "phase2a"
RESULT_DIR = ROOT / "results"
for d in (MODEL_DIR, PLOT_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Approved Phase 1 counts — the script refuses to continue if the split differs.
EXPECTED_TRAIN = 1_467_538
EXPECTED_TEST_BENIGN = 628_946
EXPECTED_TEST_ATTACK = 425_878
EXPECTED_TEST = 1_054_824

ATTACK_CATEGORIES = [
    "Heartbleed", "Infiltration", "Bot", "PortScan", "DDoS", "DoS Hulk",
    "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "FTP-Patator",
    "SSH-Patator", "Web Attack - Brute Force", "Web Attack - XSS",
    "Web Attack - Sql Injection",
]
SMALL_SAMPLE_N = 100

# Approved Phase 1 rule: drop columns with X_train.var() == 0 on the full TRAIN set.
EXPECTED_ZERO_VAR = [
    "Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]
ZERO_VAR_NOTE = (
    "The earlier EDA notebook reported 10 constant columns because it inspected a "
    "40,000-row sample (first 5,000 rows per file). 'Fwd URG Flags' and 'CWE Flag Count' "
    "are constant in that sample and in 7 of 8 files, but not in the full data: each has "
    "56 non-zero rows in TRAIN and 24 in TEST (all BENIGN, all from the Thursday-Afternoon "
    "file), and the two columns are identical. They are therefore kept. "
    "See results/zero_variance_reconciliation.txt."
)

FREEZE_MANIFEST = ROOT / "results" / "phase2a_frozen_manifest.json"
if FREEZE_MANIFEST.exists() and os.environ.get("PHASE2A_UNFREEZE") != "1":
    sys.exit(f"Phase 2A is FROZEN ({FREEZE_MANIFEST.name} exists). Refusing to overwrite "
             "frozen artifacts. Set PHASE2A_UNFREEZE=1 only with explicit approval.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()  # categories below this get an explicit statistical caveat

# Reference categorical palette (slot 1 = BENIGN, slot 2 = ATTACK); text in ink tokens.
C_BENIGN, C_ATTACK = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def log(msg=""):
    print(msg, flush=True)


def peak_rss_gb():
    """Peak working set of this process (Windows), in GB."""
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    try:
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 1e9
    except Exception:
        return float("nan")


def to_project_convention(sk_pred):
    """sklearn (+1 normal, -1 anomaly) -> project (0 normal, 1 anomaly)."""
    assert set(np.unique(sk_pred)).issubset({-1, 1})
    return (sk_pred == -1).astype(np.int8)


def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "anomaly_pct": 100.0 * float(np.mean(y_pred)),
    }


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


T0 = time.time()

# =====================================================================
# 0. REPRODUCE THE APPROVED PHASE 1 PIPELINE (identical to validate_phase1.py)
# =====================================================================
log("=" * 70)
log("0. DATA LOADING + APPROVED PHASE 1 PREPROCESSING")
log("=" * 70)
# File order must match validate_phase1.py exactly (same glob call): row positions
# drive both drop_duplicates (keep-first) and the seeded permutation split.
os.chdir(ROOT)
csv_files = list(glob.glob("*.csv"))
log(f"CSV order: {csv_files}")
dfs = []
for f in csv_files:
    d = pd.read_csv(f)
    d.columns = d.columns.str.strip()
    dfs.append(d)
combined_df = pd.concat(dfs, ignore_index=True)
del dfs
gc.collect()
combined_df = combined_df.drop_duplicates()
log(f"Rows after duplicate removal: {len(combined_df):,}")

labels = combined_df["Label"].copy()
# The raw CSVs store the Web Attack dash as a Windows-1252 byte that decodes to
# U+FFFD ("Web Attack � XSS"). Normalise to "Web Attack - XSS" for reporting.
# This happens after deduplication and does not touch the BENIGN mask or split.
labels = labels.str.replace("�", "-", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
binary_labels = (labels != "BENIGN").astype(int)

numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
combined_df[numeric_cols] = combined_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
X = combined_df.drop(columns=["Label"])
del combined_df
gc.collect()
assert "Label" not in X.columns and "binary_label" not in X.columns
assert all(np.issubdtype(t, np.number) for t in X.dtypes), "Non-numeric feature present"

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

# --- split sanity checks ---
assert len(X_train) == EXPECTED_TRAIN, len(X_train)
assert len(X_test) == EXPECTED_TEST, len(X_test)
assert int((y_test_binary == 0).sum()) == EXPECTED_TEST_BENIGN
assert int((y_test_binary == 1).sum()) == EXPECTED_TEST_ATTACK
assert (labels_train == "BENIGN").all(), "Training contains non-BENIGN rows"
assert set(labels_test.unique()) == {"BENIGN", *ATTACK_CATEGORIES}, set(labels_test.unique())
assert len(set(X_train.index) & set(X_test.index)) == 0, "Train/test overlap"
assert np.array_equal(y_test_binary, (labels_test != "BENIGN").to_numpy().astype(np.int8))
log(f"Train: {len(X_train):,} (BENIGN only) | Test: {len(X_test):,} "
    f"({EXPECTED_TEST_BENIGN:,} BENIGN + {EXPECTED_TEST_ATTACK:,} attack)  -> matches approved split")

# --- preprocessing: fitted ONCE on TRAIN; test is only transformed ---
raw_feature_count = X_train.shape[1]
variances = X_train.var(numeric_only=True)
zero_var_cols = variances[variances == 0].index.tolist()
X_train = X_train.drop(columns=zero_var_cols)
X_test = X_test.drop(columns=zero_var_cols)
feature_columns = list(X_train.columns)
assert zero_var_cols == EXPECTED_ZERO_VAR, f"zero-variance set changed: {zero_var_cols}"

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
# IsolationForest internally casts to float32, so storing float32 is lossless for
# the model and halves RAM.
X_train_s = scaler.fit_transform(imputer.fit_transform(X_train)).astype(np.float32)
X_test_s = scaler.transform(imputer.transform(X_test)).astype(np.float32)
del X_train, X_test
gc.collect()

assert "Label" not in feature_columns and "binary_label" not in feature_columns
assert X_train_s.shape[1] == X_test_s.shape[1] == len(feature_columns)
for name, arr in (("X_train", X_train_s), ("X_test", X_test_s)):
    assert not np.isnan(arr).any(), f"NaN in {name}"
    assert np.isfinite(arr).all(), f"Inf in {name}"
n_features = X_train_s.shape[1]
log(f"Features: {raw_feature_count} raw -> removed {len(zero_var_cols)} zero-variance "
    f"{zero_var_cols} -> {n_features} used (Destination Port retained)")
log(f"X_train_s {X_train_s.shape} {X_train_s.nbytes/1e9:.2f} GB | "
    f"X_test_s {X_test_s.shape} {X_test_s.nbytes/1e9:.2f} GB (float32)")

# =====================================================================
# 1-3. BASELINE ISOLATION FOREST, SCORES, PREDICTIONS
# =====================================================================
log("\n" + "=" * 70)
log("1. BASELINE ISOLATION FOREST")
log("=" * 70)
BASELINE = dict(n_estimators=200, max_samples="auto", contamination="auto",
                random_state=RANDOM_STATE, n_jobs=-1)
log(f"Parameters: {BASELINE}")
model = IsolationForest(**BASELINE)
t = time.time()
model.fit(X_train_s)                      # <- features only; no labels exist here
train_time = time.time() - t
log(f"Training time: {train_time:.2f} s | trees: {len(model.estimators_)} | "
    f"max_samples_ (effective): {model.max_samples_} | offset_: {model.offset_}")

t = time.time()
decision = model.decision_function(X_test_s)
score_time = time.time() - t
anomaly_score = -decision                 # HIGHER = MORE ANOMALOUS

t = time.time()
raw_pred = model.predict(X_test_s)
predict_time = time.time() - t
y_pred_if = to_project_convention(raw_pred)
log(f"Scoring time (decision_function): {score_time:.2f} s | predict() time: {predict_time:.2f} s")

# Consistency: predict() must equal thresholding the anomaly score at 0.
assert np.array_equal(y_pred_if, (anomaly_score > 0).astype(np.int8)), "predict/score mismatch"

# =====================================================================
# 4. BASELINE EVALUATION (labels used only from here on)
# =====================================================================
log("\n" + "=" * 70)
log("4. BASELINE EVALUATION")
log("=" * 70)
base = binary_metrics(y_test_binary, y_pred_if)
roc_auc = roc_auc_score(y_test_binary, anomaly_score)
roc_auc_reversed = roc_auc_score(y_test_binary, decision)
base["roc_auc"] = roc_auc
# Direction check: scoring with the un-negated decision_function must give 1-AUC.
assert abs(roc_auc + roc_auc_reversed - 1) < 1e-9
log(f"Mean anomaly_score  BENIGN={anomaly_score[y_test_binary == 0].mean():+.4f}  "
    f"ATTACK={anomaly_score[y_test_binary == 1].mean():+.4f}")
log(f"ROC-AUC (anomaly_score = -decision_function): {roc_auc:.4f}  "
    f"[reversed direction would give {roc_auc_reversed:.4f}]")
log(classification_report(y_test_binary, y_pred_if, labels=[0, 1],
                          target_names=["Normal (0)", "Anomaly (1)"], digits=4, zero_division=0))
for k in ("TN", "FP", "FN", "TP"):
    log(f"{k}: {base[k]:,}")
for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr"):
    log(f"{k}: {base[k]:.4f}")

# =====================================================================
# 6. ATTACK CATEGORY ANALYSIS (post-prediction only)
# =====================================================================
labels_test_arr = labels_test.to_numpy()


def category_table(y_pred):
    rows = []
    for cat in ATTACK_CATEGORIES:
        m = labels_test_arr == cat
        n = int(m.sum())
        det = int(y_pred[m].sum())
        lo, hi = wilson_ci(det, n)
        rows.append({"category": cat, "total": n, "flagged_anomalous": det, "missed": n - det,
                     "detection_rate": det / n if n else float("nan"),
                     "ci95_low": lo, "ci95_high": hi,
                     "small_sample": n < SMALL_SAMPLE_N})
    out = pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)
    assert out["total"].sum() == EXPECTED_TEST_ATTACK
    return out


cat_df = category_table(y_pred_if)
benign_test_flag_rate = float(y_pred_if[labels_test_arr == "BENIGN"].mean())

# =====================================================================
# 5. VISUALIZATIONS
# =====================================================================
cm = np.array([[base["TN"], base["FP"]], [base["FN"], base["TP"]]])
fig, ax = plt.subplots(figsize=(5.2, 4.4))
row_pct = cm / cm.sum(axis=1, keepdims=True)
ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
ax.grid(False)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}\n({row_pct[i, j]:.1%} of row)", ha="center", va="center",
                color="white" if row_pct[i, j] > 0.55 else INK, fontsize=10)
ax.set_xticks([0, 1], ["Predicted normal (0)", "Predicted anomaly (1)"])
ax.set_yticks([0, 1], ["BENIGN (0)", "ATTACK (1)"])
ax.set_title("Isolation Forest baseline — confusion matrix (test set)", fontsize=11)
fig.savefig(PLOT_DIR / "01_confusion_matrix.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.4))
bins = np.linspace(np.percentile(anomaly_score, 0.05), np.percentile(anomaly_score, 99.95), 120)
for cls, name, col in ((0, "BENIGN test traffic", C_BENIGN), (1, "ATTACK test traffic", C_ATTACK)):
    s = anomaly_score[y_test_binary == cls]
    ax.hist(s, bins=bins, density=True, histtype="stepfilled", alpha=0.25, color=col)
    ax.hist(s, bins=bins, density=True, histtype="step", linewidth=2, color=col,
            label=f"{name} (n={len(s):,})")
ax.axvline(0, color=INK, linestyle="--", linewidth=1.2)
ax.text(0, ax.get_ylim()[1] * 0.95, "  threshold (contamination='auto')", color=INK_2, va="top")
ax.set_xlabel("anomaly_score = −decision_function  (higher = more anomalous)")
ax.set_ylabel("Density")
ax.set_title("Distribution of Isolation Forest anomaly scores (test set)", fontsize=11)
ax.legend(frameon=False)
fig.savefig(PLOT_DIR / "02_anomaly_score_distribution.png")
plt.close(fig)

fpr_c, tpr_c, _ = roc_curve(y_test_binary, anomaly_score)
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.plot(fpr_c, tpr_c, color=C_BENIGN, linewidth=2, label=f"Isolation Forest (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color=INK_2, linestyle=":", linewidth=1, label="Chance (AUC = 0.5)")
ax.scatter([base["fpr"]], [base["recall"]], s=60, color=C_ATTACK, zorder=3,
           edgecolor="#fcfcfb", linewidth=2, label="Operating point ('auto')")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate (recall)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.01)
ax.set_title("ROC curve — Isolation Forest baseline", fontsize=11)
ax.legend(frameon=False, loc="lower right")
fig.savefig(PLOT_DIR / "03_roc_curve.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
counts = [int((y_pred_if == 0).sum()), int((y_pred_if == 1).sum())]
bars = axes[0].bar(["Predicted normal (0)", "Predicted anomaly (1)"], counts,
                   color=[C_BENIGN, C_ATTACK], width=0.55)
for b, c in zip(bars, counts):
    axes[0].text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({c/len(y_pred_if):.1%})",
                 ha="center", va="bottom", color=INK)
axes[0].set_ylim(0, max(counts) * 1.2)
axes[0].set_ylabel("Test rows")
axes[0].set_title("Prediction distribution (all test rows)", fontsize=11)
x = np.arange(2)
w = 0.38
for k, (cls, name, col) in enumerate(((0, "True BENIGN", C_BENIGN), (1, "True ATTACK", C_ATTACK))):
    vals = [int(((y_test_binary == cls) & (y_pred_if == p)).sum()) for p in (0, 1)]
    bb = axes[1].bar(x + (k - 0.5) * w, vals, w - 0.02, color=col, label=name)
    for b, v in zip(bb, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom",
                     fontsize=8, color=INK)
axes[1].set_xticks(x, ["Predicted normal (0)", "Predicted anomaly (1)"])
axes[1].set_title("Predictions split by ground truth (evaluation only)", fontsize=11)
axes[1].legend(frameon=False)
fig.savefig(PLOT_DIR / "04_prediction_distribution.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5.6))
plot_df = cat_df.sort_values("detection_rate")
yy = np.arange(len(plot_df))
ax.barh(yy, plot_df["detection_rate"], color=C_ATTACK, height=0.6)
ax.errorbar(plot_df["detection_rate"], yy,
            xerr=[plot_df["detection_rate"] - plot_df["ci95_low"],
                  plot_df["ci95_high"] - plot_df["detection_rate"]],
            fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
ax.set_yticks(yy, [f"{c}  (n={n:,}){' *' if s else ''}" for c, n, s in
                   zip(plot_df["category"], plot_df["total"], plot_df["small_sample"])])
ax.axvline(benign_test_flag_rate, color=C_BENIGN, linestyle="--", linewidth=1.2)
ax.text(benign_test_flag_rate, -0.9, f" BENIGN flagged: {benign_test_flag_rate:.1%}",
        color=INK_2, fontsize=8)
ax.set_xlim(0, 1.0)
ax.set_xlabel("Share of category's test traffic flagged as anomalous (95% Wilson CI)")
ax.set_title(f"Per-category anomaly flag rate  (* n < {SMALL_SAMPLE_N}: wide uncertainty)", fontsize=11)
fig.savefig(PLOT_DIR / "05_attack_category_flag_rate.png")
plt.close(fig)

# =====================================================================
# 7. CONTAMINATION — TRAINING-THRESHOLD SENSITIVITY ANALYSIS (not model selection)
# =====================================================================
# The primary configuration stays contamination="auto". The values below are
# reported as a sensitivity analysis only; none of them is selected.
# For a fixed (n_estimators, max_samples, random_state) the trees are identical
# whatever `contamination` is; contamination only sets
#     offset_ = percentile(score_samples(X_train), 100 * contamination)
# and predict() flags rows with score_samples < offset_. So every contamination
# value is evaluated from the SAME fitted forest, with the threshold computed from
# TRAINING scores only (exactly what sklearn does). One real refit verifies this.
# Candidate values are theoretically motivated: TRAIN is pure BENIGN, so
# contamination = target false-alarm rate on normal traffic (0.1%, 0.5%, 1%, 2%,
# 5%, 10%). The test attack prevalence (40.4%) is deliberately NOT a candidate and
# test labels are not used to choose among them.
log("\n" + "=" * 70)
log("7. CONTAMINATION — training-threshold sensitivity (not model selection)")
log("=" * 70)
t = time.time()
train_scores = model.score_samples(X_train_s)
train_score_time = time.time() - t
test_scores = model.score_samples(X_test_s)          # = decision + offset_ (-0.5)
assert np.allclose(test_scores - model.offset_, decision)
auto_implied = float(np.mean(train_scores < model.offset_))
log(f"'auto' threshold (offset_=-0.5) flags {auto_implied:.4%} of TRAINING rows "
    f"=> implied contamination ≈ {auto_implied:.4f}")

CONTAM_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
contam_rows = [{"contamination": "auto", "offset": float(model.offset_),
                "train_flag_rate": auto_implied,
                **binary_metrics(y_test_binary, y_pred_if)}]
for c in CONTAM_VALUES:
    off = float(np.percentile(train_scores, 100.0 * c))
    p = (test_scores < off).astype(np.int8)
    contam_rows.append({"contamination": c, "offset": off,
                        "train_flag_rate": float(np.mean(train_scores < off)),
                        **binary_metrics(y_test_binary, p)})

# Verify the shortcut against a genuine sklearn refit.
VERIFY_C = 0.05
t = time.time()
m_ver = IsolationForest(n_estimators=200, max_samples="auto", contamination=VERIFY_C,
                        random_state=RANDOM_STATE, n_jobs=-1).fit(X_train_s)
p_ver = to_project_convention(m_ver.predict(X_test_s))
verify_time = time.time() - t
off_short = float(np.percentile(train_scores, 100.0 * VERIFY_C))
p_short = (test_scores < off_short).astype(np.int8)
assert np.isclose(m_ver.offset_, off_short), (m_ver.offset_, off_short)
mismatch = int((p_ver != p_short).sum())
assert mismatch <= 10, f"{mismatch} prediction mismatches vs refit"
log(f"Verified vs real refit (contamination={VERIFY_C}): offset {m_ver.offset_:.6f} == "
    f"{off_short:.6f}; prediction mismatches: {mismatch} ({verify_time:.1f} s)")
del m_ver, p_ver
contam_df = pd.DataFrame(contam_rows)

fig, ax = plt.subplots(figsize=(8, 4.4))
cdf = contam_df[contam_df["contamination"] != "auto"]
xs = cdf["contamination"].astype(float).to_numpy()
for col_name, lab, colr, mk in (("recall", "Recall (TPR)", C_ATTACK, "o"),
                                ("precision", "Precision", C_BENIGN, "s"),
                                ("fpr", "False positive rate", "#1baf7a", "^")):
    ax.plot(xs, cdf[col_name], marker=mk, markersize=7, linewidth=2, color=colr, label=lab)
    ax.annotate(lab, (xs[-1], cdf[col_name].iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9)
ax.axvline(auto_implied, color=INK_2, linestyle="--", linewidth=1)
ax.text(auto_implied, 1.02, f"'auto' ≈ {auto_implied:.3f}", color=INK_2, fontsize=8, ha="center")
ax.set_xscale("log")
ax.set_xticks(xs, [f"{v:g}" for v in xs])
ax.set_ylim(0, 1.05)
ax.set_xlabel("contamination (threshold from TRAINING scores; log scale)")
ax.set_ylabel("Rate")
ax.set_title("Training-threshold sensitivity (contamination) — same 200-tree forest; "
             "primary = 'auto'", fontsize=10)
ax.legend(frameon=False, loc="center left")
fig.savefig(PLOT_DIR / "06_contamination_sensitivity.png")
plt.close(fig)

# =====================================================================
# 8. PARAMETER ANALYSIS — n_estimators stability / sensitivity (NOT selection)
# =====================================================================
# The primary configuration is the PRE-SPECIFIED baseline (n_estimators=200) and
# it is not changed by anything in this section. This section only describes how
# much the scores move with forest size and random seed:
#   (a) label-free stability on a BENIGN-only validation subset carved out of TRAIN
#       (10%), across 3 seeds — the test set plays no role;
#   (b) test-set metrics for n=100 / 300 (seed 42, full TRAIN), reported side by side.
log("\n" + "=" * 70)
log("8. PARAMETER ANALYSIS (n_estimators) — stability / sensitivity only")
log("=" * 70)
rng = np.random.RandomState(RANDOM_STATE)
perm = rng.permutation(len(X_train_s))
n_val = int(0.10 * len(X_train_s))
val_idx, fit_idx = perm[:n_val], perm[n_val:]
X_fit, X_val = X_train_s[fit_idx], X_train_s[val_idx]
log(f"Validation design: fit on {len(X_fit):,} TRAIN rows, evaluate on {len(X_val):,} held-out "
    f"TRAIN rows (all BENIGN). Test set untouched.")

SEEDS = [42, 43, 44]
N_EST = [100, 200, 300]
stab_rows = []
val_scores_by_n = {}
for n in N_EST:
    sc, pr, ft = [], [], []
    for s in SEEDS:
        t = time.time()
        m = IsolationForest(n_estimators=n, max_samples="auto", contamination="auto",
                            random_state=s, n_jobs=-1).fit(X_fit)
        ft.append(time.time() - t)
        v = -m.decision_function(X_val)
        sc.append(v)
        pr.append((v > 0).astype(np.int8))
    val_scores_by_n[n] = sc[0]
    pairs = [(i, j) for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))]
    rho = [spearmanr(sc[i], sc[j]).statistic for i, j in pairs]
    agree = [float(np.mean(pr[i] == pr[j])) for i, j in pairs]
    sdev = float(np.mean(np.std(np.vstack(sc), axis=0)))
    flag = [float(p.mean()) for p in pr]
    stab_rows.append({"n_estimators": n, "mean_seed_spearman": float(np.mean(rho)),
                      "min_seed_spearman": float(np.min(rho)),
                      "mean_pred_agreement": float(np.mean(agree)),
                      "mean_per_row_score_sd": sdev,
                      "val_flag_rate_mean": float(np.mean(flag)),
                      "val_flag_rate_sd": float(np.std(flag)),
                      "fit_time_s": float(np.mean(ft))})
    log(f"n={n}: seed-Spearman mean={np.mean(rho):.5f} min={np.min(rho):.5f} | "
        f"pred agreement={np.mean(agree):.5f} | per-row score SD={sdev:.5f} | "
        f"val BENIGN flag rate={np.mean(flag):.4%} ± {np.std(flag):.4%}")
stab_df = pd.DataFrame(stab_rows)
for n in (100, 300):
    log(f"Spearman(val scores n={n} vs n=200, seed 42): "
        f"{spearmanr(val_scores_by_n[n], val_scores_by_n[200]).statistic:.5f}")
del X_fit, X_val
gc.collect()

est_rows = [{"n_estimators": 200, "train_time_s": train_time, "score_time_s": score_time,
             "spearman_vs_200": 1.0, **base}]
est_cat_rates = {200: cat_df.set_index("category")["detection_rate"]}
for n in (100, 300):
    t = time.time()
    m = IsolationForest(n_estimators=n, max_samples="auto", contamination="auto",
                        random_state=RANDOM_STATE, n_jobs=-1).fit(X_train_s)
    tt = time.time() - t
    t = time.time()
    a = -m.decision_function(X_test_s)
    st = time.time() - t
    p = (a > 0).astype(np.int8)
    est_rows.append({"n_estimators": n, "train_time_s": tt, "score_time_s": st,
                     "spearman_vs_200": float(spearmanr(a, anomaly_score).statistic),
                     **binary_metrics(y_test_binary, p),
                     "roc_auc": roc_auc_score(y_test_binary, a)})
    est_cat_rates[n] = category_table(p).set_index("category")["detection_rate"]
    del m, a, p
    gc.collect()
est_df = pd.DataFrame(est_rows).sort_values("n_estimators").reset_index(drop=True)

# =====================================================================
# 10. SAVE THE PRIMARY (n=200 baseline) MODEL + PREPROCESSING ARTIFACTS
# =====================================================================
assert model.n_estimators == 200 and len(model.estimators_) == 200
assert model.contamination == "auto" and model.random_state == RANDOM_STATE
paths = {
    "isolation_forest": MODEL_DIR / "isolation_forest.pkl",
    "imputer": MODEL_DIR / "imputer.pkl",
    "scaler": MODEL_DIR / "scaler.pkl",
    "preprocessing_meta": MODEL_DIR / "preprocessing_meta.pkl",
}
joblib.dump(model, paths["isolation_forest"], compress=3)
joblib.dump(imputer, paths["imputer"])
joblib.dump(scaler, paths["scaler"])
joblib.dump({"zero_variance_dropped": zero_var_cols, "feature_columns": feature_columns,
             "zero_variance_rule": "X_train.var() == 0 on the full approved TRAIN set",
             "zero_variance_note": ZERO_VAR_NOTE,
             "inf_handling": "±inf -> NaN, then median imputation (fit on TRAIN)",
             "dtype_for_model": "float32",
             "score_convention": "anomaly_score = -decision_function (higher = more anomalous)",
             "pred_convention": "0 = normal, 1 = anomaly"}, paths["preprocessing_meta"])
sizes_mb = {k: v.stat().st_size / 1e6 for k, v in paths.items()}

# Reload round-trip: the saved file must be the same 200-tree model that produced
# every baseline number in the report (scores, predictions and offset).
reloaded = joblib.load(paths["isolation_forest"])
assert reloaded.n_estimators == 200 and len(reloaded.estimators_) == 200
assert reloaded.offset_ == model.offset_
rel_decision = reloaded.decision_function(X_test_s)
assert np.array_equal(rel_decision, decision), "reloaded model scores differ from baseline"
assert np.array_equal(to_project_convention(reloaded.predict(X_test_s)), y_pred_if)
rel_imp, rel_sc = joblib.load(paths["imputer"]), joblib.load(paths["scaler"])
assert np.array_equal(rel_imp.statistics_, imputer.statistics_)
assert np.array_equal(rel_sc.mean_, scaler.mean_) and np.array_equal(rel_sc.scale_, scaler.scale_)
log("Reload check: saved isolation_forest.pkl reproduces the n=200 baseline scores and "
    "predictions exactly on all test rows.")
del reloaded, rel_decision, rel_imp, rel_sc

np.savez_compressed(RESULT_DIR / "if_test_outputs.npz",
                    anomaly_score=anomaly_score.astype(np.float32), y_pred_if=y_pred_if,
                    y_test_binary=y_test_binary)

# =====================================================================
# 11. SANITY CHECKS (final)
# =====================================================================
sanity = {
    "no Label column in model features": "Label" not in feature_columns,
    "no binary_label in model features": "binary_label" not in feature_columns,
    "model n_features_in_ == feature count": model.n_features_in_ == n_features,
    "zero-variance columns == approved 8": zero_var_cols == EXPECTED_ZERO_VAR,
    "no NaN in X_train/X_test": not (np.isnan(X_train_s).any() or np.isnan(X_test_s).any()),
    "no inf in X_train/X_test": bool(np.isfinite(X_train_s).all() and np.isfinite(X_test_s).all()),
    "training data BENIGN only": bool((labels_train == "BENIGN").all()),
    "test data = BENIGN + all 14 attacks": set(labels_test.unique()) == {"BENIGN", *ATTACK_CATEGORIES},
    "random_state == 42": model.random_state == 42,
    "saved model is the n=200 baseline": True,   # asserted in section 10
    "score direction: mean ATTACK score > mean BENIGN score":
        bool(anomaly_score[y_test_binary == 1].mean() > anomaly_score[y_test_binary == 0].mean()),
    "score direction: AUC(-decision) == 1 - AUC(decision)": abs(roc_auc + roc_auc_reversed - 1) < 1e-9,
    "y_pred convention 0=normal,1=anomaly (matches predict==-1)":
        bool(np.array_equal(y_pred_if, (raw_pred == -1).astype(np.int8))),
    "ground-truth labels unchanged": bool(np.array_equal(
        y_test_binary, (labels_test != "BENIGN").to_numpy().astype(np.int8))),
}
for k, v in sanity.items():
    assert v, f"SANITY FAILED: {k}"

peak_gb = peak_rss_gb()
total_time = time.time() - T0

# =====================================================================
# 12. FINAL REPORT
# =====================================================================
L = []
L.append("=" * 70)
L.append("ISOLATION FOREST — BASELINE RESULTS  (primary, pre-specified configuration)")
L.append("=" * 70)
L.append("\nDataset:")
L.append(f"  Training: {EXPECTED_TRAIN:,} rows (BENIGN only)")
L.append(f"  Testing:  {EXPECTED_TEST:,} rows ({EXPECTED_TEST_BENIGN:,} unseen BENIGN + "
         f"{EXPECTED_TEST_ATTACK:,} attack; attack share {EXPECTED_TEST_ATTACK/EXPECTED_TEST:.2%})")
L.append(f"  Features: {n_features} ({raw_feature_count} raw - {len(zero_var_cols)} zero-variance; "
         f"Destination Port retained)")
L.append(f"  Zero-variance dropped (rule: X_train.var() == 0 on full TRAIN): {zero_var_cols}")
L.append(f"  Note: {ZERO_VAR_NOTE}")
L.append("\nParameters:")
L.append(f"  n_estimators  = {model.n_estimators} (trees fitted: {len(model.estimators_)})")
L.append(f"  max_samples   = 'auto' -> {model.max_samples_} rows per tree")
L.append(f"  contamination = 'auto' (offset_ = {model.offset_}; flags {auto_implied:.3%} of TRAIN)")
L.append(f"  random_state  = {model.random_state} | n_jobs = -1")
L.append("  Score: anomaly_score = -decision_function (higher = more anomalous); threshold 0")
L.append("\nConfusion Matrix (rows = truth, cols = prediction):")
L.append(f"                 Pred normal   Pred anomaly")
L.append(f"  BENIGN (0)     {base['TN']:>11,}   {base['FP']:>12,}")
L.append(f"  ATTACK (1)     {base['FN']:>11,}   {base['TP']:>12,}")
L.append(f"\n  TP={base['TP']:,}  TN={base['TN']:,}  FP={base['FP']:,}  FN={base['FN']:,}")
L.append(f"\nAccuracy:  {base['accuracy']:.4f}")
L.append(f"Precision: {base['precision']:.4f}")
L.append(f"Recall:    {base['recall']:.4f}")
L.append(f"F1:        {base['f1']:.4f}")
L.append(f"FPR:       {base['fpr']:.4f}")
L.append(f"FNR:       {base['fnr']:.4f}")
L.append(f"ROC-AUC:   {roc_auc:.4f}")
L.append(f"\nTraining Time:   {train_time:.2f} s")
L.append(f"Prediction Time: {predict_time:.2f} s for predict() on {EXPECTED_TEST:,} rows "
         f"({score_time:.2f} s for decision_function)")
L.append("\nAttack Category Detection")
L.append("(share of each category's test traffic flagged as anomalous — NOT a classification)")
L.append(f"  {'Category':<28}{'Total':>9}{'Flagged':>9}{'Missed':>9}{'Rate':>9}   95% CI")
for r in cat_df.itertuples():
    L.append(f"  {r.category:<28}{r.total:>9,}{r.flagged_anomalous:>9,}{r.missed:>9,}"
             f"{r.detection_rate:>9.2%}   [{r.ci95_low:.1%}, {r.ci95_high:.1%}]"
             f"{'  * small n' if r.small_sample else ''}")
L.append(f"  (reference: unseen BENIGN test traffic flagged {benign_test_flag_rate:.2%})")
L.append(f"  * n < {SMALL_SAMPLE_N}: the rate rests on very few flows (CI shown); it should not be "
         "generalised to the attack type.")
L.append("\n" + "-" * 70)
L.append("SENSITIVITY ANALYSES (descriptive only — none of these changes the primary model)")
L.append("-" * 70)
L.append("\nA. Training-threshold sensitivity (contamination)")
L.append("   Same 200-tree forest; threshold = percentile of TRAINING scores; NOT model selection.")
L.append(f"  {'contam':>8}{'train flag':>11}{'anom %':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}"
         f"{'FPR':>8}{'FNR':>8}")
for r in contam_df.itertuples():
    c = r.contamination if isinstance(r.contamination, str) else f"{r.contamination:g}"
    tag = "  <- primary" if c == "auto" else ""
    L.append(f"  {c:>8}{r.train_flag_rate:>11.3%}{r.anomaly_pct:>7.2f}%{r.precision:>8.4f}"
             f"{r.recall:>8.4f}{r.f1:>8.4f}{r.fpr:>8.4f}{r.fnr:>8.4f}{tag}")
L.append("\nB. n_estimators stability (validation = 10% held-out TRAIN BENIGN; 3 seeds; label-free)")
L.append(f"  {'n':>5}{'seed rho':>10}{'min rho':>10}{'pred agree':>12}{'score SD':>10}"
         f"{'val flag %':>12}{'fit s':>8}")
for r in stab_df.itertuples():
    L.append(f"  {r.n_estimators:>5}{r.mean_seed_spearman:>10.5f}{r.min_seed_spearman:>10.5f}"
             f"{r.mean_pred_agreement:>12.5f}{r.mean_per_row_score_sd:>10.5f}"
             f"{100*r.val_flag_rate_mean:>11.3f}%{r.fit_time_s:>8.1f}")
L.append("\nC. n_estimators test-set sensitivity (seed 42, full TRAIN)")
L.append(f"  {'n':>5}{'Prec':>8}{'Recall':>8}{'F1':>8}{'FPR':>8}{'AUC':>8}{'rho vs 200':>12}"
         f"{'fit s':>8}{'score s':>9}")
for r in est_df.itertuples():
    tag = "  <- primary" if r.n_estimators == 200 else ""
    L.append(f"  {r.n_estimators:>5}{r.precision:>8.4f}{r.recall:>8.4f}{r.f1:>8.4f}{r.fpr:>8.4f}"
             f"{r.roc_auc:>8.4f}{r.spearman_vs_200:>12.5f}{r.train_time_s:>8.1f}{r.score_time_s:>9.1f}{tag}")
L.append("  Per-category flag rate by forest size:")
L.append(f"    {'Category':<28}{'n=100':>9}{'n=200':>9}{'n=300':>9}")
for r in cat_df.itertuples():
    L.append(f"    {r.category:<28}{est_cat_rates[100][r.category]:>9.2%}"
             f"{est_cat_rates[200][r.category]:>9.2%}{est_cat_rates[300][r.category]:>9.2%}")
L.append("\nComputational Report")
L.append(f"  Samples: train {EXPECTED_TRAIN:,} | test {EXPECTED_TEST:,} | features {n_features}")
L.append(f"  Model input matrices (float32): train {X_train_s.nbytes/1e9:.2f} GB, "
         f"test {X_test_s.nbytes/1e9:.2f} GB")
L.append(f"  Peak process RAM (working set): {peak_gb:.2f} GB — dominated by loading/deduplicating "
         f"the raw CSVs, not by the model")
L.append(f"  Trees: {len(model.estimators_)} x {model.max_samples_} samples each "
         f"(subsampling makes fit cost independent of training-set size)")
L.append(f"  Training: {train_time:.2f} s | decision_function on test: {score_time:.2f} s | "
         f"predict() on test: {predict_time:.2f} s | scoring TRAIN: {train_score_time:.2f} s")
L.append("  Saved artifact sizes: " + ", ".join(f"{k} {v:.2f} MB" for k, v in sizes_mb.items()))
L.append(f"  Total script wall time: {total_time/60:.1f} min")
L.append("\nSanity Checks")
for k, v in sanity.items():
    L.append(f"  [PASS] {k}")
L.append("\nArtifacts")
for k, v in paths.items():
    L.append(f"  {v.relative_to(ROOT)}")
L.append(f"  {(RESULT_DIR / 'if_test_outputs.npz').relative_to(ROOT)}")
L.append(f"  {PLOT_DIR.relative_to(ROOT)}/01..06 *.png")
L.append(f"  {FREEZE_MANIFEST.relative_to(ROOT)}  (SHA-256 of every artifact)")
L.append("\nPHASE 2A FROZEN — awaiting review. (DBSCAN / PCA / port ablation NOT run.)")
report = "\n".join(L)
log("\n" + report)
(RESULT_DIR / "phase2a_report.txt").write_text(report, encoding="utf-8")

with open(RESULT_DIR / "phase2a_metrics.json", "w") as fh:
    json.dump({
        "primary_params": BASELINE,
        "n_train": EXPECTED_TRAIN, "n_test": EXPECTED_TEST, "n_features": n_features,
        "zero_variance_dropped": zero_var_cols, "zero_variance_note": ZERO_VAR_NOTE,
        "baseline_metrics": base, "auto_implied_contamination": auto_implied,
        "train_time_s": train_time, "score_time_s": score_time, "predict_time_s": predict_time,
        "attack_categories": cat_df.to_dict(orient="records"),
        "sensitivity_contamination": contam_df.astype({"contamination": str}).to_dict(orient="records"),
        "sensitivity_n_estimators_val_stability": stab_df.to_dict(orient="records"),
        "sensitivity_n_estimators_test": est_df.to_dict(orient="records"),
        "sensitivity_n_estimators_category_rates": {str(k): v.to_dict() for k, v in est_cat_rates.items()},
        "peak_ram_gb": peak_gb, "artifact_sizes_mb": sizes_mb,
    }, fh, indent=2, default=float)

# =====================================================================
# 13. FREEZE
# =====================================================================
frozen_files = [paths[k] for k in paths] + [
    RESULT_DIR / "phase2a_report.txt", RESULT_DIR / "phase2a_metrics.json",
    RESULT_DIR / "if_test_outputs.npz", Path(__file__).resolve(),
] + sorted(PLOT_DIR.glob("*.png"))
with open(FREEZE_MANIFEST, "w") as fh:
    json.dump({
        "phase": "2A", "status": "FROZEN",
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "primary_model": {"n_estimators": 200, "max_samples": "auto", "contamination": "auto",
                          "random_state": 42},
        "split": {"train": EXPECTED_TRAIN, "test": EXPECTED_TEST,
                  "test_benign": EXPECTED_TEST_BENIGN, "test_attack": EXPECTED_TEST_ATTACK,
                  "method": "glob order -> drop_duplicates -> np.random.seed(42) permutation, "
                            "first 70% of BENIGN = TRAIN"},
        "n_features": n_features, "zero_variance_dropped": zero_var_cols,
        "headline": {k: base[k] for k in ("TP", "TN", "FP", "FN", "precision", "recall", "f1",
                                          "fpr", "fnr", "roc_auc")},
        "sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in frozen_files},
    }, fh, indent=2, default=float)
log(f"\nFreeze manifest written: {FREEZE_MANIFEST.relative_to(ROOT)}")
