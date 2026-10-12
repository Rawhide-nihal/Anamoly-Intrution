"""
PHASE 2B — DBSCAN (transductive, density-based anomaly detection)

DBSCAN is TRANSDUCTIVE: it assigns cluster / noise labels only to the rows it is
run on. sklearn's DBSCAN has no predict() for unseen rows, so nothing here is a
claim about out-of-sample generalisation.

Methodological guarantees enforced in code
------------------------------------------
* Ground-truth labels live in a HiddenLabels "vault". Before clustering, the vault
  only releases aggregate composition counts (Section 1 reporting) and a leakage
  check. The label vectors are released only after EVERY clustering run (primary,
  PCA, sensitivity) has finished; any DBSCAN call after that raises.
* Preprocessing, PCA, k-distance analysis, eps / min_samples selection and DBSCAN
  receive feature matrices only.
* Phase 2A artifacts are hash-verified at start and end and never written.
* Nothing from Phase 2A (predictions, metrics, contamination, model selection) is read.

Anomaly convention: DBSCAN label -1 (noise) -> 1 = anomaly; any cluster id -> 0 = normal.
"""
import ctypes
import gc
import glob
import hashlib
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
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             pairwise_distances_chunked, precision_score, recall_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import sklearn

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Constants (all fixed before any data is seen)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
SAMPLE_N = 100_000
FALLBACK_N = 50_000
EXPECTED_DEDUP_ROWS = 2_522_362
EXPECTED_ZERO_VAR = [
    "Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]
N_FEATURES = 70
MIN_SAMPLES_CANDIDATES = [5, 10, 20]
EPS_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]          # sensitivity grid around the elbow
PCA_VARIANCE_TARGETS = [0.90, 0.95, 0.99]
# Memory model for sklearn DBSCAN (sklearn 1.9.1, brute radius neighbours): measured
# ~23.4 bytes of committed memory per stored neighbour pair (eps=0.5, 100k rows,
# 2.12e8 pairs -> +4.96 GB commit). 24 B/pair is used, with a fixed 10 GB budget.
BYTES_PER_PAIR = 24
MEMORY_BUDGET_GB = 10.0
UNSTABLE_N = 30        # category estimates with n < 30 are marked statistically unstable
ATTACK_CATEGORIES = [
    "Heartbleed", "Infiltration", "Bot", "PortScan", "DDoS", "DoS Hulk", "DoS GoldenEye",
    "DoS slowloris", "DoS Slowhttptest", "FTP-Patator", "SSH-Patator",
    "Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection",
]

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
RESULT_DIR = ROOT / "results"
PLOT_DIR = ROOT / "plots" / "phase2b"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PHASE2A_MANIFEST = RESULT_DIR / "phase2a_frozen_manifest.json"
OUT = {
    "report": RESULT_DIR / "phase2b_report.txt",
    "metrics": RESULT_DIR / "phase2b_metrics.json",
    "config": RESULT_DIR / "phase2b_config.json",
    "sample_ids": RESULT_DIR / "phase2b_sample_ids.csv",
    "assignments": RESULT_DIR / "phase2b_cluster_assignments.npz",
    "preprocessing": RESULT_DIR / "phase2b_preprocessing.pkl",
    "manifest": RESULT_DIR / "phase2b_manifest.json",
}

C_BENIGN, C_ATTACK, C_3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": INK_2,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "savefig.dpi": 150, "savefig.bbox": "tight",
})

LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_phase2a():
    m = json.load(open(PHASE2A_MANIFEST))
    bad = [p for p, d in m["sha256"].items() if sha256(ROOT / p) != d]
    return m, bad


def mem_gb():
    """(peak working set, peak commit) of this process in GB (Windows)."""
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage")]
    try:
        p = PMC()
        p.cb = ctypes.sizeof(PMC)
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb)
        return p.PeakWorkingSetSize / 1e9, p.PeakPagefileUsage / 1e9
    except Exception:
        return float("nan"), float("nan")


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------------------
# Label vault and clustering registry
# ---------------------------------------------------------------------------
class HiddenLabels:
    """Holds ground truth; releases vectors only after all clustering is complete."""

    def __init__(self, labels):
        self._labels = np.asarray(labels, dtype=object)
        self.revealed = False
        self.reveal_time = None

    def composition(self):
        """Aggregate counts only (Section 1 descriptive reporting)."""
        vc = pd.Series(self._labels).value_counts()
        return {str(k): int(v) for k, v in vc.items()}

    def assert_not_in(self, X):
        """Leakage check: no feature column may equal the binary label vector."""
        y = (self._labels != "BENIGN").astype(np.float64)
        X = np.asarray(X)
        for j in range(X.shape[1]):
            assert not np.array_equal(X[:, j], y), f"column {j} equals the label vector"
            assert abs(np.corrcoef(X[:, j], y)[0, 1]) < 0.999 if X[:, j].std() > 0 else True
        return True

    def reveal(self, registry):
        assert registry.all_finished(), "labels requested before all clustering finished"
        registry.closed = True
        self.revealed = True
        self.reveal_time = time.time()
        return (self._labels != "BENIGN").astype(np.int8), self._labels.copy()


class ClusteringRegistry:
    def __init__(self):
        self.runs = []
        self.closed = False
        self.last_end = 0.0

    def all_finished(self):
        return all(r.get("end") is not None for r in self.runs)


VAULT = None
REGISTRY = ClusteringRegistry()


def run_dbscan(name, X, eps, min_samples):
    """DBSCAN on a feature matrix only. Refuses to run once labels are revealed."""
    assert not REGISTRY.closed and not VAULT.revealed, "clustering after label reveal"
    assert np.isfinite(X).all()
    rec = {"name": name, "start": time.time(), "end": None}
    REGISTRY.runs.append(rec)
    t = time.time()
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean",
                    algorithm="brute", n_jobs=-1).fit_predict(X)
    rec["runtime_s"] = time.time() - t
    rec["end"] = time.time()
    REGISTRY.last_end = rec["end"]
    rec["peak_ws_gb"], rec["peak_commit_gb"] = mem_gb()
    return labels.astype(np.int32), rec


def noise_to_pred(labels):
    """-1 -> anomaly (1); every cluster id -> normal (0)."""
    return (labels == -1).astype(np.int8)


def cluster_summary(labels, min_samples):
    sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.array([], int)
    sizes_sorted = np.sort(sizes)[::-1]
    return {
        "n_clusters": int(len(sizes)),
        "n_noise": int((labels == -1).sum()),
        "noise_pct": 100.0 * float((labels == -1).mean()),
        "largest_cluster_sizes": [int(s) for s in sizes_sorted[:10]],
        "largest_cluster_share_pct": 100.0 * float(sizes_sorted[0] / len(labels)) if len(sizes) else 0.0,
        "clusters_smaller_than_min_samples": int((sizes < min_samples).sum()),
        "clusters_size_lt_50": int((sizes < 50).sum()),
        "median_cluster_size": float(np.median(sizes)) if len(sizes) else 0.0,
    }


# ---------------------------------------------------------------------------
# k-distance / elbow helpers (unsupervised)
# ---------------------------------------------------------------------------
def kdist_matrix(X, k_max):
    """Distances to the 1..k_max nearest points INCLUDING the point itself (col 0 = self),
    so column (m-1) is the core distance for DBSCAN min_samples=m."""
    nn = NearestNeighbors(n_neighbors=k_max, algorithm="brute", n_jobs=-1).fit(X)
    D, _ = nn.kneighbors(X)
    return D


def kneedle(y_sorted):
    """Kneedle (Satopaa et al., 2011) for an increasing convex curve: the point of
    maximum distance below the chord after min-max normalisation."""
    n = len(y_sorted)
    x = np.arange(n) / (n - 1)
    yn = (y_sorted - y_sorted[0]) / (y_sorted[-1] - y_sorted[0])
    i = int(np.argmax(x - yn))
    return i, y_sorted[i]


def elbow_eps(kd, rule="log10"):
    k = np.sort(kd)
    if rule == "log10":
        i, v = kneedle(np.log10(np.maximum(k, 1e-6)))
        return float(10 ** v), i / (len(k) - 1)
    if rule == "linear":
        i, v = kneedle(k)
        return float(v), i / (len(k) - 1)
    if rule == "trunc99":
        kt = k[: int(np.ceil(0.99 * len(k)))]
        i, v = kneedle(kt)
        return float(v), i / (len(k) - 1)
    raise ValueError(rule)


def count_pairs(X, eps_list):
    """Exact number of stored neighbour pairs (incl. self) for each eps, chunked, label-free."""
    eps_arr = np.asarray(sorted(eps_list))
    tot = np.zeros(len(eps_arr), dtype=np.int64)
    mx = np.zeros(len(eps_arr), dtype=np.int64)
    for ch in pairwise_distances_chunked(X, X, n_jobs=-1, working_memory=512):
        c = np.stack([(ch <= e).sum(axis=1) for e in eps_arr], axis=1)
        tot += c.sum(axis=0)
        mx = np.maximum(mx, c.max(axis=0))
    return {float(e): {"pairs": int(t), "max_neighbourhood": int(m),
                       "est_gb": float(t * BYTES_PER_PAIR / 1e9)}
            for e, t, m in zip(eps_arr, tot, mx)}


def binary_metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "accuracy": accuracy_score(y, p),
            "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0),
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "fnr": fn / (fn + tp) if fn + tp else 0.0}


T0 = time.time()
# =====================================================================
# 0. Phase 2A integrity (before)
# =====================================================================
p2a_manifest, bad = verify_phase2a()
assert not bad, f"Phase 2A artifacts modified before Phase 2B: {bad}"
P2A_PATHS = {str((ROOT / p).resolve()) for p in p2a_manifest["sha256"]}
assert not any(str(p.resolve()) in P2A_PATHS for p in OUT.values()), "output path collides with Phase 2A"
log(f"Phase 2A integrity (start): {len(p2a_manifest['sha256'])} frozen artifacts unchanged.")

# =====================================================================
# 1. REPRODUCIBLE SAMPLE
# =====================================================================
log("\n" + "=" * 72 + "\n1. SAMPLE CONSTRUCTION\n" + "=" * 72)
csv_files = list(glob.glob("*.csv"))
dfs, file_sizes = [], []
for f in csv_files:
    d = pd.read_csv(f)
    d.columns = d.columns.str.strip()
    dfs.append(d)
    file_sizes.append(len(d))
df = pd.concat(dfs, ignore_index=True)
del dfs
gc.collect()
df = df.drop_duplicates()
assert len(df) == EXPECTED_DEDUP_ROWS, len(df)
labels_all = (df["Label"].str.replace("�", "-", regex=False)
              .str.replace(r"\s+", " ", regex=True).str.strip())
pop_attack = int((labels_all != "BENIGN").sum())
pop_counts = labels_all.value_counts()

X_all = df.drop(columns=["Label"] + EXPECTED_ZERO_VAR)
X_all = X_all.replace([np.inf, -np.inf], np.nan)
assert X_all.shape[1] == N_FEATURES and "Destination Port" in X_all.columns
del df
gc.collect()

# Uniform random sample of the deduplicated population (no stratification, no balancing).
X_sample = X_all.sample(n=SAMPLE_N, random_state=RANDOM_STATE)
sample_index = X_sample.index.to_numpy()        # = row position in the raw concatenation
labels_sample = labels_all.loc[sample_index].to_numpy()
# The 50k fallback is the same uniform draw truncated: pandas' sampling is a seeded
# permutation, so sample(n=50k, rs=42) is exactly the first 50k rows of sample(n=100k, rs=42).
assert np.array_equal(X_all.sample(n=FALLBACK_N, random_state=RANDOM_STATE).index.to_numpy(),
                      sample_index[:FALLBACK_N])
del X_all
gc.collect()

cum = np.cumsum([0] + file_sizes)
file_idx = np.searchsorted(cum, sample_index, side="right") - 1
sample_ids = pd.DataFrame({
    "sample_order": np.arange(SAMPLE_N),
    "raw_concat_index": sample_index,
    "source_file": np.array(csv_files)[file_idx],
    "file_row": sample_index - cum[file_idx],      # 0-based data row inside that CSV
})

VAULT = HiddenLabels(labels_sample)                # y_dbscan, hidden
comp100 = VAULT.composition()
n_att100 = sum(v for k, v in comp100.items() if k != "BENIGN")
log(f"Sample: {SAMPLE_N:,} rows drawn uniformly at random from {EXPECTED_DEDUP_ROWS:,} "
    f"deduplicated rows (random_state={RANDOM_STATE})")
log(f"  BENIGN {comp100.get('BENIGN', 0):,} | ATTACK {n_att100:,} | prevalence "
    f"{n_att100 / SAMPLE_N:.2%} (population {pop_attack / EXPECTED_DEDUP_ROWS:.2%})")

# =====================================================================
# 2. PREPROCESSING (label-free) — helper
# =====================================================================
def preprocess(Xdf):
    """Median imputation + StandardScaler fitted on the rows being clustered (no y)."""
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Z = sc.fit_transform(imp.fit_transform(Xdf))
    return Z, imp, sc


# =====================================================================
# 3a. FEASIBILITY ON THE FULL 100k SAMPLE (label-free)
# =====================================================================
log("\n" + "=" * 72 + "\n3a. k-DISTANCE + MEMORY FEASIBILITY ON 100,000 ROWS\n" + "=" * 72)
Z100, _, _ = preprocess(X_sample)
assert np.isfinite(Z100).all() and Z100.shape == (SAMPLE_N, N_FEATURES)
t = time.time()
D100 = kdist_matrix(Z100, max(MIN_SAMPLES_CANDIDATES))
kdist_time_100 = time.time() - t
eps100, _ = elbow_eps(D100[:, 10 - 1], "log10")
feas100 = count_pairs(Z100, [eps100])[float(eps100)]
log(f"Elbow eps (min_samples=10, log10-Kneedle) on 100k: {eps100:.4f}; neighbour pairs "
    f"{feas100['pairs']:,} -> est. {feas100['est_gb']:.1f} GB (budget {MEMORY_BUDGET_GB} GB)")
use_fallback = feas100["est_gb"] > MEMORY_BUDGET_GB
N_WORK = FALLBACK_N if use_fallback else SAMPLE_N
log(f"=> DBSCAN working sample: {N_WORK:,} rows "
    f"({'documented 50k fallback: memory' if use_fallback else 'full 100k sample'})")
kdist100_summary = {ms: np.percentile(D100[:, ms - 1], [50, 90, 95, 99, 99.9, 100]).tolist()
                    for ms in MIN_SAMPLES_CANDIDATES}
del Z100, D100
gc.collect()

# Working sample
X_dbscan_df = X_sample.iloc[:N_WORK]
work_ids = sample_ids.iloc[:N_WORK]
VAULT = HiddenLabels(labels_sample[:N_WORK])       # y_dbscan for the working sample, hidden
comp_work = VAULT.composition()
n_att_work = sum(v for k, v in comp_work.items() if k != "BENIGN")
sample_ids["in_dbscan_working_sample"] = sample_ids["sample_order"] < N_WORK

# =====================================================================
# 2. PREPROCESSING of the working sample
# =====================================================================
log("\n" + "=" * 72 + "\n2. PREPROCESSING (working sample, label-free)\n" + "=" * 72)
nan_rows = int(X_dbscan_df.isna().any(axis=1).sum())
X_dbscan, imputer, scaler = preprocess(X_dbscan_df)
feature_columns = list(X_dbscan_df.columns)
assert X_dbscan.shape == (N_WORK, N_FEATURES)
assert "Destination Port" in feature_columns
assert not np.isnan(X_dbscan).any() and np.isfinite(X_dbscan).all()
assert VAULT.assert_not_in(X_dbscan)
const_cols = [c for c, s in zip(feature_columns, scaler.var_) if s == 0]
log(f"X_dbscan {X_dbscan.shape}; rows with NaN before imputation: {nan_rows}; "
    f"columns constant within sample: {const_cols or 'none'}")

# =====================================================================
# 3. k-DISTANCE ANALYSIS, eps AND min_samples SELECTION (unsupervised)
# =====================================================================
log("\n" + "=" * 72 + "\n3. k-DISTANCE / eps / min_samples SELECTION\n" + "=" * 72)
t = time.time()
D = kdist_matrix(X_dbscan, max(MIN_SAMPLES_CANDIDATES))
kdist_time = time.time() - t
elbows = {}
for ms in MIN_SAMPLES_CANDIDATES:
    kd = D[:, ms - 1]
    elbows[ms] = {r: elbow_eps(kd, r) for r in ("log10", "linear", "trunc99")}
    log(f"min_samples={ms:>2}: " + " | ".join(
        f"{r}: eps={e:.4f} (at {100 * q:.2f}th pct)" for r, (e, q) in elbows[ms].items()))
# Robustness of each elbow rule (label-free): trim the extreme tail / halve the sample.
kd10 = np.sort(D[:, 9])
D_half = kdist_matrix(X_dbscan[: N_WORK // 2], 10)[:, 9]
robust = {}
for r in ("log10", "linear", "trunc99"):
    v = {"full": elbow_eps(kd10, r)[0],
         "trim_top_0.1pct": elbow_eps(kd10[: int(0.999 * len(kd10))], r)[0],
         "trim_top_1pct": elbow_eps(kd10[: int(0.99 * len(kd10))], r)[0],
         "half_sample": elbow_eps(D_half, r)[0]}
    v["max_over_min"] = max(v.values()) / min(v.values())
    robust[r] = v
    log(f"  rule {r:<8} eps under perturbation: " +
        ", ".join(f"{k}={x:.3f}" for k, x in v.items() if k != "max_over_min") +
        f"  -> max/min {v['max_over_min']:.2f}")
ELBOW_RULE = "log10"
PRIMARY_MS = min(MIN_SAMPLES_CANDIDATES, key=lambda m: abs(m - np.log(N_WORK)))
PRIMARY_EPS = elbows[PRIMARY_MS][ELBOW_RULE][0]
log(f"Primary: min_samples={PRIMARY_MS} (candidate closest to ln(n)={np.log(N_WORK):.2f}), "
    f"eps={PRIMARY_EPS:.4f} ({ELBOW_RULE}-Kneedle)")

# =====================================================================
# 3b. PRIMARY DBSCAN — 70-dimensional standardized space
# =====================================================================
pair_counts = count_pairs(X_dbscan, sorted({PRIMARY_EPS * m for m in EPS_MULTIPLIERS} |
                                           {elbows[ms][ELBOW_RULE][0] for ms in MIN_SAMPLES_CANDIDATES}))
prim_pairs = pair_counts[float(PRIMARY_EPS)]
assert prim_pairs["est_gb"] <= MEMORY_BUDGET_GB, prim_pairs
log(f"Primary neighbour pairs: {prim_pairs['pairs']:,} (est. {prim_pairs['est_gb']:.1f} GB)")
primary_labels, primary_run = run_dbscan("primary_70d", X_dbscan, PRIMARY_EPS, PRIMARY_MS)
primary_summary = cluster_summary(primary_labels, PRIMARY_MS)
log(f"Primary DBSCAN: {primary_summary['n_clusters']} clusters, noise "
    f"{primary_summary['noise_pct']:.2f}%, runtime {primary_run['runtime_s']:.1f}s")

# =====================================================================
# 5. PCA (label-free) + DBSCAN per candidate dimension
# =====================================================================
log("\n" + "=" * 72 + "\n5. PCA INVESTIGATION\n" + "=" * 72)
pca_full = PCA(svd_solver="full", random_state=RANDOM_STATE).fit(X_dbscan)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
pca_runs = []
pca_models = {}
for target in PCA_VARIANCE_TARGETS:
    k = int(np.searchsorted(cumvar, target) + 1)
    if any(r["n_components"] == k for r in pca_runs):
        continue
    t = time.time()
    pca_k = PCA(n_components=k, svd_solver="full", random_state=RANDOM_STATE).fit(X_dbscan)
    Xp = pca_k.transform(X_dbscan)
    pca_time = time.time() - t
    assert np.isfinite(Xp).all() and VAULT.assert_not_in(Xp)
    Dk = kdist_matrix(Xp, PRIMARY_MS)[:, PRIMARY_MS - 1]
    eps_k, q_k = elbow_eps(Dk, ELBOW_RULE)
    pc = count_pairs(Xp, [eps_k])[float(eps_k)]
    rec = {"target_variance": target, "n_components": k, "cum_variance": float(cumvar[k - 1]),
           "eps": eps_k, "eps_pct": q_k, "min_samples": PRIMARY_MS, "pca_fit_s": pca_time,
           "pairs": pc["pairs"], "est_gb": pc["est_gb"], "kdist_curve": np.sort(Dk)}
    if pc["est_gb"] > MEMORY_BUDGET_GB:
        rec["status"] = "not run: memory budget"
    else:
        lab, run = run_dbscan(f"pca_{k}", Xp, eps_k, PRIMARY_MS)
        rec.update(status="run", labels=lab, runtime_s=run["runtime_s"], **cluster_summary(lab, PRIMARY_MS))
    pca_runs.append(rec)
    pca_models[k] = pca_k
    log(f"PCA {target:.0%}: k={k} (cum var {cumvar[k - 1]:.4f}) eps={eps_k:.4f} -> "
        f"{rec['status']}" + (f", {rec['n_clusters']} clusters, noise {rec['noise_pct']:.2f}%, "
                              f"{rec['runtime_s']:.1f}s" if rec["status"] == "run" else ""))
    del Xp
    gc.collect()

# =====================================================================
# 6. PARAMETER SENSITIVITY (descriptive)
# =====================================================================
log("\n" + "=" * 72 + "\n6. PARAMETER SENSITIVITY\n" + "=" * 72)
sens_runs = []
grid = [(ms, m, PRIMARY_EPS * m, "multiplier") for ms in MIN_SAMPLES_CANDIDATES for m in EPS_MULTIPLIERS]
grid += [(ms, None, elbows[ms][ELBOW_RULE][0], "own elbow") for ms in MIN_SAMPLES_CANDIDATES
         if ms != PRIMARY_MS]
for ms, mult, eps, kind in grid:
    rec = {"min_samples": ms, "eps_multiplier": mult, "eps": float(eps), "kind": kind,
           "pairs": pair_counts[float(eps)]["pairs"], "est_gb": pair_counts[float(eps)]["est_gb"]}
    if ms == PRIMARY_MS and mult == 1.0:
        rec.update(status="run (primary)", labels=primary_labels,
                   runtime_s=primary_run["runtime_s"], **primary_summary)
    elif rec["est_gb"] > MEMORY_BUDGET_GB:
        rec["status"] = "not run: memory budget"
    else:
        lab, run = run_dbscan(f"sens_ms{ms}_eps{eps:.4f}", X_dbscan, eps, ms)
        rec.update(status="run", labels=lab, runtime_s=run["runtime_s"], **cluster_summary(lab, ms))
    if "labels" in rec:
        a, b = rec["labels"] == -1, primary_labels == -1
        rec["noise_jaccard_vs_primary"] = float((a & b).sum() / max((a | b).sum(), 1))
    sens_runs.append(rec)
    log(f"ms={ms:>2} eps={eps:.4f} ({'x' + str(mult) if mult else kind}): {rec['status']}" +
        (f" | clusters {rec['n_clusters']}, noise {rec['noise_pct']:.2f}%, "
         f"Jaccard(noise vs primary) {rec['noise_jaccard_vs_primary']:.3f}, {rec['runtime_s']:.1f}s"
         if "labels" in rec else f" (est. {rec['est_gb']:.1f} GB)"))

# =====================================================================
# 4. EVALUATION — labels attached ONLY NOW
# =====================================================================
log("\n" + "=" * 72 + "\n4. EVALUATION (labels revealed after all clustering)\n" + "=" * 72)
n_clustering_runs = len(REGISTRY.runs)
y_dbscan, cat_dbscan = VAULT.reveal(REGISTRY)
assert VAULT.reveal_time > REGISTRY.last_end
y_pred_primary = noise_to_pred(primary_labels)
assert np.array_equal(y_pred_primary[primary_labels == -1], np.ones((primary_labels == -1).sum()))
assert np.array_equal(y_pred_primary[primary_labels != -1], np.zeros((primary_labels != -1).sum()))
prim_metrics = binary_metrics(y_dbscan, y_pred_primary)
for k_, v_ in prim_metrics.items():
    log(f"  {k_}: {v_:,}" if isinstance(v_, int) else f"  {k_}: {v_:.4f}")
for rec in pca_runs + sens_runs:
    if "labels" in rec:
        rec["metrics"] = binary_metrics(y_dbscan, noise_to_pred(rec["labels"]))

# Noise rate inside each cluster-membership type (descriptive)
benign_noise_rate = float(y_pred_primary[y_dbscan == 0].mean())

# =====================================================================
# 7. ATTACK-CATEGORY ANALYSIS
# =====================================================================
cat_rows = []
for cat in ATTACK_CATEGORIES:
    m = cat_dbscan == cat
    n = int(m.sum())
    k_ = int(y_pred_primary[m].sum()) if n else 0
    lo, hi = wilson_ci(k_, n)
    cat_rows.append({"category": cat, "n_in_sample": n, "n_population": int(pop_counts.get(cat, 0)),
                     "flagged_anomalous": k_, "pct_flagged_anomalous": 100 * k_ / n if n else float("nan"),
                     "ci95_low": 100 * lo if n else float("nan"), "ci95_high": 100 * hi if n else float("nan"),
                     "status": ("not present in sample" if n == 0 else
                                "statistically unstable (n < 30)" if n < UNSTABLE_N else "ok")})
cat_df = pd.DataFrame(cat_rows).sort_values("n_in_sample", ascending=False).reset_index(drop=True)

# =====================================================================
# 8. VISUALIZATIONS
# =====================================================================
x_frac = np.linspace(0, 100, N_WORK)
fig, ax = plt.subplots(figsize=(8.5, 5))
for ms, col in zip(MIN_SAMPLES_CANDIDATES, (C_3, C_BENIGN, C_ATTACK)):
    kd = np.sort(D[:, ms - 1])
    ax.plot(x_frac, np.maximum(kd, 1e-6), color=col, linewidth=2, label=f"min_samples = {ms}")
    e, q = elbows[ms][ELBOW_RULE]
    ax.scatter([100 * q], [e], s=64, color=col, edgecolor="#fcfcfb", linewidth=2, zorder=3)
for r, mk in (("linear", "^"), ("trunc99", "v")):
    e, q = elbows[PRIMARY_MS][r]
    ax.scatter([100 * q], [e], s=56, marker=mk, facecolor="none", edgecolor=INK_2, linewidth=1.4,
               zorder=3, label=f"{r} Kneedle (min_samples={PRIMARY_MS}): eps={e:.2f}")
ax.axhline(PRIMARY_EPS, color=INK, linestyle="--", linewidth=1)
ax.text(1, PRIMARY_EPS * 1.15, f"selected eps = {PRIMARY_EPS:.3f} (log10-Kneedle, min_samples={PRIMARY_MS})",
        color=INK, fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("Points sorted by k-distance (percentile)")
ax.set_ylabel("Distance to min_samples-th neighbour (incl. self), log scale")
ax.set_title(f"Sorted k-distance curves — standardized 70-D space, n = {N_WORK:,}", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.savefig(PLOT_DIR / "01_k_distance.png")
plt.close(fig)

sizes = np.sort(np.bincount(primary_labels[primary_labels >= 0]))[::-1]
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(np.arange(1, len(sizes) + 1), sizes, color=C_BENIGN, linewidth=2, marker="o", markersize=3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Cluster rank (largest first, log)")
ax.set_ylabel("Cluster size (rows, log)")
ax.set_title(f"Primary DBSCAN cluster-size distribution — {len(sizes)} clusters, "
             f"noise excluded ({primary_summary['n_noise']:,} noise rows)", fontsize=10)
for i in range(min(3, len(sizes))):
    ax.annotate(f"{sizes[i]:,}", (i + 1, sizes[i]), xytext=(6, 2), textcoords="offset points",
                fontsize=8, color=INK)
fig.savefig(PLOT_DIR / "02_cluster_size_distribution.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
cnt = [int((y_pred_primary == 0).sum()), int((y_pred_primary == 1).sum())]
bars = axes[0].bar(["In a cluster (normal, 0)", "Noise −1 (anomaly, 1)"], cnt,
                   color=[C_BENIGN, C_ATTACK], width=0.55)
for b, c in zip(bars, cnt):
    axes[0].text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({c / N_WORK:.1%})", ha="center", va="bottom")
axes[0].set_ylim(0, max(cnt) * 1.2)
axes[0].set_ylabel("Rows in working sample")
axes[0].set_title("Noise vs non-noise (primary DBSCAN)", fontsize=11)
xx, w = np.arange(2), 0.38
for j, (cls, name, col) in enumerate(((0, "True BENIGN", C_BENIGN), (1, "True ATTACK", C_ATTACK))):
    vals = [int(((y_dbscan == cls) & (y_pred_primary == p)).sum()) for p in (0, 1)]
    bb = axes[1].bar(xx + (j - 0.5) * w, vals, w - 0.02, color=col, label=name)
    for b, v in zip(bb, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
axes[1].set_xticks(xx, ["In a cluster (0)", "Noise (1)"])
axes[1].set_title("Split by ground truth (attached after clustering)", fontsize=11)
axes[1].legend(frameon=False)
fig.savefig(PLOT_DIR / "03_noise_vs_non_noise.png")
plt.close(fig)

cm = np.array([[prim_metrics["TN"], prim_metrics["FP"]], [prim_metrics["FN"], prim_metrics["TP"]]])
rp = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.4, 4.4))
ax.imshow(rp, cmap="Blues", vmin=0, vmax=1)
ax.grid(False)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}\n({rp[i, j]:.1%} of row)", ha="center", va="center",
                color="white" if rp[i, j] > 0.55 else INK)
ax.set_xticks([0, 1], ["Clustered (0)", "Noise (1)"])
ax.set_yticks([0, 1], ["BENIGN (0)", "ATTACK (1)"])
ax.set_title(f"Primary DBSCAN — confusion matrix (transductive, n = {N_WORK:,})", fontsize=10)
fig.savefig(PLOT_DIR / "04_confusion_matrix.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(np.arange(1, len(cumvar) + 1), cumvar, color=C_BENIGN, linewidth=2, marker="o", markersize=3)
for rec, col in zip(pca_runs, (C_3, C_ATTACK, INK_2)):
    ax.axvline(rec["n_components"], color=col, linestyle="--", linewidth=1)
    ax.text(rec["n_components"] + 0.5, 0.55 + 0.08 * pca_runs.index(rec),
            f"{rec['target_variance']:.0%} → k={rec['n_components']}", color=INK, fontsize=9)
ax.set_xlabel("Number of principal components")
ax.set_ylabel("Cumulative explained variance")
ax.set_ylim(0, 1.02)
ax.set_title("PCA cumulative explained variance (fitted on standardized working sample, no labels)",
             fontsize=10)
fig.savefig(PLOT_DIR / "05_pca_explained_variance.png")
plt.close(fig)

cfg_names = [f"70-D standardized\n(eps={PRIMARY_EPS:.2f})"] + [
    f"PCA k={r['n_components']}\n({r['cum_variance']:.1%} var, eps={r['eps']:.2f})" for r in pca_runs]
cfg_noise = [primary_summary["noise_pct"]] + [r.get("noise_pct", np.nan) for r in pca_runs]
fig, ax = plt.subplots(figsize=(8, 4.4))
bars = ax.bar(cfg_names, cfg_noise, color=[C_BENIGN] + [C_3] * len(pca_runs), width=0.55)
for b, v in zip(bars, cfg_noise):
    ax.text(b.get_x() + b.get_width() / 2, (0 if np.isnan(v) else v),
            "not run" if np.isnan(v) else f"{v:.2f}%", ha="center", va="bottom")
ax.set_ylabel("Rows labelled noise (%)")
ax.set_title(f"Noise share by feature space (min_samples={PRIMARY_MS}; eps by log10-Kneedle in each space)",
             fontsize=10)
fig.savefig(PLOT_DIR / "06_noise_pct_by_configuration.png")
plt.close(fig)

plot_cats = cat_df[cat_df["n_in_sample"] >= UNSTABLE_N].sort_values("pct_flagged_anomalous")
fig, ax = plt.subplots(figsize=(8, 0.5 * len(plot_cats) + 1.6))
yy = np.arange(len(plot_cats))
ax.barh(yy, plot_cats["pct_flagged_anomalous"], color=C_ATTACK, height=0.6)
ax.errorbar(plot_cats["pct_flagged_anomalous"], yy,
            xerr=[plot_cats["pct_flagged_anomalous"] - plot_cats["ci95_low"],
                  plot_cats["ci95_high"] - plot_cats["pct_flagged_anomalous"]],
            fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
ax.set_yticks(yy, [f"{c}  (n={n:,})" for c, n in zip(plot_cats["category"], plot_cats["n_in_sample"])])
ax.axvline(100 * benign_noise_rate, color=C_BENIGN, linestyle="--", linewidth=1.2)
ax.text(100 * benign_noise_rate, len(plot_cats) - 0.4, f" BENIGN flagged: {100 * benign_noise_rate:.1f}%",
        color=INK_2, fontsize=8)
ax.set_xlim(0, 100)
ax.set_xlabel("Percentage flagged as anomalous (DBSCAN noise), 95% Wilson CI")
ax.set_title(f"Percentage of each category flagged as anomalous — categories with n ≥ {UNSTABLE_N}",
             fontsize=10)
fig.savefig(PLOT_DIR / "07_attack_category_flagged.png")
plt.close(fig)

heat = np.full((len(MIN_SAMPLES_CANDIDATES), len(EPS_MULTIPLIERS)), np.nan)
for r in sens_runs:
    if r["kind"] == "multiplier" and "labels" in r:
        heat[MIN_SAMPLES_CANDIDATES.index(r["min_samples"]), EPS_MULTIPLIERS.index(r["eps_multiplier"])] = r["noise_pct"]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.imshow(np.nan_to_num(heat, nan=0), cmap="Blues", aspect="auto",
          vmin=0, vmax=np.nanmax(heat) if np.isfinite(heat).any() else 1)
ax.grid(False)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat[i, j]
        ax.text(j, i, "not run\n(memory)" if np.isnan(v) else f"{v:.2f}%", ha="center", va="center",
                fontsize=9, color=INK if np.isnan(v) or v < 0.6 * np.nanmax(heat) else "white")
ax.set_xticks(range(len(EPS_MULTIPLIERS)),
              [f"{m}×\n{PRIMARY_EPS * m:.2f}" for m in EPS_MULTIPLIERS])
ax.set_yticks(range(len(MIN_SAMPLES_CANDIDATES)), [f"min_samples={m}" for m in MIN_SAMPLES_CANDIDATES])
ax.set_xlabel("eps (multiple of selected elbow eps)")
ax.set_title("Sensitivity: noise share (%) across min_samples × eps (70-D space)", fontsize=10)
fig.savefig(PLOT_DIR / "08_sensitivity_noise_pct.png")
plt.close(fig)

# =====================================================================
# 10. ARTIFACTS
# =====================================================================
sample_ids.to_csv(OUT["sample_ids"], index=False)
assign = {"primary_70d": primary_labels, "raw_concat_index": work_ids["raw_concat_index"].to_numpy()}
for r in pca_runs:
    if "labels" in r:
        assign[f"pca_k{r['n_components']}"] = r["labels"]
for r in sens_runs:
    if "labels" in r and r["status"] != "run (primary)":
        assign[f"sens_ms{r['min_samples']}_eps{r['eps']:.4f}"] = r["labels"]
np.savez_compressed(OUT["assignments"], **assign)
joblib.dump({"imputer": imputer, "scaler": scaler, "feature_columns": feature_columns,
             "pca_full": pca_full, "pca_by_k": pca_models}, OUT["preprocessing"])

config = {
    "phase": "2B", "method": "DBSCAN", "transductive": True,
    "transductive_note": "Labels (cluster/noise) exist only for the rows clustered; sklearn DBSCAN has no "
                         "predict() and no out-of-sample generalisation is claimed.",
    "sample": {"population_rows": EXPECTED_DEDUP_ROWS, "sample_n": SAMPLE_N,
               "working_n": N_WORK, "fallback_used": use_fallback,
               "fallback_reason": (f"100k rows at the selected eps need ~{feas100['est_gb']:.1f} GB of "
                                   f"neighbourhood storage (> {MEMORY_BUDGET_GB} GB budget)") if use_fallback else None,
               "method": "pandas DataFrame.sample(n, random_state=42) on the deduplicated data; uniform, "
                         "no stratification; 50k fallback = first 50k rows of the same draw",
               "random_state": RANDOM_STATE},
    "features": {"n": N_FEATURES, "dropped_zero_variance": EXPECTED_ZERO_VAR,
                 "destination_port_retained": True, "columns": feature_columns},
    "preprocessing": {"inf": "±inf -> NaN", "imputation": "median (fit on working sample, no labels)",
                      "scaling": "StandardScaler (fit on working sample, no labels)"},
    "dbscan": {"metric": "euclidean", "algorithm": "brute", "eps": PRIMARY_EPS,
               "min_samples": PRIMARY_MS, "elbow_rule": "Kneedle on log10 sorted k-distance",
               "min_samples_rule": "candidate in {5,10,20} closest to ln(n)",
               "anomaly_mapping": {"-1": 1, "cluster id >= 0": 0}},
    "pca": [{"target_variance": r["target_variance"], "n_components": r["n_components"],
             "eps": r["eps"], "min_samples": r["min_samples"], "status": r["status"]} for r in pca_runs],
    "sensitivity": {"min_samples": MIN_SAMPLES_CANDIDATES, "eps_multipliers": EPS_MULTIPLIERS},
    "memory_model": {"bytes_per_pair": BYTES_PER_PAIR, "budget_gb": MEMORY_BUDGET_GB},
    "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                 "numpy": np.__version__, "pandas": pd.__version__},
}
with open(OUT["config"], "w") as fh:
    json.dump(config, fh, indent=2, default=float)


def strip(r):
    return {k: v for k, v in r.items() if k not in ("labels", "kdist_curve")}


metrics_out = {
    "transductive": True,
    "sample_composition_100k": comp100, "sample_composition_working": comp_work,
    "population_attack_prevalence": pop_attack / EXPECTED_DEDUP_ROWS,
    "feasibility_100k": {"eps": eps100, **feas100, "kdist_percentiles_50_90_95_99_99.9_100": kdist100_summary,
                         "kdist_time_s": kdist_time_100},
    "elbows": {ms: {r: {"eps": e, "percentile": q} for r, (e, q) in v.items()} for ms, v in elbows.items()},
    "elbow_robustness": robust,
    "primary": {"eps": PRIMARY_EPS, "min_samples": PRIMARY_MS, **primary_summary,
                "runtime_s": primary_run["runtime_s"], "pairs": prim_pairs["pairs"],
                "metrics": prim_metrics},
    "pca_cumulative_variance": cumvar.tolist(),
    "pca_runs": [strip(r) for r in pca_runs],
    "sensitivity_runs": [strip(r) for r in sens_runs],
    "attack_categories": cat_df.to_dict(orient="records"),
    "benign_flagged_pct": 100 * benign_noise_rate,
}

# =====================================================================
# 9. SANITY CHECKS
# =====================================================================
p2a_after, bad_after = verify_phase2a()
sanity = {
    "1. sample size 100,000 (working sample 100,000 or documented 50,000 fallback)":
        len(sample_ids) == SAMPLE_N and N_WORK in (SAMPLE_N, FALLBACK_N)
        and (N_WORK == SAMPLE_N or use_fallback),
    "2. random_state == 42": RANDOM_STATE == 42,
    "3. exactly 70 final features": X_dbscan.shape[1] == 70 == len(feature_columns),
    "4. Destination Port present": "Destination Port" in feature_columns,
    "5. labels not passed to preprocessing (feature-only fit; leakage check passed)":
        not hasattr(scaler, "classes_") and "Label" not in feature_columns,
    "6. labels not passed to PCA": pca_full.n_features_in_ == 70,
    "7. labels not passed to DBSCAN (all runs before label reveal)":
        all(r["end"] < VAULT.reveal_time for r in REGISTRY.runs),
    "8. no NaN in DBSCAN input": not np.isnan(X_dbscan).any(),
    "9. no inf in DBSCAN input": np.isfinite(X_dbscan).all(),
    "10. cluster -1 -> anomaly 1": bool((y_pred_primary[primary_labels == -1] == 1).all()),
    "11. all other cluster ids -> 0": bool((y_pred_primary[primary_labels != -1] == 0).all()),
    "12. evaluation only after clustering": VAULT.reveal_time > REGISTRY.last_end
        and len(REGISTRY.runs) == n_clustering_runs,
    "13. no Phase 2A artifact modified": not bad_after,
    "14. DBSCAN marked transductive": config["transductive"] is True,
}
for k_, v_ in sanity.items():
    assert v_, f"SANITY FAILED: {k_}"
peak_ws, peak_commit = mem_gb()
total_time = time.time() - T0
metrics_out.update(peak_ws_gb=peak_ws, peak_commit_gb=peak_commit, total_time_s=total_time,
                   kdist_time_s=kdist_time, sanity={k: True for k in sanity})
with open(OUT["metrics"], "w") as fh:
    json.dump(metrics_out, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))

# =====================================================================
# FINAL REPORT
# =====================================================================
pm = prim_metrics
L = []
A = L.append
A("=" * 78)
A("PHASE 2B — DBSCAN (TRANSDUCTIVE DENSITY-BASED ANOMALY DETECTION)")
A("=" * 78)
A("\n1. OBJECTIVE")
A("  Assess whether density-based clustering separates attack traffic from benign traffic in")
A("  CIC-IDS2017 by labelling low-density points (DBSCAN noise, label -1) as anomalies, using")
A("  only unlabelled flow features for every modelling decision.")
A("\n2. WHY DBSCAN")
A("  DBSCAN needs no number of clusters, finds arbitrarily shaped dense regions, and has an")
A("  explicit 'noise' concept: points that are neither core points nor within eps of one.")
A("  Under the hypothesis that attack flows lie in sparse regions of feature space, noise is")
A("  a natural binary anomaly indicator.")
A("\n3. HOW DBSCAN DIFFERS METHODOLOGICALLY FROM ISOLATION FOREST")
A("  * TRANSDUCTIVE: DBSCAN labels only the rows it is run on. sklearn's DBSCAN has no")
A("    predict(); there is no trained model to apply to unseen traffic, so these results do")
A("    not demonstrate out-of-sample generalisation.")
A("  * Unsupervised over a mixed sample: DBSCAN clusters a uniform random sample that contains")
A("    both benign and attack rows (labels hidden), rather than learning from benign-only data.")
A("  * Output is binary (noise vs clustered). No continuous anomaly score is produced, so")
A("    ROC-AUC is not reported; no score is manufactured from cluster ids.")
A("  * Distance-based: results depend on the scale of every feature, so standardisation is")
A("    essential here (a property of DBSCAN's Euclidean neighbourhoods, not a claim about")
A("    Isolation Forest).")
A("  * Cost grows with the number of neighbour pairs (roughly quadratic in n for dense data),")
A("    which forces sampling.")
A("\n4. SAMPLE CONSTRUCTION")
A(f"  Population: {EXPECTED_DEDUP_ROWS:,} deduplicated rows; attack prevalence "
  f"{pop_attack / EXPECTED_DEDUP_ROWS:.2%}.")
A(f"  Uniform random sample: n = {SAMPLE_N:,}, random_state = {RANDOM_STATE} "
  "(DataFrame.sample; no first-N, per-file, chronological, balanced or label-based sampling).")
A(f"  100k sample: BENIGN {comp100.get('BENIGN', 0):,} | ATTACK {n_att100:,} | prevalence "
  f"{n_att100 / SAMPLE_N:.2%}")
if use_fallback:
    A(f"  DOCUMENTED FALLBACK -> working sample n = {N_WORK:,}: at the selected eps the 100k sample")
    A(f"  needs {feas100['pairs']:,} stored neighbour pairs (~{feas100['est_gb']:.1f} GB in sklearn's")
    A(f"  DBSCAN, measured ~{BYTES_PER_PAIR} B/pair) > {MEMORY_BUDGET_GB:.0f} GB budget. The 50k rows are the first")
    A("  50,000 rows of the same seeded uniform draw (identical to sample(n=50000, random_state=42)),")
    A("  so the working sample is itself a uniform random sample. The decision used no labels.")
A(f"  Working sample: BENIGN {comp_work.get('BENIGN', 0):,} | ATTACK {n_att_work:,} | prevalence "
  f"{n_att_work / N_WORK:.2%}")
A(f"  {'Category':<28}{'population':>12}{'100k sample':>13}{'working':>10}")
for cat in ["BENIGN"] + ATTACK_CATEGORIES:
    A(f"  {cat:<28}{int(pop_counts.get(cat, 0)):>12,}{comp100.get(cat, 0):>13,}{comp_work.get(cat, 0):>10,}")
A("  No rare-attack rows were added. y_dbscan was kept in a sealed container: only these")
A("  aggregate counts were read before clustering.")
A("\n5. PREPROCESSING")
A(f"  Features: {N_FEATURES} (78 numeric - 8 confirmed zero-variance columns); Destination Port retained.")
A("  ±inf -> NaN; median imputation; StandardScaler. Imputer and scaler were fitted on the")
A("  working sample's features only (no labels). The frozen Phase 2A imputer/scaler were NOT")
A("  reused: they were fitted on BENIGN-only training rows, a label-based selection that would")
A("  let labels shape DBSCAN's distance metric.")
A(f"  Rows with NaN before imputation: {nan_rows}. Columns constant within the working sample: "
  f"{const_cols or 'none'}.")
A(f"  Checks: no NaN, no inf, shape {X_dbscan.shape}, finite standardized matrix: PASS.")
A("\n6. k-DISTANCE / eps SELECTION (unsupervised)")
A("  For each min_samples m, the distance from every point to its m-th nearest neighbour")
A("  (counting itself, matching sklearn's core-point definition) was sorted and the elbow located")
A("  with Kneedle. Three variants were computed; the k-distances span ~5 orders of magnitude:")
for ms in MIN_SAMPLES_CANDIDATES:
    A(f"    min_samples={ms:>2}: " + " | ".join(f"{r} eps={e:.3f} @{100 * q:.1f}th pct"
                                             for r, (e, q) in elbows[ms].items()))
A("  Stability of each rule under label-free perturbations (min_samples=10):")
for r, v in robust.items():
    A(f"    {r:<8} full={v['full']:.3f} trim0.1%={v['trim_top_0.1pct']:.3f} trim1%={v['trim_top_1pct']:.3f} "
      f"half-sample={v['half_sample']:.3f}  max/min={v['max_over_min']:.2f}")
A("  Selected rule: Kneedle on the log10 k-distance curve. It is invariant to rescaling of")
A("  distances and is the most stable rule above; the linear rule is set by a handful of extreme")
A("  points (its eps moves >10x when the top 0.1-1% of points is trimmed), and the truncated rule")
A("  depends on an arbitrary truncation level. The transition region spans roughly")
A(f"  eps {elbows[PRIMARY_MS]['trunc99'][0]:.2f} - {elbows[PRIMARY_MS]['linear'][0]:.2f}; the chosen eps lies inside it.")
A("  eps was NOT chosen using precision, recall, F1, accuracy or any label.")
A("\n7. min_samples SELECTION")
A(f"  Candidates {MIN_SAMPLES_CANDIDATES} were fixed in advance. Primary min_samples = {PRIMARY_MS}: the")
A(f"  candidate closest to ln(n) = {np.log(N_WORK):.2f}, a common data-size heuristic. The 2*dim")
A("  heuristic (140 for 70 features) was not used: it implies a much larger eps whose neighbour")
A("  storage is infeasible here; 5 and 20 are reported in the sensitivity analysis.")
A(f"\n8. BASELINE DBSCAN RESULTS — standardized 70-D space, eps={PRIMARY_EPS:.4f}, "
  f"min_samples={PRIMARY_MS}, Euclidean")
A(f"  Clusters (excl. noise): {primary_summary['n_clusters']:,}")
A(f"  Noise points: {primary_summary['n_noise']:,} ({primary_summary['noise_pct']:.2f}%)")
A(f"  Largest clusters: {primary_summary['largest_cluster_sizes']}")
A(f"  Largest cluster share: {primary_summary['largest_cluster_share_pct']:.1f}% of rows; median cluster "
  f"size {primary_summary['median_cluster_size']:.0f}")
A(f"  Clusters smaller than min_samples: {primary_summary['clusters_smaller_than_min_samples']}; "
  f"clusters with < 50 rows: {primary_summary['clusters_size_lt_50']}")
A(f"  Neighbour pairs stored: {prim_pairs['pairs']:,}; runtime {primary_run['runtime_s']:.1f} s")
A("  Evaluation (labels attached AFTER all clustering; noise = predicted anomaly):")
A("                     Clustered (0)   Noise (1)")
A(f"    BENIGN (0)      {pm['TN']:>13,} {pm['FP']:>11,}")
A(f"    ATTACK (1)      {pm['FN']:>13,} {pm['TP']:>11,}")
A(f"  TP={pm['TP']:,} TN={pm['TN']:,} FP={pm['FP']:,} FN={pm['FN']:,}")
A(f"  Accuracy {pm['accuracy']:.4f} | Precision {pm['precision']:.4f} | Recall {pm['recall']:.4f} | "
  f"F1 {pm['f1']:.4f} | FPR {pm['fpr']:.4f} | FNR {pm['fnr']:.4f}")
A("  ROC-AUC: not reported (no continuous anomaly score).")
A(f"  These metrics evaluate the noise assignments on the {N_WORK:,} sampled rows themselves; they")
A("  do not demonstrate out-of-sample generalisation.")
A("\n9. PCA INVESTIGATION (same working rows; PCA fitted without labels)")
A("  Cumulative explained variance: " + ", ".join(
    f"k={k}: {cumvar[k - 1]:.3f}" for k in (1, 2, 5, 10, 15, 20, 25, 30, 40, 50) if k <= len(cumvar)))
A("  eps re-derived in each space with the same rule (log10-Kneedle, min_samples="
  f"{PRIMARY_MS}); truncating components shrinks distances, so eps is not transferable.")
A(f"  {'space':<24}{'k':>4}{'cum var':>9}{'eps':>8}{'clusters':>10}{'noise %':>9}{'Prec':>8}"
  f"{'Recall':>8}{'F1':>8}{'FPR':>8}{'FNR':>8}{'time s':>8}")
A(f"  {'standardized 70-D':<24}{70:>4}{1.0:>9.3f}{PRIMARY_EPS:>8.3f}{primary_summary['n_clusters']:>10,}"
  f"{primary_summary['noise_pct']:>9.2f}{pm['precision']:>8.4f}{pm['recall']:>8.4f}{pm['f1']:>8.4f}"
  f"{pm['fpr']:>8.4f}{pm['fnr']:>8.4f}{primary_run['runtime_s']:>8.1f}")
for r in pca_runs:
    if "labels" in r:
        m = r["metrics"]
        A(f"  {'PCA ' + format(r['target_variance'], '.0%'):<24}{r['n_components']:>4}{r['cum_variance']:>9.3f}"
          f"{r['eps']:>8.3f}{r['n_clusters']:>10,}{r['noise_pct']:>9.2f}{m['precision']:>8.4f}"
          f"{m['recall']:>8.4f}{m['f1']:>8.4f}{m['fpr']:>8.4f}{m['fnr']:>8.4f}{r['runtime_s']:>8.1f}")
    else:
        A(f"  {'PCA ' + format(r['target_variance'], '.0%'):<24}{r['n_components']:>4}{r['cum_variance']:>9.3f}"
          f"{r['eps']:>8.3f}   {r['status']}")
A("  The highest-variance configuration is not assumed to be best; PCA is reported only to show")
A("  how dimensionality reduction changes local density and cost.")
A("\n10. PARAMETER SENSITIVITY (descriptive; labels not used to choose anything)")
A(f"  {'min_s':>5}{'eps':>8}{'setting':>12}{'clusters':>10}{'noise %':>9}{'Jacc.':>7}{'Prec':>8}"
  f"{'Recall':>8}{'F1':>8}{'FPR':>8}{'time s':>8}")
for r in sens_runs:
    tag = (f"x{r['eps_multiplier']}" if r["eps_multiplier"] else "own elbow")
    if "labels" in r:
        m = r["metrics"]
        A(f"  {r['min_samples']:>5}{r['eps']:>8.3f}{tag:>12}{r['n_clusters']:>10,}{r['noise_pct']:>9.2f}"
          f"{r['noise_jaccard_vs_primary']:>7.3f}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}"
          f"{m['fpr']:>8.4f}{r['runtime_s']:>8.1f}{'  <- primary' if r['status'] == 'run (primary)' else ''}")
    else:
        A(f"  {r['min_samples']:>5}{r['eps']:>8.3f}{tag:>12}   {r['status']} (est. {r['est_gb']:.1f} GB)")
ran = [r for r in sens_runs if "labels" in r]
A(f"  Noise share across the grid: {min(r['noise_pct'] for r in ran):.2f}% - "
  f"{max(r['noise_pct'] for r in ran):.2f}%; Jaccard overlap of noise sets with the primary run: "
  f"{min(r['noise_jaccard_vs_primary'] for r in ran):.3f} - "
  f"{max(r['noise_jaccard_vs_primary'] for r in ran if r['status'] != 'run (primary)'):.3f}.")
A("\n11. ATTACK-CATEGORY ANALYSIS — percentage flagged as anomalous (DBSCAN noise)")
A("  (Not attack classification: DBSCAN does not assign attack types.)")
A(f"  {'Category':<28}{'n':>8}{'flagged':>9}{'% flagged':>11}   95% CI          note")
for r in cat_df.itertuples():
    if r.n_in_sample == 0:
        A(f"  {r.category:<28}{0:>8}{'-':>9}{'-':>11}   {'-':<15} not present in sample")
    else:
        A(f"  {r.category:<28}{r.n_in_sample:>8,}{r.flagged_anomalous:>9,}{r.pct_flagged_anomalous:>10.1f}%"
          f"   [{r.ci95_low:5.1f}, {r.ci95_high:5.1f}]  {'' if r.status == 'ok' else r.status}")
A(f"  Reference: BENIGN rows flagged as noise: {100 * benign_noise_rate:.2f}%")
A("\n12. COMPUTATIONAL COST")
A(f"  Working sample {N_WORK:,} x {N_FEATURES}; brute-force Euclidean neighbourhoods (tree indexes are")
A("  ineffective in 70 dimensions); n_jobs=-1 on 24 logical CPUs.")
A(f"  k-NN for k-distance (k={max(MIN_SAMPLES_CANDIDATES)}): {kdist_time:.1f} s; 100k feasibility k-NN: {kdist_time_100:.1f} s")
A(f"  Primary DBSCAN: {primary_run['runtime_s']:.1f} s, {prim_pairs['pairs']:,} neighbour pairs "
  f"(~{prim_pairs['est_gb']:.1f} GB est.)")
A(f"  Clustering runs performed: {n_clustering_runs}; total script time {total_time / 60:.1f} min")
A(f"  Process peak working set {peak_ws:.1f} GB; peak committed memory {peak_commit:.1f} GB")
A("\n13. LIMITATIONS")
A("  * DBSCAN is TRANSDUCTIVE: noise labels exist only for the clustered rows; no predict() and")
A("    no evidence of generalisation to unseen traffic.")
A(f"  * The experiment uses a representative uniform sample ({N_WORK:,} rows), not all 2,522,362")
A("    deduplicated observations; neighbourhood storage grows ~quadratically with n.")
A("  * Noise assignment is sensitive to eps and min_samples (Section 10); the elbow itself is")
A("    ill-defined on this heavy-tailed k-distance curve (three reasonable rules disagree).")
A("  * In 70 dimensions Euclidean distances concentrate and are dominated by heavy-tailed")
A("    standardized features; local density is harder to estimate reliably.")
A("  * PCA changes local-density relationships: truncation removes low-variance directions in")
A("    which some flows may be isolated, so noise sets differ across spaces.")
A("  * Rare attack categories (e.g. Heartbleed, Infiltration, SQL Injection) are absent or have")
A("    very few rows in a uniform sample; their percentages are unstable or unavailable.")
A("  * Median imputation and StandardScaler were fitted on the same rows being clustered")
A("    (consistent with the transductive design).")
A("\n14. REPRODUCIBILITY")
A(f"  Script: scripts/phase2b.py | random_state={RANDOM_STATE} | sklearn {sklearn.__version__}, numpy "
  f"{np.__version__}, pandas {pd.__version__}, Python {sys.version.split()[0]}")
A("  Sample identifiers: results/phase2b_sample_ids.csv (raw concatenation index, source file, file row)")
A("  Cluster assignments: results/phase2b_cluster_assignments.npz | preprocessing/PCA objects:")
A("  results/phase2b_preprocessing.pkl | configuration: results/phase2b_config.json")
A("  Hashes: results/phase2b_manifest.json")
A("\nSANITY CHECKS")
for k_ in sanity:
    A(f"  [PASS] {k_}")
A("\nPHASE 2B COMPLETE — awaiting review. (No port ablation, no DBSCAN-vs-IF comparison.)")
report = "\n".join(L)
log("\n" + report)
OUT["report"].write_text(report, encoding="utf-8")

hashed = [OUT[k] for k in ("report", "metrics", "config", "sample_ids", "assignments", "preprocessing")]
hashed += [Path(__file__).resolve()] + sorted(PLOT_DIR.glob("*.png"))
with open(OUT["manifest"], "w") as fh:
    json.dump({"phase": "2B", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "transductive": True,
               "sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in hashed}},
              fh, indent=2)
_, bad_final = verify_phase2a()
assert not bad_final, bad_final
log(f"\nManifest: {OUT['manifest'].relative_to(ROOT)}; Phase 2A artifacts still unchanged.")
