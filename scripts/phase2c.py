"""
PHASE 2C — DESTINATION PORT ABLATION (feature ablation only, NOT model optimisation)

Question: how does removing Destination Port change the behaviour of Isolation Forest
and DBSCAN?

* Destination Port is the ONLY feature removed (70 -> 69).
* Isolation Forest: exact frozen Phase 2A train/test rows; preprocessing refit on BENIGN
  TRAIN (69 features); the frozen primary configuration (200 trees, max_samples="auto",
  contamination="auto", random_state=42). No contamination or tree-count experiments.
* DBSCAN: exact frozen Phase 2B 50,000-row sample; imputer/scaler refit on that sample
  (labels hidden); eps re-selected with the identical unsupervised Phase 2B procedure
  (log10-Kneedle on the k-distance curve, min_samples = candidate closest to ln(n)).
* The frozen WITH-Port results (Phase 2A / 2B) are the reference baselines and are only
  read. Both frozen manifests are hash-verified before and after.
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
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             pairwise_distances_chunked, precision_score, recall_score,
                             roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
PORT = "Destination Port"
EXPECTED_DEDUP_ROWS = 2_522_362
EXPECTED_TRAIN, EXPECTED_TEST = 1_467_538, 1_054_824
EXPECTED_TEST_BENIGN, EXPECTED_TEST_ATTACK = 628_946, 425_878
DBSCAN_N = 50_000
EXPECTED_ZERO_VAR = [
    "Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]
IF_PARAMS = dict(n_estimators=200, max_samples="auto", contamination="auto",
                 random_state=RANDOM_STATE, n_jobs=-1)
MIN_SAMPLES_CANDIDATES = [5, 10, 20]
BYTES_PER_PAIR, MEMORY_BUDGET_GB = 24, 10.0          # same memory model as Phase 2B
IF_SMALL_N, DB_UNSTABLE_N = 100, 30                  # same caveat thresholds as 2A / 2B
# Pre-declared wording thresholds (fixed before any result was seen)
MATERIAL_METRIC_DELTA = 0.01         # |delta| < 0.01 on every metric (1 pp) ...
MATERIAL_SCORE_CORR = 0.99           # ... and Pearson(score) >= 0.99 -> "little material effect"
JACCARD_LARGELY_SAME, JACCARD_SUBSTANTIAL = 0.80, 0.50
CATEGORY_NOTABLE_PP = 5.0
ATTACK_CATEGORIES = [
    "Heartbleed", "Infiltration", "Bot", "PortScan", "DDoS", "DoS Hulk", "DoS GoldenEye",
    "DoS slowloris", "DoS Slowhttptest", "FTP-Patator", "SSH-Patator",
    "Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection",
]

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
RES, MODELS = ROOT / "results", ROOT / "models"
PLOT_DIR = ROOT / "plots" / "phase2c"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
FROZEN_MANIFESTS = {"2A": RES / "phase2a_frozen_manifest.json", "2B": RES / "phase2b_manifest.json"}
OUT = {
    "report": RES / "phase2c_report.txt", "metrics": RES / "phase2c_metrics.json",
    "config": RES / "phase2c_config.json", "features": RES / "phase2c_feature_lists.json",
    "db_assign": RES / "phase2c_dbscan_cluster_assignments.npz",
    "if_model": MODELS / "isolation_forest_no_port.pkl", "if_imputer": MODELS / "imputer_no_port.pkl",
    "if_scaler": MODELS / "scaler_no_port.pkl", "manifest": RES / "phase2c_manifest.json",
}
C_WITH, C_WITHOUT, C_3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": INK_2,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def log(m=""):
    print(m, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_frozen():
    """Every artifact listed in each frozen manifest, plus the manifests themselves."""
    out = {}
    for tag, mp in FROZEN_MANIFESTS.items():
        m = json.load(open(mp))
        bad = [p for p, d in m["sha256"].items() if sha256(ROOT / p) != d]
        out[tag] = {"n": len(m["sha256"]), "modified": bad, "manifest_sha256": sha256(mp),
                    "paths": {str((ROOT / p).resolve()) for p in m["sha256"]}}
    return out


def mem_gb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in (
                "PeakWorkingSetSize", "WorkingSetSize", "a", "b", "c", "d", "PagefileUsage",
                "PeakPagefileUsage")]
    try:
        p = PMC()
        p.cb = ctypes.sizeof(PMC)
        k32, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb)
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


def binary_metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "accuracy": accuracy_score(y, p), "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0),
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "fnr": fn / (fn + tp) if fn + tp else 0.0}


def category_rates(pred, cats, categories):
    out = {}
    for c in categories:
        m = cats == c
        n = int(m.sum())
        k = int(pred[m].sum()) if n else 0
        out[c] = {"n": n, "flagged": k, "pct": 100 * k / n if n else float("nan"),
                  "ci": tuple(100 * v for v in wilson_ci(k, n))}
    return out


# ---- Phase 2B k-distance / Kneedle procedure (identical definitions) ----
def kdist_matrix(X, k_max):
    D, _ = NearestNeighbors(n_neighbors=k_max, algorithm="brute", n_jobs=-1).fit(X).kneighbors(X)
    return D


def kneedle(y_sorted):
    n = len(y_sorted)
    x = np.arange(n) / (n - 1)
    yn = (y_sorted - y_sorted[0]) / (y_sorted[-1] - y_sorted[0])
    i = int(np.argmax(x - yn))
    return i, y_sorted[i], float((x - yn)[i])


def elbow_eps(kd, rule="log10"):
    k = np.sort(kd)
    if rule == "log10":
        i, v, s = kneedle(np.log10(np.maximum(k, 1e-6)))
        return float(10 ** v), i / (len(k) - 1), s
    if rule == "linear":
        i, v, s = kneedle(k)
        return float(v), i / (len(k) - 1), s
    if rule == "trunc99":
        i, v, s = kneedle(k[: int(np.ceil(0.99 * len(k)))])
        return float(v), i / (len(k) - 1), s
    raise ValueError(rule)


def count_pairs(X, eps):
    tot, mx = 0, 0
    for ch in pairwise_distances_chunked(X, X, n_jobs=-1, working_memory=512):
        c = (ch <= eps).sum(axis=1)
        tot += int(c.sum())
        mx = max(mx, int(c.max()))
    return {"pairs": tot, "max_neighbourhood": mx, "est_gb": tot * BYTES_PER_PAIR / 1e9}


class HiddenLabels:
    def __init__(self, labels):
        self._l = np.asarray(labels, dtype=object)
        self.revealed, self.reveal_time = False, None

    def assert_not_in(self, X):
        y = (self._l != "BENIGN").astype(np.float64)
        for j in range(X.shape[1]):
            assert not np.array_equal(X[:, j], y)
        return True

    def reveal(self, clustering_end):
        assert clustering_end is not None
        self.revealed, self.reveal_time = True, time.time()
        assert self.reveal_time > clustering_end
        return (self._l != "BENIGN").astype(np.int8), self._l.copy()


T0 = time.time()
# =====================================================================
# 0. FROZEN HASH VERIFICATION (before)
# =====================================================================
frozen_before = verify_frozen()
assert not frozen_before["2A"]["modified"], frozen_before["2A"]["modified"]   # sanity 1
assert not frozen_before["2B"]["modified"], frozen_before["2B"]["modified"]   # sanity 2
frozen_paths = frozen_before["2A"]["paths"] | frozen_before["2B"]["paths"] | {
    str(p.resolve()) for p in FROZEN_MANIFESTS.values()}
assert not any(str(p.resolve()) in frozen_paths for p in OUT.values()), "Phase 2C path collides with frozen file"
log(f"Frozen verification (before): 2A {frozen_before['2A']['n']} files OK, "
    f"2B {frozen_before['2B']['n']} files OK")

# =====================================================================
# DATA (identical loading / cleaning path to Phase 2A and 2B)
# =====================================================================
csv_files = list(glob.glob("*.csv"))
dfs, sizes = [], []
for f in csv_files:
    d = pd.read_csv(f)
    d.columns = d.columns.str.strip()
    dfs.append(d)
    sizes.append(len(d))
df = pd.concat(dfs, ignore_index=True)
del dfs
gc.collect()
df = df.drop_duplicates()
assert len(df) == EXPECTED_DEDUP_ROWS
labels = (df["Label"].str.replace("�", "-", regex=False)
          .str.replace(r"\s+", " ", regex=True).str.strip())
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
X_all = df.drop(columns=["Label"] + EXPECTED_ZERO_VAR)
del df
gc.collect()

# =====================================================================
# 1. FEATURE ABLATION
# =====================================================================
features_with = list(X_all.columns)
assert len(features_with) == 70 and PORT in features_with                       # sanity 3
frozen_2a_meta = joblib.load(MODELS / "preprocessing_meta.pkl")
frozen_2b_pre = joblib.load(RES / "phase2b_preprocessing.pkl")
assert features_with == frozen_2a_meta["feature_columns"] == frozen_2b_pre["feature_columns"]
features_without = [c for c in features_with if c != PORT]
assert len(features_without) == 69                                               # sanity 4
assert set(features_with) - set(features_without) == {PORT}                      # sanity 5
assert not (set(features_without) - set(features_with))
with open(OUT["features"], "w") as fh:
    json.dump({"with_port": features_with, "without_port": features_without, "removed": [PORT],
               "n_with": 70, "n_without": 69}, fh, indent=2)
log(f"Features: WITH {len(features_with)} | WITHOUT {len(features_without)} | removed: [{PORT}]")

# =====================================================================
# 2. DESTINATION PORT — DESCRIPTIVE ANALYSIS (exploratory; changes nothing)
# =====================================================================
port = X_all[PORT]
is_att = (labels != "BENIGN").to_numpy()
vc = port.value_counts()
port_desc = {
    "n_rows": int(len(port)), "n_unique": int(port.nunique()), "min": int(port.min()),
    "max": int(port.max()), "median": float(port.median()), "n_zero": int((port == 0).sum()),
    "top10": [{"port": int(p), "rows": int(n), "share_pct": 100 * n / len(port)} for p, n in vc.head(10).items()],
    "top5_share_pct": 100 * vc.head(5).sum() / len(port),
    "top10_share_pct": 100 * vc.head(10).sum() / len(port),
    "by_class": {},
}
for name, mask in (("BENIGN", ~is_att), ("ATTACK", is_att)):
    s = port[mask]
    v = s.value_counts()
    port_desc["by_class"][name] = {
        "rows": int(len(s)), "n_unique": int(s.nunique()), "median": float(s.median()),
        "n_zero": int((s == 0).sum()), "top5": [{"port": int(p), "share_pct": 100 * n / len(s)}
                                               for p, n in v.head(5).items()],
        "top5_share_pct": 100 * v.head(5).sum() / len(s),
        "share_ephemeral_ge_49152_pct": 100 * float((s >= 49152).mean()),
        "share_well_known_lt_1024_pct": 100 * float((s < 1024).mean()),
    }
cat_ports = {}
for c in ATTACK_CATEGORIES:
    s = port[(labels == c).to_numpy()]
    v = s.value_counts()
    cat_ports[c] = {"rows": int(len(s)), "n_unique": int(s.nunique()),
                    "top_port": int(v.index[0]), "top_port_share_pct": 100 * v.iloc[0] / len(s)}
log(f"Port: {port_desc['n_unique']:,} unique, range {port_desc['min']}-{port_desc['max']}, "
    f"median {port_desc['median']:.0f}, zeros {port_desc['n_zero']:,}")

# =====================================================================
# 3. ISOLATION FOREST — exact frozen Phase 2A split
# =====================================================================
log("\n" + "=" * 72 + "\n3. ISOLATION FOREST WITHOUT DESTINATION PORT\n" + "=" * 72)
benign_mask = (labels == "BENIGN").to_numpy()
Xb, Xa = X_all[benign_mask], X_all[~benign_mask]
lb, la = labels[benign_mask], labels[~benign_mask]
train_size = int(len(Xb) * 0.7)
np.random.seed(RANDOM_STATE)
perm = np.random.permutation(len(Xb))
tr_idx, te_idx = perm[:train_size], perm[train_size:]
X_train = Xb.iloc[tr_idx]
X_test = pd.concat([Xb.iloc[te_idx], Xa])
labels_train = lb.iloc[tr_idx]
labels_test = pd.concat([lb.iloc[te_idx], la]).to_numpy()
y_test = (labels_test != "BENIGN").astype(np.int8)
del Xb, Xa
gc.collect()
assert len(X_train) == EXPECTED_TRAIN and len(X_test) == EXPECTED_TEST
assert int((y_test == 0).sum()) == EXPECTED_TEST_BENIGN and int((y_test == 1).sum()) == EXPECTED_TEST_ATTACK
assert (labels_train == "BENIGN").all()                                          # sanity 8

# Prove exact frozen rows: (a) TRAIN -> refit 70-feature imputer/scaler stats are bit-identical
# to the frozen ones (nothing saved); (b) TEST -> the frozen model reproduces the frozen
# per-row scores on the rebuilt 70-feature test matrix, in order.
fz_imp, fz_sc = joblib.load(MODELS / "imputer.pkl"), joblib.load(MODELS / "scaler.pkl")
chk_imp = SimpleImputer(strategy="median").fit(X_train)
chk_sc = StandardScaler().fit(chk_imp.transform(X_train))
train_rows_exact = (np.array_equal(chk_imp.statistics_, fz_imp.statistics_)
                    and np.array_equal(chk_sc.mean_, fz_sc.mean_) and np.array_equal(chk_sc.scale_, fz_sc.scale_))
fz_out = np.load(RES / "if_test_outputs.npz")
fz_model = joblib.load(MODELS / "isolation_forest.pkl")
Xt70 = fz_sc.transform(fz_imp.transform(X_test)).astype(np.float32)
with_score = -fz_model.decision_function(Xt70)
test_rows_exact = (np.array_equal(with_score.astype(np.float32), fz_out["anomaly_score"])
                   and np.array_equal(fz_out["y_test_binary"], y_test))
assert train_rows_exact and test_rows_exact                                      # sanity 6
with_pred = fz_out["y_pred_if"].astype(np.int8)
with_score = fz_out["anomaly_score"].astype(np.float64)
del Xt70, fz_model, chk_imp, chk_sc
gc.collect()
log("Frozen Phase 2A train rows (bit-identical preprocessing stats) and test rows "
    "(identical per-row frozen scores) reproduced exactly.")

# WITHOUT Port: preprocessing refit on BENIGN TRAIN only (69 features)
Xtr = X_train[features_without]
Xte = X_test[features_without]
assert PORT not in Xtr.columns and PORT not in Xte.columns
imp_np = SimpleImputer(strategy="median")
sc_np = StandardScaler()
Xtr_s = sc_np.fit_transform(imp_np.fit_transform(Xtr)).astype(np.float32)
Xte_s = sc_np.transform(imp_np.transform(Xte)).astype(np.float32)
del X_train, X_test, Xtr, Xte
gc.collect()
for a in (Xtr_s, Xte_s):
    assert a.shape[1] == 69 and not np.isnan(a).any() and np.isfinite(a).all()  # sanity 11/12

if_np = IsolationForest(**IF_PARAMS)
assert (if_np.n_estimators, if_np.contamination, if_np.random_state,
        if_np.max_samples) == (200, "auto", 42, "auto")                          # sanity 13
t = time.time()
if_np.fit(Xtr_s)                         # features only
if_train_time = time.time() - t
t = time.time()
dec_np = if_np.decision_function(Xte_s)
if_score_time = time.time() - t
t = time.time()
raw_np = if_np.predict(Xte_s)
if_predict_time = time.time() - t
score_np = -dec_np                       # higher = more anomalous
pred_np = (raw_np == -1).astype(np.int8)  # 0 normal, 1 anomaly
assert np.array_equal(pred_np, (score_np > 0).astype(np.int8))
if_pred_done = time.time()
if_peak_ws, if_peak_commit = mem_gb()
log(f"Fit {if_train_time:.2f}s | decision_function {if_score_time:.2f}s | predict {if_predict_time:.2f}s")

# Evaluation (after prediction)                                                  # sanity 15
assert time.time() > if_pred_done
m_np = binary_metrics(y_test, pred_np)
m_np["roc_auc"] = roc_auc_score(y_test, score_np)
assert abs(m_np["roc_auc"] + roc_auc_score(y_test, dec_np) - 1) < 1e-9
m_with = binary_metrics(y_test, with_pred)
m_with["roc_auc"] = roc_auc_score(y_test, with_score)
fz_head = json.load(open(FROZEN_MANIFESTS["2A"]))["headline"]
for k, v in fz_head.items():
    assert np.isclose(m_with[k], v), (k, m_with[k], v)
fz2a = json.load(open(RES / "phase2a_metrics.json"))
pear = float(pearsonr(with_score, score_np).statistic)
spear = float(spearmanr(with_score, score_np).statistic)
agree = float(np.mean(with_pred == pred_np))
changed = int((with_pred != pred_np).sum())
flip_to_anom = int(((with_pred == 0) & (pred_np == 1)).sum())
flip_to_norm = int(((with_pred == 1) & (pred_np == 0)).sum())
flips_by_class = {
    "benign_normal_to_anomaly": int(((with_pred == 0) & (pred_np == 1) & (y_test == 0)).sum()),
    "benign_anomaly_to_normal": int(((with_pred == 1) & (pred_np == 0) & (y_test == 0)).sum()),
    "attack_normal_to_anomaly": int(((with_pred == 0) & (pred_np == 1) & (y_test == 1)).sum()),
    "attack_anomaly_to_normal": int(((with_pred == 1) & (pred_np == 0) & (y_test == 1)).sum()),
}
DELTA_KEYS = ["precision", "recall", "f1", "fpr", "fnr", "roc_auc"]
if_delta = {k: m_np[k] - m_with[k] for k in DELTA_KEYS}
if_cat_with = category_rates(with_pred, labels_test, ATTACK_CATEGORIES)
if_cat_np = category_rates(pred_np, labels_test, ATTACK_CATEGORIES)
if_benign_with = float(with_pred[y_test == 0].mean())
if_benign_np = float(pred_np[y_test == 0].mean())
log(f"WITHOUT Port: P {m_np['precision']:.4f} R {m_np['recall']:.4f} F1 {m_np['f1']:.4f} "
    f"AUC {m_np['roc_auc']:.4f} | Pearson(scores) {pear:.4f} | agreement {agree:.4%}")
joblib.dump(if_np, OUT["if_model"], compress=3)
joblib.dump(imp_np, OUT["if_imputer"])
joblib.dump(sc_np, OUT["if_scaler"])
rel = joblib.load(OUT["if_model"])
assert np.array_equal(rel.decision_function(Xte_s[:20000]), dec_np[:20000])
del rel, Xtr_s
gc.collect()

# =====================================================================
# 6. DBSCAN — exact frozen Phase 2B 50k sample
# =====================================================================
log("\n" + "=" * 72 + "\n6-8. DBSCAN WITHOUT DESTINATION PORT\n" + "=" * 72)
ids = pd.read_csv(RES / "phase2b_sample_ids.csv")
work = ids[ids["in_dbscan_working_sample"]].sort_values("sample_order")
assert len(work) == DBSCAN_N and np.array_equal(work["sample_order"].to_numpy(), np.arange(DBSCAN_N))
db_index = work["raw_concat_index"].to_numpy()
redraw = X_all.sample(n=100_000, random_state=RANDOM_STATE).index.to_numpy()
cum = np.cumsum([0] + sizes)
fi = np.searchsorted(cum, db_index, side="right") - 1
fz_db = np.load(RES / "phase2b_cluster_assignments.npz")
db_rows_exact = (np.array_equal(redraw[:DBSCAN_N], db_index)
                 and np.array_equal(np.array(csv_files)[fi], work["source_file"].to_numpy())
                 and np.array_equal(db_index - cum[fi], work["file_row"].to_numpy())
                 and np.array_equal(fz_db["raw_concat_index"], db_index))
X_db70 = X_all.loc[db_index]
fz_imp_b, fz_sc_b = frozen_2b_pre["imputer"], frozen_2b_pre["scaler"]
chk_imp_b = SimpleImputer(strategy="median").fit(X_db70)
chk_sc_b = StandardScaler().fit(chk_imp_b.transform(X_db70))
db_rows_exact = db_rows_exact and np.array_equal(chk_imp_b.statistics_, fz_imp_b.statistics_) \
    and np.array_equal(chk_sc_b.mean_, fz_sc_b.mean_)
assert db_rows_exact                                                             # sanity 7
VAULT = HiddenLabels(labels.loc[db_index].to_numpy())
del X_all, chk_imp_b, chk_sc_b
gc.collect()
log("Frozen Phase 2B 50,000-row sample reproduced exactly (IDs, files, rows, preprocessing stats).")

# WITH-Port reference geometry (read-only: frozen 2B imputer/scaler transform, k-distance only)
Z_with = fz_sc_b.transform(fz_imp_b.transform(X_db70))
D_with = kdist_matrix(Z_with, max(MIN_SAMPLES_CANDIDATES))
del Z_with
# WITHOUT Port: imputer + scaler refit on the 50k sample itself (no labels)
imp_db, sc_db = SimpleImputer(strategy="median"), StandardScaler()
Z = sc_db.fit_transform(imp_db.fit_transform(X_db70[features_without]))
assert Z.shape == (DBSCAN_N, 69) and not np.isnan(Z).any() and np.isfinite(Z).all()   # 11/12
assert VAULT.assert_not_in(Z) and not VAULT.revealed                             # sanity 9

# 7. eps selection — identical unsupervised procedure (no labels)
t = time.time()
D = kdist_matrix(Z, max(MIN_SAMPLES_CANDIDATES))
kdist_time = time.time() - t
elb_np = {ms: {r: elbow_eps(D[:, ms - 1], r) for r in ("log10", "linear", "trunc99")} for ms in MIN_SAMPLES_CANDIDATES}
elb_with = {ms: {r: elbow_eps(D_with[:, ms - 1], r) for r in ("log10", "linear", "trunc99")} for ms in MIN_SAMPLES_CANDIDATES}
fz2b = json.load(open(RES / "phase2b_metrics.json"))
assert np.isclose(elb_with[10]["log10"][0], fz2b["primary"]["eps"])     # reproduces frozen 2B eps


def robustness(Dm, Xhalf_D, rule):
    kd = np.sort(Dm[:, 9])
    v = {"full": elbow_eps(kd, rule)[0], "trim_top_0.1pct": elbow_eps(kd[: int(0.999 * len(kd))], rule)[0],
         "trim_top_1pct": elbow_eps(kd[: int(0.99 * len(kd))], rule)[0], "half_sample": elbow_eps(Xhalf_D, rule)[0]}
    v["max_over_min"] = max(v.values()) / min(v.values())
    return v


D_half = kdist_matrix(Z[: DBSCAN_N // 2], 10)[:, 9]
robust_np = {r: robustness(D, D_half, r) for r in ("log10", "linear", "trunc99")}
PRIMARY_MS = min(MIN_SAMPLES_CANDIDATES, key=lambda m: abs(m - np.log(DBSCAN_N)))
EPS_NP = elb_np[PRIMARY_MS]["log10"][0]
clarity = {}
for tag, E in (("with_port", elb_with), ("without_port", elb_np)):
    e = E[PRIMARY_MS]
    clarity[tag] = {"eps_log10": e["log10"][0], "eps_linear": e["linear"][0], "eps_trunc99": e["trunc99"][0],
                    "rule_spread_linear_over_trunc99": e["linear"][0] / e["trunc99"][0],
                    "knee_strength_log10": e["log10"][2],
                    "log10_eps_across_min_samples": [E[m]["log10"][0] for m in MIN_SAMPLES_CANDIDATES]}
clarity["with_port"]["log10_rule_max_over_min"] = fz2b["elbow_robustness"]["log10"]["max_over_min"]
clarity["without_port"]["log10_rule_max_over_min"] = robust_np["log10"]["max_over_min"]
# Feature-identical rows (label-free geometry): zero k-distances are floored at 1e-6 by the
# Phase 2B log10 procedure, which can create a flat block at the bottom of the log curve.
Z_with_dup = fz_sc_b.transform(fz_imp_b.transform(X_db70))
geometry = {
    "with_port": {"duplicate_feature_rows": int(pd.DataFrame(Z_with_dup).duplicated().sum()),
                  "zero_kdist_rows": int((D_with[:, PRIMARY_MS - 1] == 0).sum())},
    "without_port": {"duplicate_feature_rows": int(pd.DataFrame(Z).duplicated().sum()),
                     "zero_kdist_rows": int((D[:, PRIMARY_MS - 1] == 0).sum())},
}
del Z_with_dup
for ms in MIN_SAMPLES_CANDIDATES:
    log(f"min_samples={ms:>2}: " + " | ".join(f"{r} eps={v[0]:.4f} @{100 * v[1]:.1f}th pct"
                                             for r, v in elb_np[ms].items()))
log(f"Selected (same rule as 2B): min_samples={PRIMARY_MS}, eps={EPS_NP:.4f}")

# 8. DBSCAN
pairs = count_pairs(Z, EPS_NP)
assert pairs["est_gb"] <= MEMORY_BUDGET_GB, pairs
assert not VAULT.revealed
t = time.time()
db_labels = DBSCAN(eps=EPS_NP, min_samples=PRIMARY_MS, metric="euclidean", algorithm="brute",
                   n_jobs=-1).fit_predict(Z).astype(np.int32)
db_runtime = time.time() - t
db_end = time.time()
db_peak_ws, db_peak_commit = mem_gb()
pred_db = (db_labels == -1).astype(np.int8)
assert (pred_db[db_labels == -1] == 1).all() and (pred_db[db_labels != -1] == 0).all()   # sanity 14
csz = np.sort(np.bincount(db_labels[db_labels >= 0]))[::-1]
db_summary = {"n_clusters": int(len(csz)), "n_noise": int(pred_db.sum()), "noise_pct": 100 * float(pred_db.mean()),
              "largest_cluster_sizes": csz[:10].tolist(), "largest_cluster_share_pct": 100 * csz[0] / DBSCAN_N,
              "clusters_lt_50": int((csz < 50).sum()), "runtime_s": db_runtime, **pairs}
log(f"DBSCAN WITHOUT Port: {db_summary['n_clusters']} clusters, noise {db_summary['noise_pct']:.2f}%, "
    f"{db_runtime:.1f}s")

# Evaluation — labels revealed only now                                          # sanity 15
y_db, cat_db = VAULT.reveal(db_end)
lab_with = fz_db["primary_70d"]
pred_with_db = (lab_with == -1).astype(np.int8)
dm_np = binary_metrics(y_db, pred_db)
dm_with = binary_metrics(y_db, pred_with_db)
for k in ("TP", "TN", "FP", "FN"):
    assert dm_with[k] == fz2b["primary"]["metrics"][k]
db_delta = {k: dm_np[k] - dm_with[k] for k in ("precision", "recall", "f1", "fpr", "fnr")}
nw, nn_ = pred_with_db == 1, pred_db == 1
overlap = {"noise_with": int(nw.sum()), "noise_without": int(nn_.sum()), "both": int((nw & nn_).sum()),
           "with_only": int((nw & ~nn_).sum()), "without_only": int((~nw & nn_).sum()),
           "jaccard": float((nw & nn_).sum() / max((nw | nn_).sum(), 1)),
           "changed_assignment": int((nw != nn_).sum())}
overlap["changed_attack"] = int(((nw != nn_) & (y_db == 1)).sum())
overlap["changed_benign"] = int(((nw != nn_) & (y_db == 0)).sum())
db_cat_with = category_rates(pred_with_db, cat_db, ATTACK_CATEGORIES)
db_cat_np = category_rates(pred_db, cat_db, ATTACK_CATEGORIES)
db_benign_with, db_benign_np = float(pred_with_db[y_db == 0].mean()), float(pred_db[y_db == 0].mean())


# 11. Cluster composition (post-hoc; same definitions as the Phase 2B check)
def composition(lab, y, cats):
    att = y == 1
    d = pd.DataFrame({"c": lab, "att": att, "cat": cats})
    cl = d[d.c >= 0].groupby("c").agg(n=("att", "size"), share=("att", "mean"), k=("att", "sum"))
    dom = cl[cl.share >= 0.95]
    mixed = cl[(cl.share > 0.05) & (cl.share < 0.95)]
    bdom = cl[cl.share <= 0.05]
    tot = int(att.sum())
    big = cl.sort_values("n", ascending=False).head(3)
    major = []
    for c, r in big.iterrows():
        vcats = d[d.c == c].cat.value_counts().head(5)
        major.append({"cluster": int(c), "size": int(r.n), "attack_share_pct": 100 * r.share,
                      "top_labels": {str(k): int(v) for k, v in vcats.items()}})
    return {"attack_rows": tot,
            "pct_attack_in_attack_dominated": 100 * dom.k.sum() / tot, "n_attack_dominated_clusters": int(len(dom)),
            "pct_attack_in_mixed": 100 * mixed.k.sum() / tot, "n_mixed_clusters": int(len(mixed)),
            "pct_attack_in_benign_dominated": 100 * bdom.k.sum() / tot,
            "pct_attack_in_noise": 100 * int((att & (lab == -1)).sum()) / tot, "major_clusters": major}


comp_with = composition(lab_with, y_db, cat_db)
comp_np = composition(db_labels, y_db, cat_db)
for c_ in (comp_with, comp_np):
    assert abs(c_["pct_attack_in_attack_dominated"] + c_["pct_attack_in_mixed"] +
               c_["pct_attack_in_benign_dominated"] + c_["pct_attack_in_noise"] - 100) < 1e-6

np.savez_compressed(OUT["db_assign"], dbscan_no_port=db_labels, raw_concat_index=db_index,
                    eps=np.array(EPS_NP), min_samples=np.array(PRIMARY_MS))
joblib.dump({"imputer": imp_db, "scaler": sc_db, "feature_columns": features_without},
            RES / "phase2c_dbscan_preprocessing.pkl")

# =====================================================================
# 12. PLOTS
# =====================================================================
top = vc.head(12)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
pb = port[~is_att].value_counts()
pa = port[is_att].value_counts()
yy = np.arange(len(top))[::-1]
axes[0].barh(yy, [pb.get(p, 0) for p in top.index], color=C_WITH, height=0.6, label="BENIGN")
axes[0].barh(yy, [pa.get(p, 0) for p in top.index], left=[pb.get(p, 0) for p in top.index],
             color=C_WITHOUT, height=0.6, label="ATTACK")
axes[0].set_yticks(yy, [f"{int(p)}  ({100 * n / len(port):.1f}%)" for p, n in top.items()])
axes[0].set_xlabel("Rows in deduplicated data")
axes[0].set_title(f"12 most frequent Destination Ports (of {port_desc['n_unique']:,} unique)", fontsize=10)
axes[0].legend(frameon=False)
for name, mask, col in (("BENIGN", ~is_att, C_WITH), ("ATTACK", is_att, C_WITHOUT)):
    s = np.sort(port[mask].to_numpy())
    axes[1].plot(s, np.arange(1, len(s) + 1) / len(s), color=col, linewidth=2, label=name)
axes[1].set_xscale("symlog", linthresh=10)
axes[1].set_xlabel("Destination Port (symlog)")
axes[1].set_ylabel("Cumulative share of rows")
axes[1].set_title("Port ECDF by class (descriptive only)", fontsize=10)
axes[1].legend(frameon=False)
fig.savefig(PLOT_DIR / "01_destination_port_distribution.png")
plt.close(fig)


def paired_bars(ax, keys, a, b, la_, lb_, fmt="{:.3f}"):
    x = np.arange(len(keys))
    w = 0.38
    for j, (vals, lab_, col) in enumerate(((a, la_, C_WITH), (b, lb_, C_WITHOUT))):
        bb = ax.bar(x + (j - 0.5) * w, vals, w - 0.03, color=col, label=lab_)
        for r_, v in zip(bb, vals):
            ax.text(r_.get_x() + r_.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [k.upper() if len(k) <= 3 else k.replace("_", "-").title() for k in keys])
    ax.legend(frameon=False)


fig, ax = plt.subplots(figsize=(9, 4.4))
paired_bars(ax, DELTA_KEYS, [m_with[k] for k in DELTA_KEYS], [m_np[k] for k in DELTA_KEYS],
            "WITH Port (frozen 2A, 70)", "WITHOUT Port (69)")
ax.set_ylim(0, 1.08)
ax.set_title("Isolation Forest — same rows, same configuration; only Destination Port removed", fontsize=10)
fig.savefig(PLOT_DIR / "02_if_metric_comparison.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
hb = axes[0].hexbin(with_score, score_np, gridsize=90, bins="log", cmap="Blues", mincnt=1)
lo_, hi_ = min(with_score.min(), score_np.min()), max(with_score.max(), score_np.max())
axes[0].plot([lo_, hi_], [lo_, hi_], color=INK_2, linestyle=":", linewidth=1)
axes[0].axhline(0, color=INK, linewidth=0.8, linestyle="--")
axes[0].axvline(0, color=INK, linewidth=0.8, linestyle="--")
axes[0].set_xlabel("anomaly_score WITH Port")
axes[0].set_ylabel("anomaly_score WITHOUT Port")
axes[0].set_title(f"Per-row scores, {EXPECTED_TEST:,} test rows (Pearson {pear:.3f})", fontsize=10)
fig.colorbar(hb, ax=axes[0], label="rows (log)")
bins = np.linspace(lo_, hi_, 100)
for sc_arr, name, col in ((with_score, "WITH Port", C_WITH), (score_np, "WITHOUT Port", C_WITHOUT)):
    for cls, ls in ((0, "-"), (1, "--")):
        axes[1].hist(sc_arr[y_test == cls], bins=bins, density=True, histtype="step", linewidth=1.8,
                     linestyle=ls, color=col, label=f"{name} — {'BENIGN' if cls == 0 else 'ATTACK'}")
axes[1].axvline(0, color=INK, linewidth=0.8, linestyle="--")
axes[1].set_xlabel("anomaly_score (higher = more anomalous)")
axes[1].set_ylabel("Density")
axes[1].set_title("Score distributions by ground truth (evaluation only)", fontsize=10)
axes[1].legend(frameon=False, fontsize=8)
fig.savefig(PLOT_DIR / "03_if_anomaly_score_comparison.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
xf = np.linspace(0, 100, DBSCAN_N)
for Dm, E, name, col in ((D_with, elb_with, "WITH Port (frozen 2B)", C_WITH), (D, elb_np, "WITHOUT Port", C_WITHOUT)):
    ax.plot(xf, np.maximum(np.sort(Dm[:, PRIMARY_MS - 1]), 1e-6), color=col, linewidth=2, label=name)
    e, q, _ = E[PRIMARY_MS]["log10"]
    ax.scatter([100 * q], [e], s=64, color=col, edgecolor="#fcfcfb", linewidth=2, zorder=3)
    ax.annotate(f"eps={e:.3f}", (100 * q, e), xytext=(-70, 8), textcoords="offset points", color=INK, fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("Points sorted by k-distance (percentile)")
ax.set_ylabel(f"Distance to {PRIMARY_MS}th neighbour (incl. self), log")
ax.set_title(f"k-distance curves, min_samples={PRIMARY_MS}, same 50,000 rows (log10-Kneedle elbows)", fontsize=10)
ax.legend(frameon=False)
fig.savefig(PLOT_DIR / "04_dbscan_k_distance_comparison.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.2))
vals = [fz2b["primary"]["noise_pct"], db_summary["noise_pct"]]
bb = ax.bar([f"WITH Port\neps={fz2b['primary']['eps']:.3f}", f"WITHOUT Port\neps={EPS_NP:.3f}"], vals,
            color=[C_WITH, C_WITHOUT], width=0.5)
for r_, v in zip(bb, vals):
    ax.text(r_.get_x() + r_.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom")
ax.set_ylabel("Rows labelled noise (%)")
ax.set_title(f"DBSCAN noise share (min_samples={PRIMARY_MS}, same 50k rows)", fontsize=10)
fig.savefig(PLOT_DIR / "05_dbscan_noise_pct.png")
plt.close(fig)

DBK = ["precision", "recall", "f1", "fpr", "fnr"]
fig, ax = plt.subplots(figsize=(8.5, 4.4))
paired_bars(ax, DBK, [dm_with[k] for k in DBK], [dm_np[k] for k in DBK], "WITH Port (frozen 2B)", "WITHOUT Port")
ax.set_ylim(0, 1.08)
ax.set_title("DBSCAN noise-as-anomaly metrics (transductive, same 50k rows)", fontsize=10)
fig.savefig(PLOT_DIR / "06_dbscan_metric_comparison.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 2.8))
segs = [("Noise WITH only", overlap["with_only"], C_WITH), ("Noise in both", overlap["both"], INK_2),
        ("Noise WITHOUT only", overlap["without_only"], C_WITHOUT)]
left = 0
for name, v, col in segs:
    ax.barh([0], [v], left=left, color=col, height=0.5, edgecolor="#fcfcfb", linewidth=2)
    ax.text(left + v / 2, 0, f"{name}\n{v:,}", ha="center", va="center", color="white", fontsize=9)
    left += v
ax.set_yticks([])
ax.set_xlabel("Rows")
ax.grid(False)
ax.set_title(f"DBSCAN noise-set overlap — Jaccard {overlap['jaccard']:.3f}; "
             f"{overlap['changed_assignment']:,} rows changed assignment", fontsize=10)
fig.savefig(PLOT_DIR / "07_dbscan_noise_overlap.png")
plt.close(fig)


def dumbbell(ax, cats, rw, rn, nmap, title, min_n):
    show = [c for c in cats if nmap[c] >= min_n]
    show = sorted(show, key=lambda c: rw[c])
    y = np.arange(len(show))
    for i, c in enumerate(show):
        ax.plot([rw[c], rn[c]], [i, i], color=GRID, linewidth=3, zorder=1)
    ax.scatter([rw[c] for c in show], y, s=60, color=C_WITH, label="WITH Port", zorder=2)
    ax.scatter([rn[c] for c in show], y, s=60, color=C_WITHOUT, label="WITHOUT Port", zorder=3)
    ax.set_yticks(y, [f"{c} (n={nmap[c]:,})" for c in show])
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Percentage flagged as anomalous")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, loc="lower right")


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
dumbbell(axes[0], ATTACK_CATEGORIES, {c: if_cat_with[c]["pct"] for c in ATTACK_CATEGORIES},
         {c: if_cat_np[c]["pct"] for c in ATTACK_CATEGORIES}, {c: if_cat_np[c]["n"] for c in ATTACK_CATEGORIES},
         "Isolation Forest — Phase 2A test set (all categories; n<100 unstable)", 1)
dumbbell(axes[1], ATTACK_CATEGORIES, {c: db_cat_with[c]["pct"] for c in ATTACK_CATEGORIES},
         {c: db_cat_np[c]["pct"] for c in ATTACK_CATEGORIES}, {c: db_cat_np[c]["n"] for c in ATTACK_CATEGORIES},
         f"DBSCAN — 50k sample (categories with n ≥ {DB_UNSTABLE_N})", DB_UNSTABLE_N)
fig.tight_layout()
fig.savefig(PLOT_DIR / "08_attack_category_comparison.png")
plt.close(fig)

# =====================================================================
# 13. SANITY CHECKS + frozen verification (after)
# =====================================================================
frozen_after = verify_frozen()
sanity = {
    "1. frozen Phase 2A hashes verified before": not frozen_before["2A"]["modified"],
    "2. frozen Phase 2B hashes verified before": not frozen_before["2B"]["modified"],
    "3. exactly 70 features WITH Port": len(features_with) == 70,
    "4. exactly 69 features WITHOUT Port": len(features_without) == 69 and Xte_s.shape[1] == 69 and Z.shape[1] == 69,
    "5. Destination Port is the only removed feature": set(features_with) ^ set(features_without) == {PORT},
    "6. IF uses the exact frozen train/test rows": bool(train_rows_exact and test_rows_exact),
    "7. DBSCAN uses the exact frozen 50k sample IDs": bool(db_rows_exact),
    "8. IF training contains only BENIGN": bool((labels_train == "BENIGN").all()),
    "9. DBSCAN labels hidden during clustering": VAULT.reveal_time > db_end,
    "10. DBSCAN eps selection used no labels (selection precedes reveal)": VAULT.reveal_time > db_end,
    "11. no NaN in model input": not (np.isnan(Xte_s).any() or np.isnan(Z).any()),
    "12. no inf in model input": bool(np.isfinite(Xte_s).all() and np.isfinite(Z).all()),
    "13. IF n_estimators=200, contamination='auto', random_state=42":
        (if_np.n_estimators, if_np.contamination, if_np.random_state) == (200, "auto", 42),
    "14. DBSCAN noise mapping (-1 -> 1, else 0)": bool((pred_db == (db_labels == -1)).all()),
    "15. evaluation only after prediction/clustering": VAULT.reveal_time > db_end,
    "16. frozen Phase 2A unchanged after": not frozen_after["2A"]["modified"]
        and frozen_after["2A"]["manifest_sha256"] == frozen_before["2A"]["manifest_sha256"],
    "17. frozen Phase 2B unchanged after": not frozen_after["2B"]["modified"]
        and frozen_after["2B"]["manifest_sha256"] == frozen_before["2B"]["manifest_sha256"],
}
for k, v in sanity.items():
    assert v, f"SANITY FAILED: {k}"
peak_ws, peak_commit = mem_gb()
total_time = time.time() - T0

# =====================================================================
# REPORT
# =====================================================================
if_little = (all(abs(if_delta[k]) < MATERIAL_METRIC_DELTA for k in DELTA_KEYS) and pear >= MATERIAL_SCORE_CORR)
jac = overlap["jaccard"]
jac_word = ("largely the same" if jac >= JACCARD_LARGELY_SAME else
            "substantially different" if jac < JACCARD_SUBSTANTIAL else "partially different")
pp = lambda v: f"{100 * v:+.2f} pp"
L = []
A = L.append
A("=" * 80)
A("PHASE 2C — DESTINATION PORT ABLATION (feature ablation; not model optimisation)")
A("=" * 80)
A("Reference baselines (WITH Port) are the frozen Phase 2A Isolation Forest and frozen Phase 2B DBSCAN")
A("results; they were read, not rerun. Only Destination Port is removed.")
A("\n1. FEATURE ABLATION")
A(f"  Feature name: '{PORT}' (present in the frozen 70-feature list: yes)")
A(f"  WITH Port: 70 features | WITHOUT Port: 69 features | difference: ['{PORT}'] only")
A("  Exact lists: results/phase2c_feature_lists.json")
A("\n2. DESTINATION PORT — DESCRIPTIVE ANALYSIS (deduplicated data, exploratory only)")
A(f"  Rows {port_desc['n_rows']:,} | unique values {port_desc['n_unique']:,} | min {port_desc['min']} | "
  f"max {port_desc['max']:,} | median {port_desc['median']:.0f} | zero-port rows {port_desc['n_zero']:,}")
A("  Most frequent ports: " + ", ".join(f"{t['port']} ({t['share_pct']:.1f}%)" for t in port_desc["top10"]))
A(f"  Top 5 ports cover {port_desc['top5_share_pct']:.1f}% of rows; top 10 cover {port_desc['top10_share_pct']:.1f}%.")
for name, d in port_desc["by_class"].items():
    A(f"  {name:<6}: rows {d['rows']:,} | unique {d['n_unique']:,} | median {d['median']:.0f} | "
      f"top-5 share {d['top5_share_pct']:.1f}% | <1024 {d['share_well_known_lt_1024_pct']:.1f}% | "
      f">=49152 {d['share_ephemeral_ge_49152_pct']:.1f}%")
    A("          top ports: " + ", ".join(f"{t['port']} ({t['share_pct']:.1f}%)" for t in d["top5"]))
A("  Per attack category, most common port (share of that category's rows):")
for c in ATTACK_CATEGORIES:
    d = cat_ports[c]
    A(f"    {c:<28} n={d['rows']:>7,}  unique={d['n_unique']:>6,}  top port {d['top_port']} ({d['top_port_share_pct']:.1f}%)")
A("  Destination Port is a categorical identifier stored as an integer: as a numeric feature it")
A("  imposes an arbitrary ordering/distance (port 80 is 'close' to 81, far from 8080). Its")
A("  association with attack categories reflects the CIC-IDS2017 testbed (which services were")
A("  attacked); this descriptive association alone is not evidence of label leakage.")
A("\n3. ISOLATION FOREST WITHOUT DESTINATION PORT")
A(f"  Rows: exact frozen Phase 2A split — TRAIN {EXPECTED_TRAIN:,} BENIGN; TEST {EXPECTED_TEST_BENIGN:,} BENIGN + "
  f"{EXPECTED_TEST_ATTACK:,} ATTACK (verified: refit 70-feature preprocessing stats bit-identical; frozen model")
A("  reproduces the frozen per-row test scores).")
A("  Preprocessing: ±inf->NaN, median imputer + StandardScaler refit on BENIGN TRAIN (69 features);")
A("  test transformed with the train-fitted artifacts.")
A("  Model: IsolationForest(n_estimators=200, max_samples='auto', contamination='auto', random_state=42, n_jobs=-1)")
A("  No contamination or tree-count experiments were run.")
mn = m_np
A(f"  TP={mn['TP']:,} TN={mn['TN']:,} FP={mn['FP']:,} FN={mn['FN']:,}")
A(f"  Accuracy {mn['accuracy']:.4f} | Precision {mn['precision']:.4f} | Recall {mn['recall']:.4f} | F1 {mn['f1']:.4f} | "
  f"FPR {mn['fpr']:.4f} | FNR {mn['fnr']:.4f} | ROC-AUC {mn['roc_auc']:.4f} (anomaly_score = -decision_function)")
A(f"  Training {if_train_time:.2f} s | predict() {if_predict_time:.2f} s | decision_function {if_score_time:.2f} s | "
  f"peak RAM at this stage {if_peak_ws:.1f} GB working set ({if_peak_commit:.1f} GB committed) | features 69")
A("\n4. ISOLATION FOREST ABLATION COMPARISON (paired: same 1,054,824 test rows)")
A(f"  {'':<16}{'WITH Port':>14}{'WITHOUT Port':>14}{'delta (W/O - W)':>18}")
A(f"  {'Features':<16}{70:>14}{69:>14}{-1:>18}")
for k in DELTA_KEYS:
    A(f"  {k.upper() if len(k) <= 3 else k.title():<16}{m_with[k]:>14.4f}{m_np[k]:>14.4f}{if_delta[k]:>+12.4f} ({pp(if_delta[k])})"
      if k != "roc_auc" else f"  {'ROC-AUC':<16}{m_with[k]:>14.4f}{m_np[k]:>14.4f}{if_delta[k]:>+18.4f}")
A(f"  {'Training (s)':<16}{fz2a['train_time_s']:>14.2f}{if_train_time:>14.2f}")
A(f"  {'Prediction (s)':<16}{fz2a['predict_time_s']:>14.2f}{if_predict_time:>14.2f}   (timings from separate runs)")
A(f"  Pearson(anomaly_score) {pear:.4f} | Spearman {spear:.4f} | prediction agreement {agree:.4%}")
A(f"  Rows whose prediction changed: {changed:,} ({flip_to_anom:,} normal->anomaly, {flip_to_norm:,} anomaly->normal)")
A(f"    BENIGN: {flips_by_class['benign_normal_to_anomaly']:,} newly flagged, {flips_by_class['benign_anomaly_to_normal']:,} no longer flagged")
A(f"    ATTACK: {flips_by_class['attack_normal_to_anomaly']:,} newly flagged, {flips_by_class['attack_anomaly_to_normal']:,} no longer flagged")
A("  These deltas describe the effect of the feature; they are not a model-selection result.")
A("\n5. ISOLATION FOREST — PERCENTAGE FLAGGED AS ANOMALOUS BY CATEGORY (Phase 2A test set)")
A(f"  {'Category':<28}{'n':>9}{'WITH %':>9}{'W/O %':>9}{'change':>10}  note")
for c in sorted(ATTACK_CATEGORIES, key=lambda c: -if_cat_np[c]["n"]):
    w, n_ = if_cat_with[c], if_cat_np[c]
    d = n_["pct"] - w["pct"]
    note = f"small n (<{IF_SMALL_N}): unstable" if w["n"] < IF_SMALL_N else ("notable" if abs(d) >= CATEGORY_NOTABLE_PP else "")
    A(f"  {c:<28}{w['n']:>9,}{w['pct']:>9.2f}{n_['pct']:>9.2f}{d:>+9.2f}pp  {note}")
A(f"  BENIGN test rows flagged: WITH {100 * if_benign_with:.2f}% | WITHOUT {100 * if_benign_np:.2f}%")
A("\n6. DBSCAN WITHOUT DESTINATION PORT — SAMPLE AND PREPROCESSING")
A(f"  Same 50,000 rows as frozen Phase 2B (results/phase2b_sample_ids.csv; verified against a fresh seeded")
A("  draw, source file/row, frozen assignment file and bit-identical refit 70-feature preprocessing stats).")
A("  Median imputer + StandardScaler refit on the 50k sample (69 features). Labels sealed until clustering ended.")
A("\n7. DBSCAN eps SELECTION WITHOUT PORT (identical unsupervised procedure to Phase 2B)")
for ms in MIN_SAMPLES_CANDIDATES:
    A(f"  min_samples={ms:>2}: " + " | ".join(f"{r} eps={v[0]:.3f} @{100 * v[1]:.1f}th pct" for r, v in elb_np[ms].items()))
A("  Rule robustness WITHOUT Port (min_samples=10): " + "; ".join(
    f"{r} max/min={v['max_over_min']:.2f}" for r, v in robust_np.items()))
A(f"  Selected: min_samples={PRIMARY_MS} (candidate closest to ln(n)={np.log(DBSCAN_N):.2f}), eps={EPS_NP:.4f}")
A("  (log10-Kneedle). eps 2.552 was NOT reused; no label, metric or prevalence informed the choice.")
cw, cn = clarity["with_port"], clarity["without_port"]
A(f"  Elbow clarity, min_samples={PRIMARY_MS}           WITH Port   WITHOUT Port")
A(f"    eps log10 / linear / trunc99        {cw['eps_log10']:.2f}/{cw['eps_linear']:.2f}/{cw['eps_trunc99']:.2f}"
  f"      {cn['eps_log10']:.2f}/{cn['eps_linear']:.2f}/{cn['eps_trunc99']:.2f}")
A(f"    rule disagreement (linear/trunc99)  {cw['rule_spread_linear_over_trunc99']:>9.2f}   {cn['rule_spread_linear_over_trunc99']:>12.2f}")
A(f"    log10 rule stability (max/min)      {cw['log10_rule_max_over_min']:>9.2f}   {cn['log10_rule_max_over_min']:>12.2f}")
A(f"    log10 knee strength (max chord gap) {cw['knee_strength_log10']:>9.3f}   {cn['knee_strength_log10']:>12.3f}")
gw, gn = geometry["with_port"], geometry["without_port"]
A(f"    feature-identical rows in the 50k   {gw['duplicate_feature_rows']:>9,}   {gn['duplicate_feature_rows']:>12,}")
A(f"    rows with zero {PRIMARY_MS}-NN distance        {gw['zero_kdist_rows']:>9,}   {gn['zero_kdist_rows']:>12,}")
rb = robust_np["log10"]
A("  Assessment: the elbow is ambiguous in BOTH spaces — the three Kneedle variants disagree by")
A(f"  ~{cw['rule_spread_linear_over_trunc99']:.0f}x (WITH) and ~{cn['rule_spread_linear_over_trunc99']:.0f}x (WITHOUT), and the log10 knee is weaker WITHOUT Port")
A(f"  ({cn['knee_strength_log10']:.3f} vs {cw['knee_strength_log10']:.3f}). The WITHOUT-Port curve is also LESS stable under the")
A(f"  label-free perturbations: trimming the top 1% sends the log10 knee to the 1e-6 floor (eps={rb['trim_top_1pct']:g};")
A(f"  other perturbations {rb['trim_top_0.1pct']:.2f}-{rb['half_sample']:.2f}). Cause: removing Destination Port makes "
  f"{gn['duplicate_feature_rows']:,} rows")
A(f"  feature-identical (WITH Port: {gw['duplicate_feature_rows']:,}), giving {gn['zero_kdist_rows']:,} zero k-distances that the Phase 2B")
A("  procedure floors at 1e-6; this flat block can dominate the normalised log curve. The full-curve")
A(f"  eps ({EPS_NP:.3f}) used here is unaffected by that perturbation, but it is less robust than the")
A("  WITH-Port eps. No clean elbow is claimed.")
A("\n8. DBSCAN WITHOUT PORT — RESULTS")
A(f"  Clusters {db_summary['n_clusters']} | noise {db_summary['n_noise']:,} ({db_summary['noise_pct']:.2f}%) | largest cluster "
  f"{db_summary['largest_cluster_sizes'][0]:,} ({db_summary['largest_cluster_share_pct']:.1f}%) | next: {db_summary['largest_cluster_sizes'][1:6]}")
A(f"  Runtime {db_runtime:.1f} s | neighbour pairs {pairs['pairs']:,} (~{pairs['est_gb']:.1f} GB est.) | "
  f"process peak {db_peak_ws:.1f} GB working set, {db_peak_commit:.1f} GB committed")
dn = dm_np
A(f"  TP={dn['TP']:,} TN={dn['TN']:,} FP={dn['FP']:,} FN={dn['FN']:,}")
A(f"  Accuracy {dn['accuracy']:.4f} | Precision {dn['precision']:.4f} | Recall {dn['recall']:.4f} | F1 {dn['f1']:.4f} | "
  f"FPR {dn['fpr']:.4f} | FNR {dn['fnr']:.4f} | ROC-AUC not reported (binary noise output)")
A("  Transductive: these metrics describe the noise assignment of these 50,000 rows only.")
A("\n9. DBSCAN ABLATION COMPARISON (paired: same 50,000 rows)")
fp_ = fz2b["primary"]
A(f"  {'':<14}{'WITH Port':>12}{'WITHOUT Port':>14}{'delta':>16}")
A(f"  {'Features':<14}{70:>12}{69:>14}{-1:>16}")
A(f"  {'eps':<14}{fp_['eps']:>12.4f}{EPS_NP:>14.4f}{EPS_NP - fp_['eps']:>+16.4f}")
A(f"  {'min_samples':<14}{fp_['min_samples']:>12}{PRIMARY_MS:>14}{PRIMARY_MS - fp_['min_samples']:>+16}")
A(f"  {'Clusters':<14}{fp_['n_clusters']:>12}{db_summary['n_clusters']:>14}{db_summary['n_clusters'] - fp_['n_clusters']:>+16}")
A(f"  {'Noise %':<14}{fp_['noise_pct']:>12.2f}{db_summary['noise_pct']:>14.2f}{db_summary['noise_pct'] - fp_['noise_pct']:>+13.2f} pp")
for k in ("precision", "recall", "f1", "fpr", "fnr"):
    A(f"  {k.upper() if len(k) <= 3 else k.title():<14}{dm_with[k]:>12.4f}{dm_np[k]:>14.4f}{db_delta[k]:>+9.4f} ({pp(db_delta[k])})")
A(f"  {'Runtime (s)':<14}{fp_['runtime_s']:>12.1f}{db_runtime:>14.1f}   (separate runs)")
A(f"  Noise sets: WITH {overlap['noise_with']:,} | WITHOUT {overlap['noise_without']:,} | both {overlap['both']:,} | "
  f"WITH only {overlap['with_only']:,} | WITHOUT only {overlap['without_only']:,}")
A(f"  Jaccard {jac:.3f}; rows whose noise/non-noise assignment changed: {overlap['changed_assignment']:,} "
  f"({overlap['changed_attack']:,} ATTACK, {overlap['changed_benign']:,} BENIGN)")
if overlap["without_only"] == 0:
    A("  Every WITHOUT-Port noise row was also noise WITH Port (the WITHOUT-Port noise set is a strict")
    A("  subset): removing the feature only absorbed former noise rows into clusters.")
A("  Note: this paired comparison combines two changes — the feature removal and the re-selected eps")
A(f"  ({fp_['eps']:.3f} -> {EPS_NP:.3f}); the two effects are not separated here.")
A("\n10. DBSCAN — PERCENTAGE FLAGGED AS ANOMALOUS BY CATEGORY (same 50k sample)")
A(f"  {'Category':<28}{'n':>7}{'WITH %':>9}{'W/O %':>9}{'change':>10}  note")
for c in sorted(ATTACK_CATEGORIES, key=lambda c: -db_cat_np[c]["n"]):
    w, n_ = db_cat_with[c], db_cat_np[c]
    if n_["n"] == 0:
        A(f"  {c:<28}{0:>7}{'':>9}{'':>9}{'':>10}  Not present in the DBSCAN sample")
        continue
    d = n_["pct"] - w["pct"]
    note = f"statistically unstable (n < {DB_UNSTABLE_N})" if n_["n"] < DB_UNSTABLE_N else ("notable" if abs(d) >= CATEGORY_NOTABLE_PP else "")
    A(f"  {c:<28}{n_['n']:>7,}{w['pct']:>9.2f}{n_['pct']:>9.2f}{d:>+9.2f}pp  {note}")
A(f"  BENIGN rows flagged: WITH {100 * db_benign_with:.2f}% | WITHOUT {100 * db_benign_np:.2f}%")
A("\n11. DBSCAN CLUSTER COMPOSITION (post-hoc, after clustering; Phase 2B definitions)")
A("  attack-dominated: >=95% attack rows; mixed: 5-95%; benign-dominated: <=5%.")
A(f"  {'share of attack rows in...':<34}{'WITH Port':>11}{'WITHOUT Port':>14}")
for key, nm in (("pct_attack_in_attack_dominated", "attack-dominated clusters"),
                ("pct_attack_in_mixed", "mixed clusters"),
                ("pct_attack_in_benign_dominated", "benign-dominated clusters"),
                ("pct_attack_in_noise", "noise")):
    A(f"  {nm:<34}{comp_with[key]:>10.1f}%{comp_np[key]:>13.1f}%")
A(f"  number of attack-dominated clusters: WITH {comp_with['n_attack_dominated_clusters']} | WITHOUT {comp_np['n_attack_dominated_clusters']}")
for tag, cp in (("WITH", comp_with), ("WITHOUT", comp_np)):
    for mc in cp["major_clusters"]:
        A(f"  {tag} Port cluster {mc['cluster']} (n={mc['size']:,}, {mc['attack_share_pct']:.1f}% attack): {mc['top_labels']}")
A("\n12. INTERPRETATION (neutral; no configuration is a 'winner')")
A("  Q1 Isolation Forest: " + (
    f"The ablation had little material effect on Isolation Forest: every metric moved by < {MATERIAL_METRIC_DELTA:.2f} "
    f"and per-row scores correlate at {pear:.3f}." if if_little else
    f"Removing Destination Port changed Isolation Forest's scores and/or metrics beyond the pre-declared "
    f"threshold (|delta| >= {MATERIAL_METRIC_DELTA:.2f} or Pearson < {MATERIAL_SCORE_CORR}): ROC-AUC {if_delta['roc_auc']:+.4f}, "
    f"recall {pp(if_delta['recall'])}, FPR {pp(if_delta['fpr'])}, Pearson {pear:.3f}, {changed:,} predictions changed."))
A(f"  Q2 DBSCAN density structure: the selected eps moved {fp_['eps']:.3f} -> {EPS_NP:.3f}, clusters "
  f"{fp_['n_clusters']} -> {db_summary['n_clusters']}, largest cluster {fp_['largest_cluster_sizes'][0]:,} -> "
  f"{db_summary['largest_cluster_sizes'][0]:,}; attack rows in attack-dominated clusters "
  f"{comp_with['pct_attack_in_attack_dominated']:.1f}% -> {comp_np['pct_attack_in_attack_dominated']:.1f}%, in mixed clusters "
  f"{comp_with['pct_attack_in_mixed']:.1f}% -> {comp_np['pct_attack_in_mixed']:.1f}%.")
A(f"  Q3 DBSCAN noise membership: the noise sets are {jac_word} (Jaccard {jac:.3f}); "
  f"{overlap['changed_assignment']:,} of 50,000 rows changed noise/non-noise assignment.")
notable_if = [c for c in ATTACK_CATEGORIES if if_cat_np[c]["n"] >= IF_SMALL_N
              and abs(if_cat_np[c]["pct"] - if_cat_with[c]["pct"]) >= CATEGORY_NOTABLE_PP]
notable_db = [c for c in ATTACK_CATEGORIES if db_cat_np[c]["n"] >= DB_UNSTABLE_N
              and abs(db_cat_np[c]["pct"] - db_cat_with[c]["pct"]) >= CATEGORY_NOTABLE_PP]
A(f"  Q4 Category-level: changes of >= {CATEGORY_NOTABLE_PP:.0f} pp (stable-n categories only) — Isolation Forest: "
  + (", ".join(f"{c} {if_cat_with[c]['pct']:.1f}%->{if_cat_np[c]['pct']:.1f}%" for c in notable_if) or "none")
  + "; DBSCAN: " + (", ".join(f"{c} {db_cat_with[c]['pct']:.1f}%->{db_cat_np[c]['pct']:.1f}%" for c in notable_db) or "none") + ".")
A("  Q5 Interpretation of the DBSCAN/Isolation Forest difference: in both feature spaces the")
A(f"  majority of DBSCAN attack rows sit inside clusters rather than in noise ({comp_np['pct_attack_in_noise']:.1f}% noise "
  f"without Port vs {comp_with['pct_attack_in_noise']:.1f}% with), so the Phase 2B observation that most attack")
A("  rows occur in dense regions " + ("persists" if comp_np["pct_attack_in_noise"] < 50 else "does not persist") +
  " after removing Destination Port. This is a description of the ablation only;")
A("  the formal DBSCAN-vs-Isolation-Forest comparison is deferred.")
A("\n13. LIMITATIONS")
A("  * One feature removed, one configuration per algorithm; IF uses one random seed (random_state=42) — with a")
A("    different feature count the trees draw different splits, so part of any IF change is seed variation.")
A("  * DBSCAN is transductive and evaluated on a 50k sample; eps had to be re-selected, so the DBSCAN")
A("    comparison combines the feature change with the change in the unsupervised eps.")
A("  * The k-distance elbow is ambiguous in both feature spaces; noise sets are sensitive to eps (Phase 2B).")
A("  * The Phase 2B log10 rule floors zero k-distances at 1e-6; without Port, feature-identical rows make")
A("    the rule unstable under trimming (Section 7). The procedure was kept unchanged for comparability.")
A("  * Rare categories (Heartbleed, Infiltration, SQL Injection, XSS) give unstable or no estimates.")
A("  * Timing comparisons come from separate runs on a shared machine.")
A("\n14. REPRODUCIBILITY")
A(f"  scripts/phase2c.py | random_state=42 | sklearn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}")
A("  Frozen manifests verified before/after (SHA-256 of the manifests themselves recorded in phase2c_config.json).")
A(f"  Total time {total_time / 60:.1f} min | process peak working set {peak_ws:.1f} GB, committed {peak_commit:.1f} GB")
A("\nSANITY CHECKS")
for k in sanity:
    A(f"  [PASS] {k}")
A("\nPHASE 2C COMPLETE — awaiting review. (No final comparison, ensemble, or optimisation.)")
report = "\n".join(L)
log("\n" + report)
OUT["report"].write_text(report, encoding="utf-8")

with open(OUT["config"], "w") as fh:
    json.dump({"phase": "2C", "ablation": {"removed_feature": PORT, "n_with": 70, "n_without": 69},
               "isolation_forest": {**IF_PARAMS, "rows": "frozen Phase 2A split", "preprocessing":
                                    "median imputer + StandardScaler fit on BENIGN TRAIN (69 features)"},
               "dbscan": {"rows": "frozen Phase 2B 50k sample", "eps": EPS_NP, "min_samples": PRIMARY_MS,
                          "eps_rule": "Kneedle on log10 sorted k-distance (as Phase 2B)",
                          "min_samples_rule": "candidate in {5,10,20} closest to ln(n)",
                          "metric": "euclidean", "algorithm": "brute", "transductive": True},
               "wording_thresholds": {"material_metric_delta": MATERIAL_METRIC_DELTA,
                                      "material_score_corr": MATERIAL_SCORE_CORR,
                                      "jaccard_largely_same": JACCARD_LARGELY_SAME,
                                      "jaccard_substantial": JACCARD_SUBSTANTIAL,
                                      "category_notable_pp": CATEGORY_NOTABLE_PP},
               "frozen_manifest_sha256": {k: v["manifest_sha256"] for k, v in frozen_before.items()},
               "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__}}, fh, indent=2)

with open(OUT["metrics"], "w") as fh:
    json.dump({"port_descriptive": port_desc, "port_by_category": cat_ports,
               "if": {"with_port": m_with, "without_port": m_np, "delta_without_minus_with": if_delta,
                      "pearson_score": pear, "spearman_score": spear, "prediction_agreement": agree,
                      "predictions_changed": changed, "flips": flips_by_class,
                      "train_time_s": if_train_time, "predict_time_s": if_predict_time,
                      "score_time_s": if_score_time, "peak_ws_gb": if_peak_ws,
                      "category_with": if_cat_with, "category_without": if_cat_np,
                      "benign_flag_rate_with": if_benign_with, "benign_flag_rate_without": if_benign_np},
               "dbscan": {"elbows_without": {ms: {r: list(v) for r, v in e.items()} for ms, e in elb_np.items()},
                          "elbows_with": {ms: {r: list(v) for r, v in e.items()} for ms, e in elb_with.items()},
                          "robustness_without": robust_np, "clarity": clarity, "geometry": geometry,
                          "summary_without": db_summary, "metrics_with": dm_with, "metrics_without": dm_np,
                          "delta_without_minus_with": db_delta, "noise_overlap": overlap,
                          "category_with": db_cat_with, "category_without": db_cat_np,
                          "composition_with": comp_with, "composition_without": comp_np,
                          "kdist_time_s": kdist_time},
               "sanity": {k: True for k in sanity}, "peak_ws_gb": peak_ws, "total_time_s": total_time},
              fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))

hashed = [OUT[k] for k in ("report", "metrics", "config", "features", "db_assign", "if_model", "if_imputer", "if_scaler")]
hashed += [RES / "phase2c_dbscan_preprocessing.pkl", Path(__file__).resolve()] + sorted(PLOT_DIR.glob("*.png"))
with open(OUT["manifest"], "w") as fh:
    json.dump({"phase": "2C", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "frozen_manifest_sha256_verified": {k: v["manifest_sha256"] for k, v in frozen_after.items()},
               "sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in hashed}}, fh, indent=2)
final = verify_frozen()
assert not final["2A"]["modified"] and not final["2B"]["modified"]
log(f"\nManifest written: {OUT['manifest'].relative_to(ROOT)}; frozen 2A/2B verified unchanged.")
