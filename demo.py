"""
LIVE DEMO — Anomaly-Based Intrusion Detection on CIC-IDS2017 (Isolation Forest + DBSCAN)

    python demo.py              full demo (~2 min; DBSCAN step needs ~8 GB free RAM)
    python demo.py --quick      skip the live DBSCAN run (uses the frozen cluster labels)
    python demo.py --open       also open the final comparison figures at the end

What it does, live:
  1. Verifies the frozen experiment artifacts (SHA-256 manifests).
  2. Loads and cleans all 8 CIC-IDS2017 CSV files, rebuilds the approved train/test split.
  3. Trains Isolation Forest on BENIGN traffic only and scores 1,054,824 unseen test flows.
  4. Runs DBSCAN on the 50,000-row sample with the frozen unsupervised parameters.
  5. Confirms every live result is identical to the frozen, reported result.
Nothing in models/, results/ or plots/ is modified; demo outputs go to demo_output/.
"""
import gc
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
# scikit-learn may be blocked by Windows Smart App Control (unsigned .pyd files). In that case
# the demo runs in REPLAY mode: everything except model fitting/clustering still runs live,
# and the frozen model outputs are evaluated live instead.
try:
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    SKLEARN, SKLEARN_ERR = True, None
except ImportError as e:
    SKLEARN, SKLEARN_ERR = False, str(e)
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
OUT = ROOT / "demo_output"
OUT.mkdir(exist_ok=True)
QUICK, OPEN = "--quick" in sys.argv, "--open" in sys.argv
ZERO_VAR = ["Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
            "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"]
CATS = ["DoS Hulk", "DDoS", "PortScan", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest",
        "FTP-Patator", "SSH-Patator", "Bot", "Web Attack - Brute Force", "Web Attack - XSS",
        "Infiltration", "Web Attack - Sql Injection", "Heartbleed"]
T0 = time.time()
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def step(n, title):
    say("\n" + "=" * 78)
    say(f" STEP {n}  {title}")
    say("=" * 78)


def ok(s):
    say(f"   [OK] {s}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def free_ram_gb():
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 1e9
    except Exception:
        return float("nan")


def metrics(y, p):
    """Confusion-matrix metrics in plain numpy (no scikit-learn needed)."""
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(TN=tn, FP=fp, FN=fn, TP=tp, precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0, fpr=fp / (fp + tn), fnr=fn / (fn + tp))


def roc_auc(y, s):
    """ROC-AUC via the Mann-Whitney rank statistic (ties get average ranks)."""
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def show_cm(m, name):
    say(f"   {name} confusion matrix         Pred normal   Pred anomaly")
    say(f"     True BENIGN               {m['TN']:>11,}   {m['FP']:>12,}")
    say(f"     True ATTACK               {m['FN']:>11,}   {m['TP']:>12,}")


say("#" * 78)
say("  ANOMALY-BASED INTRUSION DETECTION — CIC-IDS2017")
say("  Isolation Forest (out-of-sample)  vs  DBSCAN (transductive density clustering)")
say("#" * 78)
if not SKLEARN:
    say("\n  NOTE: REPLAY MODE — scikit-learn could not be loaded on this machine:")
    say(f"        {SKLEARN_ERR}")
    say("        (Windows Smart App Control blocks scikit-learn's unsigned compiled modules.)")
    say("        Data loading, cleaning, the train/test split and ALL metrics run live below;")
    say("        model fitting and clustering are replayed from the frozen, hash-verified outputs.")

# ---------------------------------------------------------------------------
step(1, "Verify frozen experiment artifacts (SHA-256)")
for tag, mp in (("Phase 2A  Isolation Forest", "results/phase2a_frozen_manifest.json"),
                ("Phase 2B  DBSCAN", "results/phase2b_manifest.json"),
                ("Phase 2C  Destination Port ablation", "results/phase2c_manifest.json"),
                ("Phase 3   Final synthesis", "results/phase3_manifest.json")):
    m = json.load(open(mp))
    bad = [p for p, d in m["sha256"].items() if sha256(p) != d]
    assert not bad, f"{tag}: modified files {bad}"
    ok(f"{tag:<38} {len(m['sha256']):>2} files unchanged")

# ---------------------------------------------------------------------------
step(2, "Load and clean CIC-IDS2017")
t = time.time()
dfs = []
for f in glob.glob("*.csv"):
    d = pd.read_csv(f)
    d.columns = d.columns.str.strip()
    say(f"   {f:<62}{len(d):>9,} rows")
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
del dfs
raw_rows = len(df)
df = df.drop_duplicates()
labels = df["Label"].str.replace("�", "-", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
num = df.select_dtypes(include=[np.number]).columns
df[num] = df[num].replace([np.inf, -np.inf], np.nan)
X = df.drop(columns=["Label"] + ZERO_VAR)
del df
gc.collect()
say(f"   Raw flows {raw_rows:,} -> {len(X):,} after removing exact duplicates")
say(f"   Features: {X.shape[1]} numeric (8 zero-variance columns dropped; Destination Port kept)")
say(f"   BENIGN {int((labels == 'BENIGN').sum()):,} | ATTACK {int((labels != 'BENIGN').sum()):,} "
    f"across {labels.nunique() - 1} attack categories   ({time.time() - t:.0f} s)")
assert len(X) == 2_522_362 and X.shape[1] == 70

# ---------------------------------------------------------------------------
step(3, "Isolation Forest — train on BENIGN only, test on unseen traffic")
benign = (labels == "BENIGN").to_numpy()
Xb, Xa = X[benign], X[~benign]
lb, la = labels[benign], labels[~benign]
n_tr = int(len(Xb) * 0.7)
np.random.seed(42)
perm = np.random.permutation(len(Xb))
X_train, X_test = Xb.iloc[perm[:n_tr]], pd.concat([Xb.iloc[perm[n_tr:]], Xa])
lab_test = pd.concat([lb.iloc[perm[n_tr:]], la]).to_numpy()
y_test = (lab_test != "BENIGN").astype(np.int8)
del Xb, Xa
say(f"   TRAIN: {len(X_train):,} BENIGN flows (no attacks, no labels used by the model)")
say(f"   TEST : {len(X_test):,} unseen flows = {int((y_test == 0).sum()):,} BENIGN + {int((y_test == 1).sum()):,} ATTACK")
say("   Model: IsolationForest(n_estimators=200, max_samples='auto', contamination='auto', random_state=42)")
frozen = np.load("results/if_test_outputs.npz")
assert np.array_equal(frozen["y_test_binary"], y_test)
ok("Rebuilt test rows match the frozen Phase 2A test set exactly (same rows, same order)")
if SKLEARN:
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(X_train)).astype(np.float32)
    Xte = sc.transform(imp.transform(X_test)).astype(np.float32)
    del X_train, X_test
    gc.collect()
    t = time.time()
    model = IsolationForest(n_estimators=200, max_samples="auto", contamination="auto",
                            random_state=42, n_jobs=-1).fit(Xtr)
    fit_s = time.time() - t
    t = time.time()
    score = -model.decision_function(Xte)       # higher = more anomalous
    score_s = time.time() - t
    say(f"   Trained 200 trees LIVE in {fit_s:.1f} s; scored {len(Xte):,} unseen flows in {score_s:.1f} s")
    del Xtr, Xte, model
else:
    del X_train, X_test
    score = frozen["anomaly_score"].astype(np.float64)
    say("   [REPLAY] Using the frozen Phase 2A anomaly scores for these 1,054,824 test flows")
gc.collect()
pred = (score > 0).astype(np.int8)              # 0 normal, 1 anomaly ('auto' threshold)
m_if = metrics(y_test, pred)
auc = roc_auc(y_test, score)
show_cm(m_if, "Isolation Forest")
say(f"   Precision {m_if['precision']:.4f} | Recall {m_if['recall']:.4f} | F1 {m_if['f1']:.4f} | "
    f"FPR {m_if['fpr']:.4f} | ROC-AUC {auc:.4f}")
assert np.array_equal(score.astype(np.float32), frozen["anomaly_score"])
assert np.array_equal(pred, frozen["y_pred_if"])
ok("Scores and predictions identical to the frozen, reported Phase 2A result")
say("   % flagged as anomalous, by attack category (not attack-type classification):")
for c in CATS:
    msk = lab_test == c
    say(f"     {c:<28}{int(msk.sum()):>9,} flows  {100 * pred[msk].mean():6.1f}%"
        + ("   (small n: unstable)" if msk.sum() < 100 else ""))

# ---------------------------------------------------------------------------
step(4, "DBSCAN — density clustering of a 50,000-row representative sample")
ids = pd.read_csv("results/phase2b_sample_ids.csv")
ids = ids[ids.in_dbscan_working_sample].sort_values("sample_order")
idx = ids["raw_concat_index"].to_numpy()
Xs = X.loc[idx]
ys = (labels.loc[idx] != "BENIGN").to_numpy().astype(np.int8)
cats_s = labels.loc[idx].to_numpy()
del X
gc.collect()
cfg = json.load(open("results/phase2b_config.json"))["dbscan"]
eps, ms = cfg["eps"], cfg["min_samples"]
say(f"   Sample: {len(Xs):,} flows drawn uniformly at random (seed 42): "
    f"{int((ys == 0).sum()):,} BENIGN + {int(ys.sum()):,} ATTACK (labels hidden from DBSCAN)")
say(f"   Parameters chosen WITHOUT labels (k-distance elbow): eps = {eps:.3f}, min_samples = {ms}")
fz = np.load("results/phase2b_cluster_assignments.npz")["primary_70d"]
ram = free_ram_gb()
if QUICK or ram < 9.5 or not SKLEARN:
    why = ("--quick" if QUICK else "scikit-learn blocked" if not SKLEARN
           else f"only {ram:.1f} GB RAM free; needs ~8 GB")
    say(f"   [REPLAY] Live DBSCAN skipped ({why}); using the frozen Phase 2B cluster labels.")
    lab_db, live_db = fz, False
else:
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(Xs))
    say(f"   Running DBSCAN live ({ram:.0f} GB RAM free) ...")
    t = time.time()
    lab_db = DBSCAN(eps=eps, min_samples=ms, metric="euclidean", algorithm="brute", n_jobs=-1).fit_predict(Z)
    db_s = time.time() - t
    live_db = True
    del Z
    say(f"   DBSCAN finished in {db_s:.1f} s")
pred_db = (lab_db == -1).astype(np.int8)        # noise (-1) = anomaly
n_cl = int(lab_db.max() + 1)
sizes = np.sort(np.bincount(lab_db[lab_db >= 0]))[::-1]
say(f"   {n_cl} clusters | noise {int(pred_db.sum()):,} rows ({100 * pred_db.mean():.2f}%) | "
    f"largest cluster {sizes[0]:,} rows ({100 * sizes[0] / len(lab_db):.0f}% of sample)")
m_db = metrics(ys, pred_db)
show_cm(m_db, "DBSCAN")
say(f"   Precision {m_db['precision']:.4f} | Recall {m_db['recall']:.4f} | F1 {m_db['f1']:.4f} | "
    f"FPR {m_db['fpr']:.4f} | ROC-AUC n/a (binary noise output)")
if live_db:
    assert np.array_equal(pred_db, (fz == -1).astype(np.int8))
    ok("Live noise assignments identical to the frozen Phase 2B result on all 50,000 rows")
cl = pd.DataFrame({"c": lab_db, "a": ys}).query("c >= 0").groupby("c")["a"].agg(["size", "mean", "sum"])
att = int(ys.sum())
say("   Where did DBSCAN put the attack rows? (post-hoc, labels attached after clustering)")
say(f"     inside attack-dominated clusters (>=95% attack): {100 * cl[cl['mean'] >= 0.95]['sum'].sum() / att:5.1f}%")
say(f"     inside mixed clusters (5-95% attack)           : {100 * cl[(cl['mean'] > 0.05) & (cl['mean'] < 0.95)]['sum'].sum() / att:5.1f}%")
say(f"     inside benign-dominated clusters (<=5% attack) : {100 * cl[cl['mean'] <= 0.05]['sum'].sum() / att:5.1f}%")
say(f"     labelled noise (= flagged as anomalous)        : {100 * (pred_db[ys == 1].mean()):5.1f}%")

# ---------------------------------------------------------------------------
step(5, "Summary")
say("                         Isolation Forest            DBSCAN")
say("   Evaluation            1,054,824 unseen rows       50,000 sample rows (transductive)")
for k in ("precision", "recall", "f1", "fpr", "fnr"):
    say(f"   {k.upper() if len(k) == 3 else k.title():<22}{m_if[k]:<28.4f}{m_db[k]:.4f}")
say(f"   {'ROC-AUC':<22}{auc:<28.4f}n/a")
say("\n   Isolation Forest was evaluated on an unseen test population after being trained exclusively")
say("   on BENIGN traffic, whereas DBSCAN was evaluated transductively on a representative mixed sample;")
say("   the raw metrics belong to different experimental designs, not a head-to-head benchmark.")
say("\n   Key finding: many attacks in CIC-IDS2017 are DENSE, not sparse. DBSCAN keeps them inside")
say("   clusters, so few become noise; Isolation Forest, trained on BENIGN traffic only, still gives")
say("   high-volume DoS traffic high anomaly scores. The two methods define 'anomaly' differently.")
say(f"\n   Full report : results/phase3_final_report.txt")
say(f"   Figures     : plots/phase3/   Tables: results/phase3_tables/")
say(f"   Demo log    : demo_output/demo_log.txt")
say(f"   Total demo time: {time.time() - T0:.0f} s")
(OUT / "demo_log.txt").write_text("\n".join(LOG), encoding="utf-8")
if OPEN:
    for p in ("plots/phase3/01_confusion_matrices.png", "plots/phase3/05_attack_category_profile.png",
              "plots/phase3/08_dbscan_cluster_composition.png"):
        os.startfile(ROOT / p)
