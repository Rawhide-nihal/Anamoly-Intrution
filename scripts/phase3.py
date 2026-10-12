"""
PHASE 3 — FINAL DBSCAN vs ISOLATION FOREST COMPARISON AND PROJECT SYNTHESIS

Synthesis only. No model is trained, refit, rescored or re-thresholded; every number is
read from the frozen Phase 2A / 2B / 2C artifacts, each of which must pass its SHA-256
manifest before anything is written. No ranking, overall score, ensemble or winner.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
RES = ROOT / "results"
TAB_DIR = RES / "phase3_tables"
PLOT_DIR = ROOT / "plots" / "phase3"
MANIFESTS = {"2A": RES / "phase2a_frozen_manifest.json", "2B": RES / "phase2b_manifest.json",
             "2C": RES / "phase2c_manifest.json"}
SOURCES = {  # every file Phase 3 reads; each must be covered by a frozen manifest
    "2A_metrics": "results/phase2a_metrics.json", "2A_outputs": "results/if_test_outputs.npz",
    "2B_metrics": "results/phase2b_metrics.json", "2B_config": "results/phase2b_config.json",
    "2C_metrics": "results/phase2c_metrics.json", "2C_config": "results/phase2c_config.json",
}
OUT_REPORT, OUT_JSON, OUT_MANIFEST = RES / "phase3_final_report.txt", RES / "phase3_comparison.json", RES / "phase3_manifest.json"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_all():
    res = {}
    for tag, mp in MANIFESTS.items():
        m = json.load(open(mp))
        bad = [p for p, d in m["sha256"].items() if sha256(ROOT / p) != d]
        res[tag] = {"files": len(m["sha256"]), "failed": bad, "manifest_sha256": sha256(mp),
                    "covered": set(m["sha256"])}
    return res


# =====================================================================
# SECTION 1 — VERIFY ALL FROZEN ARTIFACTS (stop on any failure, write nothing)
# =====================================================================
ver0 = verify_all()
for tag, v in ver0.items():
    print(f"Phase {tag} manifest: {v['files']} files, failed: {v['failed'] or 'none'}")
if any(v["failed"] for v in ver0.values()):
    sys.exit("STOP: a frozen artifact failed hash verification. Nothing regenerated or written.")
covered = set().union(*(v["covered"] for v in ver0.values()))
for k, p in SOURCES.items():
    assert p in covered, f"source {p} is not covered by a frozen manifest"
source_hashes = {p: sha256(ROOT / p) for p in SOURCES.values()}
TAB_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# SECTION 2 — LOAD FROZEN RESULTS
# =====================================================================
A = json.load(open(SOURCES["2A_metrics"]))
A_out = np.load(SOURCES["2A_outputs"])
B = json.load(open(SOURCES["2B_metrics"]))
Bcfg = json.load(open(SOURCES["2B_config"]))
C = json.load(open(SOURCES["2C_metrics"]))
Ccfg = json.load(open(SOURCES["2C_config"]))

IF = A["baseline_metrics"]
DB = B["primary"]["metrics"]
DBp = B["primary"]
comp = C["dbscan"]["composition_with"]          # post-hoc Phase 2B composition (frozen in 2C)
comp_np = C["dbscan"]["composition_without"]
y_test, if_score, if_pred = A_out["y_test_binary"], A_out["anomaly_score"], A_out["y_pred_if"]
n_work = Bcfg["sample"]["working_n"]
cw = B["sample_composition_working"]
db_attack_n = sum(v for k, v in cw.items() if k != "BENIGN")

# Cross-checks: frozen arrays vs frozen metrics, and the numbers quoted in the Phase 3 brief.
tn, fp = int(((y_test == 0) & (if_pred == 0)).sum()), int(((y_test == 0) & (if_pred == 1)).sum())
fn, tp = int(((y_test == 1) & (if_pred == 0)).sum()), int(((y_test == 1) & (if_pred == 1)).sum())
assert (tn, fp, fn, tp) == (IF["TN"], IF["FP"], IF["FN"], IF["TP"])
EXPECT = {
    ("IF", "precision"): 0.8387, ("IF", "recall"): 0.5789, ("IF", "f1"): 0.6850, ("IF", "fpr"): 0.0754,
    ("IF", "fnr"): 0.4211, ("IF", "roc_auc"): 0.8082,
    ("DB", "precision"): 0.1061, ("DB", "recall"): 0.0205, ("DB", "f1"): 0.0344, ("DB", "fpr"): 0.0346,
    ("DB", "fnr"): 0.9795,
}
for (alg, k), v in EXPECT.items():
    assert round((IF if alg == "IF" else DB)[k], 4) == v, (alg, k)
assert (IF["TN"], IF["FP"], IF["FN"], IF["TP"]) == (581_525, 47_421, 179_318, 246_560)
assert (DB["TN"], DB["FP"], DB["FN"], DB["TP"]) == (40_224, 1_441, 8_164, 171)
assert n_work == 50_000 and DBp["min_samples"] == 10 and round(DBp["eps"], 3) == 2.552
assert (comp["n_attack_dominated_clusters"], round(comp["pct_attack_in_attack_dominated"]),
        round(comp["pct_attack_in_mixed"]), round(comp["pct_attack_in_noise"])) == (11, 41, 56, 2)
pd_ = C["port_descriptive"]
assert (pd_["n_unique"], pd_["median"], pd_["n_zero"]) == (53_805, 80, 1_567)
top3 = sum(t["share_pct"] for t in pd_["top10"][:3])
assert [t["port"] for t in pd_["top10"][:3]] == [53, 80, 443] and round(top3) == 75
assert pd_["by_class"]["ATTACK"]["top5"][0]["port"] == 80 and round(pd_["by_class"]["ATTACK"]["top5"][0]["share_pct"]) == 76
Ci, Cd = C["if"], C["dbscan"]
assert (round(Ci["without_port"]["f1"], 4), round(Ci["without_port"]["roc_auc"], 4),
        round(Ci["pearson_score"], 3), round(100 * Ci["prediction_agreement"], 1)) == (0.6733, 0.8140, 0.997, 98.8)
assert (round(Cd["summary_without"]["noise_pct"], 2), round(Cd["metrics_without"]["recall"], 3),
        round(json.load(open(SOURCES["2C_config"]))["dbscan"]["eps"], 3)) == (2.30, 0.012, 3.1)
assert Cd["geometry"]["without_port"]["duplicate_feature_rows"] == 2_872
pca = {r["n_components"]: r for r in B["pca_runs"]}
assert sorted(pca) == [19, 23, 31] and [round(pca[k]["noise_pct"], 2) for k in (19, 23, 31)] == [3.70, 3.73, 3.15]
assert max(r["metrics"]["recall"] for r in B["pca_runs"]) < 0.03
sens = [r for r in B["sensitivity_runs"] if r["status"].startswith("run")]
noise_rng = (min(r["noise_pct"] for r in sens), max(r["noise_pct"] for r in sens))
jac_rng = (min(r["noise_jaccard_vs_primary"] for r in sens if r["status"] == "run"),
           max(r["noise_jaccard_vs_primary"] for r in sens if r["status"] == "run"))
assert (round(noise_rng[0], 2), round(noise_rng[1], 1), round(jac_rng[0], 2), round(jac_rng[1], 2)) == (1.55, 11.0, 0.29, 0.74)
assert round(B["feasibility_100k"]["est_gb"]) == 29
print("All brief-quoted values match the frozen artifacts.")

IF_TEST_BENIGN, IF_TEST_ATTACK = IF["TN"] + IF["FP"], IF["TP"] + IF["FN"]
POP_N, POP_BENIGN, POP_ATTACK = (pd_["n_rows"], pd_["by_class"]["BENIGN"]["rows"], pd_["by_class"]["ATTACK"]["rows"])
assert (IF_TEST_BENIGN, IF_TEST_ATTACK, POP_N) == (628_946, 425_878, 2_522_362)
stab200 = next(r for r in A["sensitivity_n_estimators_val_stability"] if r["n_estimators"] == 200)
IF_MODEL_MB = A["artifact_sizes_mb"]["isolation_forest"]

CATS = ["DoS Hulk", "DDoS", "PortScan", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest",
        "FTP-Patator", "SSH-Patator", "Bot", "Web Attack - Brute Force", "Web Attack - XSS",
        "Infiltration", "Web Attack - Sql Injection", "Heartbleed"]
if_cat = {r["category"]: r for r in A["attack_categories"]}
db_cat = {r["category"]: r for r in B["attack_categories"]}

POP_IF = "IF: unseen test set, 1,054,824 rows (628,946 BENIGN + 425,878 ATTACK)"
POP_DB = f"DBSCAN: transductive 50,000-row seeded sample ({cw['BENIGN']:,} BENIGN + {db_attack_n:,} ATTACK)"
MANDATORY = ("Isolation Forest was evaluated on an unseen test population after being trained exclusively on "
             "BENIGN traffic, whereas DBSCAN was evaluated transductively on a representative mixed sample. "
             "Therefore, their raw metric values should be interpreted within their respective experimental "
             "designs rather than as a direct apples-to-apples benchmark.")

# =====================================================================
# SECTION 17 — TABLES (all values from frozen artifacts)
# =====================================================================
f4 = lambda v: f"{v:.4f}"
T = {}
T["T01_populations"] = pd.DataFrame([
    {"Item": "Deduplicated population", "Isolation Forest": f"{POP_N:,} rows", "DBSCAN": f"{POP_N:,} rows"},
    {"Item": "Rows used to fit", "Isolation Forest": f"{A['n_train']:,} BENIGN (TRAIN)",
     "DBSCAN": f"{n_work:,} mixed rows (labels hidden)"},
    {"Item": "Rows evaluated", "Isolation Forest": f"{A['n_test']:,} unseen TEST rows",
     "DBSCAN": f"the same {n_work:,} rows (transductive)"},
    {"Item": "BENIGN evaluated", "Isolation Forest": f"{IF_TEST_BENIGN:,}", "DBSCAN": f"{cw['BENIGN']:,}"},
    {"Item": "ATTACK evaluated", "Isolation Forest": f"{IF_TEST_ATTACK:,}", "DBSCAN": f"{db_attack_n:,}"},
    {"Item": "Attack prevalence", "Isolation Forest": f"{IF_TEST_ATTACK / A['n_test']:.2%}",
     "DBSCAN": f"{db_attack_n / n_work:.2%}"},
    {"Item": "Selection", "Isolation Forest": "seeded 70/30 permutation of BENIGN + all attacks",
     "DBSCAN": "uniform random sample, random_state=42 (50k fallback of 100k draw: memory)"},
])
T["T02_configurations"] = pd.DataFrame([
    {"Setting": "Algorithm", "Isolation Forest": "sklearn IsolationForest", "DBSCAN": "sklearn DBSCAN (Euclidean, brute)"},
    {"Setting": "Features", "Isolation Forest": "70 (Destination Port retained)", "DBSCAN": "70 (Destination Port retained)"},
    {"Setting": "Preprocessing", "Isolation Forest": "inf->NaN, median impute, StandardScaler (fit on BENIGN TRAIN)",
     "DBSCAN": "inf->NaN, median impute, StandardScaler (fit on the 50k sample)"},
    {"Setting": "Key parameters", "Isolation Forest": "n_estimators=200, max_samples='auto' (256), contamination='auto', random_state=42",
     "DBSCAN": f"eps={DBp['eps']:.3f}, min_samples={DBp['min_samples']}"},
    {"Setting": "How parameters were set", "Isolation Forest": "pre-specified baseline",
     "DBSCAN": "log10-Kneedle on k-distance curve; min_samples closest to ln(n); no labels"},
    {"Setting": "Anomaly decision", "Isolation Forest": "-decision_function > 0", "DBSCAN": "cluster label == -1 (noise)"},
])
T["T03_performance"] = pd.DataFrame([
    {"Measure": "Training paradigm", "Isolation Forest": "trained on BENIGN only", "DBSCAN": "unsupervised clustering of a mixed sample"},
    {"Measure": "Evaluation paradigm", "Isolation Forest": "out-of-sample (unseen test set)", "DBSCAN": "transductive (same rows clustered)"},
    {"Measure": "Features", "Isolation Forest": "70", "DBSCAN": "70"},
    {"Measure": "Evaluation size", "Isolation Forest": "1,054,824", "DBSCAN": f"{n_work:,}"},
    *[{"Measure": k.upper() if len(k) <= 3 else k.title(), "Isolation Forest": f4(IF[k]), "DBSCAN": f4(DB[k])}
      for k in ("precision", "recall", "f1", "fpr", "fnr")],
    {"Measure": "ROC-AUC", "Isolation Forest": f4(IF["roc_auc"]), "DBSCAN": "not applicable (binary noise output)"},
    {"Measure": "Runtime", "Isolation Forest": f"fit {A['train_time_s']:.2f} s; predict {A['predict_time_s']:.2f} s",
     "DBSCAN": f"{DBp['runtime_s']:.1f} s (one clustering run)"},
    {"Measure": "Memory", "Isolation Forest": f"peak process {A['peak_ram_gb']:.1f} GB (data loading dominates)",
     "DBSCAN": f"~{DBp['pairs'] * 24 / 1e9:.1f} GB est. neighbour storage; process peak {B['peak_ws_gb']:.1f} GB over all 2B runs"},
])
T["T04_confusion"] = pd.DataFrame([
    {"Algorithm": "Isolation Forest", "Population": "1,054,824 test rows", "TN": IF["TN"], "FP": IF["FP"], "FN": IF["FN"], "TP": IF["TP"]},
    {"Algorithm": "DBSCAN", "Population": "50,000 sample rows", "TN": DB["TN"], "FP": DB["FP"], "FN": DB["FN"], "TP": DB["TP"]},
])
rows = []
for c in CATS:
    i, d = if_cat[c], db_cat[c]
    rows.append({"Attack category": c, "IF n": i["total"], "IF % flagged as anomalous": round(100 * i["detection_rate"], 2),
                 "IF note": "small n (<100): unstable" if i["total"] < 100 else "",
                 "DBSCAN n": d["n_in_sample"] if d["n_in_sample"] else "Not present",
                 "DBSCAN % flagged as anomalous": round(d["pct_flagged_anomalous"], 2) if d["n_in_sample"] else "Not present",
                 "DBSCAN note": ("" if d["status"] == "ok" else "statistically unstable (n<30)" if d["n_in_sample"] else "")})
T["T05_attack_categories"] = pd.DataFrame(rows)
Cdw, Cdn = Cd["metrics_with"], Cd["metrics_without"]
T["T06_port_ablation"] = pd.DataFrame([
    {"Algorithm": "Isolation Forest", "Measure": m, "WITH Port (70)": f4(Ci["with_port"][k]),
     "WITHOUT Port (69)": f4(Ci["without_port"][k]), "Delta": f"{Ci['delta_without_minus_with'][k]:+.4f}"}
    for m, k in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("FPR", "fpr"), ("FNR", "fnr"), ("ROC-AUC", "roc_auc"))
] + [
    {"Algorithm": "Isolation Forest", "Measure": "Score Pearson / prediction agreement", "WITH Port (70)": "",
     "WITHOUT Port (69)": "", "Delta": f"{Ci['pearson_score']:.3f} / {100 * Ci['prediction_agreement']:.1f}% ({Ci['predictions_changed']:,} changed)"},
    {"Algorithm": "DBSCAN", "Measure": "eps (re-selected, same rule)", "WITH Port (70)": f"{DBp['eps']:.3f}",
     "WITHOUT Port (69)": f"{Ccfg['dbscan']['eps']:.3f}", "Delta": f"{Ccfg['dbscan']['eps'] - DBp['eps']:+.3f}"},
    {"Algorithm": "DBSCAN", "Measure": "Clusters", "WITH Port (70)": str(DBp["n_clusters"]),
     "WITHOUT Port (69)": str(Cd["summary_without"]["n_clusters"]), "Delta": f"{Cd['summary_without']['n_clusters'] - DBp['n_clusters']:+d}"},
    {"Algorithm": "DBSCAN", "Measure": "Noise %", "WITH Port (70)": f"{DBp['noise_pct']:.2f}",
     "WITHOUT Port (69)": f"{Cd['summary_without']['noise_pct']:.2f}", "Delta": f"{Cd['summary_without']['noise_pct'] - DBp['noise_pct']:+.2f} pp"},
] + [
    {"Algorithm": "DBSCAN", "Measure": m, "WITH Port (70)": f4(Cdw[k]), "WITHOUT Port (69)": f4(Cdn[k]),
     "Delta": f"{Cd['delta_without_minus_with'][k]:+.4f}"}
    for m, k in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("FPR", "fpr"), ("FNR", "fnr"))
] + [
    {"Algorithm": "DBSCAN", "Measure": "Noise-set Jaccard / rows changed", "WITH Port (70)": "", "WITHOUT Port (69)": "",
     "Delta": f"{Cd['noise_overlap']['jaccard']:.3f} / {Cd['noise_overlap']['changed_assignment']:,}"},
    {"Algorithm": "DBSCAN", "Measure": "Feature-identical rows in 50k", "WITH Port (70)": str(Cd["geometry"]["with_port"]["duplicate_feature_rows"]),
     "WITHOUT Port (69)": f"{Cd['geometry']['without_port']['duplicate_feature_rows']:,}", "Delta": ""},
])
T["T07_pca"] = pd.DataFrame(
    [{"Space": "Standardized (no PCA)", "Components": 70, "Cum. variance": 1.0, "eps": round(DBp["eps"], 3),
      "Clusters": DBp["n_clusters"], "Noise %": round(DBp["noise_pct"], 2), "Precision": round(DB["precision"], 4),
      "Recall": round(DB["recall"], 4), "F1": round(DB["f1"], 4), "Runtime s": round(DBp["runtime_s"], 1)}] +
    [{"Space": f"PCA {r['target_variance']:.0%}", "Components": r["n_components"], "Cum. variance": round(r["cum_variance"], 4),
      "eps": round(r["eps"], 3), "Clusters": r["n_clusters"], "Noise %": round(r["noise_pct"], 2),
      "Precision": round(r["metrics"]["precision"], 4), "Recall": round(r["metrics"]["recall"], 4),
      "F1": round(r["metrics"]["f1"], 4), "Runtime s": round(r["runtime_s"], 1)} for r in B["pca_runs"]])
T["T08_sensitivity"] = pd.DataFrame([
    {"min_samples": r["min_samples"], "eps": round(r["eps"], 3),
     "setting": f"x{r['eps_multiplier']}" if r["eps_multiplier"] else r["kind"], "status": r["status"],
     "clusters": r.get("n_clusters", ""), "noise %": round(r["noise_pct"], 2) if "noise_pct" in r else "",
     "Jaccard vs primary": round(r["noise_jaccard_vs_primary"], 3) if "noise_jaccard_vs_primary" in r else "",
     "recall": round(r["metrics"]["recall"], 4) if "metrics" in r else "",
     "est. memory GB": round(r["est_gb"], 1)} for r in B["sensitivity_runs"]])
T["T09_computation"] = pd.DataFrame([
    {"Aspect": "Rows processed", "Isolation Forest": f"fit {A['n_train']:,}; score {A['n_test']:,}", "DBSCAN": f"{n_work:,} (clustered once per configuration)"},
    {"Aspect": "Fit / cluster time", "Isolation Forest": f"{A['train_time_s']:.2f} s (200 trees x 256 rows)", "DBSCAN": f"{DBp['runtime_s']:.1f} s primary; PCA runs {min(r['runtime_s'] for r in B['pca_runs']):.1f}-{max(r['runtime_s'] for r in B['pca_runs']):.1f} s"},
    {"Aspect": "Prediction", "Isolation Forest": f"{A['predict_time_s']:.2f} s for 1,054,824 unseen rows", "DBSCAN": "no predict(); labels exist only for clustered rows"},
    {"Aspect": "Dominant memory cost", "Isolation Forest": f"data loading (peak {A['peak_ram_gb']:.1f} GB process)", "DBSCAN": f"neighbour storage: {DBp['pairs']:,} pairs (~{DBp['pairs'] * 24 / 1e9:.1f} GB est.)"},
    {"Aspect": "Scaling limit observed", "Isolation Forest": "none reached", "DBSCAN": f"100k rows needed ~{B['feasibility_100k']['est_gb']:.1f} GB (> 10 GB budget); 1.5x eps runs skipped"},
    {"Aspect": "Saved model size", "Isolation Forest": f"{IF_MODEL_MB:.2f} MB (frozen 200-tree pickle, compressed)", "DBSCAN": "n/a (cluster labels for the sample)"},
])
T["T10_methodology"] = pd.DataFrame([
    {"Property": "Anomaly definition", "Isolation Forest": "easy to isolate by random partitions (short average path length)", "DBSCAN": "density noise: not a core point and not within eps of one"},
    {"Property": "Training data", "Isolation Forest": "BENIGN only", "DBSCAN": "unlabelled mixed sample"},
    {"Property": "Unseen observations", "Isolation Forest": "predict()/decision_function on new rows", "DBSCAN": "no standard sklearn predict(); transductive"},
    {"Property": "Output", "Isolation Forest": "continuous score + threshold", "DBSCAN": "binary noise / cluster id"},
    {"Property": "ROC analysis", "Isolation Forest": "yes (score)", "DBSCAN": "no intrinsic score"},
    {"Property": "Parameter sensitivity shown", "Isolation Forest": f"2A: n=200 cross-seed score Spearman {stab200['mean_seed_spearman']:.3f}", "DBSCAN": f"noise {noise_rng[0]:.2f}-{noise_rng[1]:.2f}% across grid"},
    {"Property": "Scale dependence", "Isolation Forest": "per-feature splits", "DBSCAN": "Euclidean distances; standardisation essential"},
])
for name, df in T.items():
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)
def to_md(df):
    cells = [[str(c) for c in df.columns]] + [[str(v).replace("|", "/") for v in r] for r in df.itertuples(index=False)]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(cells[0])]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in cells[1:]])


with open(TAB_DIR / "all_tables.md", "w", encoding="utf-8") as fh:
    for name, df in T.items():
        fh.write(f"## {name}\n\n{to_md(df)}\n\n")

# =====================================================================
# SECTION 18 — VISUALIZATIONS (from frozen values only)
# =====================================================================
C_IF, C_DB, C_3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "savefig.dpi": 150, "savefig.bbox": "tight",
})
NOTE = "Different evaluation populations — not a head-to-head benchmark"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, M, ttl in ((axes[0], IF, POP_IF), (axes[1], DB, POP_DB)):
    cm = np.array([[M["TN"], M["FP"]], [M["FN"], M["TP"]]])
    rp = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(rp, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({rp[i, j]:.1%} of row)", ha="center", va="center",
                    color="white" if rp[i, j] > 0.55 else INK, fontsize=9)
    ax.set_xticks([0, 1], ["Predicted normal", "Predicted anomaly"])
    ax.set_yticks([0, 1], ["BENIGN", "ATTACK"])
    ax.set_title(ttl.replace(": ", "\n", 1).replace(" (", "\n(", 1), fontsize=9)
fig.suptitle(f"Primary confusion matrices — {NOTE}", fontsize=11, y=1.08)
fig.savefig(PLOT_DIR / "01_confusion_matrices.png")
plt.close(fig)


def two_bars(keys, fname, title, ylim=1.0):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x, w = np.arange(len(keys)), 0.38
    for j, (M, lab, col) in enumerate(((IF, "Isolation Forest — unseen test set (1,054,824)", C_IF),
                                       (DB, f"DBSCAN — transductive sample ({n_work:,})", C_DB))):
        bb = ax.bar(x + (j - 0.5) * w, [M[k] for k in keys], w - 0.03, color=col, label=lab)
        for b, k in zip(bb, keys):
            ax.text(b.get_x() + b.get_width() / 2, M[k], f"{M[k]:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [k.upper() if len(k) <= 3 else k.title() for k in keys])
    ax.set_ylim(0, ylim * 1.12)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f"{title}\n{NOTE}", fontsize=10)
    fig.savefig(PLOT_DIR / fname)
    plt.close(fig)


two_bars(["precision", "recall", "f1"], "02_precision_recall_f1.png", "Precision / Recall / F1 (primary configurations)")
two_bars(["fpr", "fnr"], "03_fpr_fnr.png", "False-positive and false-negative rates (primary configurations)")

fpr_c, tpr_c, _ = roc_curve(y_test, if_score)
fig, ax = plt.subplots(figsize=(5.4, 5.2))
ax.plot(fpr_c, tpr_c, color=C_IF, linewidth=2, label=f"Isolation Forest (AUC = {IF['roc_auc']:.4f})")
ax.plot([0, 1], [0, 1], color=INK_2, linestyle=":", linewidth=1, label="Chance")
ax.scatter([IF["fpr"]], [IF["recall"]], s=60, color=C_DB, edgecolor="#fcfcfb", linewidth=2, zorder=3,
           label="Operating point (contamination='auto')")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC — Isolation Forest only (frozen scores)\nunseen test set, 1,054,824 rows; "
             "DBSCAN has no continuous score", fontsize=9)
ax.legend(frameon=False, loc="lower right", fontsize=8)
fig.savefig(PLOT_DIR / "04_roc_isolation_forest.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6.4))
yy = np.arange(len(CATS))[::-1]
for y, c in zip(yy, CATS):
    i, d = if_cat[c], db_cat[c]
    ax.scatter(100 * i["detection_rate"], y, s=70, color=C_IF, zorder=3, marker="o" if i["total"] >= 100 else "s")
    if d["n_in_sample"]:
        ax.scatter(d["pct_flagged_anomalous"], y, s=70, color=C_DB, zorder=3, marker="o" if d["n_in_sample"] >= 30 else "s")
    else:
        ax.text(101, y, "DBSCAN: not present", va="center", fontsize=8, color=INK_2)
ax.set_yticks(yy, [f"{c}  (IF n={if_cat[c]['total']:,}; DBSCAN n={db_cat[c]['n_in_sample'] or '—'})" for c in CATS])
ax.set_xlim(-2, 125)
ax.set_xticks(range(0, 101, 20))
ax.set_xlabel("% flagged as anomalous (not attack-type classification)")
ax.scatter([], [], color=C_IF, label="Isolation Forest — unseen test set")
ax.scatter([], [], color=C_DB, label=f"DBSCAN — {n_work:,}-row sample")
ax.scatter([], [], color=INK_2, marker="s", label="small sample (IF n<100, DBSCAN n<30): unstable")
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
ax.set_title(f"Attack-category profile — {NOTE}", fontsize=10)
fig.savefig(PLOT_DIR / "05_attack_category_profile.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
keys = ["precision", "recall", "f1", "fpr", "roc_auc"]
x, w = np.arange(len(keys)), 0.38
for j, (src, lab, col) in enumerate(((Ci["with_port"], "WITH Port (70)", C_IF), (Ci["without_port"], "WITHOUT Port (69)", C_DB))):
    bb = axes[0].bar(x + (j - 0.5) * w, [src[k] for k in keys], w - 0.03, color=col, label=lab)
    for b, k in zip(bb, keys):
        axes[0].text(b.get_x() + b.get_width() / 2, src[k], f"{src[k]:.3f}", ha="center", va="bottom", fontsize=7)
axes[0].set_xticks(x, ["Precision", "Recall", "F1", "FPR", "ROC-AUC"])
axes[0].set_ylim(0, 1.05)
axes[0].legend(frameon=False, fontsize=8)
axes[0].set_title("Isolation Forest — same unseen test set (1,054,824)", fontsize=10)
keys2 = ["noise %", "recall %", "precision %"]
wv = [DBp["noise_pct"], 100 * Cdw["recall"], 100 * Cdw["precision"]]
nv = [Cd["summary_without"]["noise_pct"], 100 * Cdn["recall"], 100 * Cdn["precision"]]
x2 = np.arange(3)
for j, (vals, lab, col) in enumerate(((wv, f"WITH Port (eps {DBp['eps']:.3f})", C_IF),
                                      (nv, f"WITHOUT Port (eps {Ccfg['dbscan']['eps']:.3f})", C_DB))):
    bb = axes[1].bar(x2 + (j - 0.5) * w, vals, w - 0.03, color=col, label=lab)
    for b, v in zip(bb, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
axes[1].set_xticks(x2, ["Noise %", "Recall (%)", "Precision (%)"])
axes[1].legend(frameon=False, fontsize=8)
axes[1].set_title("DBSCAN — same 50,000 rows; eps re-selected in each space", fontsize=10)
fig.suptitle("Destination Port ablation (Phase 2C)", fontsize=11)
fig.savefig(PLOT_DIR / "06_destination_port_ablation.png")
plt.close(fig)

ms_list, mults = [5, 10, 20], [0.5, 0.75, 1.0, 1.25, 1.5]
heat = np.full((3, 5), np.nan)
jac = np.full((3, 5), np.nan)
for r in B["sensitivity_runs"]:
    if r["kind"] == "multiplier" and "noise_pct" in r:
        heat[ms_list.index(r["min_samples"]), mults.index(r["eps_multiplier"])] = r["noise_pct"]
        jac[ms_list.index(r["min_samples"]), mults.index(r["eps_multiplier"])] = r["noise_jaccard_vs_primary"]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.imshow(np.nan_to_num(heat), cmap="Blues", aspect="auto", vmin=0, vmax=np.nanmax(heat))
ax.grid(False)
for i in range(3):
    for j in range(5):
        v = heat[i, j]
        ax.text(j, i, "not run\n(memory)" if np.isnan(v) else f"{v:.2f}%\nJ={jac[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color=INK if np.isnan(v) or v < 0.6 * np.nanmax(heat) else "white")
ax.set_xticks(range(5), [f"{m}x\n{DBp['eps'] * m:.2f}" for m in mults])
ax.set_yticks(range(3), [f"min_samples={m}" for m in ms_list])
ax.set_xlabel("eps (multiple of the primary unsupervised eps)")
ax.set_title(f"DBSCAN sensitivity — noise % and Jaccard vs primary noise set\n{POP_DB}", fontsize=9)
fig.savefig(PLOT_DIR / "07_dbscan_parameter_sensitivity.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.2))
segs = [("pct_attack_in_attack_dominated", "attack-dominated clusters (>=95%)", C_DB),
        ("pct_attack_in_mixed", "mixed clusters (5-95%)", C_IF),
        ("pct_attack_in_benign_dominated", "benign-dominated (<=5%)", C_3),
        ("pct_attack_in_noise", "noise (-1)", INK)]
for y, (cp, lab) in enumerate(((comp_np, "WITHOUT Port (Phase 2C)"), (comp, "WITH Port (Phase 2B primary)"))):
    left = 0
    for key, nm, col in segs:
        v = cp[key]
        ax.barh([y], [v], left=left, color=col, height=0.55, edgecolor="#fcfcfb", linewidth=2, label=nm if y == 0 else None)
        if v > 3:
            ax.text(left + v / 2, y, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=9)
        left += v
ax.set_yticks([0, 1], ["WITHOUT Port", "WITH Port\n(primary)"])
ax.set_xlim(0, 100)
ax.set_xlabel(f"Share of the {comp['attack_rows']:,} attack rows in the 50k DBSCAN sample (post-hoc)")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32))
ax.grid(False)
ax.set_title("Where DBSCAN placed attack rows — post-hoc diagnostic on the frozen sample", fontsize=10)
fig.savefig(PLOT_DIR / "08_dbscan_cluster_composition.png")
plt.close(fig)

# =====================================================================
# REPORT
# =====================================================================
def tbl(df):
    return "\n".join("    " + ln for ln in df.to_string(index=False).splitlines())


hi = [c for c in ["DoS Hulk", "DDoS", "DoS Slowhttptest", "DoS slowloris", "DoS GoldenEye"]]
L = []
P = L.append
P("=" * 92)
P("PHASE 3 — FINAL SYNTHESIS: ISOLATION FOREST AND DBSCAN FOR ANOMALY-BASED INTRUSION DETECTION")
P("CIC-IDS2017 | all values read from frozen Phase 2A/2B/2C artifacts (hash-verified); no model rerun")
P("=" * 92)
P("\n1. EXECUTIVE SUMMARY")
P(f"  * Isolation Forest, trained only on BENIGN flows, reached ROC-AUC {IF['roc_auc']:.4f}, precision "
  f"{IF['precision']:.4f}, recall {IF['recall']:.4f} and F1 {IF['f1']:.4f} on an unseen test set of 1,054,824 rows.")
P(f"  * DBSCAN, run transductively on a {n_work:,}-row representative sample, labelled {DBp['noise_pct']:.2f}% of rows as")
P(f"    density noise; {100 * DB['recall']:.2f}% of the sample's attack rows were noise (precision {DB['precision']:.4f}).")
P(f"  * Post-hoc, {comp['pct_attack_in_attack_dominated']:.1f}% of DBSCAN's attack rows sat in {comp['n_attack_dominated_clusters']} attack-dominated "
  f"clusters and {comp['pct_attack_in_mixed']:.1f}% in mixed clusters:")
P("    many attacks are dense, not sparse, in this feature space.")
P("  * PCA (19/23/31 components) and removing Destination Port did not materially change DBSCAN's low")
P("    attack recall; removing Destination Port changed Isolation Forest only modestly.")
P("  * The two algorithms operationalise 'anomaly' differently, which explains much of their different behaviour.")
P(f"  * {MANDATORY}")
P("\n2. DATASET")
P("  CIC-IDS2017 (8 CSV files, 78 numeric features + Label). Exact duplicate rows removed ->")
P(f"  {POP_N:,} rows (BENIGN {POP_BENIGN:,}; ATTACK {POP_ATTACK:,}; prevalence {B['population_attack_prevalence']:.2%}). ±inf -> NaN.")
P("  8 zero-variance columns removed -> 70 features (Destination Port retained). 14 attack categories.")
P("  TABLE 1 — Dataset and experimental populations")
P(tbl(T["T01_populations"]))
P("\n3. EXPERIMENTAL METHODOLOGY")
P("  Isolation Forest (Phase 2A): seeded 70/30 split of BENIGN rows; TRAIN = BENIGN only; TEST = remaining BENIGN +")
P("  all attacks. Median imputer and StandardScaler fitted on TRAIN. Pre-specified configuration; contamination and")
P("  tree-count variations were reported only as sensitivity analyses.")
P(f"  DBSCAN (Phase 2B): uniform random sample (random_state=42); 100k rows were infeasible at the selected eps")
P(f"  (~{B['feasibility_100k']['est_gb']:.1f} GB of neighbour storage vs a 10 GB budget), so the documented 50k fallback (first half of")
P("  the same draw) was used. Imputer/scaler fitted on the sample; eps chosen by Kneedle on the log10 k-distance")
P("  curve and min_samples as the candidate closest to ln(n); labels sealed until all clustering finished.")
P("  Phase 2C removed only Destination Port and repeated each primary procedure unchanged.")
P("  TABLE 2 — Primary algorithm configurations")
P(tbl(T["T02_configurations"]))
P("\n  TABLE 3 — Primary performance results")
P(tbl(T["T03_performance"]))
P(f"\n  {MANDATORY}")
P("  No overall score or ranking is computed.")
P("\n4. ISOLATION FOREST")
P("  TABLE 4 — Confusion matrices (rows: truth; populations differ)")
P(tbl(T["T04_confusion"]))
P(f"  Isolation Forest flagged {IF['TP']:,} of 425,878 attack rows ({100 * IF['recall']:.1f}%) while raising false alarms on")
P(f"  {IF['FP']:,} of 628,946 unseen BENIGN rows ({100 * IF['fpr']:.2f}%); {IF['FN']:,} attack rows were not flagged. Its")
P(f"  continuous score ranks attacks above benign traffic with ROC-AUC {IF['roc_auc']:.4f}. Flag rates were high for several")
P("  high-volume DoS categories (" + ", ".join(f"{c} {100 * if_cat[c]['detection_rate']:.1f}%" for c in hi) + ")")
P("  and near zero for PortScan, FTP-/SSH-Patator, Bot and the web attacks.")
P("\n5. DBSCAN")
P(f"  Primary: eps={DBp['eps']:.3f} (the primary unsupervised eps selected by the predefined parameter-selection")
P(f"  procedure — not an 'optimal' eps), min_samples={DBp['min_samples']}; {DBp['n_clusters']} clusters, {DBp['n_noise']:,} noise rows "
  f"({DBp['noise_pct']:.2f}%), largest cluster {DBp['largest_cluster_sizes'][0]:,} rows.")
P(f"  DBSCAN produced relatively few false positives ({DB['FP']:,} of {cw['BENIGN']:,} BENIGN rows, FPR {DB['fpr']:.4f}) but")
P(f"  {DB['FN']:,} of {db_attack_n:,} attack rows in its sample were assigned to clusters rather than noise (FNR {DB['fnr']:.4f}).")
P("  Post-hoc cluster composition (frozen sample; diagnostic only, not used for any selection):")
P(f"    {comp['pct_attack_in_attack_dominated']:.1f}% of attack rows in {comp['n_attack_dominated_clusters']} highly attack-dominated clusters (>=95% attack)")
P(f"    {comp['pct_attack_in_mixed']:.1f}% in {comp['n_mixed_clusters']} mixed clusters (5-95% attack), chiefly the largest cluster "
  f"({comp['major_clusters'][0]['size']:,} rows, {100 - comp['major_clusters'][0]['attack_share_pct']:.1f}% BENIGN)")
P(f"    {comp['pct_attack_in_benign_dominated']:.1f}% in benign-dominated clusters (<=5% attack); {comp['pct_attack_in_noise']:.1f}% in noise")
P("  These figures describe this experiment's sample and parameters and are not generalised beyond it.")
P("\n  TABLE 5 — Attack-category behaviour: % flagged as anomalous (not attack-type classification accuracy)")
P(tbl(T["T05_attack_categories"]))
P("  Isolation Forest showed substantially higher flagging rates for the high-volume DoS categories (DoS Hulk, DDoS,")
P("  GoldenEye, slowloris, Slowhttptest), while DBSCAN generally assigned those observations to clusters rather than")
P("  noise (its highest rates were the slower DoS variants and Web Attack - Brute Force). Both flagged almost none of the")
P("  FTP-/SSH-Patator or PortScan rows. Heartbleed (11), Infiltration (36) and SQL Injection (21) have too few rows for")
P("  stable Isolation Forest estimates and are absent from the DBSCAN sample; XSS (10 in the sample) is unstable.")
P("\n6. DESTINATION PORT ABLATION (Phase 2C)")
P("  The analysis treated Destination Port as a legitimate network-flow feature and investigated its influence through")
P(f"  ablation. It has {pd_['n_unique']:,} unique values, median {pd_['median']:.0f}, {pd_['n_zero']:,} zero-port rows; ports 53, 80 and 443 cover")
P(f"  {top3:.1f}% of rows, and {pd_['by_class']['ATTACK']['top5'][0]['share_pct']:.1f}% of attack rows target port 80.")
P("  TABLE 6 — Destination Port ablation")
P(tbl(T["T06_port_ablation"]))
P(f"  Isolation Forest: removing the port changed F1 by {Ci['delta_without_minus_with']['f1']:+.4f} and ROC-AUC by "
  f"{Ci['delta_without_minus_with']['roc_auc']:+.4f}; per-row scores correlate at")
P(f"  {Ci['pearson_score']:.3f} and {100 * Ci['prediction_agreement']:.1f}% of predictions were unchanged — a modest effect.")
P(f"  DBSCAN: removing the port did not change the basic behaviour — most attack rows still occurred inside clusters")
P(f"  ({comp_np['pct_attack_in_attack_dominated']:.1f}% attack-dominated, {comp_np['pct_attack_in_mixed']:.1f}% mixed, {comp_np['pct_attack_in_noise']:.1f}% noise). Every WITHOUT-Port noise row")
P("  was also noise WITH Port (Jaccard 0.712).")
P("  The DBSCAN ablation uses independently selected eps values in each feature space, so it measures the effect of")
P("  removing Destination Port under the complete unsupervised DBSCAN procedure rather than a pure feature effect at a")
P(f"  fixed eps. Removing the port made {Cd['geometry']['without_port']['duplicate_feature_rows']:,} sample rows feature-identical; they were retained because")
P("  deduplicating them would introduce an additional preprocessing change.")
P("\n7. PCA ANALYSIS (Phase 2B)")
P("  TABLE 7 — DBSCAN PCA analysis (same 50,000 rows; eps re-derived in each space by the same rule)")
P(tbl(T["T07_pca"]))
P("  Within the tested configurations, PCA did not materially change DBSCAN's low attack recall (all < 3%); noise")
P(f"  stayed between {min(DBp['noise_pct'], *(r['noise_pct'] for r in B['pca_runs'])):.2f}% and {max(DBp['noise_pct'], *(r['noise_pct'] for r in B['pca_runs'])):.2f}%. "
  f"Runtime was ~{DBp['runtime_s']:.1f} s without PCA and ~{np.mean([r['runtime_s'] for r in B['pca_runs']]):.0f} s with PCA. This does not show that")
P("  PCA is ineffective in general.")
P("\n8. PARAMETER SENSITIVITY (Phase 2B)")
P("  TABLE 8 — DBSCAN parameter sensitivity")
P(tbl(T["T08_sensitivity"]))
el = B["elbows"]["10"]
P(f"  Noise ranged {noise_rng[0]:.2f}%-{noise_rng[1]:.2f}% and the Jaccard overlap with the primary noise set "
  f"{jac_rng[0]:.2f}-{jac_rng[1]:.2f}. The k-distance curve spans")
P(f"  about five orders of magnitude and standard elbow rules disagreed (min_samples=10: log10 {el['log10']['eps']:.2f}, linear "
  f"{el['linear']['eps']:.2f}, truncated {el['trunc99']['eps']:.2f}).")
P("  DBSCAN's noise assignments are sensitive to density parameters in this feature space; eps=2.552 is the primary")
P("  unsupervised eps selected by the predefined parameter-selection procedure.")
P("\n9. COMPARATIVE ANALYSIS")
P("  Anomaly definitions. Isolation Forest: an observation is anomalous when it is relatively easy to isolate through")
P("  randomised tree partitioning. DBSCAN: an observation is anomalous only when it is assigned to density noise (-1).")
P("  Consequences observed here: attacks can receive high Isolation Forest scores even when they belong to dense")
P(f"  traffic patterns (e.g. DoS Hulk, flagged at {100 * if_cat['DoS Hulk']['detection_rate']:.1f}%). This is consistent with the forest being fitted on")
P("  BENIGN data only, where such patterns are rare, whereas DBSCAN, run on a mixed sample, found the same high-volume")
P(f"  attacks forming dense groups (DoS Hulk {db_cat['DoS Hulk']['pct_flagged_anomalous']:.1f}% noise) and so did not label them noise. In short, 'attack' != 'DBSCAN noise', while an attack can still receive a")
P("  high Isolation Forest anomaly score — a central finding of this project.")
P("  TABLE 10 — Methodological comparison")
P(tbl(T["T10_methodology"]))
P("  Generalisation and deployment characteristics demonstrated by this implementation:")
P("    Isolation Forest — trained on BENIGN traffic; predicts unseen observations; continuous score with a threshold for")
P("    the binary decision; supports ROC analysis; evaluated on a separate test set.")
P("    DBSCAN — clusters the supplied sample directly; no standard sklearn predict() for unseen observations; noise")
P("    assignment is sample-dependent; no intrinsic continuous score; eps/min_samples materially influence results;")
P("    the primary experiment is transductive. (Adaptations for deployment exist but were not evaluated here.)")
P("\n10. COMPUTATIONAL ANALYSIS")
P("  TABLE 9 — Computational characteristics")
P(tbl(T["T09_computation"]))
P("  The runtimes are not directly equivalent: the algorithms processed different populations in different workflows.")
P("  Isolation Forest's cost is fitting 200 trees on 256-row subsamples plus a linear-time scoring pass over unseen rows.")
P("  DBSCAN's cost here is dominated by storing every eps-neighbourhood; with dense regions this grows roughly with the")
P("  square of the sample size, which is why 100,000 rows exceeded the ~10 GB budget and some sensitivity settings were")
P("  skipped. Neither algorithm is claimed to be computationally better in general.")
P("\n11. LIMITATIONS")
lims = [
    "CIC-IDS2017 is a benchmark/testbed dataset and may not represent production traffic.",
    "The DBSCAN experiment used 50,000 observations because the 100,000-row configuration exceeded the available memory budget.",
    "Isolation Forest and DBSCAN were evaluated under different experimental paradigms.",
    "DBSCAN is transductive in this implementation.",
    "DBSCAN is sensitive to eps and min_samples.",
    "High-dimensional distance structure affects DBSCAN.",
    "PCA changes local-density geometry.",
    "Rare attack categories have unstable detection estimates.",
    f"The test attack prevalence ({IF_TEST_ATTACK / A['n_test']:.1%}) is not intended to represent real-world traffic prevalence.",
    "Destination Port was investigated by ablation but not declared leakage.",
    "The experiments evaluate binary anomaly detection rather than attack-type classification.",
    "No real-time production deployment was evaluated.",
    "No external dataset validation was performed.",
]
for i, l in enumerate(lims, 1):
    P(f"  {i:>2}. {l}")
P("\n12. RESEARCH-QUESTION ANSWERS")
P("  RQ1 — Can Isolation Forest detect anomalous traffic when trained only on BENIGN observations?")
P(f"    Yes, partially. On the unseen test set it reached ROC-AUC {IF['roc_auc']:.4f}, precision {IF['precision']:.4f}, recall "
  f"{IF['recall']:.4f}, F1 {IF['f1']:.4f}")
P(f"    (TP {IF['TP']:,}, FP {IF['FP']:,}, FN {IF['FN']:,}, TN {IF['TN']:,}). Detection was uneven across categories: high for")
P("    volumetric DoS traffic, near zero for PortScan, brute-force (Patator), Bot and web attacks.")
P("  RQ2 — Can DBSCAN identify anomalous traffic by treating low-density observations as noise?")
P(f"    Only to a very limited extent in this experiment: {DBp['noise_pct']:.2f}% of the sample was noise, with precision "
  f"{DB['precision']:.4f}, recall {DB['recall']:.4f}, F1 {DB['f1']:.4f}.")
P(f"    Most attack rows were in dense clusters ({comp['pct_attack_in_attack_dominated']:.1f}% attack-dominated, "
  f"{comp['pct_attack_in_mixed']:.1f}% mixed). Noise share varied {noise_rng[0]:.2f}-{noise_rng[1]:.2f}%")
P("    across the sensitivity grid, so the result depends on the density parameters; no tested setting changed the")
P("    picture of low attack recall (max recall in the grid "
  f"{max(r['metrics']['recall'] for r in sens):.3f}).")
P("  RQ3 — How does Destination Port affect the two algorithms?")
P(f"    Modestly. Isolation Forest: F1 {Ci['with_port']['f1']:.4f} -> {Ci['without_port']['f1']:.4f}, ROC-AUC {Ci['with_port']['roc_auc']:.4f} -> "
  f"{Ci['without_port']['roc_auc']:.4f}, score correlation {Ci['pearson_score']:.3f}.")
P(f"    DBSCAN (eps re-selected {DBp['eps']:.3f} -> {Ccfg['dbscan']['eps']:.3f}): noise {DBp['noise_pct']:.2f}% -> "
  f"{Cd['summary_without']['noise_pct']:.2f}%, recall {Cdw['recall']:.3f} -> {Cdn['recall']:.3f};")
P("    attacks remained predominantly inside clusters. The largest category-level change was DoS GoldenEye, which moved")
P("    in opposite directions for the two algorithms.")
P("  RQ4 — How do the algorithms differ in anomaly-detection behaviour?")
P("    Isolation vs density: Isolation Forest scores how easily a flow separates from BENIGN-trained partitions; DBSCAN")
P("    asks whether a flow lies in a sparse region of the mixed sample. Out-of-sample vs transductive: Isolation Forest")
P("    scored 1,054,824 unseen rows; DBSCAN labelled only the 50,000 rows it clustered. Continuous vs binary: Isolation")
P("    Forest supports ROC analysis; DBSCAN gives noise/cluster only. Computation: Isolation Forest's cost was small and")
P("    linear in scored rows; DBSCAN's neighbour storage limited the sample size. Category behaviour: high-volume DoS")
P("    traffic was largely flagged by Isolation Forest but clustered by DBSCAN.")
P("\n13. FINAL CONCLUSION")
P(f"  Isolation Forest produced meaningful anomaly separation under the BENIGN-only training formulation, with ROC-AUC")
P(f"  {IF['roc_auc']:.4f} and F1 {IF['f1']:.4f} on its frozen unseen test set. DBSCAN identified a small fraction of observations as")
P(f"  density noise, but only {100 * DB['recall']:.2f}% of attack observations in its primary sample were flagged as noise. Post-hoc")
P("  cluster analysis showed that many attacks occurred inside dense clusters rather than as isolated observations. PCA")
P("  did not materially change this DBSCAN behaviour, and Destination Port removal produced only modest changes in the")
P("  Isolation Forest results and did not resolve DBSCAN's low attack recall. The algorithms therefore exhibit different")
P("  anomaly-detection behaviour because they operationalise 'anomaly' differently. These conclusions are specific to")
P("  CIC-IDS2017 and to the frozen experimental designs described above.")

# =====================================================================
# SECTION 20 — REPRODUCIBILITY CHECK
# =====================================================================
report_body = "\n".join(L)
OUT_REPORT.write_text(report_body + "\n", encoding="utf-8")
comparison = {
    "mandatory_statement": MANDATORY,
    "populations": {"isolation_forest": {"train_benign": A["n_train"], "test": A["n_test"], "test_benign": 628946, "test_attack": 425878},
                    "dbscan": {"sample": n_work, "benign": cw["BENIGN"], "attack": db_attack_n, "composition": cw}},
    "isolation_forest_primary": {"config": A["primary_params"], "metrics": IF, "train_time_s": A["train_time_s"],
                                 "predict_time_s": A["predict_time_s"], "peak_ram_gb": A["peak_ram_gb"]},
    "dbscan_primary": {"eps": DBp["eps"], "min_samples": DBp["min_samples"], "metrics": DB, "n_clusters": DBp["n_clusters"],
                       "noise_pct": DBp["noise_pct"], "runtime_s": DBp["runtime_s"], "pairs": DBp["pairs"],
                       "cluster_composition_posthoc": comp},
    "attack_categories": T["T05_attack_categories"].to_dict(orient="records"),
    "port_ablation": {"isolation_forest": Ci["delta_without_minus_with"], "if_pearson": Ci["pearson_score"],
                      "if_agreement": Ci["prediction_agreement"], "dbscan": Cd["delta_without_minus_with"],
                      "dbscan_eps_without": Ccfg["dbscan"]["eps"], "dbscan_noise_overlap": Cd["noise_overlap"],
                      "duplicate_rows_without_port": Cd["geometry"]["without_port"]["duplicate_feature_rows"]},
    "pca": T["T07_pca"].to_dict(orient="records"),
    "sensitivity_ranges": {"noise_pct": noise_rng, "jaccard": jac_rng},
    "no_ranking": True, "no_models_rerun": True,
    "source_artifact_sha256": source_hashes,
    "frozen_manifest_sha256": {k: v["manifest_sha256"] for k, v in ver0.items()},
}
with open(OUT_JSON, "w") as fh:
    json.dump(comparison, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))

generated = [OUT_REPORT, OUT_JSON, Path(__file__).resolve()] + sorted(TAB_DIR.iterdir()) + sorted(PLOT_DIR.glob("*.png"))
with open(OUT_MANIFEST, "w") as fh:
    json.dump({"phase": "3", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "frozen_manifests_verified": {k: v["manifest_sha256"] for k, v in ver0.items()},
               "source_artifact_sha256": source_hashes,
               "sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in generated}}, fh, indent=2)

ver1 = verify_all()
assert all(not v["failed"] for v in ver1.values()), {k: v["failed"] for k, v in ver1.items()}
assert all(ver1[k]["manifest_sha256"] == ver0[k]["manifest_sha256"] for k in ver0)
assert {p: sha256(ROOT / p) for p in SOURCES.values()} == source_hashes
m3 = json.load(open(OUT_MANIFEST))
assert all(sha256(ROOT / p) == d for p, d in m3["sha256"].items())

repro = ["\nREPRODUCIBILITY CHECK"]
for k, v in ver1.items():
    repro.append(f"  Phase {k} manifest: {v['files']} files verified, 0 changed (manifest sha256 {v['manifest_sha256'][:16]}...)")
repro.append(f"  Phase 3 manifest: {len(m3['sha256'])} generated artifacts verified")
repro.append(f"  Isolation Forest config: {A['primary_params']}")
repro.append(f"  DBSCAN config: eps={DBp['eps']}, min_samples={DBp['min_samples']}, metric=euclidean, sample n={n_work}, random_state=42")
repro.append("  Source artifacts used (sha256):")
for p, h in source_hashes.items():
    repro.append(f"    {p}: {h}")
repro.append("  Assertion: no frozen artifact changed — PASS")
repro.append("\nPHASE 3 COMPLETE. No experiment started, nothing tuned, no ensemble, no additional model.")
print(report_body + "\n" + "\n".join(repro))
