## T01_populations

| Item | Isolation Forest | DBSCAN |
|---|---|---|
| Deduplicated population | 2,522,362 rows | 2,522,362 rows |
| Rows used to fit | 1,467,538 BENIGN (TRAIN) | 50,000 mixed rows (labels hidden) |
| Rows evaluated | 1,054,824 unseen TEST rows | the same 50,000 rows (transductive) |
| BENIGN evaluated | 628,946 | 41,665 |
| ATTACK evaluated | 425,878 | 8,335 |
| Attack prevalence | 40.37% | 16.67% |
| Selection | seeded 70/30 permutation of BENIGN + all attacks | uniform random sample, random_state=42 (50k fallback of 100k draw: memory) |

## T02_configurations

| Setting | Isolation Forest | DBSCAN |
|---|---|---|
| Algorithm | sklearn IsolationForest | sklearn DBSCAN (Euclidean, brute) |
| Features | 70 (Destination Port retained) | 70 (Destination Port retained) |
| Preprocessing | inf->NaN, median impute, StandardScaler (fit on BENIGN TRAIN) | inf->NaN, median impute, StandardScaler (fit on the 50k sample) |
| Key parameters | n_estimators=200, max_samples='auto' (256), contamination='auto', random_state=42 | eps=2.552, min_samples=10 |
| How parameters were set | pre-specified baseline | log10-Kneedle on k-distance curve; min_samples closest to ln(n); no labels |
| Anomaly decision | -decision_function > 0 | cluster label == -1 (noise) |

## T03_performance

| Measure | Isolation Forest | DBSCAN |
|---|---|---|
| Training paradigm | trained on BENIGN only | unsupervised clustering of a mixed sample |
| Evaluation paradigm | out-of-sample (unseen test set) | transductive (same rows clustered) |
| Features | 70 | 70 |
| Evaluation size | 1,054,824 | 50,000 |
| Precision | 0.8387 | 0.1061 |
| Recall | 0.5789 | 0.0205 |
| F1 | 0.6850 | 0.0344 |
| FPR | 0.0754 | 0.0346 |
| FNR | 0.4211 | 0.9795 |
| ROC-AUC | 0.8082 | not applicable (binary noise output) |
| Runtime | fit 0.95 s; predict 5.33 s | 15.5 s (one clustering run) |
| Memory | peak process 5.9 GB (data loading dominates) | ~7.7 GB est. neighbour storage; process peak 8.1 GB over all 2B runs |

## T04_confusion

| Algorithm | Population | TN | FP | FN | TP |
|---|---|---|---|---|---|
| Isolation Forest | 1,054,824 test rows | 581525 | 47421 | 179318 | 246560 |
| DBSCAN | 50,000 sample rows | 40224 | 1441 | 8164 | 171 |

## T05_attack_categories

| Attack category | IF n | IF % flagged as anomalous | IF note | DBSCAN n | DBSCAN % flagged as anomalous | DBSCAN note |
|---|---|---|---|---|---|---|
| DoS Hulk | 172849 | 89.28 |  | 3414 | 0.97 |  |
| DDoS | 128016 | 62.03 |  | 2441 | 1.31 |  |
| PortScan | 90819 | 0.24 |  | 1830 | 0.49 |  |
| DoS GoldenEye | 10286 | 49.12 |  | 190 | 23.68 |  |
| DoS slowloris | 5385 | 57.72 |  | 123 | 14.63 |  |
| DoS Slowhttptest | 5228 | 82.06 |  | 86 | 33.72 |  |
| FTP-Patator | 5933 | 0.0 |  | 105 | 0.0 |  |
| SSH-Patator | 3219 | 0.0 |  | 63 | 0.0 |  |
| Bot | 1953 | 2.25 |  | 35 | 2.86 |  |
| Web Attack - Brute Force | 1470 | 4.97 |  | 38 | 10.53 |  |
| Web Attack - XSS | 652 | 2.45 |  | 10 | 0.0 | statistically unstable (n<30) |
| Infiltration | 36 | 83.33 | small n (<100): unstable | Not present | Not present |  |
| Web Attack - Sql Injection | 21 | 0.0 | small n (<100): unstable | Not present | Not present |  |
| Heartbleed | 11 | 100.0 | small n (<100): unstable | Not present | Not present |  |

## T06_port_ablation

| Algorithm | Measure | WITH Port (70) | WITHOUT Port (69) | Delta |
|---|---|---|---|---|
| Isolation Forest | Precision | 0.8387 | 0.8273 | -0.0114 |
| Isolation Forest | Recall | 0.5789 | 0.5677 | -0.0113 |
| Isolation Forest | F1 | 0.6850 | 0.6733 | -0.0117 |
| Isolation Forest | FPR | 0.0754 | 0.0802 | +0.0048 |
| Isolation Forest | FNR | 0.4211 | 0.4323 | +0.0113 |
| Isolation Forest | ROC-AUC | 0.8082 | 0.8140 | +0.0058 |
| Isolation Forest | Score Pearson / prediction agreement |  |  | 0.997 / 98.8% (12,197 changed) |
| DBSCAN | eps (re-selected, same rule) | 2.552 | 3.100 | +0.547 |
| DBSCAN | Clusters | 56 | 44 | -12 |
| DBSCAN | Noise % | 3.22 | 2.30 | -0.93 pp |
| DBSCAN | Precision | 0.1061 | 0.0880 | -0.0181 |
| DBSCAN | Recall | 0.0205 | 0.0121 | -0.0084 |
| DBSCAN | F1 | 0.0344 | 0.0213 | -0.0131 |
| DBSCAN | FPR | 0.0346 | 0.0251 | -0.0095 |
| DBSCAN | FNR | 0.9795 | 0.9879 | +0.0084 |
| DBSCAN | Noise-set Jaccard / rows changed |  |  | 0.712 / 464 |
| DBSCAN | Feature-identical rows in 50k | 0 | 2,872 |  |

## T07_pca

| Space | Components | Cum. variance | eps | Clusters | Noise % | Precision | Recall | F1 | Runtime s |
|---|---|---|---|---|---|---|---|---|---|
| Standardized (no PCA) | 70 | 1.0 | 2.552 | 56 | 3.22 | 0.1061 | 0.0205 | 0.0344 | 15.5 |
| PCA 90% | 19 | 0.904 | 1.683 | 52 | 3.7 | 0.1315 | 0.0292 | 0.0477 | 20.2 |
| PCA 95% | 23 | 0.9509 | 1.813 | 52 | 3.73 | 0.1223 | 0.0274 | 0.0447 | 20.6 |
| PCA 99% | 31 | 0.9908 | 2.402 | 58 | 3.15 | 0.1036 | 0.0196 | 0.0329 | 20.3 |

## T08_sensitivity

| min_samples | eps | setting | status | clusters | noise % | Jaccard vs primary | recall | est. memory GB |
|---|---|---|---|---|---|---|---|---|
| 5 | 1.276 | x0.5 | run | 228 | 6.58 | 0.438 | 0.0496 | 4.7 |
| 5 | 1.914 | x0.75 | run | 135 | 3.63 | 0.674 | 0.015 | 6.6 |
| 5 | 2.552 | x1.0 | run | 94 | 2.27 | 0.705 | 0.0076 | 7.7 |
| 5 | 3.19 | x1.25 | run | 71 | 1.55 | 0.481 | 0.0053 | 9.2 |
| 5 | 3.828 | x1.5 | not run: memory budget |  |  |  |  | 12.7 |
| 10 | 1.276 | x0.5 | run | 136 | 8.51 | 0.379 | 0.0822 | 4.7 |
| 10 | 1.914 | x0.75 | run | 72 | 5.0 | 0.645 | 0.038 | 6.6 |
| 10 | 2.552 | x1.0 | run (primary) | 56 | 3.22 | 1.0 | 0.0205 | 7.7 |
| 10 | 3.19 | x1.25 | run | 43 | 2.29 | 0.712 | 0.0119 | 9.2 |
| 10 | 3.828 | x1.5 | not run: memory budget |  |  |  |  | 12.7 |
| 20 | 1.276 | x0.5 | run | 84 | 11.0 | 0.293 | 0.1293 | 4.7 |
| 20 | 1.914 | x0.75 | run | 32 | 6.76 | 0.477 | 0.0629 | 6.6 |
| 20 | 2.552 | x1.0 | run | 28 | 4.62 | 0.698 | 0.037 | 7.7 |
| 20 | 3.19 | x1.25 | run | 26 | 3.24 | 0.736 | 0.0264 | 9.2 |
| 20 | 3.828 | x1.5 | not run: memory budget |  |  |  |  | 12.7 |
| 5 | 2.335 | own elbow | run | 100 | 2.65 | 0.737 | 0.0095 | 7.4 |
| 20 | 3.219 | own elbow | run | 26 | 3.2 | 0.729 | 0.0264 | 9.3 |

## T09_computation

| Aspect | Isolation Forest | DBSCAN |
|---|---|---|
| Rows processed | fit 1,467,538; score 1,054,824 | 50,000 (clustered once per configuration) |
| Fit / cluster time | 0.95 s (200 trees x 256 rows) | 15.5 s primary; PCA runs 20.2-20.6 s |
| Prediction | 5.33 s for 1,054,824 unseen rows | no predict(); labels exist only for clustered rows |
| Dominant memory cost | data loading (peak 5.9 GB process) | neighbour storage: 320,310,172 pairs (~7.7 GB est.) |
| Scaling limit observed | none reached | 100k rows needed ~29.3 GB (> 10 GB budget); 1.5x eps runs skipped |
| Saved model size | 0.36 MB (frozen 200-tree pickle, compressed) | n/a (cluster labels for the sample) |

## T10_methodology

| Property | Isolation Forest | DBSCAN |
|---|---|---|
| Anomaly definition | easy to isolate by random partitions (short average path length) | density noise: not a core point and not within eps of one |
| Training data | BENIGN only | unlabelled mixed sample |
| Unseen observations | predict()/decision_function on new rows | no standard sklearn predict(); transductive |
| Output | continuous score + threshold | binary noise / cluster id |
| ROC analysis | yes (score) | no intrinsic score |
| Parameter sensitivity shown | 2A: n=200 cross-seed score Spearman 0.991 | noise 1.55-11.00% across grid |
| Scale dependence | per-feature splits | Euclidean distances; standardisation essential |

