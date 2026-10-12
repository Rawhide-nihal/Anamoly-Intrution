import pandas as pd
import numpy as np
import glob
import gc
import sys

sys.stdout.reconfigure(encoding='utf-8')

dataset_path = "."
csv_files = list(glob.glob("*.csv"))

dfs = []
for file in csv_files:
    df_temp = pd.read_csv(file)
    df_temp.columns = df_temp.columns.str.strip()
    dfs.append(df_temp)

combined_df = pd.concat(dfs, ignore_index=True)
del dfs
gc.collect()

combined_df = combined_df.drop_duplicates()
total_dedup_rows = combined_df.shape[0]

labels = combined_df['Label'].copy()
binary_labels = (labels != 'BENIGN').astype(int)

benign_mask = labels == 'BENIGN'

X_benign = combined_df[benign_mask]
labels_benign = labels[benign_mask]
y_benign = binary_labels[benign_mask]

X_attack = combined_df[~benign_mask]
labels_attack = labels[~benign_mask]
y_attack = binary_labels[~benign_mask]

train_size = int(len(X_benign) * 0.7)
np.random.seed(42)
indices = np.random.permutation(len(X_benign))
train_idx, test_idx = indices[:train_size], indices[train_size:]

labels_train = labels_benign.iloc[train_idx]
y_train = y_benign.iloc[train_idx]

labels_test_b = labels_benign.iloc[test_idx]
y_test_benign = y_benign.iloc[test_idx]

labels_test = pd.concat([labels_test_b, labels_attack])
y_test = pd.concat([y_test_benign, y_attack])

print(f"1. Total unique rows after duplicate removal: {total_dedup_rows}")

print("\n2. Training:")
print(f"- total rows: {len(labels_train)}")
print(f"- BENIGN count: {sum(labels_train == 'BENIGN')}")
print(f"- attack count: {sum(labels_train != 'BENIGN')}")
print(f"- number of unique original Label values: {labels_train.nunique()}")

print("\n3. Testing:")
print(f"- total rows: {len(labels_test)}")
print(f"- BENIGN count: {sum(labels_test == 'BENIGN')}")
print(f"- attack count: {sum(labels_test != 'BENIGN')}")
print(f"- number of unique original Label values: {labels_test.nunique()}")

print("\n4. Complete original attack-category distribution in the TEST set (sorted):")
print(labels_test[labels_test != 'BENIGN'].value_counts().to_string())

print("\n5. Verifications:")
train_only_benign = (labels_train.unique() == ['BENIGN']).all() and len(labels_train.unique()) == 1
print(f"- training contains only BENIGN: {train_only_benign}")
attacks_absent_train = sum(labels_train != 'BENIGN') == 0
print(f"- every attack category is absent from training: {attacks_absent_train}")
all_attacks_in_test = set(labels_attack.unique()).issubset(set(labels_test.unique()))
print(f"- every attack category present in the deduplicated dataset is present in the test set: {all_attacks_in_test}")
print(f"- no row exists in both train and test (indices overlap): {len(set(labels_train.index).intersection(set(labels_test.index))) == 0}")
print(f"- train + test = deduplicated dataset: {len(labels_train) + len(labels_test) == total_dedup_rows}")

print("\n6. Exact number of rows for:")
attack_counts = labels_attack.value_counts()
for att in ["Heartbleed", "Infiltration", "Bot", "PortScan", "DDoS"]:
    print(f"- {att}: {attack_counts.get(att, 0)}")

print("- each DoS category:")
for att in attack_counts.index:
    if "DoS " in att and att != "DDoS":
        print(f"  - {att}: {attack_counts.get(att, 0)}")

print("- each Web Attack category:")
for att in attack_counts.index:
    if "Web Attack" in att:
        print(f"  - {att}: {attack_counts.get(att, 0)}")

print(f"- FTP-Patator: {attack_counts.get('FTP-Patator', 0)}")
print(f"- SSH-Patator: {attack_counts.get('SSH-Patator', 0)}")

print("\n7. Verify binary label mapping:")
benign_binary = y_test[labels_test == 'BENIGN'].unique()
attack_binary = y_test[labels_test != 'BENIGN'].unique()
print(f"BENIGN = {benign_binary[0] if len(benign_binary)==1 else 'ERROR'}")
print(f"all non-BENIGN labels = {attack_binary[0] if len(attack_binary)==1 else 'ERROR'}")

print("\n8. Label check:")
cols = combined_df.drop(columns=['Label']).columns
print(f"- Label in X_train/X_test: {'Label' in cols}")

pass_cond = (
    train_only_benign and 
    attacks_absent_train and 
    all_attacks_in_test and 
    len(set(labels_train.index).intersection(set(labels_test.index))) == 0 and 
    len(labels_train) + len(labels_test) == total_dedup_rows and
    len(benign_binary) == 1 and benign_binary[0] == 0 and
    len(attack_binary) == 1 and attack_binary[0] == 1
)

if pass_cond:
    print("\nPHASE 1 VALIDATION: PASS")
else:
    print("\nPHASE 1 VALIDATION: FAIL")
