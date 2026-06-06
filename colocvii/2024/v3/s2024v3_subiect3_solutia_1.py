import numpy as np
from sklearn.metrics import accuracy_score

# ================================================================
# EXERCITIUL 3 (2024 V3) - KNN cu distanta Minkowski p=3
# ================================================================
# V3 specific: p=3
# D(x,y) = ( sum(|xi-yi|^3) )^(1/3)
# p=3 e intre Manhattan (p=1) si p=5: penalizeaza diferente mari
# dar nu la fel de agresiv ca p=5

train_features = np.load('train_markov_features.npy')
test_features = np.load('test_markov_features.npy')
train_labels = np.load('train_labels_saved.npy')

test_files = []
with open('test_filenames.txt', 'r') as f:
    for line in f:
        test_files.append(line.strip())

P = 3  # V3 specific


def knn_classify(train_feat, train_lab, test_sample, k=3):
    """KNN cu Minkowski p=3."""
    dists = np.sum(np.abs(train_feat - test_sample) ** P, axis=1) ** (1.0 / P)
    nearest = np.argsort(dists)[:k]
    return np.bincount(train_lab[nearest].astype(int)).argmax()


# Cautare K
np.random.seed(42)
idx = np.arange(len(train_labels))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, val_idx = idx[:split], idx[split:]

print(f"--- Cautare K (Minkowski p={P}) ---")
best_k, best_acc = 3, 0
for k in [1, 3, 5, 7, 9, 11]:
    preds = [knn_classify(train_features[tr_idx], train_labels[tr_idx],
                          train_features[i], k) for i in val_idx]
    acc = accuracy_score(train_labels[val_idx], preds)
    print(f"  K={k:2d} -> {acc*100:.2f}%")
    if acc > best_acc:
        best_acc, best_k = acc, k

print(f"Cel mai bun K={best_k} ({best_acc*100:.2f}%)")

# Predictii test
print(f"\nClasificam cu K={best_k}...")
test_pred = []
for i in range(len(test_features)):
    test_pred.append(knn_classify(train_features, train_labels, test_features[i], best_k))
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(test_features)}")

with open('subiect3_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fname, pred in zip(test_files, test_pred):
        f.write(f"{fname},{pred}\n")
print(f"Salvat: {len(test_pred)} predictii")
