import numpy as np
from sklearn.metrics import accuracy_score

# ================================================================
# EXERCITIUL 3 (2024 V2) - KNN cu distanta Minkowski p=5
# ================================================================
# Diferenta fata de V1: distanta Minkowski cu p=5 (nu Manhattan/L1).
#
# Distanta Minkowski generalizata:
#   D(x, y) = ( sum( |xi - yi|^p ) )^(1/p)
#
#   p=1 -> Manhattan (L1)
#   p=2 -> Euclidiana (L2)
#   p=5 -> penalizeaza mai mult diferentele mari
#          (se apropie de norma infinit = max|xi-yi|)

train_features = np.load('train_markov_features.npy')
test_features = np.load('test_markov_features.npy')
train_labels = np.load('train_labels_saved.npy')

test_files = []
with open('test_filenames.txt', 'r') as f:
    for line in f:
        test_files.append(line.strip())

print(f"Train: {train_features.shape}, Test: {test_features.shape}")


def minkowski_distance(x, y, p=5):
    """
    Distanta Minkowski cu parametrul p.
    D = ( sum(|xi - yi|^p) )^(1/p)
    """
    return np.sum(np.abs(x - y) ** p) ** (1.0 / p)


def knn_classify(train_features, train_labels, test_sample, k=3, p=5):
    """KNN cu distanta Minkowski."""
    # Calculam distantele vectorizat
    diffs = np.abs(train_features - test_sample)
    distances = np.sum(diffs ** p, axis=1) ** (1.0 / p)

    nearest = np.argsort(distances)[:k]
    neighbor_labels = train_labels[nearest].astype(int)
    return np.bincount(neighbor_labels).argmax()


# ================================================================
# CAUTARE HIPERPARAMETRI (K)
# ================================================================
np.random.seed(42)
indices = np.arange(len(train_labels))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]

print("\n--- Cautare K (distanta Minkowski p=5) ---")
best_k, best_acc = 3, 0

for k in [1, 3, 5, 7, 9, 11]:
    val_pred = []
    for i in val_idx:
        pred = knn_classify(train_features[train_idx], train_labels[train_idx],
                           train_features[i], k=k, p=5)
        val_pred.append(pred)
    acc = accuracy_score(train_labels[val_idx], val_pred)
    print(f"  K={k:2d} -> Acc: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc, best_k = acc, k

print(f"\nCel mai bun K: {best_k} ({best_acc*100:.2f}%)")


# ================================================================
# PREDICTII
# ================================================================
print(f"\nClasificam testul cu K={best_k}, p=5...")
test_pred = []
for i in range(len(test_features)):
    pred = knn_classify(train_features, train_labels, test_features[i], k=best_k, p=5)
    test_pred.append(pred)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(test_features)}")

with open('subiect3_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fname, pred in zip(test_files, test_pred):
        f.write(f"{fname},{pred}\n")

print(f"Predictii salvate: {len(test_pred)} etichete")
