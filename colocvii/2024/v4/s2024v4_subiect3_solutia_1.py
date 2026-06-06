import numpy as np
from sklearn.metrics import accuracy_score

# ================================================================
# EXERCITIUL 3 (2024 V4) - KNN cu distanta Euclidiana (L2)
# ================================================================
# V4 specific: distanta Euclidiana = Minkowski p=2
# D(x,y) = sqrt( sum( (xi-yi)^2 ) )
# Cea mai comuna distanta - "linia dreapta" intre doua puncte.

train_features = np.load('train_markov_features.npy')
test_features = np.load('test_markov_features.npy')
train_labels = np.load('train_labels_saved.npy')
test_files = open('test_filenames.txt').read().strip().split('\n')


def knn_euclidean(train_f, train_l, test_sample, k=3):
    """KNN cu distanta Euclidiana (L2)."""
    # sqrt(sum((xi-yi)^2)) - vectorizat
    dists = np.sqrt(np.sum((train_f - test_sample) ** 2, axis=1))
    nearest = np.argsort(dists)[:k]
    return np.bincount(train_l[nearest].astype(int)).argmax()


# Cautare K
np.random.seed(42)
idx = np.arange(len(train_labels))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, val_idx = idx[:split], idx[split:]

print("--- Cautare K (Euclidiana) ---")
best_k, best_acc = 3, 0
for k in [1, 3, 5, 7, 9, 11]:
    preds = [knn_euclidean(train_features[tr_idx], train_labels[tr_idx],
                           train_features[i], k) for i in val_idx]
    acc = accuracy_score(train_labels[val_idx], preds)
    print(f"  K={k:2d} -> {acc*100:.2f}%")
    if acc > best_acc: best_acc, best_k = acc, k

print(f"Cel mai bun K={best_k} ({best_acc*100:.2f}%)")

# Predictii test
print(f"\nClasificam cu K={best_k}...")
test_pred = []
for i in range(len(test_features)):
    test_pred.append(knn_euclidean(train_features, train_labels, test_features[i], best_k))
    if (i+1) % 100 == 0: print(f"  {i+1}/{len(test_features)}")

with open('subiect3_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fn, p in zip(test_files, test_pred): f.write(f"{fn},{p}\n")
print(f"Salvat: {len(test_pred)} predictii")
