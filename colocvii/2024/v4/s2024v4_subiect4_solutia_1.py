import numpy as np
from sklearn.metrics import accuracy_score

# ================================================================
# EXERCITIUL 4 (2024 V4) - KRR cu kernel intersectie (precomputed)
# ================================================================
# KRR = Kernel Ridge Regression
#   alpha = (K + lambda * I)^(-1) * y_one_hot
#   predictie = K_test * alpha -> argmax
#
# Kernel Intersectie: K(x,y) = sum( min(xi, yi) )
# Perfect pt features de tip probabilitate (matrici Markov).

train_features = np.load('train_markov_features.npy')
test_features = np.load('test_markov_features.npy')
train_labels = np.load('train_labels_saved.npy')
test_files = open('test_filenames.txt').read().strip().split('\n')


def intersection_kernel(X, Y):
    """K(x,y) = sum(min(xi, yi)) - vectorizat pe randuri."""
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        K[i] = np.minimum(X[i], Y).sum(axis=1)
    return K


# One-hot encoding
num_classes = len(np.unique(train_labels))
y_oh = np.zeros((len(train_labels), num_classes))
for i in range(len(train_labels)):
    y_oh[i, int(train_labels[i])] = 1

# Cautare lambda
np.random.seed(42)
idx = np.arange(len(train_labels))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, val_idx = idx[:split], idx[split:]

print("Calculam kernel intersectie pe subset...")
K_tr = intersection_kernel(train_features[tr_idx], train_features[tr_idx])
K_val = intersection_kernel(train_features[val_idx], train_features[tr_idx])

print("--- Cautare lambda ---")
best_lam, best_acc = 1.0, 0
for lam in [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
    n = K_tr.shape[0]
    alpha_c = np.linalg.solve(K_tr + lam * np.eye(n), y_oh[tr_idx])
    pred = np.argmax(K_val.dot(alpha_c), axis=1)
    acc = accuracy_score(train_labels[val_idx], pred)
    print(f"  lambda={lam:8.4f} -> {acc*100:.2f}%")
    if acc > best_acc: best_acc, best_lam = acc, lam

print(f"Cel mai bun lambda={best_lam} ({best_acc*100:.2f}%)")

# Antrenare finala
print("\nKernel-uri finale...")
K_train = intersection_kernel(train_features, train_features)
K_test = intersection_kernel(test_features, train_features)

n = len(train_labels)
alpha_final = np.linalg.solve(K_train + best_lam * np.eye(n), y_oh)
predictions = np.argmax(K_test.dot(alpha_final), axis=1)

with open('subiect4_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fn, p in zip(test_files, predictions): f.write(f"{fn},{p}\n")
print(f"Salvat: {len(predictions)} predictii")
