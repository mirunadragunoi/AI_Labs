import numpy as np
from sklearn.metrics import accuracy_score

# ================================================================
# EXERCITIUL 4 (2024 V2) - KRR cu kernel Hellinger (precomputed)
# ================================================================
# KRR = Kernel Ridge Regression
#   alpha = (K + lambda * I)^(-1) * y_one_hot
#   predictie = K_test * alpha -> argmax per clasa
#
# Hellinger kernel: K(x, y) = sum( sqrt(xi * yi) )
#   Masoara similaritatea intre distributii de probabilitate.
#   Perfect pt features Markov (care sunt probabilitati).

train_features = np.load('train_markov_features.npy')
test_features = np.load('test_markov_features.npy')
train_labels = np.load('train_labels_saved.npy')

test_files = []
with open('test_filenames.txt', 'r') as f:
    for line in f:
        test_files.append(line.strip())

print(f"Train: {train_features.shape}, Test: {test_features.shape}")


# ================================================================
# KERNEL HELLINGER
# ================================================================

def hellinger_kernel(X, Y):
    """
    K(x, y) = sum( sqrt(xi * yi) )
    Implementare: sqrt(X) @ sqrt(Y).T
    """
    X_safe = np.maximum(X, 0)
    Y_safe = np.maximum(Y, 0)
    return np.sqrt(X_safe).dot(np.sqrt(Y_safe).T)


# ================================================================
# ONE-HOT ENCODING
# ================================================================
num_classes = len(np.unique(train_labels))
n_train = len(train_labels)

y_one_hot = np.zeros((n_train, num_classes))
for i in range(n_train):
    y_one_hot[i, int(train_labels[i])] = 1


# ================================================================
# CAUTARE HIPERPARAMETRI (lambda)
# ================================================================
np.random.seed(42)
indices = np.arange(n_train)
np.random.shuffle(indices)
split = int(0.8 * n_train)
train_idx, val_idx = indices[:split], indices[split:]

print("\nCalculam kernel Hellinger pe subset...")
K_sub_tr = hellinger_kernel(train_features[train_idx], train_features[train_idx])
K_sub_val = hellinger_kernel(train_features[val_idx], train_features[train_idx])

print("--- Cautare lambda ---")
best_lambda, best_acc = 1.0, 0

for lambd in [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
    n = K_sub_tr.shape[0]
    alpha_coefs = np.linalg.solve(K_sub_tr + lambd * np.eye(n), y_one_hot[train_idx])
    val_pred = np.argmax(K_sub_val.dot(alpha_coefs), axis=1)
    acc = accuracy_score(train_labels[val_idx], val_pred)
    print(f"  lambda={lambd:8.4f} -> Acc: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc, best_lambda = acc, lambd

print(f"\nCel mai bun lambda: {best_lambda} ({best_acc*100:.2f}%)")


# ================================================================
# ANTRENARE FINALA
# ================================================================
print("\nCalculam kernel-uri finale...")
K_train = hellinger_kernel(train_features, train_features)
K_test = hellinger_kernel(test_features, train_features)

alpha_final = np.linalg.solve(K_train + best_lambda * np.eye(n_train), y_one_hot)
predictions = np.argmax(K_test.dot(alpha_final), axis=1)

with open('subiect4_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fname, pred in zip(test_files, predictions):
        f.write(f"{fname},{pred}\n")

print(f"Predictii salvate: {len(predictions)} etichete")
