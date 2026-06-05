import numpy as np
from sklearn.metrics import accuracy_score

# ================================================================
# EXERCITIUL 5 (2024 V2) - Raport
# ================================================================

train_features = np.load('train_markov_features.npy')
train_labels = np.load('train_labels_saved.npy')

np.random.seed(42)
indices = np.arange(len(train_labels))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]

num_classes = len(np.unique(train_labels))
y_one_hot = np.zeros((len(train_labels), num_classes))
for i in range(len(train_labels)):
    y_one_hot[i, int(train_labels[i])] = 1

def hellinger_kernel(X, Y):
    return np.sqrt(np.maximum(X, 0)).dot(np.sqrt(np.maximum(Y, 0)).T)

raport = []
raport.append("=" * 60)
raport.append("RAPORT - 2024 Varianta 2")
raport.append("=" * 60)

# --- KNN Minkowski p=5 ---
raport.append("\n--- KNN (Minkowski p=5) ---")
raport.append(f"{'K':<6} {'Acc':<10}")

best_k, best_acc_k = 3, 0
for k in [1, 3, 5, 7, 9, 11]:
    preds = []
    for i in val_idx:
        d = np.sum(np.abs(train_features[train_idx] - train_features[i]) ** 5, axis=1) ** 0.2
        nn = np.argsort(d)[:k]
        preds.append(np.bincount(train_labels[train_idx][nn].astype(int)).argmax())
    acc = accuracy_score(train_labels[val_idx], preds)
    raport.append(f"K={k:<4d} {acc*100:.2f}%")
    print(f"KNN K={k}: {acc*100:.2f}%")
    if acc > best_acc_k: best_acc_k, best_k = acc, k

raport.append(f"Cel mai bun: K={best_k} ({best_acc_k*100:.2f}%)")

# --- KRR Hellinger ---
raport.append("\n--- KRR Hellinger ---")
raport.append(f"{'Lambda':<12} {'Acc':<10}")

K_tr = hellinger_kernel(train_features[train_idx], train_features[train_idx])
K_val = hellinger_kernel(train_features[val_idx], train_features[train_idx])

best_lam, best_acc_h = 1.0, 0
for lam in [0.001, 0.01, 0.1, 1.0, 10.0]:
    n = K_tr.shape[0]
    a = np.linalg.solve(K_tr + lam * np.eye(n), y_one_hot[train_idx])
    pred = np.argmax(K_val.dot(a), axis=1)
    acc = accuracy_score(train_labels[val_idx], pred)
    raport.append(f"{lam:<12.3f} {acc*100:.2f}%")
    print(f"KRR lam={lam}: {acc*100:.2f}%")
    if acc > best_acc_h: best_acc_h, best_lam = acc, lam

raport.append(f"Cel mai bun: lambda={best_lam} ({best_acc_h*100:.2f}%)")

# Sumar
raport.append("\n" + "=" * 60)
raport.append("SUMAR")
raport.append("=" * 60)
raport.append(f"Retea neuronala: SGD lr=1e-3, [256,128], 100 epoci")
raport.append(f"KNN Minkowski p=5: {best_acc_k*100:.2f}% (K={best_k})")
raport.append(f"KRR Hellinger:     {best_acc_h*100:.2f}% (lambda={best_lam})")

raport_text = "\n".join(raport)
print("\n" + raport_text)

with open('raport_experimente.txt', 'w') as f:
    f.write(raport_text)
print("\nRaport salvat.")
