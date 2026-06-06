import numpy as np
from sklearn.metrics import accuracy_score

# ================================================================
# EXERCITIUL 5 (2024 V4) - Raport
# ================================================================

train_features = np.load('train_markov_features.npy')
train_labels = np.load('train_labels_saved.npy')

np.random.seed(42)
idx = np.arange(len(train_labels))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, val_idx = idx[:split], idx[split:]

num_classes = len(np.unique(train_labels))
y_oh = np.zeros((len(train_labels), num_classes))
for i in range(len(train_labels)): y_oh[i, int(train_labels[i])] = 1

def intersection_kernel(X, Y):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        K[i] = np.minimum(X[i], Y).sum(axis=1)
    return K

raport = []
raport.append("=" * 60)
raport.append("RAPORT - 2024 V4 (k=5, Euclidiana, KRR Intersectie)")
raport.append("=" * 60)

# --- KNN Euclidiana ---
raport.append("\n--- KNN (Euclidiana) ---")
best_k, best_acc_k = 3, 0
for k in [1, 3, 5, 7, 9, 11]:
    preds = []
    for i in val_idx:
        d = np.sqrt(np.sum((train_features[tr_idx] - train_features[i]) ** 2, axis=1))
        nn = np.argsort(d)[:k]
        preds.append(np.bincount(train_labels[tr_idx][nn].astype(int)).argmax())
    acc = accuracy_score(train_labels[val_idx], preds)
    raport.append(f"K={k:<4d} -> {acc*100:.2f}%")
    print(f"KNN K={k}: {acc*100:.2f}%")
    if acc > best_acc_k: best_acc_k, best_k = acc, k
raport.append(f"Cel mai bun: K={best_k} ({best_acc_k*100:.2f}%)")

# --- KRR Intersectie ---
raport.append("\n--- KRR Intersectie ---")
K_tr = intersection_kernel(train_features[tr_idx], train_features[tr_idx])
K_val = intersection_kernel(train_features[val_idx], train_features[tr_idx])

best_lam, best_acc_r = 1.0, 0
for lam in [0.001, 0.01, 0.1, 1.0, 10.0, 50.0]:
    n = K_tr.shape[0]
    a = np.linalg.solve(K_tr + lam * np.eye(n), y_oh[tr_idx])
    pred = np.argmax(K_val.dot(a), axis=1)
    acc = accuracy_score(train_labels[val_idx], pred)
    raport.append(f"lambda={lam:<8.3f} -> {acc*100:.2f}%")
    print(f"KRR lam={lam}: {acc*100:.2f}%")
    if acc > best_acc_r: best_acc_r, best_lam = acc, lam
raport.append(f"Cel mai bun: lambda={best_lam} ({best_acc_r*100:.2f}%)")

raport.append("\n" + "=" * 60)
raport.append("SUMAR")
raport.append("=" * 60)
raport.append(f"Retea neuronala: SGD lr=1e-2 momentum=0.9, [256,128]")
raport.append(f"KNN Euclidiana:      {best_acc_k*100:.2f}% (K={best_k})")
raport.append(f"KRR Intersectie:     {best_acc_r*100:.2f}% (lambda={best_lam})")
raport.append("")
raport.append("Observatii:")
raport.append("- k=5 ofera 75 features, un compromis bun intre detaliu si sparsitate")
raport.append("- Euclidiana (L2) e cea mai comuna distanta, functioneaza bine cu Markov")
raport.append("- KRR cu Intersection e potrivit pt features probabilistice")

txt = "\n".join(raport)
print("\n" + txt)
with open('raport_experimente.txt', 'w') as f: f.write(txt)
print("\nRaport salvat.")
