import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

# ================================================================
# EXERCITIUL 5 (2024 V3) - Raport
# ================================================================

train_features = np.load('train_markov_features.npy')
train_labels = np.load('train_labels_saved.npy')

np.random.seed(42)
idx = np.arange(len(train_labels))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, val_idx = idx[:split], idx[split:]

def hellinger_kernel(X, Y):
    return np.sqrt(np.maximum(X, 0)).dot(np.sqrt(np.maximum(Y, 0)).T)

raport = []
raport.append("=" * 60)
raport.append("RAPORT - 2024 Varianta 3 (k=7, Minkowski p=3, SVM Hellinger)")
raport.append("=" * 60)

# --- KNN Minkowski p=3 ---
raport.append("\n--- KNN (Minkowski p=3) ---")
P = 3
best_k, best_acc_k = 3, 0
for k in [1, 3, 5, 7, 9, 11]:
    preds = []
    for i in val_idx:
        d = np.sum(np.abs(train_features[tr_idx] - train_features[i]) ** P, axis=1) ** (1.0/P)
        nn = np.argsort(d)[:k]
        preds.append(np.bincount(train_labels[tr_idx][nn].astype(int)).argmax())
    acc = accuracy_score(train_labels[val_idx], preds)
    raport.append(f"K={k:<4d} -> {acc*100:.2f}%")
    print(f"KNN K={k}: {acc*100:.2f}%")
    if acc > best_acc_k: best_acc_k, best_k = acc, k
raport.append(f"Cel mai bun: K={best_k} ({best_acc_k*100:.2f}%)")

# --- SVM Hellinger ---
raport.append("\n--- SVM Hellinger ---")
K_tr = hellinger_kernel(train_features[tr_idx], train_features[tr_idx])
K_val = hellinger_kernel(train_features[val_idx], train_features[tr_idx])

best_C, best_acc_s = 1.0, 0
for C in [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]:
    m = svm.SVC(C=C, kernel='precomputed')
    m.fit(K_tr, train_labels[tr_idx])
    acc = accuracy_score(train_labels[val_idx], m.predict(K_val))
    raport.append(f"C={C:<8.2f} -> {acc*100:.2f}%")
    print(f"SVM C={C}: {acc*100:.2f}%")
    if acc > best_acc_s: best_acc_s, best_C = acc, C
raport.append(f"Cel mai bun: C={best_C} ({best_acc_s*100:.2f}%)")

# Sumar
raport.append("\n" + "=" * 60)
raport.append("SUMAR")
raport.append("=" * 60)
raport.append(f"Retea neuronala: SGD lr=1e-3, [256,128], 100 epoci")
raport.append(f"KNN Minkowski p=3: {best_acc_k*100:.2f}% (K={best_k})")
raport.append(f"SVM Hellinger:     {best_acc_s*100:.2f}% (C={best_C})")
raport.append("")
raport.append("Observatii:")
raport.append("- k=7 ofera mai multe stari Markov -> 147 features (vs 48 pt k=4)")
raport.append("- Minkowski p=3 e un compromis bun intre L1 si L2")
raport.append("- SVM Hellinger e cel mai performant pt features probabilistice")

raport_text = "\n".join(raport)
print("\n" + raport_text)

with open('raport_experimente.txt', 'w') as f:
    f.write(raport_text)
print("\nRaport salvat.")
