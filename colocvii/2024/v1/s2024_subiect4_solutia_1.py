import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

# ================================================================
# EXERCITIUL 4 - SVM cu kernel intersectie (precomputed)
# ================================================================
# IDEEA:
#   SVM cu kernel intersectie pe features Markov.
#   K(x, y) = sum( min(xi, yi) ) - masoara suprapunerea.
#   Perfect pentru features de tip probabilitate/histograma
#   (cum sunt matricele Markov normalizate).


# ================================================================
# INCARCAM FEATURES
# ================================================================
train_features = np.load('train_markov_features.npy')
test_features = np.load('test_markov_features.npy')
train_labels = np.load('train_labels_saved.npy')

test_files = []
with open('test_filenames.txt', 'r') as f:
    for line in f:
        test_files.append(line.strip())

print(f"Train: {train_features.shape}, Test: {test_features.shape}")


# ================================================================
# KERNEL INTERSECTIE
# ================================================================

def intersection_kernel(X, Y):
    """
    K(x, y) = sum( min(xi, yi) )

    Implementare vectorizata (mai rapida decat for-uri):
    Pentru fiecare pereche (i, j), calculam suma minimelor element cu element.
    """
    n1 = X.shape[0]
    n2 = Y.shape[0]
    K = np.zeros((n1, n2))

    for i in range(n1):
        # Calculam min(X[i], Y[j]) pentru toti j simultan
        # X[i] are shape (d,), Y are shape (n2, d)
        # np.minimum face min element cu element (broadcasting)
        mins = np.minimum(X[i], Y)  # shape (n2, d)
        K[i] = mins.sum(axis=1)     # suma pe features -> shape (n2,)

    return K


# ================================================================
# CAUTARE HIPERPARAMETRI (C)
# ================================================================
np.random.seed(42)
indices = np.arange(len(train_labels))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx = indices[:split]
val_idx = indices[split:]

print("\nCalculam kernel pe subset de validare...")
K_sub_train = intersection_kernel(train_features[train_idx], train_features[train_idx])
K_sub_val = intersection_kernel(train_features[val_idx], train_features[train_idx])

print("--- Cautare hiperparametri (C) ---")
best_C = 1.0
best_acc = 0

for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
    model = svm.SVC(C=C, kernel='precomputed')
    model.fit(K_sub_train, train_labels[train_idx])
    val_pred = model.predict(K_sub_val)
    val_acc = accuracy_score(train_labels[val_idx], val_pred)

    print(f"  C={C:7.2f} -> Acc validare: {val_acc * 100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        best_C = C

print(f"\nCel mai bun C: {best_C} (acc: {best_acc * 100:.2f}%)")


# ================================================================
# ANTRENARE FINALA SI PREDICTII
# ================================================================
print("\nCalculam matricele de kernel finale...")
K_train = intersection_kernel(train_features, train_features)
K_test = intersection_kernel(test_features, train_features)

print(f"K_train: {K_train.shape}, K_test: {K_test.shape}")

model_final = svm.SVC(C=best_C, kernel='precomputed')
model_final.fit(K_train, train_labels)
predictions = model_final.predict(K_test).astype(int)

print(f"\nPredictii generate: {len(predictions)}")

# Salvam
with open('subiect4_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fname, pred in zip(test_files, predictions):
        f.write(f"{fname},{pred}\n")

print("Predictii salvate in subiect4_solutia_1.txt")
