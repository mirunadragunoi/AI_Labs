import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

# ================================================================
# EXERCITIUL 4 (2024 V3) - SVM cu kernel Hellinger (precomputed)
# ================================================================
# K(x, y) = sum( sqrt(xi * yi) )
# Hellinger e ideal pt features de tip probabilitate (matrici Markov).

train_features = np.load('train_markov_features.npy')
test_features = np.load('test_markov_features.npy')
train_labels = np.load('train_labels_saved.npy')

test_files = []
with open('test_filenames.txt', 'r') as f:
    for line in f:
        test_files.append(line.strip())


def hellinger_kernel(X, Y):
    """K(x,y) = sum(sqrt(xi*yi)) = sqrt(X) @ sqrt(Y).T"""
    return np.sqrt(np.maximum(X, 0)).dot(np.sqrt(np.maximum(Y, 0)).T)


# Cautare C
np.random.seed(42)
idx = np.arange(len(train_labels))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, val_idx = idx[:split], idx[split:]

K_tr = hellinger_kernel(train_features[tr_idx], train_features[tr_idx])
K_val = hellinger_kernel(train_features[val_idx], train_features[tr_idx])

print("--- Cautare C ---")
best_C, best_acc = 1.0, 0
for C in [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
    model = svm.SVC(C=C, kernel='precomputed')
    model.fit(K_tr, train_labels[tr_idx])
    acc = accuracy_score(train_labels[val_idx], model.predict(K_val))
    print(f"  C={C:7.2f} -> {acc*100:.2f}%")
    if acc > best_acc:
        best_acc, best_C = acc, C

print(f"Cel mai bun C={best_C} ({best_acc*100:.2f}%)")

# Antrenare finala
K_train = hellinger_kernel(train_features, train_features)
K_test = hellinger_kernel(test_features, train_features)

model_final = svm.SVC(C=best_C, kernel='precomputed')
model_final.fit(K_train, train_labels)
predictions = model_final.predict(K_test).astype(int)

with open('subiect4_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fname, pred in zip(test_files, predictions):
        f.write(f"{fname},{pred}\n")
print(f"Salvat: {len(predictions)} predictii")
