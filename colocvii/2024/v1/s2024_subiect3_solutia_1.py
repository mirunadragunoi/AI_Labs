import numpy as np
from sklearn.metrics import accuracy_score

# ================================================================
# EXERCITIUL 3 - KNN cu distanta Manhattan pe features Markov
# ================================================================
# IDEEA:
#   KNN = clasificam fiecare exemplu de test uitandu-ne la cei mai
#   apropiati K vecini din setul de train. Eticheta majoritara = predictia.
#
#   Distanta Manhattan (L1): sum( |xi - yi| )
#   Intuitie: distanta "pe strazi" (mergi doar pe orizontala si verticala,
#   nu in diagonala ca la distanta euclidiana).
#
#   Exemplu: x=[1,3], y=[4,1] -> |1-4| + |3-1| = 3 + 2 = 5


# ================================================================
# INCARCAM FEATURES MARKOV (de la exercitiul 2)
# ================================================================
train_features = np.load('train_markov_features.npy')
test_features = np.load('test_markov_features.npy')
train_labels = np.load('train_labels_saved.npy')

# Citim filenames pentru output
test_files = []
with open('test_filenames.txt', 'r') as f:
    for line in f:
        test_files.append(line.strip())

print(f"Train: {train_features.shape}, Test: {test_features.shape}")


# ================================================================
# IMPLEMENTARE KNN CU DISTANTA MANHATTAN
# ================================================================

def manhattan_distance(x, y):
    """Distanta Manhattan: suma valorilor absolute ale diferentelor."""
    return np.sum(np.abs(x - y))


def knn_classify(train_features, train_labels, test_sample, k=3):
    """
    Clasificam un singur exemplu de test cu KNN.

    1. Calculam distanta Manhattan de la test la TOATE exemplele de train
    2. Sortam distantele si luam primii k vecini
    3. Vot majoritar: eticheta care apare cel mai des
    """
    # Calculam toate distantele (vectorizat pentru eficienta)
    # |test - fiecare_train| pe fiecare feature, apoi suma pe features
    distances = np.sum(np.abs(train_features - test_sample), axis=1)

    # Indicii celor mai apropiati k vecini
    nearest_indices = np.argsort(distances)[:k]

    # Etichetele vecinilor
    neighbor_labels = train_labels[nearest_indices].astype(int)

    # Vot majoritar
    prediction = np.bincount(neighbor_labels).argmax()
    return prediction


# ================================================================
# CAUTARE HIPERPARAMETRI (K)
# ================================================================
# Impartim train in train/validare pentru a gasi cel mai bun K

np.random.seed(42)
indices = np.arange(len(train_labels))
np.random.shuffle(indices)
split = int(0.8 * len(indices))

train_idx = indices[:split]
val_idx = indices[split:]

print("\n--- Cautare hiperparametri (K) ---")
best_k = 3
best_acc = 0

for k in [1, 3, 5, 7, 9, 11, 15]:
    val_pred = []
    for i in val_idx:
        pred = knn_classify(train_features[train_idx], train_labels[train_idx],
                           train_features[i], k=k)
        val_pred.append(pred)

    val_acc = accuracy_score(train_labels[val_idx], val_pred)
    print(f"  K={k:2d} -> Acuratete validare: {val_acc * 100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        best_k = k

print(f"\nCel mai bun K: {best_k} (acc: {best_acc * 100:.2f}%)")


# ================================================================
# PREDICTII PE TEST
# ================================================================
print(f"\nClasificam testul cu K={best_k}...")

test_pred = []
for i in range(len(test_features)):
    pred = knn_classify(train_features, train_labels, test_features[i], k=best_k)
    test_pred.append(pred)
    if (i + 1) % 100 == 0:
        print(f"  Clasificate {i + 1}/{len(test_features)}")

test_pred = np.array(test_pred)
print(f"Predictii generate: {len(test_pred)}")

# Salvam in formatul cerut
with open('subiect3_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fname, pred in zip(test_files, test_pred):
        f.write(f"{fname},{pred}\n")

print("Predictii salvate in subiect3_solutia_1.txt")
