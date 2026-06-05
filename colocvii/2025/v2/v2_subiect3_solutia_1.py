import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

# ================================================================
# EXERCITIUL 3 (V2) - SVM cu kernel RBF
# ================================================================
# IDEEA GENERALA:
#   SVM (Support Vector Machine) gaseste hiperplanul care separa
#   cel mai bine clasele, maximizand marginea (distanta minima
#   de la hiperplan la cel mai apropiat punct).
#
#   Kernel RBF (Radial Basis Function):
#     K(x, y) = exp(-gamma * ||x - y||^2)
#
#   Intuitie: RBF masoara cat de "aproape" sunt doua puncte.
#     - Daca x si y sunt identice: K = exp(0) = 1
#     - Daca x si y sunt foarte diferite: K -> 0
#     - gamma controleaza cat de repede scade similaritatea cu distanta
#       gamma mare = punctele trebuie sa fie foarte aproape ca sa fie similare
#       gamma mic = si punctele departate sunt considerate similare
#
#   Parametri importanti:
#     C = cat de mult penalizam erorile de clasificare
#     gamma = "raza de influenta" a fiecarui punct


# ================================================================
# PASUL 1: INCARCAM FEATURES DE CONVOLUTIE
# ================================================================
try:
    train_features = np.load('train_conv_features.npy')
    test_features = np.load('test_conv_features.npy')
    print("Features de convolutie incarcate.")
except FileNotFoundError:
    print("EROARE: Ruleaza intai subiect2 pentru a genera features!")
    exit()

train_labels = np.load('train_labels.npy')

print(f"Train features: {train_features.shape}")
print(f"Test features: {test_features.shape}")


# ================================================================
# PASUL 2: NORMALIZARE
# ================================================================
# Normalizam cu L2 pentru SVM (important pentru kernel RBF!)
# RBF depinde de distanta euclidiana, deci datele trebuie scalate.
train_norm = normalize(train_features, norm='l2')
test_norm = normalize(test_features, norm='l2')


# ================================================================
# PASUL 3: CAUTARE HIPERPARAMETRI (C si gamma)
# ================================================================
np.random.seed(42)
indices = np.arange(len(train_labels))
np.random.shuffle(indices)
split = int(0.8 * len(indices))

train_idx = indices[:split]
val_idx = indices[split:]

print("\n--- Cautare hiperparametri (C, gamma) ---")
best_C = 1.0
best_gamma = 'scale'
best_acc = 0

# Testam combinatii de C si gamma
for C in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
    for gamma in ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]:
        model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        model.fit(train_norm[train_idx], train_labels[train_idx])

        val_pred = model.predict(train_norm[val_idx])
        val_acc = accuracy_score(train_labels[val_idx], val_pred)

        print(f"  C={C:6.1f}, gamma={str(gamma):6s} -> Acc validare: {val_acc * 100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_C = C
            best_gamma = gamma

print(f"\nCel mai bun: C={best_C}, gamma={best_gamma} (acc: {best_acc * 100:.2f}%)")


# ================================================================
# PASUL 4: ANTRENARE FINALA SI PREDICTII
# ================================================================
# Antrenam pe TOATE datele de train cu cei mai buni hiperparametri
model_final = svm.SVC(C=best_C, kernel='rbf', gamma=best_gamma)
model_final.fit(train_norm, train_labels)

predictions = model_final.predict(test_norm)
print(f"\nPredictii generate: {len(predictions)} etichete")

# Salvam
np.save('subiect3_solutia_1.npy', predictions)
print("Predictii salvate in subiect3_solutia_1.npy")

# Distributia
unique, counts = np.unique(predictions, return_counts=True)
print("\nDistributia predictiilor:")
for cls, cnt in zip(unique, counts):
    print(f"  Clasa {int(cls)}: {cnt} exemple")
