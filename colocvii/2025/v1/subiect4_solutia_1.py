import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

# ================================================================
# EXERCITIUL 4 - SVM cu kernel Hellinger (precomputed)
# ================================================================
# IDEEA GENERALA:
#   Hellinger kernel masoara similaritatea intre doua distributii.
#   Formula: K(x, y) = sum( sqrt(xi * yi) )
#
#   Intuitie: daca x si y sunt histograme (frecvente),
#   Hellinger masoara cat de mult se "suprapun".
#   Similar cu histogram intersection, dar foloseste sqrt.
#
#   "precomputed" = ii dam SVM-ului direct matricea de kernel,
#   nu datele brute. SVM nu mai calculeaza el kernel-ul,
#   ci il primeste gata calculat.
#
#   IMPORTANT: 
#   - La antrenare: matricea (n_train x n_train)
#   - La testare: matricea (n_test x n_train)


# ================================================================
# PASUL 1: INCARCAM FEATURES DE CONVOLUTIE (de la exercitiul 2)
# ================================================================
try:
    train_features = np.load('train_conv_features.npy')
    test_features = np.load('test_conv_features.npy')
    print("Features de convolutie incarcate din fisiere.")
except FileNotFoundError:
    print("EROARE: Ruleaza intai subiect2_solutia_1.py pentru a genera features!")
    exit()

train_labels = np.load('train_labels.npy')

print(f"Train features: {train_features.shape}")
print(f"Test features: {test_features.shape}")


# ================================================================
# PASUL 2: NORMALIZARE
# ================================================================
# Normalizam features inainte de kernel
# L1 normalizare e buna pentru Hellinger (face vectorii sa fie distributii)
train_norm = normalize(train_features, norm='l1')
test_norm = normalize(test_features, norm='l1')


# ================================================================
# PASUL 3: IMPLEMENTAREA KERNEL-ULUI HELLINGER
# ================================================================

def hellinger_kernel(X, Y):
    """
    Calculeaza matricea de kernel Hellinger intre X si Y.

    Formula: K(x, y) = sum( sqrt(xi * yi) )

    PASII:
    1. Luam sqrt din X si Y (element cu element)
       - sqrt(xi) si sqrt(yi)
    2. Inmultim matricele: sqrt(X) @ sqrt(Y).T
       - Asta calculeaza sum(sqrt(xi) * sqrt(yi)) = sum(sqrt(xi*yi))
       - Functioneaza din cauza proprietatii produsului matriceal

    ATENTIE: valorile trebuie sa fie >= 0 (altfel sqrt da eroare)
    Dupa L1 normalizare, toate valorile sunt >= 0.

    X: matrice (n1 x d)
    Y: matrice (n2 x d)
    Returneaza: matrice (n1 x n2)
    """
    # Ne asiguram ca nu avem valori negative
    X_safe = np.maximum(X, 0)
    Y_safe = np.maximum(Y, 0)

    # sqrt element cu element, apoi produs matriceal
    sqrt_X = np.sqrt(X_safe)
    sqrt_Y = np.sqrt(Y_safe)

    K = sqrt_X.dot(sqrt_Y.T)
    return K


# ================================================================
# PASUL 4: CAUTARE HIPERPARAMETRI (C)
# ================================================================
# C = parametru de penalizare din SVM
#   C mare: SVM incearca sa clasifice TOTUL corect (risc de overfitting)
#   C mic: SVM accepta greseli (risc de underfitting)

np.random.seed(42)
indices = np.arange(len(train_labels))
np.random.shuffle(indices)
split = int(0.8 * len(indices))

train_idx = indices[:split]
val_idx = indices[split:]

print("\n--- Cautare hiperparametri (C) ---")
best_C = 1.0
best_acc = 0

for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
    # Calculam kernel pe subsetul de train
    K_sub_train = hellinger_kernel(train_norm[train_idx], train_norm[train_idx])
    K_sub_val = hellinger_kernel(train_norm[val_idx], train_norm[train_idx])

    # Antrenam SVM cu kernel precomputed
    model = svm.SVC(C=C, kernel='precomputed')
    model.fit(K_sub_train, train_labels[train_idx])

    # Predicam pe validare
    val_pred = model.predict(K_sub_val)
    val_acc = accuracy_score(train_labels[val_idx], val_pred)

    print(f"  C={C:7.2f} -> Acuratete validare: {val_acc * 100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        best_C = C

print(f"\nCel mai bun C: {best_C} (acc validare: {best_acc * 100:.2f}%)")


# ================================================================
# PASUL 5: ANTRENARE FINALA SI PREDICTII
# ================================================================
print("\nCalculam matricele de kernel finale...")

# Kernel train vs train (pentru antrenare)
K_train = hellinger_kernel(train_norm, train_norm)
# Kernel test vs train (pentru testare)
K_test = hellinger_kernel(test_norm, train_norm)

print(f"K_train shape: {K_train.shape}")
print(f"K_test shape: {K_test.shape}")

# Antrenam SVM-ul final
model_final = svm.SVC(C=best_C, kernel='precomputed')
model_final.fit(K_train, train_labels)

# Predictii pe test
predictions = model_final.predict(K_test)
print(f"\nPredictii generate: {len(predictions)} etichete")

# Salvam
np.save('subiect4_solutia_1.npy', predictions)
print("Predictii salvate in subiect4_solutia_1.npy")

# Distributia predictiilor
unique, counts = np.unique(predictions, return_counts=True)
print("\nDistributia predictiilor:")
for cls, cnt in zip(unique, counts):
    print(f"  Clasa {int(cls)}: {cnt} exemple")
