import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

# ================================================================
# EXERCITIUL 3 - Kernel Ridge Regression cu kernel liniar
# ================================================================
# IDEEA GENERALA:
#   Kernel Ridge = Ridge Regression dar in "spatiul kernel".
#   Cu kernel liniar: K(x, y) = x · y (produs scalar)
#
#   Ridge Regression: gaseste ponderi W care minimizeaza
#     ||y - X*W||^2 + alpha * ||W||^2
#   Termenul alpha * ||W||^2 = "regularizare" = forteaza ponderile
#   sa fie mici, prevenind overfitting.
#
#   In versiunea kernel:
#     alpha_coefs = (K + lambda * I)^(-1) * y
#     predictie = K_test * alpha_coefs
#
#   Unde K = matricea de kernel (train x train)
#         K_test = matricea de kernel (test x train)
#
# PENTRU CLASIFICARE:
#   Kernel Ridge e de fapt un model de REGRESIE.
#   Il adaptam pentru clasificare prin:
#   - Fiecare clasa e codificata ca un vector one-hot
#   - Ridge prezice un scor pentru fiecare clasa
#   - Clasa finala = clasa cu scorul maxim


# ================================================================
# PASUL 1: INCARCAM FEATURES DE CONVOLUTIE (de la exercitiul 2)
# ================================================================
# Daca ai rulat exercitiul 2, features sunt salvate.
# Daca nu, ruleaza intai subiect2_solutia_1.py

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
# Normalizam features-urile pentru rezultate mai bune
train_norm = normalize(train_features, norm='l2')
test_norm = normalize(test_features, norm='l2')


# ================================================================
# PASUL 3: IMPLEMENTARE KERNEL RIDGE REGRESSION
# ================================================================

def linear_kernel(X, Y):
    """
    Kernel liniar: K(x, y) = x · y (produs scalar).
    Pentru matrice: K = X @ Y.T

    X: (n1 x d), Y: (n2 x d)
    Returneaza: (n1 x n2)
    """
    return X.dot(Y.T)


def kernel_ridge_fit(K_train, y_train, lambd=1.0):
    """
    Antreneaza Kernel Ridge Regression.

    K_train: matricea de kernel (n x n) = kernel(train, train)
    y_train: etichetele (n,) sau matrice one-hot (n x num_clase)
    lambd: parametrul de regularizare (lambda)

    Formula: alpha = (K + lambda * I)^(-1) * y
      - K = matricea de kernel
      - I = matricea identitate
      - lambda = cat de mult penalizam ponderile mari
      - Inversam matricea (K + lambda*I) si inmultim cu y

    Returneaza: coeficientii alpha
    """
    n = K_train.shape[0]
    # (K + lambda * I) -> adaugam lambda pe diagonala
    # np.linalg.solve rezolva sistemul liniar (mai stabil decat inversarea)
    A = K_train + lambd * np.eye(n)
    alpha_coefs = np.linalg.solve(A, y_train)
    return alpha_coefs


def kernel_ridge_predict(K_test, alpha_coefs):
    """
    Prezice folosind Kernel Ridge.

    K_test: matricea de kernel (n_test x n_train) = kernel(test, train)
    alpha_coefs: coeficientii din antrenare

    Formula: y_pred = K_test * alpha
    """
    return K_test.dot(alpha_coefs)


# ================================================================
# PASUL 4: PREGATIREA ETICHETELOR (one-hot encoding)
# ================================================================
# Kernel Ridge e regresie, dar noi facem clasificare.
# Trick: transformam etichetele in format one-hot:
#   clasa 0 -> [1, 0, 0, ...]
#   clasa 1 -> [0, 1, 0, ...]
#   clasa 2 -> [0, 0, 1, ...]
# Ridge prezice un scor per clasa, alegem clasa cu scorul maxim.

num_classes = len(np.unique(train_labels))
n_train = len(train_labels)

# Cream matricea one-hot
y_one_hot = np.zeros((n_train, num_classes))
for i in range(n_train):
    y_one_hot[i, int(train_labels[i])] = 1

print(f"Numar clase: {num_classes}")
print(f"One-hot shape: {y_one_hot.shape}")


# ================================================================
# PASUL 5: CAUTARE HIPERPARAMETRI (lambda)
# ================================================================
# Impartim train in train/validare pentru a gasi cel mai bun lambda

np.random.seed(42)
indices = np.arange(n_train)
np.random.shuffle(indices)
split = int(0.8 * n_train)

train_idx = indices[:split]
val_idx = indices[split:]

print("\n--- Cautare hiperparametri (lambda) ---")
best_lambda = 1.0
best_acc = 0

for lambd in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
    # Kernel pe subsetul de train
    K_sub_train = linear_kernel(train_norm[train_idx], train_norm[train_idx])
    K_sub_val = linear_kernel(train_norm[val_idx], train_norm[train_idx])

    # Antrenam
    alpha_coefs = kernel_ridge_fit(K_sub_train, y_one_hot[train_idx], lambd=lambd)

    # Predicam pe validare
    val_scores = kernel_ridge_predict(K_sub_val, alpha_coefs)
    val_pred = np.argmax(val_scores, axis=1)  # clasa cu scorul maxim
    val_acc = accuracy_score(train_labels[val_idx], val_pred)

    print(f"  lambda={lambd:7.3f} -> Acuratete validare: {val_acc * 100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        best_lambda = lambd

print(f"\nCel mai bun lambda: {best_lambda} (acc validare: {best_acc * 100:.2f}%)")


# ================================================================
# PASUL 6: ANTRENARE FINALA SI PREDICTII
# ================================================================
# Antrenam pe TOATE datele de antrenare

K_train = linear_kernel(train_norm, train_norm)
K_test = linear_kernel(test_norm, train_norm)

alpha_coefs_final = kernel_ridge_fit(K_train, y_one_hot, lambd=best_lambda)
test_scores = kernel_ridge_predict(K_test, alpha_coefs_final)
predictions = np.argmax(test_scores, axis=1)

print(f"\nPredictii generate: {len(predictions)} etichete")

# Salvam
np.save('subiect3_solutia_1.npy', predictions)
print("Predictii salvate in subiect3_solutia_1.npy")

# Distributia predictiilor
unique, counts = np.unique(predictions, return_counts=True)
print("\nDistributia predictiilor:")
for cls, cnt in zip(unique, counts):
    print(f"  Clasa {int(cls)}: {cnt} exemple")
