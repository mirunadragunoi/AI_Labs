import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

# ================================================================
# EXERCITIUL 4 (V2) - Kernel Ridge cu kernel intersectie (precomputed)
# ================================================================
# IDEEA GENERALA:
#   Kernel Ridge = Ridge Regression in spatiul kernel.
#   In loc sa lucram direct cu features, lucram cu o matrice de
#   similaritati (kernel) intre exemple.
#
#   Kernel Intersectie:
#     K(x, y) = sum( min(xi, yi) )
#   Masoara cat de mult se "suprapun" doi vectori.
#   Perfect pentru features de tip count/histograma (ca cele de la ex2).
#
#   Exemplu:
#     x = [3, 0, 2, 1]
#     y = [1, 2, 2, 0]
#     K = min(3,1) + min(0,2) + min(2,2) + min(1,0) = 1 + 0 + 2 + 0 = 3
#
#   "precomputed" = calculam noi matricea de kernel si o dam modelului.
#
#   Kernel Ridge formula:
#     alpha = (K + lambda * I)^(-1) * y
#     predictie = K_test * alpha


# ================================================================
# PASUL 1: INCARCAM FEATURES
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
# L1 normalizare e naturala pentru kernel intersectie
# (face vectorii sa fie "distributii" cu suma = 1)
train_norm = normalize(train_features, norm='l1')
test_norm = normalize(test_features, norm='l1')


# ================================================================
# PASUL 3: IMPLEMENTARE KERNEL INTERSECTIE
# ================================================================

def intersection_kernel(X, Y):
    """
    Calculeaza matricea de kernel intersectie intre X si Y.

    K(x, y) = sum( min(xi, yi) )

    Parcurgem fiecare pereche de randuri din X si Y.
    La fiecare pereche, luam minimul element cu element si facem suma.

    X: matrice (n1 x d)
    Y: matrice (n2 x d)
    Returneaza: matrice (n1 x n2)
    """
    n1 = X.shape[0]
    n2 = Y.shape[0]
    K = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            # min element cu element, apoi suma
            K[i, j] = np.sum(np.minimum(X[i], Y[j]))

    return K


# ================================================================
# PASUL 4: IMPLEMENTARE KERNEL RIDGE
# ================================================================

def kernel_ridge_fit(K_train, y_one_hot, lambd=1.0):
    """
    Formula: alpha = (K + lambda * I)^(-1) * y

    K_train: matrice kernel (n x n)
    y_one_hot: etichete one-hot (n x num_classes)
    lambd: regularizare

    Returneaza: coeficientii alpha (n x num_classes)
    """
    n = K_train.shape[0]
    A = K_train + lambd * np.eye(n)
    alpha_coefs = np.linalg.solve(A, y_one_hot)
    return alpha_coefs


def kernel_ridge_predict(K_test, alpha_coefs):
    """
    Formula: scores = K_test * alpha
    Returneaza: etichete prezise (argmax pe scoruri)
    """
    scores = K_test.dot(alpha_coefs)
    return np.argmax(scores, axis=1)


# ================================================================
# PASUL 5: PREGATIRE ETICHETE ONE-HOT
# ================================================================
num_classes = len(np.unique(train_labels))
n_train = len(train_labels)

y_one_hot = np.zeros((n_train, num_classes))
for i in range(n_train):
    y_one_hot[i, int(train_labels[i])] = 1

print(f"Numar clase: {num_classes}")


# ================================================================
# PASUL 6: CAUTARE HIPERPARAMETRI (lambda)
# ================================================================
np.random.seed(42)
indices = np.arange(n_train)
np.random.shuffle(indices)
split = int(0.8 * n_train)

train_idx = indices[:split]
val_idx = indices[split:]

print("\n--- Cautare hiperparametri (lambda) ---")
print("Calculam kernel pe subset... (poate dura)")

best_lambda = 1.0
best_acc = 0

# Calculam kernel pe subset o singura data
K_sub_train = intersection_kernel(train_norm[train_idx], train_norm[train_idx])
K_sub_val = intersection_kernel(train_norm[val_idx], train_norm[train_idx])

for lambd in [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
    alpha_coefs = kernel_ridge_fit(K_sub_train, y_one_hot[train_idx], lambd=lambd)
    val_pred = kernel_ridge_predict(K_sub_val, alpha_coefs)
    val_acc = accuracy_score(train_labels[val_idx], val_pred)

    print(f"  lambda={lambd:8.4f} -> Acc validare: {val_acc * 100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        best_lambda = lambd

print(f"\nCel mai bun lambda: {best_lambda} (acc: {best_acc * 100:.2f}%)")


# ================================================================
# PASUL 7: ANTRENARE FINALA SI PREDICTII
# ================================================================
print("\nCalculam matricele de kernel finale...")

K_train = intersection_kernel(train_norm, train_norm)
K_test = intersection_kernel(test_norm, train_norm)

print(f"K_train: {K_train.shape}, K_test: {K_test.shape}")

alpha_coefs_final = kernel_ridge_fit(K_train, y_one_hot, lambd=best_lambda)
predictions = kernel_ridge_predict(K_test, alpha_coefs_final)

print(f"\nPredictii generate: {len(predictions)} etichete")

# Salvam
np.save('subiect4_solutia_1.npy', predictions)
print("Predictii salvate in subiect4_solutia_1.npy")

# Distributia
unique, counts = np.unique(predictions, return_counts=True)
print("\nDistributia predictiilor:")
for cls, cnt in zip(unique, counts):
    print(f"  Clasa {int(cls)}: {cnt} exemple")
