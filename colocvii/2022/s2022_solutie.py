import numpy as np
from sklearn.metrics import accuracy_score

# ================================================================
# INCARCARE DATE
# ================================================================
# train_data.npy: lista de 1000 string-uri (texte)
# test_data.npy: lista de 323 string-uri
# train_labels.npy: etichetele (clase)

train_data = np.load('train_data.npy', allow_pickle=True)
train_labels = np.load('train_labels.npy')
test_data = np.load('test_data.npy', allow_pickle=True)

print(f"Train: {len(train_data)} texte")
print(f"Test: {len(test_data)} texte")
print(f"Clase: {np.unique(train_labels)}")
print(f"Exemplu text: '{str(train_data[0])[:80]}...'")


# ================================================================
# EXERCITIUL 1 - String Kernel (similaritate pe baza n-grams)
# ================================================================
# IDEEA:
#   Un n-gram = o secventa de n caractere consecutive dintr-un text.
#   Exemplu: "ananas", n=3 -> n-grams: {"ana", "nan", "ana", "nas"}
#            ca set unic: {"ana", "nan", "nas"}
#
#   String Kernel cu biti de prezenta:
#     1. Extragem TOATE n-gramele unice din string-ul s
#     2. Extragem TOATE n-gramele unice din string-ul t
#     3. Numaram cate n-grame au IN COMUN (intersectia multimilor)
#
#   Exemplu din cerinta (p=4):
#     s = "ananas copt"  -> 4-grams: {"anan", "nana", "anas", "nas ", "as c", "s co", " cop", "copt"}
#     t = "banana verde" -> 4-grams: {"bana", "anan", "nana", "ana ", "na v", "a ve", " ver", "verd", "erde"}
#     Comune: {"anan", "nana"} -> similaritate = 2
#
#   IMPORTANT: cerinta spune n-grams de lungime 8 (p=8)

def get_ngrams(text, p=8):
    """
    Extrage multimea de n-grame unice de lungime p dintr-un text.

    text: string-ul de analizat
    p: lungimea n-gramelor

    Returneaza: set de string-uri (n-grame unice)

    Exemplu: get_ngrams("ananas", 3) -> {"ana", "nan", "ana", "nas"}
             ca set: {"ana", "nan", "nas"}
    """
    ngrams = set()
    for i in range(len(text) - p + 1):
        ngram = text[i: i + p]
        ngrams.add(ngram)
    return ngrams


def string_kernel_similarity(s, t, p=8):
    """
    Calculeaza similaritatea string kernel intre doua string-uri.

    Bazata pe BITI DE PREZENTA = numaram n-gramele UNICE comune.
    (nu numaram de cate ori apare fiecare, ci doar DACA apare)

    s, t: doua string-uri
    p: lungimea n-gramelor

    Returneaza: numarul de n-grame comune (int)
    """
    ngrams_s = get_ngrams(s, p)
    ngrams_t = get_ngrams(t, p)

    # Intersectia multimilor = n-grame care apar in AMBELE string-uri
    common = ngrams_s & ngrams_t  # operatorul & = intersectie de seturi

    return len(common)


# Test rapid
s_test = "ananas copt"
t_test = "banana verde"
print(f"\nTest: similarity('{s_test}', '{t_test}', p=4) = "
      f"{string_kernel_similarity(s_test, t_test, p=4)}")
# Trebuie sa dea 2


# ================================================================
# EXERCITIUL 3 - Matricea kernel (calculata inainte de ex2 si ex4)
# ================================================================
# Matricea kernel K[i][j] = similaritatea dintre exemplul i si exemplul j.
#
# K_train: (1000 x 1000) - similaritate train vs train
# K_test:  (323 x 1000)  - similaritate test vs train
#
# Calculul dureaza! 1000*1000 = 1.000.000 perechi pentru K_train.
# Optimizare: K e SIMETRICA (K[i][j] = K[j][i]), deci calculam doar jumatate.

def compute_kernel_matrix(data1, data2, p=8):
    """
    Calculeaza matricea kernel intre doua multimi de texte.

    data1: lista de n1 texte
    data2: lista de n2 texte
    p: lungimea n-gramelor

    Returneaza: matrice (n1 x n2)

    Optimizare: pre-calculam n-gramele pentru data2 (se refolosesc).
    """
    n1 = len(data1)
    n2 = len(data2)
    K = np.zeros((n1, n2))

    # Pre-calculam n-gramele pentru al doilea set (se refolosesc la fiecare rand)
    ngrams_2 = [get_ngrams(str(text), p) for text in data2]

    for i in range(n1):
        ngrams_i = get_ngrams(str(data1[i]), p)

        for j in range(n2):
            # Numarul de n-grame comune
            K[i][j] = len(ngrams_i & ngrams_2[j])

        if (i + 1) % 100 == 0:
            print(f"  Rand {i + 1}/{n1} calculat...")

    return K


def compute_symmetric_kernel(data, p=8):
    """
    Calculeaza matricea kernel SIMETRICA (data vs data).
    Optimizare: calculam doar triunghiul superior, apoi copiem.
    """
    n = len(data)
    K = np.zeros((n, n))

    # Pre-calculam toate n-gramele
    all_ngrams = [get_ngrams(str(text), p) for text in data]

    for i in range(n):
        for j in range(i, n):  # doar de la i in sus (triunghiul superior)
            K[i][j] = len(all_ngrams[i] & all_ngrams[j])
            K[j][i] = K[i][j]  # simetria: K[j][i] = K[i][j]

        if (i + 1) % 100 == 0:
            print(f"  Rand {i + 1}/{n} calculat...")

    return K


# Calculam matricele kernel
P = 8  # lungimea n-gramelor conform cerintei

print(f"\nCalculam K_train ({len(train_data)}x{len(train_data)}) cu p={P}...")
K_train = compute_symmetric_kernel(train_data, p=P)

print(f"Calculam K_test ({len(test_data)}x{len(train_data)}) cu p={P}...")
K_test = compute_kernel_matrix(test_data, train_data, p=P)

print(f"K_train shape: {K_train.shape}")
print(f"K_test shape: {K_test.shape}")

# Salvam pentru reutilizare
np.save('K_train.npy', K_train)
np.save('K_test.npy', K_test)


# ================================================================
# EXERCITIUL 2 - KNN cu string kernel similarity
# ================================================================
# IDEEA:
#   In loc de distanta euclidiana, folosim SIMILARITATEA string kernel.
#   Similaritate MARE = texte asemanatoare (opus distantei!).
#   Deci luam vecinii cu similaritatea CEA MAI MARE (nu cea mai mica).
#
#   Putem folosi direct K_train si K_test:
#   K_test[i][j] = similaritatea dintre test_i si train_j
#   -> sortam descrescator, luam primii K vecini

print("\n" + "=" * 60)
print("EXERCITIUL 2 - KNN cu string kernel")
print("=" * 60)

# Cautare K pe validare
np.random.seed(42)
indices = np.arange(len(train_labels))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
tr_idx, val_idx = indices[:split], indices[split:]

print("--- Cautare K ---")
best_k, best_acc_knn = 3, 0

for k in [1, 3, 5, 7, 9, 11]:
    val_pred = []
    for i in val_idx:
        # Similaritatea exemplului i cu toate din train subset
        sims = K_train[i, tr_idx]
        # Sortam DESCRESCATOR (similaritate mare = mai aproape)
        nearest = np.argsort(-sims)[:k]
        neighbor_labels = train_labels[tr_idx][nearest].astype(int)
        val_pred.append(np.bincount(neighbor_labels).argmax())

    acc = accuracy_score(train_labels[val_idx], val_pred)
    print(f"  K={k:2d} -> Acc: {acc * 100:.2f}%")
    if acc > best_acc_knn:
        best_acc_knn, best_k = acc, k

print(f"Cel mai bun K={best_k} ({best_acc_knn * 100:.2f}%)")

# Predictii test cu KNN
test_pred_knn = []
for i in range(len(test_data)):
    sims = K_test[i]  # similaritatea cu TOATE exemplele de train
    nearest = np.argsort(-sims)[:best_k]
    neighbor_labels = train_labels[nearest].astype(int)
    test_pred_knn.append(np.bincount(neighbor_labels).argmax())

# Salvam
with open('subiect2_solutia_1.txt', 'w') as f:
    for pred in test_pred_knn:
        f.write(f"{pred}\n")
print(f"KNN salvat: subiect2_solutia_1.txt ({len(test_pred_knn)} predictii)")


# ================================================================
# EXERCITIUL 4 - KRR cu kernel precomputed
# ================================================================
# Kernel Ridge Regression cu matricea kernel deja calculata.
#   alpha = (K + lambda * I)^(-1) * y_one_hot
#   predictie = K_test * alpha -> argmax

print("\n" + "=" * 60)
print("EXERCITIUL 4 - KRR cu kernel precomputed")
print("=" * 60)

num_classes = len(np.unique(train_labels))
n_train = len(train_labels)

# One-hot encoding
y_one_hot = np.zeros((n_train, num_classes))
for i in range(n_train):
    y_one_hot[i, int(train_labels[i])] = 1

# Cautare lambda
print("--- Cautare lambda ---")
best_lambda, best_acc_krr = 1.0, 0

# Submatrici kernel pentru validare
K_sub_train = K_train[np.ix_(tr_idx, tr_idx)]
K_sub_val = K_train[np.ix_(val_idx, tr_idx)]

for lambd in [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
    n = K_sub_train.shape[0]
    alpha_coefs = np.linalg.solve(K_sub_train + lambd * np.eye(n), y_one_hot[tr_idx])
    val_scores = K_sub_val.dot(alpha_coefs)
    val_pred = np.argmax(val_scores, axis=1)
    acc = accuracy_score(train_labels[val_idx], val_pred)
    print(f"  lambda={lambd:8.4f} -> Acc: {acc * 100:.2f}%")
    if acc > best_acc_krr:
        best_acc_krr, best_lambda = acc, lambd

print(f"Cel mai bun lambda={best_lambda} ({best_acc_krr * 100:.2f}%)")

# Antrenare finala pe TOT train-ul
alpha_final = np.linalg.solve(K_train + best_lambda * np.eye(n_train), y_one_hot)
test_scores = K_test.dot(alpha_final)
test_pred_krr = np.argmax(test_scores, axis=1)

# Salvam
with open('subiect4_solutia_1.txt', 'w') as f:
    for pred in test_pred_krr:
        f.write(f"{pred}\n")
print(f"KRR salvat: subiect4_solutia_1.txt ({len(test_pred_krr)} predictii)")


# ================================================================
# EXERCITIUL 5 - Raport
# ================================================================
print("\n" + "=" * 60)
print("RAPORT")
print("=" * 60)

raport = []
raport.append("=" * 60)
raport.append("RAPORT - 2022 Varianta 2")
raport.append(f"String Kernel cu n-grams de lungime p={P}")
raport.append("=" * 60)

raport.append(f"\n--- KNN (similaritate string kernel) ---")
for k in [1, 3, 5, 7, 9, 11]:
    preds = []
    for i in val_idx:
        sims = K_train[i, tr_idx]
        nearest = np.argsort(-sims)[:k]
        preds.append(np.bincount(train_labels[tr_idx][nearest].astype(int)).argmax())
    acc = accuracy_score(train_labels[val_idx], preds)
    raport.append(f"  K={k:2d} -> {acc * 100:.2f}%")

raport.append(f"  Cel mai bun: K={best_k} ({best_acc_knn * 100:.2f}%)")

raport.append(f"\n--- KRR (kernel precomputed) ---")
for lam in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    n = K_sub_train.shape[0]
    a = np.linalg.solve(K_sub_train + lam * np.eye(n), y_one_hot[tr_idx])
    pred = np.argmax(K_sub_val.dot(a), axis=1)
    acc = accuracy_score(train_labels[val_idx], pred)
    raport.append(f"  lambda={lam:8.3f} -> {acc * 100:.2f}%")

raport.append(f"  Cel mai bun: lambda={best_lambda} ({best_acc_krr * 100:.2f}%)")

raport.append("\n" + "=" * 60)
raport.append("SUMAR")
raport.append("=" * 60)
raport.append(f"KNN string kernel:  {best_acc_knn * 100:.2f}% (K={best_k})")
raport.append(f"KRR precomputed:    {best_acc_krr * 100:.2f}% (lambda={best_lambda})")
raport.append("")
raport.append("Observatii:")
raport.append("- String kernel cu p=8 capteaza sub-secvente de 8 caractere")
raport.append("- KRR cu kernel precomputed obtine de obicei rezultate mai bune")
raport.append("  decat KNN deoarece optimizeaza global, nu local")
raport.append("- Lambda mare = regularizare puternica (model simplu)")
raport.append("- Lambda mic = model complex (risc overfitting)")

raport_text = "\n".join(raport)
print(raport_text)

with open('raport_experimente.txt', 'w') as f:
    f.write(raport_text)
print("\nRaport salvat.")
