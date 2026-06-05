import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

# ================================================================
# EXERCITIUL 1 (V2) - Ridge Regression cu Bag of Words la nivel de caracter
# ================================================================
# IDEEA GENERALA:
#   Ridge Regression e un model de REGRESIE (prezice numere, nu clase).
#   Il adaptam pentru clasificare prin:
#     1. Transformam etichetele in format one-hot
#        clasa 0 -> [1, 0, 0, ...], clasa 1 -> [0, 1, 0, ...], etc.
#     2. Ridge prezice un scor pentru fiecare clasa
#     3. Clasa finala = clasa cu scorul cel mai mare (argmax)
#
#   Formula Ridge: W = (X^T * X + alpha * I)^(-1) * X^T * y
#     - X = matricea de features (fiecare rand = un document)
#     - y = etichetele one-hot
#     - alpha = regularizare (previne overfitting)
#     - I = matricea identitate
#
#   Bag of Words la nivel de caracter:
#     - numaram de cate ori apare fiecare caracter in fiecare document
#     - vocabularul = toate caracterele unice din antrenare


# ================================================================
# PASUL 1: INCARCAREA DATELOR
# ================================================================
train_sentences = []
with open('train_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f:
        train_sentences.append(line.strip())

test_sentences = []
with open('test_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f:
        test_sentences.append(line.strip())

train_labels = np.load('train_labels.npy')

print(f"Train: {len(train_sentences)} propozitii")
print(f"Test: {len(test_sentences)} propozitii")
print(f"Clase unice: {np.unique(train_labels)}")


# ================================================================
# PASUL 2: BAG OF WORDS LA NIVEL DE CARACTER
# ================================================================

class CharBagOfWords:
    def __init__(self):
        self.vocabulary = {}   # caracter -> id
        self.char_list = []    # lista caracterelor in ordinea adaugarii

    def build_vocabulary(self, sentences):
        """
        Parcurgem fiecare caracter din fiecare propozitie de antrenare.
        Fiecare caracter unic primeste un ID (index) in vocabular.
        """
        idx = 0
        for sentence in sentences:
            for char in sentence:
                if char not in self.vocabulary:
                    self.vocabulary[char] = idx
                    self.char_list.append(char)
                    idx += 1
        print(f"Vocabular: {len(self.vocabulary)} caractere unice")

    def get_features(self, sentences):
        """
        Pentru fiecare propozitie, cream un vector de frecvente.
        features[i][j] = de cate ori apare caracterul j in propozitia i.

        Exemplu: "ana", vocab={'a':0, 'n':1} -> [2, 1] (a apare de 2 ori, n de 1)
        """
        num_samples = len(sentences)
        num_features = len(self.vocabulary)
        features = np.zeros((num_samples, num_features))

        for i, sentence in enumerate(sentences):
            for char in sentence:
                if char in self.vocabulary:
                    features[i][self.vocabulary[char]] += 1

        return features


# Construim vocabularul si extragem features
bow = CharBagOfWords()
bow.build_vocabulary(train_sentences)

train_features = bow.get_features(train_sentences)
test_features = bow.get_features(test_sentences)

print(f"Train features: {train_features.shape}")
print(f"Test features: {test_features.shape}")


# ================================================================
# PASUL 3: NORMALIZARE
# ================================================================
# Normalizam cu L2 ca sa scalam toate documentele la aceeasi magnitudine
train_norm = normalize(train_features, norm='l2')
test_norm = normalize(test_features, norm='l2')


# ================================================================
# PASUL 4: PREGATIRE ETICHETE ONE-HOT
# ================================================================
# Ridge e regresie, nu clasificare. Trick-ul:
#   clasa 0 -> [1, 0, 0]
#   clasa 1 -> [0, 1, 0]
#   clasa 2 -> [0, 0, 1]
# Ridge prezice scoruri, luam argmax.

num_classes = len(np.unique(train_labels))
n_train = len(train_labels)

y_one_hot = np.zeros((n_train, num_classes))
for i in range(n_train):
    y_one_hot[i, int(train_labels[i])] = 1

print(f"Numar clase: {num_classes}")


# ================================================================
# PASUL 5: IMPLEMENTARE RIDGE REGRESSION
# ================================================================

def ridge_fit(X, y, alpha=1.0):
    """
    Antreneaza Ridge Regression.

    Formula: W = (X^T * X + alpha * I)^(-1) * X^T * y

    X: matrice de features (n_samples x n_features)
    y: etichete one-hot (n_samples x n_classes)
    alpha: parametru de regularizare
      - alpha mare -> ponderi mici, model simplu (risc underfitting)
      - alpha mic -> ponderi mari, model complex (risc overfitting)

    Returneaza: matricea de ponderi W (n_features x n_classes)
    """
    n_features = X.shape[1]
    # X^T * X -> matrice (n_features x n_features)
    XtX = X.T.dot(X)
    # Adaugam alpha pe diagonala (regularizare)
    XtX_reg = XtX + alpha * np.eye(n_features)
    # X^T * y -> matrice (n_features x n_classes)
    Xty = X.T.dot(y)
    # Rezolvam sistemul liniar (mai stabil decat inversarea matricei)
    W = np.linalg.solve(XtX_reg, Xty)
    return W


def ridge_predict(X, W):
    """
    Prezice scoruri si returneaza clasele.

    X: features (n_samples x n_features)
    W: ponderi (n_features x n_classes)

    Returneaza: etichete prezise (n_samples,)
    """
    scores = X.dot(W)          # scor per clasa
    return np.argmax(scores, axis=1)  # clasa cu scorul maxim


# ================================================================
# PASUL 6: CAUTARE HIPERPARAMETRI (alpha)
# ================================================================
np.random.seed(42)
indices = np.arange(n_train)
np.random.shuffle(indices)
split = int(0.8 * n_train)

train_idx = indices[:split]
val_idx = indices[split:]

print("\n--- Cautare hiperparametri (alpha) ---")
best_alpha = 1.0
best_acc = 0

for alpha in [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
    W = ridge_fit(train_norm[train_idx], y_one_hot[train_idx], alpha=alpha)
    val_pred = ridge_predict(train_norm[val_idx], W)
    val_acc = accuracy_score(train_labels[val_idx], val_pred)
    print(f"  alpha={alpha:8.4f} -> Acuratete validare: {val_acc * 100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        best_alpha = alpha

print(f"\nCel mai bun alpha: {best_alpha} (acc validare: {best_acc * 100:.2f}%)")


# ================================================================
# PASUL 7: ANTRENARE FINALA SI PREDICTII
# ================================================================
W_final = ridge_fit(train_norm, y_one_hot, alpha=best_alpha)
predictions = ridge_predict(test_norm, W_final)

print(f"\nPredictii generate: {len(predictions)} etichete")

# Salvam
np.save('subiect1_solutia_1.npy', predictions)
print("Predictii salvate in subiect1_solutia_1.npy")

# Distributia
unique, counts = np.unique(predictions, return_counts=True)
print("\nDistributia predictiilor:")
for cls, cnt in zip(unique, counts):
    print(f"  Clasa {int(cls)}: {cnt} exemple")
