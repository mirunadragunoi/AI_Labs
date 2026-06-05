import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ================================================================
# EXERCITIUL 1 - Naive Bayes cu Bag of Words la nivel de CARACTER
# ================================================================
# IDEEA GENERALA:
#   In loc sa numaram cuvinte (ca la lab4), numaram CARACTERE.
#   Fiecare document devine un vector de frecvente ale caracterelor.
#   Exemplu: "ana" -> {'a': 2, 'n': 1} -> vector [2, 0, ..., 1, ..., 0]
#   Apoi dam acesti vectori la Naive Bayes care invata P(caracter | clasa).


# ================================================================
# PASUL 1: INCARCAREA DATELOR
# ================================================================
# train_sentences.txt - 1025 propozitii de antrenare (una per linie)
# test_sentences.txt  - 776 propozitii de testare
# train_labels.npy    - etichetele (clasele) pentru antrenare

# Citim propozitiile din fisierele text
train_sentences = []
with open('train_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f:
        train_sentences.append(line.strip())

test_sentences = []
with open('test_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f:
        test_sentences.append(line.strip())

# Incarcam etichetele de antrenare
train_labels = np.load('train_labels.npy')

print(f"Train: {len(train_sentences)} propozitii")
print(f"Test: {len(test_sentences)} propozitii")
print(f"Labels: {len(train_labels)}")
print(f"Clase unice: {np.unique(train_labels)}")
print(f"Exemplu propozitie: '{train_sentences[0][:80]}...'")


# ================================================================
# PASUL 2: CONSTRUIREA VOCABULARULUI DE CARACTERE
# ================================================================
# Parcurgem TOATE propozitiile de antrenare si colectam fiecare caracter unic.
# Fiecare caracter primeste un ID unic (index in vector).
#
# Exemplu:
#   Daca vocabularul e {'a': 0, 'b': 1, 'c': 2, 'n': 3}
#   "abba" -> [2, 2, 0, 0] (a apare de 2 ori, b de 2 ori, c de 0, n de 0)

class CharBagOfWords:
    def __init__(self):
        self.vocabulary = {}   # caracter -> id
        self.char_list = []    # lista caracterelor in ordinea adaugarii

    def build_vocabulary(self, sentences):
        """
        Construieste vocabularul de caractere din propozitiile de antrenare.
        Parcurge fiecare caracter din fiecare propozitie.
        Daca e un caracter nou, ii atribuie urmatorul ID.
        """
        idx = 0
        for sentence in sentences:
            for char in sentence:
                if char not in self.vocabulary:
                    self.vocabulary[char] = idx
                    self.char_list.append(char)
                    idx += 1

        print(f"Vocabular construit: {len(self.vocabulary)} caractere unice")

    def get_features(self, sentences):
        """
        Transforma propozitiile in vectori de frecvente ale caracterelor.

        Pentru fiecare propozitie:
          - cream un vector de zerouri de lungimea vocabularului
          - pentru fiecare caracter din propozitie, incrementam pozitia corespunzatoare
          - ignoram caracterele care nu sunt in vocabular (caractere noi din test)

        Returneaza: matrice (num_propozitii x dimensiune_vocabular)
        """
        num_samples = len(sentences)
        num_features = len(self.vocabulary)
        features = np.zeros((num_samples, num_features))

        for i, sentence in enumerate(sentences):
            for char in sentence:
                if char in self.vocabulary:
                    char_idx = self.vocabulary[char]
                    features[i][char_idx] += 1

        return features


# Construim vocabularul si extragem features
bow = CharBagOfWords()
bow.build_vocabulary(train_sentences)

train_features = bow.get_features(train_sentences)
test_features = bow.get_features(test_sentences)

print(f"Train features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")


# ================================================================
# PASUL 3: NORMALIZARE (optional, poate imbunatati acuratetea)
# ================================================================
# Normalizam cu L2 ca fiecare propozitie sa aiba aceeasi "greutate"
# indiferent de lungimea ei.
from sklearn.preprocessing import normalize

train_norm = normalize(train_features, norm='l2')
test_norm = normalize(test_features, norm='l2')


# ================================================================
# PASUL 4: ANTRENAREA MODELULUI NAIVE BAYES
# ================================================================
# MultinomialNB functioneaza cu frecvente (valori >= 0).
# alpha = parametru de smoothing (Laplace smoothing):
#   - adauga alpha la fiecare numaratoare ca sa evitam probabilitati de 0
#   - alpha=1 e clasic, valori mai mici pot fi mai bune

# Testam mai multe valori alpha pentru a gasi cea mai buna
best_alpha = 1.0
best_acc = 0

# Cross-validation simpla: impartim train in 80% train / 20% validare
np.random.seed(42)
indices = np.arange(len(train_features))
np.random.shuffle(indices)
split = int(0.8 * len(indices))

train_idx = indices[:split]
val_idx = indices[split:]

print("\n--- Cautare hiperparametri (alpha) ---")
for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    model = MultinomialNB(alpha=alpha)
    # Folosim features NENORMALIZATE pentru MultinomialNB (are nevoie de valori >= 0)
    model.fit(train_features[train_idx], train_labels[train_idx])
    val_pred = model.predict(train_features[val_idx])
    val_acc = accuracy_score(train_labels[val_idx], val_pred)
    print(f"  alpha={alpha:6.3f} -> Acuratete validare: {val_acc * 100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        best_alpha = alpha

print(f"\nCel mai bun alpha: {best_alpha} (acc validare: {best_acc * 100:.2f}%)")


# ================================================================
# PASUL 5: ANTRENARE FINALA SI PREDICTII
# ================================================================
# Antrenam pe TOATE datele de antrenare cu cel mai bun alpha
model_final = MultinomialNB(alpha=best_alpha)
model_final.fit(train_features, train_labels)

# Predictii pe test
predictions = model_final.predict(test_features)
print(f"\nPredictii generate: {len(predictions)} etichete")

# Salvam predictiile in format .npy
np.save('subiect1_solutia_1.npy', predictions)
print("Predictii salvate in subiect1_solutia_1.npy")

# Afisam distributia predictiilor
unique, counts = np.unique(predictions, return_counts=True)
print("\nDistributia predictiilor:")
for cls, cnt in zip(unique, counts):
    print(f"  Clasa {int(cls)}: {cnt} exemple")
