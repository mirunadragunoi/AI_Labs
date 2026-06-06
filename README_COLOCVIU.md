# GHID COMPLET — Inteligență Artificială (Laboratoare + Colocviu)

---

## CUPRINS

1. [Structura colocviului — ce trebuie să știi](#structura-colocviului)
2. [Cum organizezi codul și fișierele](#organizare-cod)
3. [Încărcarea datelor — toate variantele](#incarcarea-datelor)
4. [Normalizare — toate metodele](#normalizare)
5. [Bag of Words (BoW)](#bag-of-words)
6. [Naive Bayes](#naive-bayes)
7. [KNN — K Nearest Neighbors](#knn)
8. [SVM — Support Vector Machines](#svm)
9. [Ridge Regression / Kernel Ridge (KRR)](#ridge-krr)
10. [Rețele neuronale (PyTorch)](#retele-neuronale)
11. [Funcții Kernel — toate variantele](#functii-kernel)
12. [Matrice de tranziție Markov](#markov)
13. [Convoluție cu n-grams](#convolutie)
14. [String Kernel](#string-kernel)
15. [Matricea de confuzie](#matrice-confuzie)
16. [Tabel rezumat — ce a apărut la fiecare colocviu](#tabel-rezumat)

---

## 1. STRUCTURA COLOCVIULUI {#structura-colocviului}

Colocviul are de obicei 5 exerciții (10 puncte total, 1p oficiu):

| Exercițiu | Ce se cere de obicei | Punctaj |
|-----------|---------------------|---------|
| Ex 1 | Model simplu (Naive Bayes / Ridge / Rețea neuronală) pe features BoW | 2-3.5p |
| Ex 2 | Extragere features (convoluție cu n-grams / Markov / string kernel) | 2-2.5p |
| Ex 3 | Model pe features de la Ex 2 (KNN / SVM / KRR) | 2-2.5p |
| Ex 4 | Alt model cu kernel precomputed (SVM / KRR + Hellinger/Intersection) | 2-2.5p |
| Ex 5 | Raport cu hiperparametri pe validare | 1.5p (1p oficiu) |

Punctajul la Ex 1, 3, 4 depinde de acuratețea obținută pe test.

---

## 2. CUM ORGANIZEZI CODUL ȘI FIȘIERELE {#organizare-cod}

### Structura folderului de submisie

```
Nume_Prenume_Grupa_Varianta/
├── Nume_Prenume_Grupa_subiect1_solutia_1.py     # cod ex 1
├── Nume_Prenume_Grupa_subiect1_solutia_1.npy     # predictii ex 1 (sau .txt)
├── Nume_Prenume_Grupa_subiect2_solutia_1.py     # cod ex 2
├── Nume_Prenume_Grupa_subiect3_solutia_1.py     # cod ex 3
├── Nume_Prenume_Grupa_subiect3_solutia_1.npy     # predictii ex 3
├── Nume_Prenume_Grupa_subiect4_solutia_1.py     # cod ex 4
├── Nume_Prenume_Grupa_subiect4_solutia_1.npy     # predictii ex 4
└── raport_experimente.txt                        # raport ex 5
```

### Formatul predictiilor

**2025 (text classification):** fișiere `.npy`
```python
np.save('subiect1_solutia_1.npy', predictions)
```

**2024 (accelerometru):** fișiere `.txt` cu format `filename,label`
```python
with open('subiect3_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fname, pred in zip(test_files, predictions):
        f.write(f"{fname},{pred}\n")
```

**2022 (text):** fișiere `.txt` cu o predicție per linie
```python
with open('subiect4_solutia_1.txt', 'w') as f:
    for pred in predictions:
        f.write(f"{pred}\n")
```

### Ordinea de rulare

**MEREU:** Ex 2 (features) → Ex 1, 3, 4 (modele) → Ex 5 (raport)

Ex 2 generează features pe care le salvezi cu `np.save()` și le reîncarci în Ex 3, 4.

### Submisii multiple

La Ex 3 și 4 ai voie cel mult 3 submisii. Schimbă hiperparametrii și salvează cu `_solutia_2.npy`, `_solutia_3.npy`.

---

## 3. ÎNCĂRCAREA DATELOR {#incarcarea-datelor}

### Fișiere .npy (cel mai comun)
```python
import numpy as np
train_data = np.load('train_data.npy', allow_pickle=True)  # allow_pickle pt string-uri
train_labels = np.load('train_labels.npy').astype(int)
test_data = np.load('test_data.npy', allow_pickle=True)
```

### Fișiere .txt cu text (propozitii, una per linie)
```python
train_sentences = []
with open('train_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f:
        train_sentences.append(line.strip())
```

### Fișiere .txt cu numere (MNIST)
```python
train_images = np.loadtxt('data/train_images.txt')       # matrice de numere
train_labels = np.loadtxt('data/train_labels.txt').astype(int)
```

### Fișiere semnal accelerometru (2024)
```python
import os

def load_signal(filepath):
    return np.loadtxt(filepath)  # matrice (timestamps x 3)

# Citim train.txt: header + filename,label per linie
def load_dataset(data_dir, labels_file):
    signals, filenames, labels = [], [], []
    with open(labels_file, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:  # sarim header
        parts = line.strip().split(',')
        signals.append(load_signal(os.path.join(data_dir, parts[0])))
        filenames.append(parts[0])
        labels.append(int(parts[1]))
    return signals, filenames, np.array(labels)

train_signals, train_files, train_labels = load_dataset('data/train', 'data/train.txt')
```

### Mapping caracter → număr (2025)
```python
char_to_num = {}
with open('mapping.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            if line.startswith(',,'):
                char_to_num[','] = int(line[2:])
            else:
                parts = line.split(',')
                if len(parts) == 2:
                    char_to_num[parts[0]] = int(parts[1])
```

---

## 4. NORMALIZARE {#normalizare}

### Standardizare (z-score) — medie=0, deviatie=1
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_data)                        # calculeaza media/std doar pe train
train_norm = scaler.transform(train_data)
test_norm = scaler.transform(test_data)       # aplica aceeasi media/std
```

SAU manual:
```python
mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)
std[std == 0] = 1
train_norm = (train_data - mean) / std
test_norm = (test_data - mean) / std
```

### Normalizare L1 / L2
```python
from sklearn.preprocessing import normalize

train_l1 = normalize(train_data, norm='l1')   # fiecare rand: suma |xi| = 1
train_l2 = normalize(train_data, norm='l2')   # fiecare rand: sqrt(suma xi^2) = 1
```

### Normalizare lungime semnale (2024 — accelerometru)
```python
FIXED_LEN = int(np.median([len(s) for s in signals]))

def normalize_length(signal, target_len):
    if len(signal) >= target_len:
        return signal[:target_len]      # truncam
    padding = np.zeros((target_len - len(signal), signal.shape[1]))
    return np.vstack([signal, padding])  # padding cu zerouri
```

---

## 5. BAG OF WORDS (BoW) {#bag-of-words}

### BoW la nivel de CUVÂNT (Lab 4 — spam)
```python
class BagOfWords:
    def __init__(self):
        self.vocabulary = {}
        self.words_list = []

    def build_vocabulary(self, data):
        """data = lista de mesaje (fiecare mesaj = lista de cuvinte)"""
        idx = 0
        for mesaj in data:
            for cuvant in mesaj:
                if cuvant not in self.vocabulary:
                    self.vocabulary[cuvant] = idx
                    self.words_list.append(cuvant)
                    idx += 1

    def get_features(self, data):
        features = np.zeros((len(data), len(self.vocabulary)))
        for i, mesaj in enumerate(data):
            for cuvant in mesaj:
                if cuvant in self.vocabulary:
                    features[i][self.vocabulary[cuvant]] += 1
        return features
```

### BoW la nivel de CARACTER (2025 — colocviu)
```python
class CharBagOfWords:
    def __init__(self):
        self.vocabulary = {}

    def build_vocabulary(self, sentences):
        """sentences = lista de string-uri"""
        idx = 0
        for sentence in sentences:
            for char in sentence:
                if char not in self.vocabulary:
                    self.vocabulary[char] = idx
                    idx += 1

    def get_features(self, sentences):
        features = np.zeros((len(sentences), len(self.vocabulary)))
        for i, sentence in enumerate(sentences):
            for char in sentence:
                if char in self.vocabulary:
                    features[i][self.vocabulary[char]] += 1
        return features
```

**Utilizare:**
```python
bow = CharBagOfWords()
bow.build_vocabulary(train_sentences)    # doar pe train!
train_features = bow.get_features(train_sentences)
test_features = bow.get_features(test_sentences)
```

---

## 6. NAIVE BAYES {#naive-bayes}

**Folosit la:** Lab 2 (MNIST), Colocviu 2025 V1 Ex1

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=best_alpha)   # alpha = smoothing
model.fit(train_features, train_labels)   # ATENTIE: features >= 0!
predictions = model.predict(test_features)
accuracy = model.score(test_features, test_labels)
```

**Discretizare pentru Naive Bayes (Lab 2 MNIST):**
```python
def values_to_bins(data, num_bins):
    bins = np.linspace(0, 255, num_bins + 1)
    return np.digitize(data, bins)
```

**Căutare alpha optim:**
```python
for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    acc = model.score(X_val, y_val)
```

---

## 7. KNN — K NEAREST NEIGHBORS {#knn}

**Folosit la:** Lab 3, Colocviu 2024 Ex3, Colocviu 2022 Ex2

### Implementare manuală (cerută la colocviu!)

```python
def knn_classify(train_features, train_labels, test_sample, k=3, metric='l2'):
    if metric == 'l2':  # Euclidiana
        dists = np.sqrt(np.sum((train_features - test_sample) ** 2, axis=1))
    elif metric == 'l1':  # Manhattan
        dists = np.sum(np.abs(train_features - test_sample), axis=1)
    elif metric == 'minkowski':
        p = 3  # sau 5, depinde de cerinta
        dists = np.sum(np.abs(train_features - test_sample) ** p, axis=1) ** (1.0/p)
    elif metric == 'hamming':
        dists = np.sum(train_features != test_sample, axis=1)

    nearest = np.argsort(dists)[:k]
    neighbor_labels = train_labels[nearest].astype(int)
    return np.bincount(neighbor_labels).argmax()
```

### KNN cu similaritate (în loc de distanță) — pentru string kernel
```python
# Similaritate MARE = mai aproape -> sortam DESCRESCATOR
sims = K_test[i]  # similaritati precalculate
nearest = np.argsort(-sims)[:k]  # minus pentru descrescator!
```

### Distanțe — formulele
| Distanță | Formula | Cod |
|----------|---------|-----|
| Manhattan (L1) | Σ\|xi-yi\| | `np.sum(np.abs(x-y), axis=1)` |
| Euclidiana (L2) | √(Σ(xi-yi)²) | `np.sqrt(np.sum((x-y)**2, axis=1))` |
| Minkowski p | (Σ\|xi-yi\|^p)^(1/p) | `np.sum(np.abs(x-y)**p, axis=1)**(1/p)` |
| Hamming | nr poziții diferite | `np.sum(x != y, axis=1)` |

---

## 8. SVM — SUPPORT VECTOR MACHINES {#svm}

**Folosit la:** Lab 4, Colocviu 2024, 2025

### SVM cu kernel standard
```python
from sklearn import svm

# Kernel linear
model = svm.SVC(C=1.0, kernel='linear')

# Kernel RBF
model = svm.SVC(C=10.0, kernel='rbf', gamma='scale')

model.fit(train_features, train_labels)
predictions = model.predict(test_features)
```

### SVM cu kernel PRECOMPUTED
```python
# 1. Calculezi matricea kernel (vezi sectiunea Kernel)
K_train = my_kernel(train_features, train_features)  # (n_train x n_train)
K_test = my_kernel(test_features, train_features)    # (n_test x n_train)

# 2. Antrenezi SVM cu kernel='precomputed'
model = svm.SVC(C=best_C, kernel='precomputed')
model.fit(K_train, train_labels)
predictions = model.predict(K_test)
```

### Cele mai spam/non-spam cuvinte (Lab 4)
```python
weights = model.coef_[0]
sorted_idx = np.argsort(weights)
spam_words = [bow.words_list[i] for i in sorted_idx[:10]]      # cele mai negative
ham_words = [bow.words_list[i] for i in sorted_idx[-10:]]      # cele mai pozitive
```

---

## 9. RIDGE REGRESSION / KERNEL RIDGE (KRR) {#ridge-krr}

**Folosit la:** Lab 5, Colocviu 2024 Ex4, 2025 Ex1/Ex3/Ex4, 2022 Ex4

### Ridge Regression simplu (fără kernel)
```python
def ridge_fit(X, y_one_hot, alpha=1.0):
    n_features = X.shape[1]
    W = np.linalg.solve(X.T.dot(X) + alpha * np.eye(n_features), X.T.dot(y_one_hot))
    return W

def ridge_predict(X, W):
    return np.argmax(X.dot(W), axis=1)
```

### Kernel Ridge Regression (KRR) cu kernel precomputed
```python
def krr_fit(K_train, y_one_hot, lambd=1.0):
    """alpha = (K + lambda * I)^(-1) * y"""
    n = K_train.shape[0]
    return np.linalg.solve(K_train + lambd * np.eye(n), y_one_hot)

def krr_predict(K_test, alpha_coefs):
    """predictie = K_test * alpha -> argmax"""
    return np.argmax(K_test.dot(alpha_coefs), axis=1)
```

### Pregătirea etichetelor ONE-HOT (obligatoriu pt Ridge/KRR!)
```python
num_classes = len(np.unique(train_labels))
y_one_hot = np.zeros((len(train_labels), num_classes))
for i in range(len(train_labels)):
    y_one_hot[i, int(train_labels[i])] = 1
```

### Utilizare completă KRR
```python
# 1. Calculezi kernel
K_train = my_kernel(train_norm, train_norm)
K_test = my_kernel(test_norm, train_norm)

# 2. One-hot
y_oh = np.zeros((len(train_labels), num_classes))
for i in range(len(train_labels)): y_oh[i, int(train_labels[i])] = 1

# 3. Fit + predict
alpha_coefs = krr_fit(K_train, y_oh, lambd=best_lambda)
predictions = krr_predict(K_test, alpha_coefs)
```

---

## 10. REȚELE NEURONALE (PyTorch) {#retele-neuronale}

**Folosit la:** Lab 6, Colocviu 2024 Ex1

### Template complet
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Definirea retelei
class FeedForwardNet(nn.Module):
    def __init__(self, input_size, h1, h2, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Pregatirea datelor
X_train = torch.FloatTensor(train_scaled)
y_train = torch.LongTensor(train_labels)
X_test = torch.FloatTensor(test_scaled)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# Model + optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FeedForwardNet(num_features, 256, 128, num_classes).to(device)

# ALEGEREA OPTIMIZATORULUI (depinde de cerinta!):
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)           # Adam
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)             # SGD lr constant
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)  # SGD + momentum

loss_fn = nn.CrossEntropyLoss()

# Antrenare
for epoch in range(50):
    model.train()
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        loss = loss_fn(model(bx), by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Predictii
model.eval()
with torch.no_grad():
    predictions = model(X_test.to(device)).argmax(1).cpu().numpy()
```

### Variante de optimizator (ce a apărut la colocvii)
| Cerință | Cod |
|---------|-----|
| Adam | `torch.optim.Adam(model.parameters(), lr=0.001)` |
| SGD lr=1e-3 | `torch.optim.SGD(model.parameters(), lr=1e-3)` |
| SGD generic | `torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)` |

---

## 11. FUNCȚII KERNEL {#functii-kernel}

### Kernel Liniar
```python
def linear_kernel(X, Y):
    """K(x,y) = x · y"""
    return X.dot(Y.T)
```

### Kernel Intersecție
```python
def intersection_kernel(X, Y):
    """K(x,y) = sum( min(xi, yi) ) — bun pt histograme/probabilitati"""
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        K[i] = np.minimum(X[i], Y).sum(axis=1)
    return K
```

### Kernel Hellinger
```python
def hellinger_kernel(X, Y):
    """K(x,y) = sum( sqrt(xi * yi) ) — bun pt distributii"""
    return np.sqrt(np.maximum(X, 0)).dot(np.sqrt(np.maximum(Y, 0)).T)
```

### String Kernel (2022)
```python
def get_ngrams(text, p=8):
    return {text[i:i+p] for i in range(len(text) - p + 1)}

def string_kernel(s, t, p=8):
    """K(s,t) = nr n-grame comune (biti de prezenta)"""
    return len(get_ngrams(s, p) & get_ngrams(t, p))
```

### Cum se folosesc cu SVM/KRR
```python
# Calculeaza DOUA matrici:
K_train = my_kernel(train_data, train_data)   # (n_train x n_train) pt antrenare
K_test = my_kernel(test_data, train_data)     # (n_test x n_train) pt testare

# SVM precomputed
model = svm.SVC(C=C, kernel='precomputed')
model.fit(K_train, train_labels)
pred = model.predict(K_test)

# SAU KRR precomputed
alpha = np.linalg.solve(K_train + lambd * np.eye(n), y_one_hot)
pred = np.argmax(K_test.dot(alpha), axis=1)
```

### Tabel: ce kernel s-a folosit unde
| Colocviu | Ex 4 model | Kernel |
|----------|-----------|--------|
| 2025 V1 | SVM | Hellinger |
| 2025 V2 | KRR | Intersection |
| 2024 V1 | SVM | Intersection |
| 2024 V2 | KRR | Hellinger |
| 2024 V3 | SVM | Hellinger |
| 2024 V4 | KRR | Intersection |
| 2022 | KRR | String Kernel |

---

## 12. MATRICE DE TRANZIȚIE MARKOV (2024) {#markov}

**Folosit la:** Colocviu 2024 Ex2 (toate variantele)

### Ce face
Transformă un semnal temporal (accelerometru) într-un vector de features de dimensiune fixă.

### Pași
1. **Discretizare:** împarte range-ul [min, max] în k intervale egale, înlocuiește fiecare valoare cu indexul intervalului
2. **Matrice de tranziție:** numără tranzițiile între stări consecutive, normalizează pe rânduri
3. **Concatenare:** liniarizează cele 3 matrici (x, y, z) într-un vector

### Cod complet
```python
# Range din TRAIN
all_values = np.vstack(train_signals)
axis_ranges = [(all_values[:, i].min(), all_values[:, i].max()) for i in range(3)]

def discretize(signal, axis_ranges, k):
    disc = np.zeros_like(signal, dtype=int)
    for ax in range(3):
        bins = np.linspace(axis_ranges[ax][0], axis_ranges[ax][1], k + 1)
        disc[:, ax] = np.clip(np.digitize(signal[:, ax], bins) - 1, 0, k - 1)
    return disc

def transition_matrix(axis_vals, k):
    A = np.zeros((k, k))
    for t in range(len(axis_vals) - 1):
        A[axis_vals[t]][axis_vals[t+1]] += 1
    sums = A.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    return A / sums

def markov_features(signal, axis_ranges, k):
    disc = discretize(signal, axis_ranges, k)
    feats = []
    for ax in range(3):
        feats.extend(transition_matrix(disc[:, ax], k).flatten())
    return np.array(feats)

# Aplicare
train_markov = np.array([markov_features(s, axis_ranges, k) for s in train_signals])
```

### Valori k la fiecare variantă
| V1 | V2 | V3 | V4 |
|----|----|----|-----|
| k=6 (108 feat) | k=4 (48 feat) | k=7 (147 feat) | k=5 (75 feat) |

---

## 13. CONVOLUȚIE CU N-GRAMS (2025) {#convolutie}

**Folosit la:** Colocviu 2025 Ex2

### Ce face
Glisează 500 de filtre (3-grams) peste fiecare document, calculează cosinus similarity la fiecare poziție, numără câte valori depășesc pragul 0.9.

### Cod complet
```python
def text_to_numbers(text, mapping):
    return np.array([mapping.get(c, 0) for c in text], dtype=float)

def convolution_1d(doc_nums, filter_nums):
    L, n = len(doc_nums), len(filter_nums)
    if L < n: return np.array([])
    result = np.zeros(L - n + 1)
    f_norm = np.linalg.norm(filter_nums)
    if f_norm == 0: return result
    for i in range(L - n + 1):
        sub = doc_nums[i:i+n]
        s_norm = np.linalg.norm(sub)
        if s_norm > 0:
            result[i] = np.dot(sub, filter_nums) / (s_norm * f_norm)
    return result

def extract_conv_features(sentences, trigrams, mapping, threshold=0.9):
    features = np.zeros((len(sentences), len(trigrams)))
    tg_nums = [text_to_numbers(tg, mapping) for tg in trigrams]
    for i, sent in enumerate(sentences):
        doc_nums = text_to_numbers(sent, mapping)
        for k, tg in enumerate(tg_nums):
            conv = convolution_1d(doc_nums, tg)
            if len(conv) > 0:
                features[i][k] = np.sum(conv > threshold)
    return features
```

---

## 14. STRING KERNEL (2022) {#string-kernel}

**Folosit la:** Colocviu 2022

### Ce face
Măsoară similaritatea între două texte prin numărul de n-grame (sub-secvențe de p caractere) comune.

### Cod
```python
def get_ngrams(text, p=8):
    return {text[i:i+p] for i in range(len(text) - p + 1)}

def string_kernel_similarity(s, t, p=8):
    return len(get_ngrams(str(s), p) & get_ngrams(str(t), p))

# Matrice kernel simetrica (optimizata)
def compute_symmetric_kernel(data, p=8):
    n = len(data)
    K = np.zeros((n, n))
    all_ng = [get_ngrams(str(t), p) for t in data]
    for i in range(n):
        for j in range(i, n):
            K[i][j] = len(all_ng[i] & all_ng[j])
            K[j][i] = K[i][j]
    return K
```

---

## 15. MATRICEA DE CONFUZIE {#matrice-confuzie}

```python
# Manual
def confusion_matrix(y_true, y_pred):
    n = len(np.unique(y_true))
    C = np.zeros((n, n))
    for real, pred in zip(y_true, y_pred):
        C[int(real)][int(pred)] += 1
    return C

# Sau cu sklearn
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, predictions)
```

---

## 16. TABEL REZUMAT {#tabel-rezumat}

### Colocvii 2025 (clasificare text)
| | V1 Ex1 | V1 Ex3 | V1 Ex4 | V2 Ex1 | V2 Ex3 | V2 Ex4 |
|-|--------|--------|--------|--------|--------|--------|
| Model | Naive Bayes | KRR liniar | SVM Hellinger | Ridge | SVM RBF | KRR Intersection |
| Features | Char BoW | Convoluție | Convoluție | Char BoW | Convoluție | Convoluție |

### Colocvii 2024 (clasificare accelerometru)
| | V1 | V2 | V3 | V4 |
|-|----|----|----|----|
| Ex1 | NN+Adam | NN+SGD 1e-3 | NN+SGD 1e-3 | NN+SGD |
| Ex2 k | 6 | 4 | 7 | 5 |
| Ex3 dist | Manhattan | Minkowski p=5 | Minkowski p=3 | Euclidiana |
| Ex4 | SVM+Intersection | KRR+Hellinger | SVM+Hellinger | KRR+Intersection |

### Colocviu 2022 (clasificare text)
| Ex | Ce |
|----|----|
| 1 | String kernel similarity (8-grams) |
| 2 | KNN cu string kernel |
| 3 | Matrice kernel precomputed |
| 4 | KRR cu kernel precomputed |

---

## 17. CROSS-VALIDATION (validare încrucișată)

**Folosit la:** Lab 5 (Ridge), căutarea hiperparametrilor la colocvii

### Split simplu train/validare (80/20) — cel mai rapid
```python
np.random.seed(42)
indices = np.arange(len(train_labels))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx = indices[:split]
val_idx = indices[split:]

# Folosire:
model.fit(X[train_idx], y[train_idx])
pred = model.predict(X[val_idx])
acc = accuracy_score(y[val_idx], pred)
```

### K-Fold Cross-Validation (3 fold-uri — Lab 5)
```python
num_folds = 3
fold_size = len(training_data) // num_folds

mse_list, mae_list = [], []

for fold in range(num_folds):
    start = fold * fold_size
    end = (fold + 1) * fold_size

    # Validare = fold-ul curent
    val_data = training_data[start:end]
    val_labels = labels[start:end]

    # Train = restul
    train_data = np.concatenate([training_data[:start], training_data[end:]])
    train_lab = np.concatenate([labels[:start], labels[end:]])

    # Normalizare (calculata pe TRAIN, aplicata pe ambele!)
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    std[std == 0] = 1
    train_norm = (train_data - mean) / std
    val_norm = (val_data - mean) / std

    # Antrenare + evaluare
    model.fit(train_norm, train_lab)
    pred = model.predict(val_norm)
    mse_list.append(mean_squared_error(val_labels, pred))

print(f"MSE mediu: {np.mean(mse_list):.4f}")
```

---

## 18. REGRESIE LINIARĂ / RIDGE / LASSO (Lab 5)

### Regresie Liniară
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_norm, train_labels)
pred = model.predict(test_norm)
```

### Ridge Regression (cu sklearn)
```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(train_norm, train_labels)
pred = model.predict(test_norm)
```

### Lasso Regression
```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=1.0)
model.fit(train_norm, train_labels)
pred = model.predict(test_norm)
```

### MSE și MAE
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
```

### Coeficienți și atributul cel mai semnificativ (Lab 5 Ex4)
```python
model = Ridge(alpha=best_alpha)
model.fit(data_norm, labels)

print(f"Bias: {model.intercept_}")
print(f"Coeficienti: {model.coef_}")

# Cel mai semnificativ = coeficientul cu |valoare| cea mai mare
coef_abs = np.abs(model.coef_)
idx_max = np.argmax(coef_abs)       # cel mai semnificativ
idx_min = np.argmin(coef_abs)       # cel mai putin semnificativ
```

---

## 19. PERCEPTRONUL ȘI WIDROW-HOFF (Lab 6)

### Perceptronul simplu — algoritmul Widrow-Hoff
```python
# Date: X = [[0,0], [0,1], [1,0], [1,1]], y = [-1, 1, 1, 1]
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
y = np.array([-1, 1, 1, 1], dtype=float)

W = np.zeros(2)       # ponderi initializate cu 0
b = 0.0               # bias = 0
lr = 0.1              # rata de invatare
epochs = 70

for epoch in range(epochs):
    X_s, y_s = shuffle(X, y, random_state=epoch)
    for t in range(len(X_s)):
        # Predictie (functia de activare = identitatea)
        y_hat = X_s[t].dot(W) + b

        # Actualizare ponderi (gradient descent pe MSE)
        W = W - lr * (y_hat - y_s[t]) * X_s[t]
        b = b - lr * (y_hat - y_s[t])

# Predictie finala cu sign
predictii = np.sign(X.dot(W) + b)
```

**Observație importantă:** Un singur perceptron NU poate rezolva XOR. Ai nevoie de o rețea (strat ascuns).

### Rețea neuronală manuală pentru XOR (Lab 6 Ex4)
```python
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh_deriv(x): return 1 - np.tanh(x) ** 2

# Date XOR cu etichete 0/1 (pentru sigmoid)
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

# Initializare ponderi
W1 = np.random.normal(0, 1, (2, 5))    # 2 intrari -> 5 neuroni ascunsi
b1 = np.zeros(5)
W2 = np.random.normal(0, 1, (5, 1))    # 5 neuroni ascunsi -> 1 iesire
b2 = np.zeros(1)

lr = 0.5
for epoch in range(70):
    # Forward
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)

    # Backward
    dz2 = a2 - y
    dW2 = a1.T.dot(dz2) / len(X)
    db2 = np.sum(dz2, axis=0) / len(X)

    da1 = dz2.dot(W2.T)
    dz1 = da1 * tanh_deriv(z1)
    dW1 = X.T.dot(dz1) / len(X)
    db1 = np.sum(dz1, axis=0) / len(X)

    # Update
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2
```

### sklearn MLPClassifier
```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(100,),      # tuple: neuroni per strat ascuns
    activation='relu',               # 'relu', 'tanh', 'logistic'
    solver='sgd',                    # 'sgd', 'adam'
    learning_rate_init=0.001,
    max_iter=200,
    shuffle=True,
    momentum=0.9
)
model.fit(train_data, train_labels)
pred = model.predict(test_data)
acc = model.score(test_data, test_labels)
```

---

## 20. LBP — LOCAL BINARY PATTERN (Lab 7)

### Ce face
Pentru fiecare pixel, compară cu vecinii din vecinătatea d×d. Vecin >= centru → 1, altfel → 0. Rezultă un pattern binar per pixel. Histograma pattern-urilor = feature-ul imaginii.

```python
def compute_lbp(image, d=3):
    h, w = image.shape
    half = d // 2
    patterns = []
    for i in range(half, h - half):
        for j in range(half, w - half):
            centru = image[i, j]
            vecini = image[i-half:i+half+1, j-half:j+half+1]
            binar = (vecini >= centru).astype(int).flatten()
            centru_idx = half * d + half
            binar = np.delete(binar, centru_idx)
            patterns.append(''.join(binar.astype(str)))
    return patterns

def lbp_histogram(image, d=3):
    patterns = compute_lbp(image, d)
    num_bits = d * d - 1
    histogram = np.zeros(2 ** num_bits)
    for p in patterns:
        histogram[int(p, 2)] += 1
    if histogram.sum() > 0:
        histogram /= histogram.sum()
    return histogram
```

---

## 21. GRADIENT MAGNITUDE + NMS (Lab 7)

### Magnitudinea gradientului
```python
def compute_gradient(image):
    Gx = np.zeros_like(image, dtype=float)
    Gy = np.zeros_like(image, dtype=float)
    Gx[:, :-1] = image[:, 1:] - image[:, :-1]   # diferenta orizontala
    Gy[:-1, :] = image[1:, :] - image[:-1, :]    # diferenta verticala
    G = np.sqrt(Gx**2 + Gy**2)                    # magnitudine
    return Gx, Gy, G
```

### Non-Maximum Suppression
```python
def nms(image):
    Gx, Gy, G = compute_gradient(image)
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta[theta < 0] += 180
    result = np.zeros_like(G)

    for i in range(1, G.shape[0]-1):
        for j in range(1, G.shape[1]-1):
            angle = theta[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                v1, v2 = G[i, j-1], G[i, j+1]
            elif 22.5 <= angle < 67.5:
                v1, v2 = G[i-1, j+1], G[i+1, j-1]
            elif 67.5 <= angle < 112.5:
                v1, v2 = G[i-1, j], G[i+1, j]
            else:
                v1, v2 = G[i-1, j-1], G[i+1, j+1]

            if G[i, j] >= v1 and G[i, j] >= v2:
                result[i, j] = G[i, j]
    return result
```

---

## 22. F1-SCORE ȘI METRICI

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)                          # binar
f1_multi = f1_score(y_true, y_pred, average='weighted') # multiclass
cm = confusion_matrix(y_true, y_pred)
```

---

## 23. TEMPLATE RAPID — SCHELET COMPLET PENTRU COLOCVIU

### Dacă ai date TEXT (2025-style)
```python
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.metrics import accuracy_score

# === INCARCARE ===
train_sentences = []
with open('train_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f: train_sentences.append(line.strip())
test_sentences = []
with open('test_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f: test_sentences.append(line.strip())
train_labels = np.load('train_labels.npy').astype(int)

# === CHAR BOW ===
vocab = {}
idx = 0
for s in train_sentences:
    for c in s:
        if c not in vocab: vocab[c] = idx; idx += 1

def get_feat(sentences):
    F = np.zeros((len(sentences), len(vocab)))
    for i, s in enumerate(sentences):
        for c in s:
            if c in vocab: F[i][vocab[c]] += 1
    return F

train_feat = get_feat(train_sentences)
test_feat = get_feat(test_sentences)

# === NORMALIZARE ===
train_n = normalize(train_feat, norm='l2')
test_n = normalize(test_feat, norm='l2')

# === SPLIT VALIDARE ===
np.random.seed(42)
idx = np.arange(len(train_labels)); np.random.shuffle(idx)
tr, va = idx[:int(0.8*len(idx))], idx[int(0.8*len(idx)):]

# === MODEL + CAUTARE HIPERPARAMETRI ===
# ... (inlocuieste cu modelul cerut)

# === SALVARE ===
np.save('subiect1_solutia_1.npy', predictions)
```

### Dacă ai date ACCELEROMETRU (2024-style)
```python
import numpy as np, os

# === INCARCARE ===
def load_ds(d, lf=None):
    sigs, fns, labs = [], [], []
    if lf:
        with open(lf) as f: lines = f.readlines()
        for l in lines[1:]:
            l = l.strip()
            if l:
                p = l.split(',')
                sigs.append(np.loadtxt(os.path.join(d, p[0])))
                fns.append(p[0]); labs.append(int(p[1]))
        return sigs, fns, np.array(labs)
    else:
        with open('data/test.txt') as f: lines = f.readlines()
        for l in lines:
            l = l.strip()
            if l: sigs.append(np.loadtxt(os.path.join(d, l))); fns.append(l)
        return sigs, fns, None

tr_sig, tr_fn, tr_lab = load_ds('data/train', 'data/train.txt')
te_sig, te_fn, _ = load_ds('data/test')

# === MARKOV FEATURES ===
all_v = np.vstack(tr_sig)
ax_r = [(all_v[:,i].min(), all_v[:,i].max()) for i in range(3)]
k = 6  # SCHIMBA CONFORM CERINTEI

def markov(sig):
    disc = np.zeros_like(sig, dtype=int)
    for ax in range(3):
        bins = np.linspace(ax_r[ax][0], ax_r[ax][1], k+1)
        disc[:,ax] = np.clip(np.digitize(sig[:,ax], bins)-1, 0, k-1)
    feats = []
    for ax in range(3):
        A = np.zeros((k,k))
        for t in range(len(disc[:,ax])-1):
            A[disc[t,ax]][disc[t+1,ax]] += 1
        s = A.sum(1, keepdims=True); s[s==0]=1
        feats.extend((A/s).flatten())
    return np.array(feats)

tr_f = np.array([markov(s) for s in tr_sig])
te_f = np.array([markov(s) for s in te_sig])

# === MODEL ===
# ... (KNN / SVM / KRR conform cerintei)

# === SALVARE ===
with open('subiect3_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fn, p in zip(te_fn, preds): f.write(f"{fn},{p}\n")
```

---

## 24. CUM FACI RAPORTUL (Ex 5)

```python
raport = []
raport.append("=" * 60)
raport.append("RAPORT EXPERIMENTAL")
raport.append("=" * 60)

# Pentru fiecare model, tabel cu hiperparametri testati
raport.append("\n--- Model X ---")
raport.append(f"{'Param':<12} {'Acc validare':<15}")
for param in [0.01, 0.1, 1.0, 10.0]:
    # antrenezi, evaluezi pe validare
    raport.append(f"{param:<12.3f} {acc*100:.2f}%")

raport.append(f"\nCel mai bun: {best_param} ({best_acc*100:.2f}%)")

# Sumar + observatii
raport.append("\nSUMAR:")
raport.append(f"Model 1: {acc1*100:.2f}%")
raport.append(f"Model 2: {acc2*100:.2f}%")
raport.append("Observatii: ...")

# Salvare
with open('raport_experimente.txt', 'w') as f:
    f.write("\n".join(raport))
```

---

## 25. NUMPY — FUNCȚII UTILE RAPID

```python
np.argsort(x)          # indicii care sorteaza array-ul
np.argmax(x)           # indexul valorii maxime
np.bincount(x)         # numara aparitiile fiecarei valori
np.unique(x)           # valorile unice
np.where(x == 3)       # indicii unde conditia e adevarata
np.concatenate([a, b]) # concateneaza array-uri
np.vstack([a, b])      # concateneaza pe verticala
np.hstack([a, b])      # concateneaza pe orizontala
np.linalg.solve(A, b)  # rezolva A*x = b (mai bun decat inversare)
np.linalg.norm(x)      # norma L2 a vectorului
np.minimum(a, b)       # min element cu element
np.clip(x, 0, k-1)    # limiteaza valorile in [0, k-1]
np.digitize(x, bins)   # in ce interval cade fiecare valoare
np.linspace(0, 255, k) # k valori uniform distribuite intre 0 si 255
np.eye(n)              # matricea identitate n x n
```

---

## CHECKLIST RAPID PENTRU COLOCVIU

1. **Citește cerința** — identifică: tipul datelor, modelul cerut, metrica, kernel-ul
2. **Încarcă datele** — `.npy` cu `np.load`, `.txt` cu `open`/`np.loadtxt`
3. **Fă split train/validare** (80/20) pentru căutarea hiperparametrilor:
```python
np.random.seed(42)
idx = np.arange(len(train_labels))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, val_idx = idx[:split], idx[split:]
```
4. **Extrage features** (Ex 2) și salvează-le:
```python
np.save('train_features.npy', train_features)
np.save('test_features.npy', test_features)
```
5. **Normalizează** (L1 pt Hellinger/Intersection, L2 pt SVM RBF/KRR liniar)
6. **Caută hiperparametri** pe validare
7. **Antrenează pe TOT train-ul** cu cei mai buni hiperparametri
8. **Salvează predicțiile** în formatul cerut
9. **Fă raportul** — tabel cu hiperparametri testați și acuratețe pe validare

---

## ERORI FRECVENTE

| Eroare | Cauză | Soluție |
|--------|-------|---------|
| `MultinomialNB()` vs `MultinomialNB` | Lipsesc `()` | Pune paranteze! |
| `np.load` vs `np.loadtxt` | `.npy` ≠ `.txt` | `.npy` → `np.load`, `.txt` → `np.loadtxt` |
| `dtype='int'` nu merge | Fișierul are float-uri | `.astype(int)` după încărcare |
| `IndexError` la matrice confuzie | Labels sunt float | `.astype(int)` la labels |
| `plt.subplot` vs `plt.subplots` | Lipsește `s` | `plt.subplots(2, 5)` cu **s** |
| `model.fit(X, y_validare)` | Antrenezi cu labels greșite | `model.fit(X_train, y_train)` |

---

## 26. VARIANTE PREDICT — TIPARE POSIBILE + COD GATA

Am analizat pattern-urile din 2022, 2024 (4 variante) și 2025 (2 variante). Mai jos sunt **6 combinații noi** care NU au apărut dar sunt foarte probabile.

---

### VARIANTA PREDICT A — Text: Naive Bayes + Convoluție + KNN Euclidian + SVM Intersection

```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.metrics import accuracy_score

# ===================== INCARCARE =====================
train_sentences = []
with open('train_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f: train_sentences.append(line.strip())
test_sentences = []
with open('test_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f: test_sentences.append(line.strip())
train_labels = np.load('train_labels.npy').astype(int)

# ===================== EX 1: NAIVE BAYES + CHAR BOW =====================
vocab = {}
idx = 0
for s in train_sentences:
    for c in s:
        if c not in vocab: vocab[c] = idx; idx += 1

def bow_features(sentences):
    F = np.zeros((len(sentences), len(vocab)))
    for i, s in enumerate(sentences):
        for c in s:
            if c in vocab: F[i][vocab[c]] += 1
    return F

tr_bow = bow_features(train_sentences)
te_bow = bow_features(test_sentences)

# Cautare alpha
np.random.seed(42)
ii = np.arange(len(train_labels)); np.random.shuffle(ii)
sp = int(0.8*len(ii)); tri, vai = ii[:sp], ii[sp:]

best_a, best_acc = 1.0, 0
for a in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
    m = MultinomialNB(alpha=a)
    m.fit(tr_bow[tri], train_labels[tri])
    ac = accuracy_score(train_labels[vai], m.predict(tr_bow[vai]))
    if ac > best_acc: best_acc, best_a = ac, a

m = MultinomialNB(alpha=best_a)
m.fit(tr_bow, train_labels)
np.save('subiect1_solutia_1.npy', m.predict(te_bow))

# ===================== EX 2: CONVOLUTIE CU 3-GRAMS =====================
trigrams = []
with open('words.txt', 'r', encoding='utf-8') as f:
    for line in f: trigrams.append(line.strip())

char_to_num = {}
with open('mapping.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            if line.startswith(',,'): char_to_num[','] = int(line[2:])
            else:
                p = line.split(',')
                if len(p) == 2: char_to_num[p[0]] = int(p[1])

def txt2num(text):
    return np.array([char_to_num.get(c, 0) for c in text], dtype=float)

def conv1d(doc, filt):
    L, n = len(doc), len(filt)
    if L < n: return np.array([])
    res = np.zeros(L - n + 1)
    fn = np.linalg.norm(filt)
    if fn == 0: return res
    for i in range(L - n + 1):
        sub = doc[i:i+n]; sn = np.linalg.norm(sub)
        if sn > 0: res[i] = np.dot(sub, filt) / (sn * fn)
    return res

def conv_features(sentences, threshold=0.9):
    tg_n = [txt2num(t) for t in trigrams]
    F = np.zeros((len(sentences), len(trigrams)))
    for i, s in enumerate(sentences):
        dn = txt2num(s)
        for k, tn in enumerate(tg_n):
            c = conv1d(dn, tn)
            if len(c) > 0: F[i][k] = np.sum(c > threshold)
    return F

tr_conv = conv_features(train_sentences)
te_conv = conv_features(test_sentences)
np.save('train_conv.npy', tr_conv)
np.save('test_conv.npy', te_conv)

# ===================== EX 3: KNN EUCLIDIAN =====================
tr_n = normalize(tr_conv, norm='l2')
te_n = normalize(te_conv, norm='l2')

best_k, best_acc = 3, 0
for k in [1, 3, 5, 7, 9]:
    preds = []
    for i in vai:
        d = np.sqrt(np.sum((tr_n[tri] - tr_n[i])**2, axis=1))
        nn = np.argsort(d)[:k]
        preds.append(np.bincount(train_labels[tri][nn]).argmax())
    ac = accuracy_score(train_labels[vai], preds)
    if ac > best_acc: best_acc, best_k = ac, k

te_pred = []
for i in range(len(te_n)):
    d = np.sqrt(np.sum((tr_n - te_n[i])**2, axis=1))
    nn = np.argsort(d)[:best_k]
    te_pred.append(np.bincount(train_labels[nn]).argmax())
np.save('subiect3_solutia_1.npy', np.array(te_pred))

# ===================== EX 4: SVM INTERSECTION =====================
def intersection_kernel(X, Y):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        K[i] = np.minimum(X[i], Y).sum(axis=1)
    return K

tr_l1 = normalize(tr_conv, norm='l1')
te_l1 = normalize(te_conv, norm='l1')

K_tr_s = intersection_kernel(tr_l1[tri], tr_l1[tri])
K_va_s = intersection_kernel(tr_l1[vai], tr_l1[tri])

best_C, best_acc = 1.0, 0
for C in [0.1, 1.0, 10.0, 50.0, 100.0]:
    m = svm.SVC(C=C, kernel='precomputed')
    m.fit(K_tr_s, train_labels[tri])
    ac = accuracy_score(train_labels[vai], m.predict(K_va_s))
    if ac > best_acc: best_acc, best_C = ac, C

K_tr = intersection_kernel(tr_l1, tr_l1)
K_te = intersection_kernel(te_l1, tr_l1)
m = svm.SVC(C=best_C, kernel='precomputed')
m.fit(K_tr, train_labels)
np.save('subiect4_solutia_1.npy', m.predict(K_te))
```

---

### VARIANTA PREDICT B — Text: Ridge + Convoluție + SVM RBF + KRR Hellinger

```python
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.metrics import accuracy_score

# === INCARCARE (identic) ===
train_sentences = []
with open('train_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f: train_sentences.append(line.strip())
test_sentences = []
with open('test_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f: test_sentences.append(line.strip())
train_labels = np.load('train_labels.npy').astype(int)

np.random.seed(42)
ii = np.arange(len(train_labels)); np.random.shuffle(ii)
sp = int(0.8*len(ii)); tri, vai = ii[:sp], ii[sp:]
nc = len(np.unique(train_labels))

# === EX 1: RIDGE PE CHAR BOW ===
vocab = {}
idx = 0
for s in train_sentences:
    for c in s:
        if c not in vocab: vocab[c] = idx; idx += 1

def bow_feat(sents):
    F = np.zeros((len(sents), len(vocab)))
    for i, s in enumerate(sents):
        for c in s:
            if c in vocab: F[i][vocab[c]] += 1
    return F

tr_bow = normalize(bow_feat(train_sentences), norm='l2')
te_bow = normalize(bow_feat(test_sentences), norm='l2')

y_oh = np.zeros((len(train_labels), nc))
for i in range(len(train_labels)): y_oh[i, train_labels[i]] = 1

best_al, best_acc = 1.0, 0
for al in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    nf = tr_bow.shape[1]
    W = np.linalg.solve(tr_bow[tri].T.dot(tr_bow[tri]) + al*np.eye(nf), tr_bow[tri].T.dot(y_oh[tri]))
    pred = np.argmax(tr_bow[vai].dot(W), axis=1)
    ac = accuracy_score(train_labels[vai], pred)
    if ac > best_acc: best_acc, best_al = ac, al

nf = tr_bow.shape[1]
W = np.linalg.solve(tr_bow.T.dot(tr_bow) + best_al*np.eye(nf), tr_bow.T.dot(y_oh))
np.save('subiect1_solutia_1.npy', np.argmax(te_bow.dot(W), axis=1))

# === EX 2: CONVOLUTIE (identic cu Varianta A) ===
# ... (copiaza din Varianta A)
tr_conv = np.load('train_conv.npy')  # sau recalculeaza
te_conv = np.load('test_conv.npy')

# === EX 3: SVM RBF ===
tr_n = normalize(tr_conv, norm='l2')
te_n = normalize(te_conv, norm='l2')

best_C, best_g, best_acc = 1.0, 'scale', 0
for C in [1.0, 10.0, 50.0, 100.0]:
    for g in ['scale', 0.01, 0.1]:
        m = svm.SVC(C=C, kernel='rbf', gamma=g)
        m.fit(tr_n[tri], train_labels[tri])
        ac = accuracy_score(train_labels[vai], m.predict(tr_n[vai]))
        if ac > best_acc: best_acc, best_C, best_g = ac, C, g

m = svm.SVC(C=best_C, kernel='rbf', gamma=best_g)
m.fit(tr_n, train_labels)
np.save('subiect3_solutia_1.npy', m.predict(te_n))

# === EX 4: KRR HELLINGER ===
def hellinger_kernel(X, Y):
    return np.sqrt(np.maximum(X, 0)).dot(np.sqrt(np.maximum(Y, 0)).T)

tr_l1 = normalize(tr_conv, norm='l1')
te_l1 = normalize(te_conv, norm='l1')

K_tr_s = hellinger_kernel(tr_l1[tri], tr_l1[tri])
K_va_s = hellinger_kernel(tr_l1[vai], tr_l1[tri])

best_lam, best_acc = 1.0, 0
for lam in [0.001, 0.01, 0.1, 1.0, 10.0]:
    n = K_tr_s.shape[0]
    a = np.linalg.solve(K_tr_s + lam*np.eye(n), y_oh[tri])
    ac = accuracy_score(train_labels[vai], np.argmax(K_va_s.dot(a), axis=1))
    if ac > best_acc: best_acc, best_lam = ac, lam

K_tr = hellinger_kernel(tr_l1, tr_l1)
K_te = hellinger_kernel(te_l1, tr_l1)
a = np.linalg.solve(K_tr + best_lam*np.eye(len(train_labels)), y_oh)
np.save('subiect4_solutia_1.npy', np.argmax(K_te.dot(a), axis=1))
```

---

### VARIANTA PREDICT C — Accelerometru: NN Adam + Markov k=8 + KNN Manhattan + KRR Intersection

```python
import numpy as np, os, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# === INCARCARE SEMNALE ===
def load_ds(d, lf=None):
    sigs, fns, labs = [], [], []
    if lf:
        with open(lf) as f: lines = f.readlines()
        for l in lines[1:]:
            l = l.strip()
            if l:
                p = l.split(',')
                sigs.append(np.loadtxt(os.path.join(d, p[0])))
                fns.append(p[0]); labs.append(int(p[1]))
        return sigs, fns, np.array(labs)
    else:
        with open('data/test.txt') as f: lines = f.readlines()
        for l in lines:
            l = l.strip()
            if l: sigs.append(np.loadtxt(os.path.join(d, l))); fns.append(l)
        return sigs, fns, None

tr_sig, tr_fn, tr_lab = load_ds('data/train', 'data/train.txt')
te_sig, te_fn, _ = load_ds('data/test')

np.random.seed(42)
ii = np.arange(len(tr_lab)); np.random.shuffle(ii)
sp = int(0.8*len(ii)); tri, vai = ii[:sp], ii[sp:]

# === EX 1: RETEA NEURONALA + ADAM ===
FL = int(np.median([len(s) for s in tr_sig]))
def norm_len(s):
    if len(s)>=FL: return s[:FL]
    return np.vstack([s, np.zeros((FL-len(s), s.shape[1]))])

tr_d = np.array([norm_len(s).flatten() for s in tr_sig])
te_d = np.array([norm_len(s).flatten() for s in te_sig])
mn, sd = tr_d.mean(0), tr_d.std(0); sd[sd==0]=1
tr_sc, te_sc = (tr_d-mn)/sd, (te_d-mn)/sd

nf, nc = tr_sc.shape[1], len(np.unique(tr_lab))
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf,256),nn.ReLU(),nn.Dropout(0.3),
                                 nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.3),
                                 nn.Linear(128,nc))
    def forward(self,x): return self.net(x)

dev = "cuda" if torch.cuda.is_available() else "cpu"
Xtr,ytr = torch.FloatTensor(tr_sc), torch.LongTensor(tr_lab)
loader = DataLoader(TensorDataset(Xtr,ytr), batch_size=32, shuffle=True)
model = Net().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
lf = nn.CrossEntropyLoss()

for ep in range(50):
    model.train()
    for bx,by in loader:
        bx,by = bx.to(dev),by.to(dev)
        loss = lf(model(bx),by); opt.zero_grad(); loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    preds = model(torch.FloatTensor(te_sc).to(dev)).argmax(1).cpu().numpy()
with open('subiect1_solutia_1.txt','w') as f:
    f.write('filename,label\n')
    for fn,p in zip(te_fn,preds): f.write(f"{fn},{p}\n")

# === EX 2: MARKOV k=8 ===
k = 8  # SCHIMBA DUPA CERINTA
all_v = np.vstack(tr_sig)
ax_r = [(all_v[:,i].min(), all_v[:,i].max()) for i in range(3)]

def markov(sig):
    disc = np.zeros_like(sig, dtype=int)
    for ax in range(3):
        bins = np.linspace(ax_r[ax][0], ax_r[ax][1], k+1)
        disc[:,ax] = np.clip(np.digitize(sig[:,ax], bins)-1, 0, k-1)
    feats = []
    for ax in range(3):
        A = np.zeros((k,k))
        for t in range(len(disc[:,ax])-1): A[disc[t,ax]][disc[t+1,ax]] += 1
        s = A.sum(1, keepdims=True); s[s==0]=1
        feats.extend((A/s).flatten())
    return np.array(feats)

tr_f = np.array([markov(s) for s in tr_sig])
te_f = np.array([markov(s) for s in te_sig])

# === EX 3: KNN MANHATTAN ===
best_k, best_acc = 3, 0
for kk in [1,3,5,7,9]:
    preds = []
    for i in vai:
        d = np.sum(np.abs(tr_f[tri] - tr_f[i]), axis=1)
        nn = np.argsort(d)[:kk]
        preds.append(np.bincount(tr_lab[tri][nn]).argmax())
    ac = accuracy_score(tr_lab[vai], preds)
    if ac > best_acc: best_acc, best_k = ac, kk

te_pred = []
for i in range(len(te_f)):
    d = np.sum(np.abs(tr_f - te_f[i]), axis=1)
    nn = np.argsort(d)[:best_k]
    te_pred.append(np.bincount(tr_lab[nn]).argmax())
with open('subiect3_solutia_1.txt','w') as f:
    f.write('filename,label\n')
    for fn,p in zip(te_fn,te_pred): f.write(f"{fn},{p}\n")

# === EX 4: KRR INTERSECTION ===
def inter_kernel(X, Y):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]): K[i] = np.minimum(X[i], Y).sum(axis=1)
    return K

y_oh = np.zeros((len(tr_lab), nc))
for i in range(len(tr_lab)): y_oh[i, tr_lab[i]] = 1

K_ts = inter_kernel(tr_f[tri], tr_f[tri])
K_vs = inter_kernel(tr_f[vai], tr_f[tri])

best_lam, best_acc = 1.0, 0
for lam in [0.001,0.01,0.1,1.0,10.0,50.0]:
    n = K_ts.shape[0]
    a = np.linalg.solve(K_ts + lam*np.eye(n), y_oh[tri])
    ac = accuracy_score(tr_lab[vai], np.argmax(K_vs.dot(a), axis=1))
    if ac > best_acc: best_acc, best_lam = ac, lam

K_tr = inter_kernel(tr_f, tr_f)
K_te = inter_kernel(te_f, tr_f)
a = np.linalg.solve(K_tr + best_lam*np.eye(len(tr_lab)), y_oh)
preds = np.argmax(K_te.dot(a), axis=1)
with open('subiect4_solutia_1.txt','w') as f:
    f.write('filename,label\n')
    for fn,p in zip(te_fn,preds): f.write(f"{fn},{p}\n")
```

---

### VARIANTA PREDICT D — Accelerometru: NN SGD + Markov k=3 + KNN Minkowski p=4 + SVM Hellinger

```python
import numpy as np, os
from sklearn import svm
from sklearn.metrics import accuracy_score
# + torch imports la fel ca Varianta C

# === INCARCARE (identic cu C) ===
# ... (copiaza load_ds din Varianta C)
tr_sig, tr_fn, tr_lab = load_ds('data/train', 'data/train.txt')
te_sig, te_fn, _ = load_ds('data/test')

np.random.seed(42)
ii = np.arange(len(tr_lab)); np.random.shuffle(ii)
sp = int(0.8*len(ii)); tri, vai = ii[:sp], ii[sp:]
nc = len(np.unique(tr_lab))

# === EX 1: NN SGD (schimba doar optimizatorul din Varianta C) ===
# opt = torch.optim.SGD(model.parameters(), lr=1e-3)  # sau lr=1e-2, momentum=0.9

# === EX 2: MARKOV k=3 ===
k = 3  # features: 3*3*3 = 27
# ... (aceleasi functii markov din Varianta C, schimba doar k)

# === EX 3: KNN MINKOWSKI p=4 ===
P = 4
best_k, best_acc = 3, 0
for kk in [1,3,5,7,9]:
    preds = []
    for i in vai:
        d = np.sum(np.abs(tr_f[tri] - tr_f[i])**P, axis=1)**(1.0/P)
        nn = np.argsort(d)[:kk]
        preds.append(np.bincount(tr_lab[tri][nn]).argmax())
    ac = accuracy_score(tr_lab[vai], preds)
    if ac > best_acc: best_acc, best_k = ac, kk

te_pred = []
for i in range(len(te_f)):
    d = np.sum(np.abs(tr_f - te_f[i])**P, axis=1)**(1.0/P)
    nn = np.argsort(d)[:best_k]
    te_pred.append(np.bincount(tr_lab[nn]).argmax())
# ... salvare

# === EX 4: SVM HELLINGER ===
def hellinger_kernel(X, Y):
    return np.sqrt(np.maximum(X, 0)).dot(np.sqrt(np.maximum(Y, 0)).T)

K_ts = hellinger_kernel(tr_f[tri], tr_f[tri])
K_vs = hellinger_kernel(tr_f[vai], tr_f[tri])

best_C, best_acc = 1.0, 0
for C in [0.1, 1.0, 10.0, 50.0, 100.0]:
    m = svm.SVC(C=C, kernel='precomputed')
    m.fit(K_ts, tr_lab[tri])
    ac = accuracy_score(tr_lab[vai], m.predict(K_vs))
    if ac > best_acc: best_acc, best_C = ac, C

K_tr = hellinger_kernel(tr_f, tr_f)
K_te = hellinger_kernel(te_f, tr_f)
m = svm.SVC(C=best_C, kernel='precomputed')
m.fit(K_tr, tr_lab)
preds = m.predict(K_te).astype(int)
# ... salvare
```

---

### VARIANTA PREDICT E — Text: SVM Linear pe BoW + Convoluție + KRR Liniar + SVM RBF precomputed

```python
import numpy as np
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

# === INCARCARE TEXT (identic) ===
# ...
train_labels = np.load('train_labels.npy').astype(int)

# === EX 1: SVM LINEAR PE CHAR BOW ===
# ... (bow_features identic)
tr_bow = normalize(bow_feat(train_sentences), norm='l2')
te_bow = normalize(bow_feat(test_sentences), norm='l2')

best_C, best_acc = 1.0, 0
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    m = svm.SVC(C=C, kernel='linear')
    m.fit(tr_bow[tri], train_labels[tri])
    ac = accuracy_score(train_labels[vai], m.predict(tr_bow[vai]))
    if ac > best_acc: best_acc, best_C = ac, C

m = svm.SVC(C=best_C, kernel='linear')
m.fit(tr_bow, train_labels)
np.save('subiect1_solutia_1.npy', m.predict(te_bow))

# === EX 2: CONVOLUTIE (identic) ===
# ...

# === EX 3: KRR LINIAR ===
def linear_kernel(X, Y): return X.dot(Y.T)

tr_n = normalize(tr_conv, norm='l2')
te_n = normalize(te_conv, norm='l2')
nc = len(np.unique(train_labels))
y_oh = np.zeros((len(train_labels), nc))
for i in range(len(train_labels)): y_oh[i, train_labels[i]] = 1

K_ts = linear_kernel(tr_n[tri], tr_n[tri])
K_vs = linear_kernel(tr_n[vai], tr_n[tri])

best_lam, best_acc = 1.0, 0
for lam in [0.001, 0.01, 0.1, 1.0, 10.0]:
    n = K_ts.shape[0]
    a = np.linalg.solve(K_ts + lam*np.eye(n), y_oh[tri])
    ac = accuracy_score(train_labels[vai], np.argmax(K_vs.dot(a), axis=1))
    if ac > best_acc: best_acc, best_lam = ac, lam

K_tr = linear_kernel(tr_n, tr_n)
K_te = linear_kernel(te_n, tr_n)
a = np.linalg.solve(K_tr + best_lam*np.eye(len(train_labels)), y_oh)
np.save('subiect3_solutia_1.npy', np.argmax(K_te.dot(a), axis=1))

# === EX 4: SVM CU KERNEL RBF PRECOMPUTED ===
# Uneori se cere RBF dar precomputed! Calculam noi matricea.
from scipy.spatial.distance import cdist

def rbf_kernel(X, Y, gamma=1.0):
    """K(x,y) = exp(-gamma * ||x-y||^2)"""
    dists_sq = cdist(X, Y, 'sqeuclidean')  # distante patratice
    return np.exp(-gamma * dists_sq)

# SAU fara scipy:
def rbf_kernel_manual(X, Y, gamma=1.0):
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y
    X_sq = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_sq = np.sum(Y**2, axis=1).reshape(1, -1)
    dists_sq = X_sq + Y_sq - 2 * X.dot(Y.T)
    return np.exp(-gamma * np.maximum(dists_sq, 0))

best_C, best_g, best_acc = 1.0, 0.1, 0
for C in [1.0, 10.0, 50.0]:
    for g in [0.001, 0.01, 0.1, 1.0]:
        K_ts = rbf_kernel_manual(tr_n[tri], tr_n[tri], g)
        K_vs = rbf_kernel_manual(tr_n[vai], tr_n[tri], g)
        m = svm.SVC(C=C, kernel='precomputed')
        m.fit(K_ts, train_labels[tri])
        ac = accuracy_score(train_labels[vai], m.predict(K_vs))
        if ac > best_acc: best_acc, best_C, best_g = ac, C, g

K_tr = rbf_kernel_manual(tr_n, tr_n, best_g)
K_te = rbf_kernel_manual(te_n, tr_n, best_g)
m = svm.SVC(C=best_C, kernel='precomputed')
m.fit(K_tr, train_labels)
np.save('subiect4_solutia_1.npy', m.predict(K_te))
```

---

### VARIANTA PREDICT F — Text: Lasso + String Kernel + KNN similarity + SVM precomputed

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn import svm
from sklearn.metrics import accuracy_score

# === INCARCARE ===
train_data = np.load('train_data.npy', allow_pickle=True)
train_labels = np.load('train_labels.npy').astype(int)
test_data = np.load('test_data.npy', allow_pickle=True)

np.random.seed(42)
ii = np.arange(len(train_labels)); np.random.shuffle(ii)
sp = int(0.8*len(ii)); tri, vai = ii[:sp], ii[sp:]
nc = len(np.unique(train_labels))

# === EX 1: LASSO PE CHAR BOW ===
vocab = {}
idx = 0
for s in train_data:
    for c in str(s):
        if c not in vocab: vocab[c] = idx; idx += 1

def bow_feat(data):
    F = np.zeros((len(data), len(vocab)))
    for i, s in enumerate(data):
        for c in str(s):
            if c in vocab: F[i][vocab[c]] += 1
    return F

tr_bow = bow_feat(train_data)
te_bow = bow_feat(test_data)

# Lasso e regresie -> one-hot + argmax
y_oh = np.zeros((len(train_labels), nc))
for i in range(len(train_labels)): y_oh[i, train_labels[i]] = 1

# Lasso per clasa (MultiOutputRegressor style)
from sklearn.linear_model import Lasso
best_al, best_acc = 1.0, 0
for al in [0.0001, 0.001, 0.01, 0.1, 1.0]:
    preds_all = np.zeros((len(vai), nc))
    for c in range(nc):
        m = Lasso(alpha=al, max_iter=5000)
        m.fit(tr_bow[tri], y_oh[tri, c])
        preds_all[:, c] = m.predict(tr_bow[vai])
    ac = accuracy_score(train_labels[vai], np.argmax(preds_all, axis=1))
    if ac > best_acc: best_acc, best_al = ac, al

preds_all = np.zeros((len(te_bow), nc))
for c in range(nc):
    m = Lasso(alpha=best_al, max_iter=5000)
    m.fit(tr_bow, y_oh[:, c])
    preds_all[:, c] = m.predict(te_bow)
np.save('subiect1_solutia_1.npy', np.argmax(preds_all, axis=1))

# === EX 2+3: STRING KERNEL + KNN ===
def get_ngrams(text, p=8):
    return {str(text)[i:i+p] for i in range(len(str(text)) - p + 1)}

# Matrice kernel simetrica
all_ng_tr = [get_ngrams(t) for t in train_data]
all_ng_te = [get_ngrams(t) for t in test_data]

K_train = np.zeros((len(train_data), len(train_data)))
for i in range(len(train_data)):
    for j in range(i, len(train_data)):
        K_train[i][j] = len(all_ng_tr[i] & all_ng_tr[j])
        K_train[j][i] = K_train[i][j]

K_test = np.zeros((len(test_data), len(train_data)))
for i in range(len(test_data)):
    for j in range(len(train_data)):
        K_test[i][j] = len(all_ng_te[i] & all_ng_tr[j])

# KNN cu similaritate (SORTAM DESCRESCATOR!)
best_k, best_acc = 3, 0
for kk in [1,3,5,7]:
    preds = []
    for i in vai:
        sims = K_train[i, tri]
        nn = np.argsort(-sims)[:kk]
        preds.append(np.bincount(train_labels[tri][nn]).argmax())
    ac = accuracy_score(train_labels[vai], preds)
    if ac > best_acc: best_acc, best_k = ac, kk

te_pred = []
for i in range(len(test_data)):
    nn = np.argsort(-K_test[i])[:best_k]
    te_pred.append(np.bincount(train_labels[nn]).argmax())
np.save('subiect3_solutia_1.npy', np.array(te_pred))

# === EX 4: KRR PRECOMPUTED (string kernel) ===
K_ts = K_train[np.ix_(tri, tri)]
K_vs = K_train[np.ix_(vai, tri)]

best_lam, best_acc = 1.0, 0
for lam in [0.001, 0.01, 0.1, 1.0, 10.0]:
    n = K_ts.shape[0]
    a = np.linalg.solve(K_ts + lam*np.eye(n), y_oh[tri])
    ac = accuracy_score(train_labels[vai], np.argmax(K_vs.dot(a), axis=1))
    if ac > best_acc: best_acc, best_lam = ac, lam

a = np.linalg.solve(K_train + best_lam*np.eye(len(train_labels)), y_oh)
np.save('subiect4_solutia_1.npy', np.argmax(K_test.dot(a), axis=1))
```

---

### CHEAT SHEET — CE SCHIMBI RAPID DACA CERINTA DIFERA

| Dacă cerința spune... | Schimbă doar... |
|----------------------|----------------|
| k=3/4/5/6/7/8 (Markov) | variabila `k = X` |
| Minkowski p=3/4/5 | variabila `P = X` și formula `**P ... **(1.0/P)` |
| Manhattan | `np.sum(np.abs(...), axis=1)` (e Minkowski p=1) |
| Euclidiana | `np.sqrt(np.sum((...)**2, axis=1))` (e Minkowski p=2) |
| Adam vs SGD | linia cu `torch.optim.Adam(...)` vs `torch.optim.SGD(...)` |
| lr=1e-3 vs 1e-2 | parametrul `lr=` din optimizer |
| SVM vs KRR | SVM: `svm.SVC(kernel='precomputed')` / KRR: `np.linalg.solve(K+λI, y_oh)` |
| Hellinger vs Intersection | funcția kernel: `sqrt(X)@sqrt(Y).T` vs `minimum(X[i],Y).sum` |
| n-grams p=5/6/7/8 | variabila `p = X` in `get_ngrams` |
| threshold t=0.8/0.9/0.95 | variabila `threshold = X` in `conv_features` |
| .npy vs .txt output | `np.save(...)` vs `open(...).write(...)` |
