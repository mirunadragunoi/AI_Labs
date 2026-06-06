import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# =============================================================================
# NUMPY
# =============================================================================

# --- creare / forma ---
a = np.zeros((3, 4))          # matrice de zerouri, shape (3,4)
a = np.ones((3, 4))           # matrice de 1
a = np.array([[1, 2], [3, 4]])
a = np.linspace(0, 1, num=5)  # [0, 0.25, 0.5, 0.75, 1.0] - num puncte uniform distribuite
a = np.arange(0, 10, 2)       # [0, 2, 4, 6, 8] - pas fix
a = np.eye(3)                 # matrice identitate 3x3

# --- shape / reshape ---
a.shape          # (3, 4)
a.reshape(2, 6)  # nu copiaza, schimba doar "vederea"
a.flatten()      # -> array 1D (copie)
a.reshape(-1)    # -> array 1D (acelasi lucru, fara copie)

# --- indexare ---
a[0]         # primul rand
a[:, 1]      # a doua coloana
a[1:3, :]    # randurile 1 si 2
mask = a > 0
a[mask]      # elementele pozitive

# --- operatii pe axe ---
np.sum(a, axis=0)          # suma pe coloane -> shape (4,)
np.sum(a, axis=1)          # suma pe randuri  -> shape (3,)
np.sum(a, axis=1, keepdims=True)  # -> shape (3,1), util pentru broadcasting
np.mean(a, axis=0)
np.min(a); np.max(a)
np.abs(a)
np.sqrt(a)
np.minimum(a, b)           # element-wise min intre doi array
np.maximum(a, b)           # element-wise max

# --- algebra liniara ---
np.linalg.norm(a, ord=2, axis=1, keepdims=True)  # norma L2 per rand
np.linalg.norm(a, ord=1, axis=1, keepdims=True)  # norma L1 per rand
a @ b          # inmultire matriceala (echivalent np.dot)
np.dot(a, b)

# --- sortare / argsort ---
np.argsort(a)         # indici care sorteaza crescator
np.argsort(a)[:K]     # primii K indici (cei mai mici K)

# --- unic / numarare ---
np.unique(a)                         # valori unice
np.unique(a, return_counts=True)     # (valori_unice, conturi)

# --- digitize - folosit pt discretizare (Markov) ---
bins = np.linspace(vmin, vmax + 1e-6, num=k + 1)  # k+1 limite -> k intervale
idx = np.digitize(valoare, bins) - 1               # returneaza 1..k, scadem 1 -> 0..k-1
# IMPORTANT: adaugam 1e-6 la vmax ca sa includem valoarea maxima in ultimul interval

# --- load / save ---
np.load("file.npy", allow_pickle=True)
np.save("file.npy", array)


# =============================================================================
# SKLEARN — MODELE
# =============================================================================

# --- SVM ---
# kernel="precomputed": primeste matrice kernel in loc de date brute
# C: parametru de regularizare (mai mare = mai putin regularizat, mai flexibil)
model = SVC(C=3, kernel="precomputed")
model.fit(K_train, y_train)    # K_train: (N_train, N_train)
preds = model.predict(K_test)  # K_test:  (N_test, N_train)  <- nu (N_test, N_test) !

# SVM cu kernel standard (rbf, linear, poly, sigmoid):
model = SVC(C=1.0, kernel="rbf", gamma="scale")
model.fit(X_train, y_train)
preds = model.predict(X_test)

# --- Kernel Ridge Regression (KRR) ---
# Produce valori continue -> trebuie rotunjit la clasa cea mai apropiata
model = KernelRidge(alpha=1.0, kernel="precomputed")
model.fit(K_train, y_train)
raw = model.predict(K_test)                                  # valori continue
valid_labels = np.unique(y_train)
preds = np.array([valid_labels[np.argmin(np.abs(valid_labels - p))] for p in raw])

# --- Naive Bayes ---
# MultinomialNB: pentru date de numarat (frecvente), NECESITA valori >= 0
# GaussianNB: pentru date continue (orice valori)
model = MultinomialNB()
model.fit(X_train, y_train)   # X_train: (N, n_features), toate valorile >= 0
preds = model.predict(X_test)

model = GaussianNB()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# --- accuracy ---
accuracy_score(y_true, y_pred)   # returneaza float intre 0 si 1

# --- train/test split (pt validare hiperparametri) ---
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# =============================================================================
# SKLEARN — KERNEL FUNCTIONS (de construit manual pt kernel="precomputed")
# =============================================================================
# K_train = kernel(train, train)  -> (N_train, N_train)
# K_test  = kernel(test,  train)  -> (N_test,  N_train)  <- INTOTDEAUNA fata de train!

def kernel_hellinger(A, B):
    # K[i,j] = sum(sqrt(A[i] * B[j]))  — pt date nonnegative (probabilitati, frecvente)
    K = np.zeros((A.shape[0], B.shape[0]))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            K[i, j] = np.sum(np.sqrt(a * b))
    return K


def kernel_intersection(A, B):
    # K[i,j] = sum(min(A[i], B[j]))  — pt histograme / frecvente
    K = np.zeros((A.shape[0], B.shape[0]))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            K[i, j] = np.sum(np.minimum(a, b))
    return K


def kernel_linear(A, B):
    # K[i,j] = A[i] . B[j]  — produs scalar
    return A @ B.T


# =============================================================================
# PYTORCH — RETEA NEURONALA FEED-FORWARD
# =============================================================================

# --- Definire model ---
class Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, 512)   # strat liniar: (in, out)
        self.drop = nn.Dropout(0.3)             # anuleaza aleator 30% neuroni (doar la train)
        self.fc2  = nn.Linear(512, 64)
        self.out  = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))   # activare ReLU: max(0, x)
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        return self.out(x)        # fara activare la final (CrossEntropyLoss o include intern)

# --- Conversie date numpy -> tensori pytorch ---
X_train = torch.tensor(X_np, dtype=torch.float32)  # date float
y_train = torch.tensor(y_np, dtype=torch.long)      # etichete intregi (pentru CrossEntropy)

# --- Optimizatori ---
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=1e-4)   # fara momentum

# --- Loss ---
criterion = nn.CrossEntropyLoss()   # clasifica multi-clasa, include softmax intern
# primeste: outputs (N, num_classes), targets (N,) intregi

# --- Loop de antrenare ---
model = Net(input_dim, num_classes)
for epoch in range(500):
    model.train()              # activeaza Dropout, BatchNorm etc.
    optimizer.zero_grad()      # reseteaza gradientii acumulati
    outputs = model(X_train)   # forward pass
    loss = criterion(outputs, y_train)
    loss.backward()            # calculeaza gradientii
    optimizer.step()           # actualizeaza ponderile

# --- Predictii ---
model.eval()                       # dezactiveaza Dropout
with torch.no_grad():              # nu calculeaza gradiente (economiseste memorie)
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)  # argmax pe dim 1 = clasa prezisa
    # predicted: tensor (N,)

# torch.max(tensor, dim) returneaza (valori_maxime, indici_maximi)
# _ ignora valorile, pastram doar indicii (clasele)

# --- Acuratete in pytorch ---
correct = (predicted == y_test).sum().item()
accuracy = correct / len(y_test)


# =============================================================================
# MARKOV — MATRICE DE TRANZITIE
# =============================================================================
# Folosita pt date secventiale (semnale). Captureaza probabilitatile de tranzitie
# intre intervale consecutive de valori.
#
# Pipeline:
#   semnal (float[]) -> discretizat (int[]) -> matrice A (k,k) -> normalizata -> flatten
#
# Legatura cu kernelele: vectorii Markov sunt nonneg -> compatibili cu Hellinger, Intersection

def markovize(signal_1d, vmin, vmax, k):
    """
    signal_1d: lista/array de valori float pentru O axa
    vmin, vmax: min/max GLOBAL (calculat pe tot train+test, nu doar pe un sample)
    k: numarul de intervale
    -> returneaza vector (k^2,) de probabilitati de tranzitie

    De ce vmin/vmax global? Ca sa avem aceleasi intervale la train si test.
    """
    bins = np.linspace(vmin, vmax + 1e-6, num=k + 1)
    disc = [min(np.digitize(v, bins) - 1, k - 1) for v in signal_1d]

    A = np.zeros((k, k))
    for t in range(len(disc) - 1):
        A[disc[t]][disc[t + 1]] += 1

    row_sums = A.sum(axis=1, keepdims=True)
    B = np.zeros_like(A)
    nonzero = row_sums[:, 0] != 0
    B[nonzero] = A[nonzero] / row_sums[nonzero]

    return B.flatten()  # (k^2,)

# Daca semnalul are mai multe axe (ex: x,y,z), apeleaza markovize() pt fiecare
# si concateneaza: np.concatenate([vec_x, vec_y, vec_z]) -> (3*k^2,)


# =============================================================================
# BAG OF WORDS
# =============================================================================
# Transforma un document (sir de tokeni) intr-un vector de frecvente.
# Token = caracter sau cuvant, depinde de cum iterezi documentul.
#
# Legatura cu modelele:
#   - vectorii BoW -> MultinomialNB (direct, valori nonneg)
#   - vectorii BoW normalizati -> SVM, KRR (dupa normalizare L1/L2)

class BagOfWords:
    def __init__(self):
        self.vocab = {}         # {token: index_coloana}

    def fit(self, documents):
        """Construieste vocabularul din documentele de ANTRENARE (nu aplica pe test)."""
        for doc in documents:
            for token in doc:   # doc e iterat direct: caracter cu caracter sau cuvant cu cuvant
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def transform(self, documents):
        """Transforma documentele in matrice (N, |vocab|) de frecvente."""
        X = np.zeros((len(documents), len(self.vocab)), dtype=np.float32)
        for i, doc in enumerate(documents):
            for token in doc:
                if token in self.vocab:
                    X[i][self.vocab[token]] += 1
        return X

# Normalizare BoW (necesar pt MultinomialNB daca se normalizeaza, si pt alte modele)
def l1_normalize(X):
    norms = np.sum(np.abs(X), axis=1, keepdims=True)
    return X / (norms + 1e-9)

def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-9)


# =============================================================================
# CONVOLUTIE 1D PE TEXT (cu mapping caracter->numar)
# =============================================================================
# Aplica un filtru (n-gram) pe un document prin sliding window.
# La fiecare pozitie calculeaza cosine similarity locala intre fereastra si filtru.
# Numara cate pozitii depasesc un prag (ex: 0.9).
#
# Legatura: vectorii de convolutie -> SVM/KRR cu kernel Hellinger sau liniar

def char_convolution(document, gram, mapping, threshold=0.9):
    """
    document, gram: string-uri
    mapping: dict {char: float}  — transforma fiecare caracter in numar
    threshold: prag pt numarare
    -> int: numar de ferestre cu similaritate > threshold
    """
    n = len(gram)
    norm_gram = sum(mapping.get(c, 0) ** 2 for c in gram) ** 0.5

    count = 0
    for i in range(len(document) - n + 1):  # L - n + 1 pozitii posibile
        window   = document[i:i + n]
        norm_win = sum(mapping.get(c, 0) ** 2 for c in window) ** 0.5
        dot      = sum(mapping.get(window[j], 0) * mapping.get(gram[j], 0) for j in range(n))
        if dot / (norm_gram * norm_win + 1e-6) > threshold:
            count += 1
    return count

# pentru un document si lista de filtre:
# vec = [char_convolution(doc, gram, mapping) for gram in grams]  -> lista de 500 int-uri


# =============================================================================
# UTILITARE GENERALE
# =============================================================================

# --- padding secvente la lungime egala ---
def pad_to_max(sequences, pad_value=0.0):
    """sequences: lista de liste (sau liste de liste de liste pt multivariate)"""
    max_len = max(len(s) for s in sequences)
    return [s + [pad_value] * (max_len - len(s)) for s in sequences]

# --- salvare predictii ---
def save_txt(predictions, path):
    with open(path, "w") as f:
        for p in predictions:
            f.write(f"{int(p)}\n")

def save_npy(predictions, path):
    np.save(path, np.array(predictions))
