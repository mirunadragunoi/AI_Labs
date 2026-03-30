import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# ALGORITMUL COBORARII PE GRADIENT

def sigmoid(x):
    """sigmoid --->> transforma orice nr intr o valoare intre 0 si 1"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivatie(x):
    """derivata sigmoid -->>> sigmoid(x) * (1 - sigmoid(x))"""
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    """tanh --->> transforma orice nr intr o valoare intre -1 si 1"""
    return np.tanh(x)


def tanh_derivatie(x):
    """derivata tanh -->> 1 - tanh(x) ** 2"""
    return 1 - np.tanh(x) ** 2


# pas 1 -->> initializarea ponderilor ->> ponderile si bias ul retelei se initializeaza aleator cu valori mici
# aproape de 0 sau cu valoare 0

def initializare_ponderi(nr_intrari, num_hidden_neurons, miu=0, sigma=0):
    """ initializizeaza ponderilor retelei
        nr_intrari -->> numarul de intrari (2 pt XOR)
        num_hidden_neurons -->> numarul de neuroni din stratul ascuns
        miu, sigma --->> media si deviatia standard pt generarea aleatoare
    """
    W1 = np.random.normal(miu, sigma, (nr_intrari, num_hidden_neurons))
    b1 = np.zeros(num_hidden_neurons)
    W2 = np.random.normal(miu, sigma, (num_hidden_neurons, 1))
    b2 = np.zeros(1)
    return W1, b1, W2, b2


# pas 2 --->> pasul forward -->> calculeaza predictia retelei folosind ponderile actuale si datele de intrare ca
# parametrii

def forward(X, W1, b1, W2, b2):
    """ pasul forward -->> calculeaza predictia retelei folosind ponderile actuale si datele de intrare ca
        parametrii
        X -->> datele de intrare (n_samples * nr_intrari)
        W1, b1, W2, b2 -->> ponderile si bias ul retelei
    """
    # strat ascuns
    z1 = X.dot(W1) + b1
    a1 = tanh(z1)

    # strat iesire
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)  # aplic sigmoid --->> predictia finala

    return z1, a1, z2, a2


# pas 3 --->>> calcul valoarea functiei de eroare si acuratetea
def eroare(y_true, y_pred):
    """ calcul logistic loss """
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    pierdere = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return pierdere.mean()


def acuratete(y_true, y_pred):
    """ calcul acuratetea -->> cate predictii sunt corecte """
    predictii = np.round(y_pred)  # rotunjire la 0 sau 1
    return (predictii == y_true).mean()


# pas 4 --->> pasul backward --->> calculeaza derivata functiei de roare pe directiile ponderilor, respectiv
# a fiecarui bias

def backward(X, y, a1, a2, z1, W2, num_samples):
    """ calculeaza gradientii --->> derivatele erorii pe directia fiecarei ponderi """

    # stratul de iesire
    dz2 = a2 - y  # derivata erorii fata de z2
    dW2 = a1.T.dot(dz2) / num_samples  # derivata fata de W2
    db2 = np.sum(dz2, axis=0) / num_samples  # derivata fata de b2

    # stratul ascuns -->>> propagam eroarea inapoi prin W2
    da1 = dz2.dot(W2.T)  # cat contribuie fiecare neuron ascuns la eroare
    dz1 = da1 * tanh_derivatie(z1)  # ferivata lui tanh
    dW1 = X.T.dot(dz1) / num_samples  # derivata fata de W1
    db1 = np.sum(dz1, axis=0) / num_samples  # derivata fata de b1

    return dW1, db1, dW2, db2


# pas 5 --->> actualizarea ponderilor -->> ponderile se actualizeaza proportional cu negativul mediei derivatelor din
# batch

def update_ponderi(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    """ actualizeaza ponderile retelei folosind gradientii calculati in pasul backward """
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2


# pas 6 --->>> antrenam

def antrenare(X, y, num_hidden_neurons=5, epoci=70, lr=0.5, miu=0, sigma=1, ok=True):
    """
        antrenam reteaua neuronala
        pt fiecare epoca:
        1. amestecam datele
        2. forward --->> calculam predictia
        3. calculam eroarea
        4. backward -->>> calculam gradientii
        5. actualizam ponderile
    """

    num_samples, nr_intrari = X.shape
    y = y.reshape(-1, 1)

    # pas 1 -->> initializare ponderi
    W1, b1, W2, b2 = initializare_ponderi(nr_intrari, num_hidden_neurons, miu, sigma)

    pierderi = []
    acurateti = []

    for epoca in range(epoci):
        # amestecam datele
        X_amestecat, y_amestecat = shuffle(X, y, random_state=epoca)

        # pas 2 -->> forward
        z1, a1, z2, a2 = forward(X_amestecat, W1, b1, W2, b2)

        # pas 3 -->> calculam eroarea si acuratetea
        pierdere = eroare(y_amestecat, a2)
        acuratetea = acuratete(y_amestecat, a2)
        pierderi.append(pierdere)
        acurateti.append(acuratetea)

        # pas 4 -->> backward
        dW1, db1, dW2, db2 = backward(X_amestecat, y_amestecat, a1, a2, z1, W2, num_samples)

        # pas 5 -->> actualizam ponderile
        W1, b1, W2, b2 = update_ponderi(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

        if ok and (epoca + 1) % 10 == 0:
            print(f"epoca {epoca + 1}/{epoci} -->> pierdere: {pierdere:.4f} - acuratete: {acuratetea:.4f}")

    return W1, b1, W2, b2, pierderi, acurateti
