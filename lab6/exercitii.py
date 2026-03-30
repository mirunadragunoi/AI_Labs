import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


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


def compute_y(x, W, bias):
    # dreapta de decizie
    # [x, y] * [W[0], W[1]] + b = 0
    return (-x * W[0] - bias) / (W[1] + 1e-10)


def plot_decision_boundary(X, y, W, b, current_x, current_y):
    x1 = -0.5
    y1 = compute_y(x1, W, b)
    x2 = 0.5
    y2 = compute_y(x2, W, b)
    # sterge continutul ferestrei
    plt.clf()
    # ploteaza multimea de antrenare
    color = 'r'
    if (current_y == -1):
        color = 'b'
    plt.ylim((-1, 2))
    plt.xlim((-1, 2))
    plt.plot(X[y == -1, 0], X[y == -1, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
    # ploteaza exemplul curent
    plt.plot(current_x[0], current_x[1], color + 's')
    # afisarea dreptei de decizie
    plt.plot([x1, x2], [y1, y2], 'black')
    plt.show(block=False)
    plt.pause(0.3)


# EX1 -->> multime de antrenare X -->> dreapta care separa perfect multimea de antrenare
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, 1, 1, 1])

W_manual = np.array([1.0, 1.0])
b_manual = -0.5

print("ponderile gasite manual --->> W = ", W_manual, ", b =", b_manual)
for i in range(len(X)):
    val = X[i].dot(W_manual) + b_manual
    pred = 1 if val > 0 else -1
    print(f"  X={X[i]} -> {val:.1f} -->> predicția={pred}, real={y[i]} {'DA' if pred == y[i] else 'NU'}")

# EX2 -->> antrenare perception cu alg Windrow Hoff pe multimea de antrenare de la 1 timp de 70 de epoci cu
# rata de invatare 0.1
# acuratetea pe mult de antrenare??

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([-1, 1, 1, 1], dtype=float)

# onitializare ponderi cu 0
W = np.zeros(2)
b = 0.0
lr = 0.1
epoci = 70

plt.figure(figsize=(6, 6))

for epoca in range(epoci):
    # amestecam datele
    X_s, y_s = shuffle(X, y, random_state=epoca)

    for t in range(len(X_s)):
        x_t = X_s[t]  # exemplul curent
        y_t = y_s[t]  # eticheta curenta

        # pas 1 --->> predictie
        y_hat = x_t.dot(W) + b

        # pas 2 --->> eroare
        loss = (y_hat - y_t) ** 2 / 2

        # pas 3 --->> actualizam ponderile
        # W = W - lr * (y_hat - y) * x
        W = W - lr * (y_hat - y_t) * x_t

        # pas 4 --->> actualizam bias-ul
        # b = b - lr * (y_hat - y)
        b = b - lr * (y_hat - y_t)

        # afisam dreapta de decizie
        plot_decision_boundary(X, y, W, b, x_t, y_t)

predictii = np.sign(X.dot(W) + b)
acuratete = np.mean(predictii == y)
print(f"ponderi finale -->> W = {W}, b = {b:.4f}")
print(f"acuratete pe multimea de antrenare --->> {acuratete * 100:.1f}%")

plt.close()

# EX3 -->> Antrenati un Perceptron cu algoritmul Widrow-Hoff pe multimea de antrenare
# X =[ [0, 0], [0, 1], [1, 0], [1, 1] ], y = [-1, 1, 1, -1]. Care este acuratetea pe
# multimea de antrenare? Apelati functia plot_decision_boundary la fiecare pas al
# algoritmului pentru a afisa dreapta de decizie.antrenare

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([-1, 1, 1, -1], dtype=float)

W_xor = np.zeros(2)
b_xor = 0.0
lr = 0.1
epoci = 70

plt.figure(figsize=(6, 6))

for epoca in range(epoci):
    X_s, y_s = shuffle(X_xor, y_xor, random_state=epoca)

    for t in range(len(X_s)):
        x_t = X_s[t]
        y_t = y_s[t]

        y_hat = x_t.dot(W_xor) + b_xor
        W_xor = W_xor - lr * (y_hat - y_t) * x_t
        b_xor = b_xor - lr * (y_hat - y_t)

        plot_decision_boundary(X_xor, y_xor, W_xor, b_xor, x_t, y_t)

predictii_xor = np.sign(X_xor.dot(W_xor) + b_xor)
acuratete_xor = np.mean(predictii_xor == y_xor)
print(f"ponderi finale --->> W = {W_xor}, b = {b_xor:.4f}")
print(f"acuratete pe XOR --->> {acuratete_xor * 100:.1f}%")

plt.close()

## EX4 --->> Antrenati o retea neuronala pentru rezolvarea problemei XOR cu arhitectura
# retelei descrise in 3, si algoritmul coborarii pe gradient descris in 4, folosind
# 70 epoci, rata de invatare 0.5, media si deviatia standard pentru initializarea
# ponderilor 0, respectiv 1, si 5 neuroni pe stratul ascuns. Afisati valoarea
# erorii si a acuratetii la fiecare epoca. Apelati functia plot_decision la fiecare pas
# al algoritmului pentru a afisa functia de decizie.

def compute_y(x, W, bias):
    # dreapta de decizie
    # [x, y] * [W[0], W[1]] + b = 0
    return (-x*W[0] - bias) / (W[1] + 1e-10)

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

def plot_decision(X_, W_1, W_2, b_1, b_2):
    # sterge continutul ferestrei
    plt.clf()
    # ploteaza multimea de antrenare
    plt.ylim((-0.5, 1.5))
    plt.xlim((-0.5, 1.5))
    xx = np.random.normal(0, 1, (100000))
    yy = np.random.normal(0, 1, (100000))
    X = np.array([xx, yy]).transpose()
    X = np.concatenate((X, X_))
    _, _, _, output = forward(X, W_1, b_1, W_2, b_2)
    y = np.squeeze(np.round(output))
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
    plt.show(block=False)
    plt.pause(0.1)


X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor_nn = np.array([[0], [1], [1], [0]], dtype=float)  # 0 si 1 pentru sigmoid

num_hidden = 5
epoci = 70
lr = 0.5
miu = 0
sigma = 1

# initializare
np.random.seed(42)
W1 = np.random.normal(miu, sigma, (2, num_hidden))
b1 = np.zeros(num_hidden)
W2 = np.random.normal(miu, sigma, (num_hidden, 1))
b2 = np.zeros(1)

num_samples = len(X_xor)
plt.figure(figsize=(6, 6))

for epoca in range(epoci):
    # amestecam
    X_s, y_s = shuffle(X_xor, y_xor_nn, random_state=epoca)

    # forward
    z1, a1, z2, a2 = forward(X_s, W1, b1, W2, b2)

    # eroare
    a2_clipped = np.clip(a2, 1e-10, 1 - 1e-10)
    loss = -(y_s * np.log(a2_clipped) + (1 - y_s) * np.log(1 - a2_clipped)).mean()
    acc = (np.round(a2) == y_s).mean()

    if (epoca + 1) % 10 == 0:
        print(f"epoca {epoca + 1:3d} -->> pierdere: {loss:.4f}, acuratete: {acc * 100:.1f}%")

    # backward
    dW1, db1, dW2, db2 = backward(X_s, y_s, a1, a2, z1, W2, num_samples)

    # actualizare ponderi
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    # afisam functia de decizie
    plot_decision(X_xor, W1, W2, b1, b2)

# predictie finala
_, _, _, output_final = forward(X_xor, W1, b1, W2, b2)
print(f"\npredictii finale --->>> ")
for i in range(len(X_xor)):
    print(f"  X = {X_xor[i]} -->> y_hat = {output_final[i][0]:.4f} -->> "
          f"rotunjit={int(np.round(output_final[i][0]))}, real={int(y_xor_nn[i][0])}")

plt.close()