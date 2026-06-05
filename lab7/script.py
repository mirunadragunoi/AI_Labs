import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from collections import Counter

# ============================================================
# INCARCARE DATE MNIST (acelasi subset ca la laboratoarele anterioare)
# ============================================================
train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt').astype(int)
test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt').astype(int)

# Reshape: din vectori de 784 in imagini 28x28
train_imgs = train_images.reshape(-1, 28, 28)
test_imgs = test_images.reshape(-1, 28, 28)

print(f"Train: {train_imgs.shape}, Test: {test_imgs.shape}")


# ================================================================
# EXERCITIUL 1 - Local Binary Pattern (LBP)
# ================================================================
# IDEEA:
#   Pentru fiecare pixel, te uiti la vecinii lui (intr-un patrat d x d).
#   Compari: daca vecinul >= pixel central -> 1, altfel -> 0.
#   Asta iti da o matrice binara d x d pentru fiecare pixel.
#   O liniarizezi (faci un vector din ea).
#   La final, numeri de cate ori apare fiecare pattern unic -> histograma.
#   Histograma = feature-ul imaginii, pe care il dai la un clasificator.
#
# EXEMPLU vizual (d=3, pixel central = 5):
#   Vecini:  [3, 7, 2]     Binar:  [0, 1, 0]
#            [8, 5, 1]  ->         [1, -, 0]    (5 e centrul, nu il comparam)
#            [6, 4, 9]             [1, 0, 1]
#   Vector liniarizat (fara centru): [0, 1, 0, 1, 0, 1, 0, 1]
#   Sau ca string: "01010101"

def compute_lbp(image, d=3):
    """
    Calculeaza Local Binary Pattern pentru o imagine.

    image: matrice 2D (28x28)
    d: dimensiunea vecinatatii (d x d)

    Returneaza: lista de pattern-uri (string-uri binare), cate unul per pixel valid
    """
    h, w = image.shape
    half = d // 2  # cat ne extindem in fiecare directie de la centru
    # Exemplu: d=3 -> half=1, deci vedem pixelii de la -1 la +1

    patterns = []

    # Parcurgem doar pixelii care au vecini completi (nu iesim din imagine)
    for i in range(half, h - half):
        for j in range(half, w - half):
            centru = image[i, j]

            # Extragem vecinatatea d x d
            vecini = image[i - half: i + half + 1, j - half: j + half + 1]

            # Comparam fiecare vecin cu centrul -> binar
            binar = (vecini >= centru).astype(int)

            # Liniarizam matricea binara intr-un vector
            vector = binar.flatten()

            # Scoatem elementul central (pozitia half*d + half in vectorul liniarizat)
            centru_idx = half * d + half
            vector = np.delete(vector, centru_idx)

            # Convertim in string ca sa putem numara pattern-uri unice
            pattern = ''.join(vector.astype(str))
            patterns.append(pattern)

    return patterns


def lbp_histogram(image, d=3):
    """
    Calculeaza histograma LBP pentru o imagine.

    Histograma = de cate ori apare fiecare pattern unic.
    Toate imaginile trebuie sa aiba aceleasi bin-uri (pattern-uri posibile),
    asa ca returnam un vector de frecvente.

    d=3 -> 8 biti (d*d - 1) -> 2^8 = 256 pattern-uri posibile
    """
    patterns = compute_lbp(image, d)

    # Numarul de pattern-uri posibile: d*d - 1 biti -> 2^(d*d-1) combinatii
    num_bits = d * d - 1
    num_patterns = 2 ** num_bits

    # Cream histograma: pentru fiecare pattern posibil, numaram aparitiile
    histogram = np.zeros(num_patterns)

    for pattern in patterns:
        # Convertim string-ul binar in numar intreg (index in histograma)
        idx = int(pattern, 2)
        histogram[idx] += 1

    # Normalizam histograma (ca suma sa fie 1)
    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()

    return histogram


# Calculam histogramele LBP pentru toate imaginile
print("\n" + "=" * 60)
print("EX 1 - Local Binary Pattern")
print("=" * 60)

d = 3  # dimensiunea vecinatatii
print(f"Calculam LBP cu d={d}...")

train_lbp = np.array([lbp_histogram(img, d) for img in train_imgs])
test_lbp = np.array([lbp_histogram(img, d) for img in test_imgs])

print(f"Train LBP shape: {train_lbp.shape}")  # (1000, 256) pentru d=3
print(f"Test LBP shape: {test_lbp.shape}")

# Antrenam un clasificator (SVM cu kernel rbf)
model_lbp = svm.SVC(C=10, kernel='rbf', gamma='scale')
model_lbp.fit(train_lbp, train_labels)
pred_lbp = model_lbp.predict(test_lbp)
acc_lbp = accuracy_score(test_labels, pred_lbp)
print(f"Acuratete LBP + SVM(rbf): {acc_lbp * 100:.2f}%")


# ================================================================
# EXERCITIUL 2 - Magnitudinea gradientului + regiuni top-k
# ================================================================
# IDEEA:
#   Gradientul masoara cat de repede se schimba intensitatea pixelilor.
#   Gx = schimbarea pe orizontala (stanga-dreapta)
#   Gy = schimbarea pe verticala (sus-jos)
#   G = sqrt(Gx^2 + Gy^2) = magnitudinea totala a schimbarii
#
#   Intuitie: zonele cu gradient mare = contururi/margini ale cifrei
#             zonele cu gradient mic = fundal uniform
#
#   Apoi impartim imaginea in regiuni 3x3 care NU se suprapun.
#   Imagine 28x28 -> 9 regiuni pe fiecare axa (28//3 = 9, rest 1 pixel ignorat)
#   Pastram doar k regiuni cu cea mai mare magnitudine medie.
#   Ideea: pastram doar zonele "interesante" (contururile), ignoram fundalul.

def compute_gradient_magnitude(image):
    """
    Calculeaza magnitudinea gradientului pentru fiecare pixel.

    Gradientul pe x: diferenta intre pixelul curent si cel din dreapta
    Gradientul pe y: diferenta intre pixelul curent si cel de dedesubt

    Returneaza: Gx, Gy, G (toate matrici de aceeasi dimensiune ca imaginea)
    """
    h, w = image.shape

    # Initializam cu zerouri
    Gx = np.zeros_like(image, dtype=float)
    Gy = np.zeros_like(image, dtype=float)

    # Gradient pe x (orizontal): f(x+1) - f(x)
    # Pentru fiecare pixel, diferenta cu vecinul din dreapta
    Gx[:, :-1] = image[:, 1:] - image[:, :-1]

    # Gradient pe y (vertical): f(y+1) - f(y)
    # Pentru fiecare pixel, diferenta cu vecinul de dedesubt
    Gy[:-1, :] = image[1:, :] - image[:-1, :]

    # Magnitudinea: G = sqrt(Gx^2 + Gy^2)
    G = np.sqrt(Gx ** 2 + Gy ** 2)

    return Gx, Gy, G


def extract_top_k_regions(image, region_size=3, k=30):
    """
    Imparte imaginea in regiuni region_size x region_size care NU se suprapun.
    Returneaza pixelii din cele mai importante k regiuni (dupa magnitudinea medie).
    """
    Gx, Gy, G = compute_gradient_magnitude(image)
    h, w = image.shape

    # Cate regiuni avem pe fiecare axa
    rows = h // region_size  # 28 // 3 = 9
    cols = w // region_size  # 28 // 3 = 9

    # Calculam magnitudinea medie pentru fiecare regiune
    region_scores = []
    for r in range(rows):
        for c in range(cols):
            # Coordonatele regiunii
            r_start = r * region_size
            r_end = r_start + region_size
            c_start = c * region_size
            c_end = c_start + region_size

            # Media magnitudinii in aceasta regiune
            mean_mag = G[r_start:r_end, c_start:c_end].mean()
            region_scores.append((mean_mag, r, c))

    # Sortam regiunile descrescator dupa magnitudine medie
    region_scores.sort(key=lambda x: x[0], reverse=True)

    # Pastram doar primele k regiuni
    top_k = region_scores[:k]

    # Extragem pixelii din regiunile selectate si ii concatenam intr-un vector
    features = []
    for _, r, c in top_k:
        r_start = r * region_size
        c_start = c * region_size
        region_pixels = image[r_start:r_start + region_size,
                              c_start:c_start + region_size]
        features.extend(region_pixels.flatten())

    return np.array(features)


print("\n" + "=" * 60)
print("EX 2 - Gradient Magnitude + Top-K regiuni")
print("=" * 60)

k = 30  # cate regiuni pastram (din 81 totale = 9x9)
print(f"Extragem top {k} regiuni din {(28//3) * (28//3)} totale...")

train_grad = np.array([extract_top_k_regions(img, region_size=3, k=k) for img in train_imgs])
test_grad = np.array([extract_top_k_regions(img, region_size=3, k=k) for img in test_imgs])

print(f"Train gradient features shape: {train_grad.shape}")

# Antrenam SVM
model_grad = svm.SVC(C=10, kernel='rbf', gamma='scale')
model_grad.fit(train_grad, train_labels)
pred_grad = model_grad.predict(test_grad)
acc_grad = accuracy_score(test_labels, pred_grad)
print(f"Acuratete Gradient Top-K + SVM(rbf): {acc_grad * 100:.2f}%")


# ================================================================
# EXERCITIUL 3 - Non-Maximum Suppression (NMS)
# ================================================================
# IDEEA:
#   Subtierea contururilor: pastram doar pixelii care sunt maxime locale
#   pe directia gradientului.
#
#   Pasii:
#   1. Calculam Gx, Gy, G (magnitudine)
#   2. Calculam directia: theta = arctan(Gy / Gx)
#   3. Rotunjim theta la una din 4 directii (0°, 45°, 90°, 135°)
#   4. Pentru fiecare pixel, comparam magnitudinea lui cu cei 2 vecini
#      de-a lungul directiei gradientului
#   5. Daca e mai mare decat ambii vecini -> pastram valoarea
#      Altfel -> punem 0
#
#   Rezultat: contururi subtiri de 1 pixel grosime

def non_maximum_suppression(image):
    """
    Aplica Non-Maximum Suppression pe o imagine.

    Returneaza: matrice cu aceeasi dimensiune, dar doar cu maximele locale
    """
    Gx, Gy, G = compute_gradient_magnitude(image)
    h, w = image.shape

    # Calculam directia gradientului in grade
    # np.arctan2 e mai sigur decat arctan (gestioneaza Gx=0)
    theta = np.arctan2(Gy, Gx) * 180 / np.pi

    # Normalizam unghiurile in [0, 180) (directia e aceeasi la 180°)
    theta[theta < 0] += 180

    # Matricea rezultat
    nms = np.zeros_like(G)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Determinam vecinii de comparat in functie de directia gradientului

            angle = theta[i, j]

            # 0° (orizontal) -> comparam cu vecinii din stanga si dreapta
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                vecin1 = G[i, j - 1]
                vecin2 = G[i, j + 1]

            # 45° (diagonala /) -> comparam cu vecinii din dreapta-sus si stanga-jos
            elif 22.5 <= angle < 67.5:
                vecin1 = G[i - 1, j + 1]
                vecin2 = G[i + 1, j - 1]

            # 90° (vertical) -> comparam cu vecinii de sus si de jos
            elif 67.5 <= angle < 112.5:
                vecin1 = G[i - 1, j]
                vecin2 = G[i + 1, j]

            # 135° (diagonala \) -> comparam cu vecinii din stanga-sus si dreapta-jos
            else:
                vecin1 = G[i - 1, j - 1]
                vecin2 = G[i + 1, j + 1]

            # Pastram doar daca e mai mare decat ambii vecini
            if G[i, j] >= vecin1 and G[i, j] >= vecin2:
                nms[i, j] = G[i, j]
            else:
                nms[i, j] = 0

    return nms


print("\n" + "=" * 60)
print("EX 3 - Non-Maximum Suppression")
print("=" * 60)

# Aplicam NMS si folosim rezultatul ca feature
train_nms = np.array([non_maximum_suppression(img).flatten() for img in train_imgs])
test_nms = np.array([non_maximum_suppression(img).flatten() for img in test_imgs])

print(f"Train NMS shape: {train_nms.shape}")

# Antrenam SVM
model_nms = svm.SVC(C=10, kernel='rbf', gamma='scale')
model_nms.fit(train_nms, train_labels)
pred_nms = model_nms.predict(test_nms)
acc_nms = accuracy_score(test_labels, pred_nms)
print(f"Acuratete NMS + SVM(rbf): {acc_nms * 100:.2f}%")

# Vizualizare: comparatie imagine originala vs NMS
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for idx in range(5):
    axes[0, idx].imshow(train_imgs[idx], cmap='gray')
    axes[0, idx].set_title(f"Original: {train_labels[idx]}")
    axes[0, idx].axis('off')

    axes[1, idx].imshow(non_maximum_suppression(train_imgs[idx]), cmap='gray')
    axes[1, idx].set_title("NMS")
    axes[1, idx].axis('off')
plt.tight_layout()
plt.savefig('nms_comparison.png')
plt.show()


# ================================================================
# EXERCITIUL 4 - Regiuni binarizate + distanta Hamming + KNN
# ================================================================
# IDEEA:
#   Similar cu Ex1, dar la nivel de REGIUNI:
#   1. Impartim imaginea in regiuni (ca la Ex1)
#   2. In fiecare regiune, binarizam: pixel >= medie_regiune -> 1, altfel -> 0
#   3. Liniarizam si concatenam toate regiunile -> un vector binar lung
#   4. Folosim distanta Hamming (cati biti sunt diferiti) cu KNN
#
#   Distanta Hamming: numarul de pozitii in care doi vectori binari difera
#   Exemplu: [1,0,1,1] vs [1,1,0,1] -> difera la pozitia 1 si 2 -> Hamming = 2

def binarize_regions(image, region_size=3):
    """
    Imparte imaginea in regiuni si binarizeaza fiecare regiune.
    Binarizare: pixel >= media regiunii -> 1, altfel -> 0
    Concateneaza toti vectorii binari.
    """
    h, w = image.shape
    rows = h // region_size
    cols = w // region_size

    binary_vector = []

    for r in range(rows):
        for c in range(cols):
            r_start = r * region_size
            c_start = c * region_size
            region = image[r_start:r_start + region_size,
                           c_start:c_start + region_size]

            # Binarizam: comparam fiecare pixel cu media regiunii
            mean_val = region.mean()
            binary_region = (region >= mean_val).astype(int)

            # Liniarizam si adaugam
            binary_vector.extend(binary_region.flatten())

    return np.array(binary_vector)


def hamming_distance(x, y):
    """
    Distanta Hamming: numarul de pozitii in care doi vectori difera.
    """
    return np.sum(x != y)


def knn_hamming(train_features, train_labels, test_feature, k=3):
    """
    KNN cu distanta Hamming.
    Calculeaza distanta Hamming de la test_feature la toate train_features,
    ia primii k vecini, returneaza eticheta majoritara.
    """
    distante = np.array([hamming_distance(test_feature, train_f)
                         for train_f in train_features])

    # Indicii celor mai apropiati k vecini
    indici_sortati = np.argsort(distante)[:k]
    etichete_vecini = train_labels[indici_sortati]

    # Vot majoritar
    eticheta = np.bincount(etichete_vecini).argmax()
    return eticheta


print("\n" + "=" * 60)
print("EX 4 - Regiuni binarizate + Hamming KNN")
print("=" * 60)

# Binarizam toate imaginile
train_binary = np.array([binarize_regions(img, region_size=3) for img in train_imgs])
test_binary = np.array([binarize_regions(img, region_size=3) for img in test_imgs])

print(f"Train binary shape: {train_binary.shape}")

# KNN cu distanta Hamming
k_knn = 3
print(f"Clasificam cu KNN Hamming (k={k_knn})... (dureaza cateva minute)")

pred_hamming = []
for i in range(len(test_binary)):
    pred = knn_hamming(train_binary, train_labels, test_binary[i], k=k_knn)
    pred_hamming.append(pred)
    if (i + 1) % 100 == 0:
        print(f"  Clasificate {i + 1}/{len(test_binary)} imagini...")

pred_hamming = np.array(pred_hamming)
acc_hamming = accuracy_score(test_labels, pred_hamming)
print(f"Acuratete Hamming KNN (k={k_knn}): {acc_hamming * 100:.2f}%")


# ================================================================
# EXERCITIUL 5 - Histograme LBP + SVM cu kernel intersectie
# ================================================================
# IDEEA:
#   Refolosim histogramele LBP de la Ex1.
#   Dar acum folosim un kernel special: histogram intersection.
#
#   Kernel Intersection: K(x, y) = sum(min(xi, yi))
#   Intuitie: cat de mult se "suprapun" doua histograme.
#   Daca doua histograme sunt identice, suprapunerea e maxima.
#
#   SVM permite kernel-uri custom prin precomputarea matricei de kernel.

def histogram_intersection_kernel(X, Y):
    """
    Calculeaza matricea de kernel intersection intre X si Y.

    K(x, y) = sum(min(xi, yi)) pentru fiecare pereche de randuri

    X: matrice (n1 x d)
    Y: matrice (n2 x d)
    Returneaza: matrice (n1 x n2)
    """
    n1 = X.shape[0]
    n2 = Y.shape[0]
    K = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            K[i, j] = np.sum(np.minimum(X[i], Y[j]))

    return K


print("\n" + "=" * 60)
print("EX 5 - LBP + SVM Histogram Intersection Kernel")
print("=" * 60)

# Refolosim train_lbp si test_lbp de la Ex1
print("Calculam matricea de kernel (poate dura)...")

# Pentru SVM cu kernel precomputat:
# La antrenare: matricea de kernel train vs train
# La testare: matricea de kernel test vs train
K_train = histogram_intersection_kernel(train_lbp, train_lbp)
K_test = histogram_intersection_kernel(test_lbp, train_lbp)

print(f"K_train shape: {K_train.shape}")
print(f"K_test shape: {K_test.shape}")

# SVM cu kernel precomputat
model_hi = svm.SVC(C=10, kernel='precomputed')
model_hi.fit(K_train, train_labels)
pred_hi = model_hi.predict(K_test)
acc_hi = accuracy_score(test_labels, pred_hi)
print(f"Acuratete LBP + SVM(histogram intersection): {acc_hi * 100:.2f}%")


# ================================================================
# SUMAR REZULTATE
# ================================================================
print("\n" + "=" * 60)
print("SUMAR REZULTATE")
print("=" * 60)
print(f"  Ex1 - LBP + SVM(rbf):                    {acc_lbp * 100:.2f}%")
print(f"  Ex2 - Gradient Top-K + SVM(rbf):          {acc_grad * 100:.2f}%")
print(f"  Ex3 - NMS + SVM(rbf):                     {acc_nms * 100:.2f}%")
print(f"  Ex4 - Binary Regions + Hamming KNN(k={k_knn}):  {acc_hamming * 100:.2f}%")
print(f"  Ex5 - LBP + SVM(histogram intersection):  {acc_hi * 100:.2f}%")