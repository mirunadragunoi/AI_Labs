import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

# incarc datele mnist
train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt').astype(int)
test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt').astype(int)

# ex 1 -->> clasa KnnClassifier
class KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    # ex 2 -->> classify_image(self, test_image, num_neighbors = 3, metric = '/2') clasifica imaginea test_image
    # cu metoda celor mai apropiati vecini

    """
    trebuie sa calculez distanta de la test_image la fiecare imagine din train
    trebuie sa sortez distantele crescator
    trebuie sa iau etichetele primilor K vecini cei mai apropiati
    eticheta care apare cel mai des este predictia
    
    ca distante --->>> L1 e Manhattan iar L2 e euclidiana
    """

    def classify_image(self, test_image, num_neighbors = 3, metric = '/2'):

        # step 1 --->> calculez distanta de la test_image la fiecare imagine de train
        if metric == 'l2':
            distante = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis=1))
        elif metric == 'l1':
            distante = np.sum(np.abs(self.train_images - test_image), axis=1)

        # step 2 --->> sortez si iau indicii celor mai mici K distante
        indici_sortati = np.argsort(distante)

        # primii k indici, cei mai apropiati vecini
        indici_vecini = indici_sortati[:num_neighbors]

        # step 3 --->> etichetele vecinilor
        etichete_vecini = self.train_labels[indici_vecini]

        # step 4 --->> eticheta care apare cel mai des
        nr_aparitii = np.bincount(etichete_vecini)
        eticheta_prezisa = np.argmax(nr_aparitii)

        return eticheta_prezisa


# ex 3 --->> acuratetea metodei celor mai apropiati vecini pe multimea de testare avand ca distanta l2 si nr
# vecini 3
# salvare predictii in fisierul predictii_3nn_l2_mnist.txt

knn = KnnClassifier(train_images, train_labels)

# clasific toate imaginile de test

predictii = []
for i in range(len(test_images)):

    # iau imaginea i din test
    test_imagine = test_images[i]

    # clasific
    eticheta = knn.classify_image(test_imagine, num_neighbors=3, metric='l2')
    predictii.append(eticheta)

    # afisez progresul
    if (i + 1) % 100 == 0:
        print(f"clasificate {i + 1} / {len(test_images)} imagini ...............")

predictii = np.array(predictii)

# calculam acuratetea
acuratete = np.mean(predictii == test_labels)
print(f"acuratetea cu 3-NN cu L2: {acuratete}")

# salvam predictiile in fisier
np.savetxt('predictii_3nn_l2_mnist.txt', predictii)


# ex 4 --->>> calculati acuratetea metodei celor mai apropiati vecini pe multimea de testare avand ca distanta L2 si
# nr de vecini 1 3 5 7 9
# a -->> plotati un grafic cu acuratetea obtinuta pt fiecare vecin si save scoruri in fisierul acuratete_l2.txt
# b -->> repet punct anterior pt distanta l1 ++ tot asa plot

valori_k = [1, 3, 5, 7, 9]
acurateti_l2 = []

for k in valori_k:
    predictii_k = []
    for i in range(len(test_images)):
        eticheta = knn.classify_image(test_images[i], num_neighbors=k, metric='l2')
        predictii_k.append(eticheta)

    predictii_k = np.array(predictii_k)
    acur = np.mean(predictii_k == test_labels)
    acurateti_l2.append(acur)
    print(f"K = {k}, L2 ---->> acuratete = {acur}")

# salvam acuratetile l2
np.savetxt('acuratete_l2.txt', acurateti_l2)

# plot grafic
plt.figure(figsize=(8, 5))
plt.plot(valori_k, acurateti_l2, marker='o', label='L2')
plt.xlabel('numar vecini')
plt.ylabel('acuratete')
plt.title('KNN - acuratete vs nr vecini pt L2')
plt.xticks(valori_k)
plt.legend()
plt.grid(True)
plt.savefig('grafic_l2.png')
plt.show()

# pt b
acurateti_l1 = []

for k in valori_k:
    predictii_k = []
    for i in range(len(test_images)):
        eticheta = knn.classify_image(test_images[i], num_neighbors=k, metric='l1')
        predictii_k.append(eticheta)

    predictii_k = np.array(predictii_k)
    acur = np.mean(predictii_k == test_labels)
    acurateti_l1.append(acur)
    print(f"K = {k}, L1 ---->> acuratete = {acur}")

# luam acuratetile de la L2 din fisier
acurateti_l2_fisier = np.loadtxt('acuratete_l2.txt')

# grafic combinat l1 si l2
plt.figure(figsize=(8, 5))
plt.plot(valori_k, acurateti_l1, marker='o', label='L1')
plt.plot(valori_k, acurateti_l2_fisier, marker='o', label='L2')
plt.xlabel('numar vecini')
plt.ylabel('acuratete')
plt.title('KNN - acuratete L1 vs L2')
plt.xticks(valori_k)
plt.legend()
plt.grid(True)
plt.savefig('grafic_l1_l2.png')
plt.show()


