import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

# incarc datele mnist
train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt').astype('int')
test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt').astype('int')


# ex 2 --->> capetele a num_bins invervale
# values_to_bins -->> primeste matrice (n_samples, n_features)
# plus capetele intervalelor
# pt fiecare exemplu + atribut calculeaza indexul intervalului

def values_to_bins(date, num_bins):
    # capetele intervalelor, avem pixeli de la 0 la 255
    bins = np.linspace(start=0, stop=256, num=num_bins + 1)

    # calculam pt fiecare valoare in ce interval cade
    date_binned = np.digitize(date, bins) - 1

    return date_binned


# ex 3 -->> acuratetea pe multimea de testare cu 4 subintervale

# pas 1 --->> discretizam
num_bins = 5
train_binned = values_to_bins(train_images, num_bins)
test_binned = values_to_bins(test_images, num_bins)

# pas 2 -->> trebuie sa cream si antrenam modelul
model = MultinomialNB()
model.fit(train_binned, train_labels)

# pas 3 -->> acuratetea
acuratete = model.score(test_binned, test_labels)
print(f"acuratete: {acuratete}")

# ex 4 -->> test clasificator pe subset num_bins - 3 5 7 9 11
valori_k = [3, 5, 7, 9, 11]
best_interval = []
best_acuratete = 0

for k in valori_k:
    train_k = values_to_bins(train_images, k)
    test_k = values_to_bins(test_images, k)

    modelk = MultinomialNB()
    modelk.fit(train_k, train_labels)

    acuratete_k = model.score(test_k, test_labels)

    print(f"acuratete pt {k} -->> {acuratete_k}")

    if acuratete_k > best_acuratete:
        best_acuratete = acuratete_k
        best_interval = k

# ex 5 -->> pt nr de subintervale cele mai bune -->> 10 exemple misclasate
print(f"best interval: {best_interval} cu acuratetea {best_acuratete}")

# antrenez etc etc
train_misclasate = values_to_bins(train_images, best_interval)
test_misclasate = values_to_bins(test_images, best_interval)

model_miscasate = MultinomialNB()
model_miscasate.fit(train_misclasate, train_labels)

# trebuie sa fac predictii pe test
predictii = model.predict(test_misclasate)

# trebuie sa gasesc indicii unde predictia este gresita --->>> adica predictii sa fie diferit de test_labels
misclasate_indici = np.where(predictii != test_labels)[0]

# afisez primele 10 exemple misclasate
fig, axes = plt.subplots(5, 2, figsize=(15, 6))
for indice, aux in enumerate(axes.flat):
    if indice < len(misclasate_indici):
        i = misclasate_indici[indice]
        imagine = test_images[i].reshape(28, 28)
        aux.imshow(imagine.astype(np.uint8), cmap='gray')
        aux.set_title(f"real: {test_labels[i]}, prezis = {predictii[i]}")
    aux.axis('off')
plt.suptitle("exemple misclasate")
plt.tight_layout()
plt.show()


# ex 6 -->> matricea de confuzie??? cu predictii clasificator
# cum ar veni alea gresite
def confusion_matrix(y_true, y_pred):
    # cate clase avem
    nr_clase = len(np.unique(y_true))

    # initializam matricea cu null
    matrice = np.zeros((nr_clase, nr_clase))

    # parcurgem fiecare pereche (eticheta reala, eticheta prezisa)
    for real, prezis in zip(y_true, y_pred):
        matrice[real][prezis] += 1

    return matrice

matrice_confuzie = confusion_matrix(test_labels, predictii)
print("matricea de confuzie: ")
print(matrice_confuzie)

matrice2 = confusion_matrix(test_labels, predictii)
print("cu scikit learn matricea:")
print(matrice2)
