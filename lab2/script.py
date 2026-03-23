import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

# incarc datele mnist
train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt').astype(int)
test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt').astype(int)

# test prima imagine
imagine = train_images[0, :]
imagine = np.reshape(imagine, (28, 28))
plt.imshow(imagine.astype(np.uint8), cmap='gray')
plt.show()


# ex 2 -->> calculati captele a num_bins intervale
# values_to_bins -->> primeste matrice de dimensiune (n_samples, n_features) si captele intervalelor
# pentru fiecare exemplu si fiecare atribut calculeaza indexul intervalului corespunzatori
# folositi fct definita pt a discretica multimea de antrenare si cea de testare

def values_to_bins(data, num_bins):
    # cream captele intervalelor -->> num_bins valori intre 0 si 255
    bins = np.linspace(start=0, stop=255, num=num_bins)

    # np.digitize -->> pt fiecare valoare gaseste in ce interval cade
    data_binned = np.digitize(data, bins)
    return data_binned


# ex 3 -->> acuratetea pe multimea de testare a clasificatorului Multinomial Naive Bayes, impartind intervalul
# pixelilor in 4 subintervale

# step 1 --->> discretizam train si test cu values_to_bins
num_bins = 4
train_binned = values_to_bins(train_images, num_bins)
test_binned = values_to_bins(test_images, num_bins)

# step 2 --->> cream si antrenam modelul
model = MultinomialNB()
model.fit(train_binned, train_labels)

# step 3 --->> acuratete
acuratete = model.score(test_binned, test_labels)
print("Acuratete: %s " % acuratete)

# scurta verificare cu num_binds = 5 unde tre sa ne dea 83,6%
train_binned_5 = values_to_bins(train_images, 5)
test_binned_5 = values_to_bins(test_images, 5)
model_5 = MultinomialNB()
model_5.fit(train_binned_5, train_labels)
acuratete_5 = model_5.score(test_binned_5, test_labels)
print("Acuratete cu num_binds = 5: %s" % acuratete_5)

# ex 4 --->> test clasificatorul mutinomial naive bayes pe subsetul MNIST folosind num_binds 3 5 7 9 11

bins_values = [3, 5, 7, 9, 11]
acurateti = []
best_binds = 0
best_acuratete = 0

for bins in bins_values:
    # discretizez
    train_b = values_to_bins(train_images, bins)
    test_b = values_to_bins(test_images, bins)

    # antrenez si evaluez
    mod = MultinomialNB()
    mod.fit(train_b, train_labels)
    acur = mod.score(test_b, test_labels)
    acurateti.append(acur)

    print(f"num_binds = {bins} are acuratetea {acur}")

    # cel mai bun bins
    if acur > best_acuratete:
        best_acuratete = acur
        best_binds = bins

print(f"Cel mai bun num_binds = {best_binds} cu acuratete {best_acuratete}")

# ex 5 --->> nr subintervale care obtine cea mai buna acuratete la 4, afisati cel putin 10 exemple misclasate

# antrenez modelul final cu best_bins
train_best = values_to_bins(train_images, best_binds)
test_best = values_to_bins(test_images, best_binds)
model_best = MultinomialNB()
model_best.fit(train_best, train_labels)

# predictii pe test
predictii = model_best.predict(test_best)

# trebuie sa gasesc indicii unde predictia este gresita --->>> adica predictii sa fie diferit de test_labels
misclasate_indici = np.where(predictii != test_labels)[0]
print(f"total misclasate: {len(misclasate_indici)} din {len(test_labels)}")

# afisez primele 10 exemple misclasate
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
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

# ex 6 --->> confusion_matrix(y_true, y_pred) calculeaza matricea de confuzie

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

