import numpy as np
from sklearn import svm, preprocessing
from sklearn.metrics import f1_score

# incarcare date
train_data = np.load('data/training_sentences.npy', allow_pickle=True)
test_data = np.load('data/test_sentences.npy', allow_pickle=True)
train_labels = np.load('data/training_labels.npy').astype(int)
test_labels = np.load('data/test_labels.npy').astype(int)

print(f"train: {len(train_data)} mesaje, test: {len(test_data)} mesaje")


# ex 2 -->>  normalize_data(train_data, test_data, type=None) care primește ca
# parametri datele de antrenare, respectiv de testare și tipul de normalizare ({None,
# ‘standard’, ‘l1’, ‘l2’}) și întoarce aceste date normalizate.

def normalize_data(train_data, test_data, type=None):
    if type is None:
        return train_data, test_data

    if type == 'standard':
        # StandardScaler -- calculeaza media si deviatia pe TRAIN, aplica pe ambele
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        train_norm = scaler.transform(train_data)
        test_norm = scaler.transform(test_data)

    elif type == 'l1':
        # Normalizer L1 --- fiecare exemplu (rand) e impartit la suma valorilor absolute
        # se aplica independent pe train si test
        train_norm = preprocessing.normalize(train_data, norm='l1')
        test_norm = preprocessing.normalize(test_data, norm='l1')

    elif type == 'l2':
        # Normalizer L2 --- fiecare exemplu e impartit la radacina sumei patratelor
        train_norm = preprocessing.normalize(train_data, norm='l2')
        test_norm = preprocessing.normalize(test_data, norm='l2')

    return train_norm, test_norm


# ex 3 --->> BagOfWords în al cărui constructor se inițializează vocabularul (un
# dicționar gol). În cadrul ei implementați metoda build_vocabulary(self, data) care
# primește ca parametru o listă de mesaje(listă de liste de strings) și construiește
# vocabularul pe baza acesteia. Cheile dicționarului sunt reprezentate de cuvintele din
# eseuri, iar valorile de id-urile unice atribuite cuvintelor. Pe lângă vocabularul pe care-l
# construiți, rețineți și o listă cu cuvintele în ordinea adăugării în vocabular.
# Afișați dimensiunea vocabularul construit (9522)


class BagOfWords:
    def __init__(self):
        self.vocabulary = {}  # dictionar --- cuvant -> id
        self.words_list = []  # lista cuvintelor in ordinea adaugarii

    def build_vocabulary(self, data):
        """
        construieste vocabularul din datele de antrenare.

        data: lista de mesaje (fiecare mesaj = lista de cuvinte)
        """
        idx = 0
        for mesaj in data:
            for cuvant in mesaj:
                if cuvant not in self.vocabulary:
                    self.vocabulary[cuvant] = idx
                    self.words_list.append(cuvant)
                    idx += 1

    # ex 4 --->> get_features(self, data)
    def get_features(self, data):
        """
        transforma mesajele in vectori numerici (matrice)

        data: lista de mesaje
        returneaza: matrice (num_mesaje x dimensiune_vocabular)
                    unde features[i][j] = de cate ori apare cuvantul j in mesajul i
        """
        num_samples = len(data)
        dict_length = len(self.vocabulary)
        features = np.zeros((num_samples, dict_length))

        for i, mesaj in enumerate(data):
            for cuvant in mesaj:
                # numaram doar cuvintele care sunt in vocabular
                # (cuvinte noi din test care nu erau in train le ignoram)
                if cuvant in self.vocabulary:
                    word_idx = self.vocabulary[cuvant]
                    features[i][word_idx] += 1

        return features

# construim vocabularul
bow = BagOfWords()
bow.build_vocabulary(train_data)
print(f"dimensiune vocabular: {len(bow.vocabulary)}")

# ex 5
# obtinem matricea de features
train_features = bow.get_features(train_data)
test_features = bow.get_features(test_data)
print(f"train features shape: {train_features.shape}")
print(f"test features shape: {test_features.shape}")

# normalizam cu L2
train_norm, test_norm = normalize_data(train_features, test_features, type='l2')
print("normalizare L2 aplicata")

# ex 6 --->> Antrenați un SVM cu kernel linear care să clasifice mesaje în mesaje
# spam/non-spam. Pentru parametrul C setați valoarea 1. Calculați acuratețea și
# F1-score pentru mulțimea de testare.
# Afișați cele mai negative (spam) 10 cuvinte și cele mai pozitive (non-spam) 10
# cuvinte.

# antrenam SVM
model = svm.SVC(C=1, kernel='linear')
model.fit(train_norm, train_labels)

# predictii
predictii = model.predict(test_norm)

# acuratete
acuratete = model.score(test_norm, test_labels)
print(f"Acuratete: {acuratete * 100:.2f}%")

# F1-score
f1 = f1_score(test_labels, predictii)
print(f"F1-score: {f1:.4f}")

weights = model.coef_[0]  # vectorul de ponderi

# sortam indicii dupa pondere
sorted_indices = np.argsort(weights)

# cele mai negative (spam) ->>> primele 10 din sortare
spam_indices = sorted_indices[:10]
spam_words = [bow.words_list[i] for i in spam_indices]
print(f"\nCele mai negative (spam) 10 cuvinte: {spam_words}")

# cele mai pozitive (non-spam) -->> ultimele 10 din sortare
ham_indices = sorted_indices[-10:]
ham_words = [bow.words_list[i] for i in ham_indices]
print(f"Cele mai pozitive (non-spam) 10 cuvinte: {ham_words}")


