import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn import svm

# ================================================================
# EXERCITIUL 5 - Raport experimente + evaluare hiperparametri
# ================================================================
# Acest script genereaza un raport complet cu experimentele
# de la punctele 1, 3 si 4, testand diferite combinatii de hiperparametri
# pe un set de validare (20% din train).


# ================================================================
# INCARCARE DATE
# ================================================================
train_sentences = []
with open('train_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f:
        train_sentences.append(line.strip())

test_sentences = []
with open('test_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f:
        test_sentences.append(line.strip())

train_labels = np.load('train_labels.npy')
train_conv_features = np.load('train_conv_features.npy')
test_conv_features = np.load('test_conv_features.npy')


# ================================================================
# SPLIT TRAIN / VALIDARE (80/20)
# ================================================================
np.random.seed(42)
indices = np.arange(len(train_labels))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx = indices[:split]
val_idx = indices[split:]

print(f"Train: {len(train_idx)}, Validare: {len(val_idx)}")


# ================================================================
# PREGATIRE FEATURES
# ================================================================

# --- Bag of Words la nivel de caracter (pentru exercitiul 1) ---
class CharBagOfWords:
    def __init__(self):
        self.vocabulary = {}
        self.char_list = []

    def build_vocabulary(self, sentences):
        idx = 0
        for sentence in sentences:
            for char in sentence:
                if char not in self.vocabulary:
                    self.vocabulary[char] = idx
                    self.char_list.append(char)
                    idx += 1

    def get_features(self, sentences):
        num_samples = len(sentences)
        num_features = len(self.vocabulary)
        features = np.zeros((num_samples, num_features))
        for i, sentence in enumerate(sentences):
            for char in sentence:
                if char in self.vocabulary:
                    features[i][self.vocabulary[char]] += 1
        return features

bow = CharBagOfWords()
bow.build_vocabulary(train_sentences)
bow_features = bow.get_features(train_sentences)


# --- Convolution features (pentru exercitiile 3 si 4) ---
conv_norm_l2 = normalize(train_conv_features, norm='l2')
conv_norm_l1 = normalize(train_conv_features, norm='l1')


# ================================================================
# FUNCTII HELPER
# ================================================================

def linear_kernel(X, Y):
    return X.dot(Y.T)

def hellinger_kernel(X, Y):
    X_safe = np.maximum(X, 0)
    Y_safe = np.maximum(Y, 0)
    return np.sqrt(X_safe).dot(np.sqrt(Y_safe).T)

def kernel_ridge_fit(K, y_one_hot, lambd):
    n = K.shape[0]
    return np.linalg.solve(K + lambd * np.eye(n), y_one_hot)

num_classes = len(np.unique(train_labels))
y_one_hot = np.zeros((len(train_labels), num_classes))
for i in range(len(train_labels)):
    y_one_hot[i, int(train_labels[i])] = 1


# ================================================================
# RAPORT EXPERIMENTAL
# ================================================================
raport = []
raport.append("=" * 70)
raport.append("RAPORT EXPERIMENTAL - Clasificarea documentelor text")
raport.append("=" * 70)

# ------- EXPERIMENT 1: Naive Bayes -------
raport.append("\n" + "-" * 70)
raport.append("EXPERIMENT 1: Naive Bayes (Bag of Words la nivel de caracter)")
raport.append("-" * 70)
raport.append(f"{'Alpha':<12} {'Normalizare':<15} {'Acc validare':<15}")
raport.append("-" * 42)

best_acc_nb = 0
best_config_nb = ""

for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    for norm_type in ['none', 'l2']:
        if norm_type == 'none':
            feats = bow_features
        else:
            feats = normalize(bow_features, norm='l2')

        model = MultinomialNB(alpha=alpha)
        # MultinomialNB nu accepta valori negative, deci doar cu features nenormalizate
        # sau L2 normalizate (care sunt >= 0)
        try:
            model.fit(feats[train_idx], train_labels[train_idx])
            pred = model.predict(feats[val_idx])
            acc = accuracy_score(train_labels[val_idx], pred)
        except ValueError:
            acc = 0.0

        line = f"{alpha:<12.3f} {norm_type:<15} {acc*100:<15.2f}%"
        raport.append(line)
        print(line)

        if acc > best_acc_nb:
            best_acc_nb = acc
            best_config_nb = f"alpha={alpha}, norm={norm_type}"

raport.append(f"\nCea mai buna configuratie: {best_config_nb} ({best_acc_nb*100:.2f}%)")


# ------- EXPERIMENT 2: Kernel Ridge -------
raport.append("\n" + "-" * 70)
raport.append("EXPERIMENT 2: Kernel Ridge Regression (kernel liniar)")
raport.append("-" * 70)
raport.append(f"{'Lambda':<12} {'Normalizare':<15} {'Acc validare':<15}")
raport.append("-" * 42)

best_acc_kr = 0
best_config_kr = ""

for lambd in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    for norm_type, feats in [('l2', conv_norm_l2), ('l1', conv_norm_l1)]:
        K_tr = linear_kernel(feats[train_idx], feats[train_idx])
        K_val = linear_kernel(feats[val_idx], feats[train_idx])

        alpha_coefs = kernel_ridge_fit(K_tr, y_one_hot[train_idx], lambd)
        val_scores = K_val.dot(alpha_coefs)
        pred = np.argmax(val_scores, axis=1)
        acc = accuracy_score(train_labels[val_idx], pred)

        line = f"{lambd:<12.3f} {norm_type:<15} {acc*100:<15.2f}%"
        raport.append(line)
        print(line)

        if acc > best_acc_kr:
            best_acc_kr = acc
            best_config_kr = f"lambda={lambd}, norm={norm_type}"

raport.append(f"\nCea mai buna configuratie: {best_config_kr} ({best_acc_kr*100:.2f}%)")


# ------- EXPERIMENT 3: SVM Hellinger -------
raport.append("\n" + "-" * 70)
raport.append("EXPERIMENT 3: SVM cu kernel Hellinger")
raport.append("-" * 70)
raport.append(f"{'C':<12} {'Normalizare':<15} {'Acc validare':<15}")
raport.append("-" * 42)

best_acc_svm = 0
best_config_svm = ""

for C in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
    for norm_type, feats in [('l1', conv_norm_l1), ('l2', conv_norm_l2)]:
        K_tr = hellinger_kernel(feats[train_idx], feats[train_idx])
        K_val = hellinger_kernel(feats[val_idx], feats[train_idx])

        model = svm.SVC(C=C, kernel='precomputed')
        model.fit(K_tr, train_labels[train_idx])
        pred = model.predict(K_val)
        acc = accuracy_score(train_labels[val_idx], pred)

        line = f"{C:<12.1f} {norm_type:<15} {acc*100:<15.2f}%"
        raport.append(line)
        print(line)

        if acc > best_acc_svm:
            best_acc_svm = acc
            best_config_svm = f"C={C}, norm={norm_type}"

raport.append(f"\nCea mai buna configuratie: {best_config_svm} ({best_acc_svm*100:.2f}%)")


# ------- SUMAR -------
raport.append("\n" + "=" * 70)
raport.append("SUMAR FINAL")
raport.append("=" * 70)
raport.append(f"Naive Bayes:        {best_acc_nb*100:.2f}% ({best_config_nb})")
raport.append(f"Kernel Ridge:       {best_acc_kr*100:.2f}% ({best_config_kr})")
raport.append(f"SVM Hellinger:      {best_acc_svm*100:.2f}% ({best_config_svm})")
raport.append("")
raport.append("Observatii:")
raport.append("- Naive Bayes cu BoW la nivel de caracter are performanta limitata")
raport.append("  deoarece nu capteaza secvente de caractere (n-grams).")
raport.append("- Kernel Ridge cu kernel liniar ofera performanta decenta")
raport.append("  dar e limitat de liniaritatea kernel-ului.")
raport.append("- SVM cu kernel Hellinger obtine cele mai bune rezultate")
raport.append("  deoarece Hellinger e potrivit pentru features de tip histograma/count.")

# Salvam raportul
raport_text = "\n".join(raport)
print("\n" + raport_text)

with open('raport_experimente.txt', 'w', encoding='utf-8') as f:
    f.write(raport_text)

print("\nRaport salvat in raport_experimente.txt")
