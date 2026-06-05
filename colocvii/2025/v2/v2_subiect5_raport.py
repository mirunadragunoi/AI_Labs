import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

# ================================================================
# EXERCITIUL 5 (V2) - Raport experimente
# ================================================================

# --- Incarcare date ---
train_sentences = []
with open('train_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f:
        train_sentences.append(line.strip())

train_labels = np.load('train_labels.npy')
train_conv_features = np.load('train_conv_features.npy')

# --- Char BoW ---
class CharBagOfWords:
    def __init__(self):
        self.vocabulary = {}
    def build_vocabulary(self, sentences):
        idx = 0
        for s in sentences:
            for c in s:
                if c not in self.vocabulary:
                    self.vocabulary[c] = idx
                    idx += 1
    def get_features(self, sentences):
        features = np.zeros((len(sentences), len(self.vocabulary)))
        for i, s in enumerate(sentences):
            for c in s:
                if c in self.vocabulary:
                    features[i][self.vocabulary[c]] += 1
        return features

bow = CharBagOfWords()
bow.build_vocabulary(train_sentences)
bow_features = bow.get_features(train_sentences)

# --- Normalizari ---
conv_norm_l1 = normalize(train_conv_features, norm='l1')
conv_norm_l2 = normalize(train_conv_features, norm='l2')
bow_norm = normalize(bow_features, norm='l2')

# --- One-hot ---
num_classes = len(np.unique(train_labels))
y_one_hot = np.zeros((len(train_labels), num_classes))
for i in range(len(train_labels)):
    y_one_hot[i, int(train_labels[i])] = 1

# --- Split ---
np.random.seed(42)
indices = np.arange(len(train_labels))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]

# --- Helpers ---
def intersection_kernel(X, Y):
    n1, n2 = X.shape[0], Y.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = np.sum(np.minimum(X[i], Y[j]))
    return K

raport = []
raport.append("=" * 70)
raport.append("RAPORT EXPERIMENTAL - Varianta 2")
raport.append("=" * 70)

# ======= EXPERIMENT 1: Ridge Regression =======
raport.append("\n--- EXP 1: Ridge Regression (Char BoW) ---")
raport.append(f"{'Alpha':<12} {'Acc validare':<15}")

best_acc1, best_cfg1 = 0, ""
for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    X_tr, X_val = bow_norm[train_idx], bow_norm[val_idx]
    n_feat = X_tr.shape[1]
    W = np.linalg.solve(X_tr.T.dot(X_tr) + alpha * np.eye(n_feat), X_tr.T.dot(y_one_hot[train_idx]))
    pred = np.argmax(X_val.dot(W), axis=1)
    acc = accuracy_score(train_labels[val_idx], pred)
    line = f"{alpha:<12.4f} {acc*100:.2f}%"
    raport.append(line)
    print(line)
    if acc > best_acc1:
        best_acc1, best_cfg1 = acc, f"alpha={alpha}"

raport.append(f"Cel mai bun: {best_cfg1} ({best_acc1*100:.2f}%)")

# ======= EXPERIMENT 2: SVM RBF =======
raport.append("\n--- EXP 2: SVM RBF (Conv features) ---")
raport.append(f"{'C':<8} {'Gamma':<10} {'Norm':<6} {'Acc validare':<15}")

best_acc2, best_cfg2 = 0, ""
for C in [1.0, 10.0, 50.0, 100.0]:
    for gamma in ['scale', 0.01, 0.1]:
        for norm_name, feats in [('l2', conv_norm_l2)]:
            model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
            model.fit(feats[train_idx], train_labels[train_idx])
            pred = model.predict(feats[val_idx])
            acc = accuracy_score(train_labels[val_idx], pred)
            line = f"{C:<8.1f} {str(gamma):<10} {norm_name:<6} {acc*100:.2f}%"
            raport.append(line)
            print(line)
            if acc > best_acc2:
                best_acc2, best_cfg2 = acc, f"C={C}, gamma={gamma}, norm={norm_name}"

raport.append(f"Cel mai bun: {best_cfg2} ({best_acc2*100:.2f}%)")

# ======= EXPERIMENT 3: Kernel Ridge Intersectie =======
raport.append("\n--- EXP 3: Kernel Ridge + Intersectie ---")
raport.append(f"{'Lambda':<12} {'Acc validare':<15}")

best_acc3, best_cfg3 = 0, ""
print("Calculam kernel intersectie pe subset...")
K_sub_tr = intersection_kernel(conv_norm_l1[train_idx], conv_norm_l1[train_idx])
K_sub_val = intersection_kernel(conv_norm_l1[val_idx], conv_norm_l1[train_idx])

for lambd in [0.001, 0.01, 0.1, 1.0, 10.0]:
    n = K_sub_tr.shape[0]
    alpha_c = np.linalg.solve(K_sub_tr + lambd * np.eye(n), y_one_hot[train_idx])
    pred = np.argmax(K_sub_val.dot(alpha_c), axis=1)
    acc = accuracy_score(train_labels[val_idx], pred)
    line = f"{lambd:<12.3f} {acc*100:.2f}%"
    raport.append(line)
    print(line)
    if acc > best_acc3:
        best_acc3, best_cfg3 = acc, f"lambda={lambd}"

raport.append(f"Cel mai bun: {best_cfg3} ({best_acc3*100:.2f}%)")

# ======= SUMAR =======
raport.append("\n" + "=" * 70)
raport.append("SUMAR")
raport.append("=" * 70)
raport.append(f"Ridge Regression:          {best_acc1*100:.2f}% ({best_cfg1})")
raport.append(f"SVM RBF:                   {best_acc2*100:.2f}% ({best_cfg2})")
raport.append(f"Kernel Ridge Intersectie:  {best_acc3*100:.2f}% ({best_cfg3})")
raport.append("")
raport.append("Observatii:")
raport.append("- Ridge cu BoW caracter captureaza doar frecvente individuale.")
raport.append("- SVM RBF pe features convolutie capteaza n-grams, performanta mai buna.")
raport.append("- Kernel Ridge cu intersectie e potrivit pt features de tip count.")

raport_text = "\n".join(raport)
print("\n" + raport_text)

with open('raport_experimente.txt', 'w', encoding='utf-8') as f:
    f.write(raport_text)
print("\nRaport salvat in raport_experimente.txt")
