import numpy as np

# ================================================================
# EXERCITIUL 2 (V2) - Convolutie cu 3-grams
# ================================================================
# IDENTIC cu Varianta 1 - extrage features de convolutie.
# Aceste features sunt folosite de exercitiile 3 si 4.
#
# CE FACE:
#   Pentru fiecare document si fiecare 3-gram (din 500):
#     1. Convertim textul si 3-gram-ul in numere (folosind mapping.txt)
#     2. Glisam 3-gram-ul peste document (sliding window)
#     3. La fiecare pozitie calculam cosinus similarity (produs scalar normalizat)
#     4. Numaram cate valori > 0.9 (pragul t)
#   Rezultat: vector de 500 features per document


# ================================================================
# INCARCAREA DATELOR
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

# 3-grams (500 de filtre)
trigrams = []
with open('words.txt', 'r', encoding='utf-8') as f:
    for line in f:
        trigrams.append(line.strip())

print(f"Numar de 3-grams: {len(trigrams)}")

# Mapping caracter -> numar
char_to_num = {}
with open('mapping.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            if line.startswith(',,'):
                char_to_num[','] = int(line[2:])
            else:
                parts = line.split(',')
                if len(parts) == 2:
                    char_to_num[parts[0]] = int(parts[1])

print(f"Dimensiune mapping: {len(char_to_num)} caractere")


# ================================================================
# FUNCTII
# ================================================================

def text_to_numbers(text, mapping):
    """Converteste text in vector numeric folosind mapping-ul."""
    numbers = []
    for char in text:
        if char in mapping:
            numbers.append(mapping[char])
        else:
            numbers.append(0)
    return np.array(numbers, dtype=float)


def convolution_1d(document_nums, filter_nums):
    """
    Convolutie 1D: gliseaza filtrul peste document, calculeaza
    cosinus similarity la fiecare pozitie.

    Cosinus similarity = (a · b) / (||a|| * ||b||)
    Valoare intre -1 si 1. 1 = identice, 0 = ortogonale.
    """
    L = len(document_nums)
    n = len(filter_nums)

    if L < n:
        return np.array([])

    result = np.zeros(L - n + 1)
    filter_norm = np.linalg.norm(filter_nums)

    if filter_norm == 0:
        return result

    for i in range(L - n + 1):
        subvector = document_nums[i: i + n]
        sub_norm = np.linalg.norm(subvector)

        if sub_norm > 0:
            result[i] = np.dot(subvector, filter_nums) / (sub_norm * filter_norm)
        else:
            result[i] = 0

    return result


def extract_convolution_features(sentences, trigrams, mapping, threshold=0.9):
    """
    Pentru fiecare propozitie si fiecare 3-gram:
      - aplica convolutia
      - numara valorile > threshold
    Returneaza matrice (num_propozitii x 500)
    """
    num_samples = len(sentences)
    num_filters = len(trigrams)
    features = np.zeros((num_samples, num_filters))

    trigram_nums = [text_to_numbers(tg, mapping) for tg in trigrams]

    for i, sentence in enumerate(sentences):
        doc_nums = text_to_numbers(sentence, mapping)

        for k, tg_nums in enumerate(trigram_nums):
            conv_result = convolution_1d(doc_nums, tg_nums)
            if len(conv_result) > 0:
                features[i][k] = np.sum(conv_result > threshold)

        if (i + 1) % 100 == 0:
            print(f"  Procesate {i + 1}/{num_samples} propozitii...")

    return features


# ================================================================
# CALCULAM FEATURES
# ================================================================
print("\nCalculam features de convolutie pentru train...")
train_conv_features = extract_convolution_features(
    train_sentences, trigrams, char_to_num, threshold=0.9
)

print("Calculam features de convolutie pentru test...")
test_conv_features = extract_convolution_features(
    test_sentences, trigrams, char_to_num, threshold=0.9
)

print(f"\nTrain features: {train_conv_features.shape}")
print(f"Test features: {test_conv_features.shape}")

# Salvam pentru exercitiile 3 si 4
np.save('train_conv_features.npy', train_conv_features)
np.save('test_conv_features.npy', test_conv_features)
print("Features salvate.")
