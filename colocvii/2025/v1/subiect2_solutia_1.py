import numpy as np

# ================================================================
# EXERCITIUL 2 - Convolutie cu 3-grams
# ================================================================
# IDEEA GENERALA:
#   Avem 500 de "filtre" (3-grams = secvente de 3 caractere).
#   Pentru fiecare document, "glisam" fiecare filtru peste text
#   si calculam cat de similar e filtrul cu fiecare bucata de text.
#
#   E ca si cum ai avea un sablon de 3 litere si il plimbi peste text
#   ca sa vezi unde se potriveste cel mai bine.
#
# PASII:
#   1. Incarcam mapping.txt: fiecare caracter -> un numar
#   2. Convertim textele si 3-grams in vectori numerici folosind mapping-ul
#   3. Pentru fiecare document si fiecare 3-gram:
#      - glisam 3-gram-ul peste document (sliding window)
#      - la fiecare pozitie calculam produsul scalar normalizat
#      - numaram cate valori depasesc pragul t=0.9
#   4. Rezultat: vector de 500 componente per document


# ================================================================
# PASUL 1: INCARCAREA DATELOR
# ================================================================

# Citim propozitiile
train_sentences = []
with open('train_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f:
        train_sentences.append(line.strip())

test_sentences = []
with open('test_sentences.txt', 'r', encoding='utf-8') as f:
    for line in f:
        test_sentences.append(line.strip())

train_labels = np.load('train_labels.npy')

# Citim 3-grams (cate un 3-gram per linie)
trigrams = []
with open('words.txt', 'r', encoding='utf-8') as f:
    for line in f:
        trigrams.append(line.strip())

print(f"Numar de 3-grams: {len(trigrams)}")
print(f"Primele 5 3-grams: {trigrams[:5]}")

# Citim mapping-ul caracter -> numar
# Formatul: caracter,numar pe fiecare linie
char_to_num = {}
with open('mapping.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            # Separator e virgula, dar primul element poate fi virgula insasi
            # Tratam cazul special: daca linia incepe cu "," (virgula e caracterul)
            if line.startswith(',,'):
                # Virgula mapat la numarul de dupa a doua virgula
                char_to_num[','] = int(line[2:])
            else:
                parts = line.split(',')
                if len(parts) == 2:
                    char_to_num[parts[0]] = int(parts[1])

print(f"Dimensiune mapping: {len(char_to_num)} caractere")


# ================================================================
# PASUL 2: CONVERTIRE TEXT -> VECTORI NUMERICI
# ================================================================
# Folosim mapping-ul pentru a inlocui fiecare caracter cu numarul corespunzator.
# Caracterele care nu sunt in mapping le ignoram sau le punem 0.

def text_to_numbers(text, mapping):
    """
    Converteste un text intr-un vector de numere folosind mapping-ul.

    Exemplu: "ana" cu mapping {'a': 37, 'n': 20} -> [37, 20, 37]
    """
    numbers = []
    for char in text:
        if char in mapping:
            numbers.append(mapping[char])
        else:
            numbers.append(0)  # caracter necunoscut -> 0
    return np.array(numbers, dtype=float)


# ================================================================
# PASUL 3: OPERATIA DE CONVOLUTIE
# ================================================================
# Convolutia = glisare filtru peste document + produs scalar normalizat
#
# Pentru un document de L caractere si un filtru de n=3 caractere:
#   - avem L - n + 1 pozitii de glisare
#   - la fiecare pozitie, extragem un subvector de dimensiune n din document
#   - calculam produsul scalar normalizat cu filtrul:
#     cos_sim = (subvector · filtru) / (||subvector|| * ||filtru||)
#   - asta e practic cosinus similarity (similaritate cosinus)
#
# Exemplu vizual:
#   Document: [37, 20, 37, 18, 37, 7, 24, 18, 30, 24, 7, 24]
#   Filtru:   [37, 20, 37]
#
#   Pozitia 0: [37, 20, 37] · [37, 20, 37] / (norme) = 1.0 (identic!)
#   Pozitia 1: [20, 37, 18] · [37, 20, 37] / (norme) = 0.837...
#   Pozitia 2: [37, 18, 37] · [37, 20, 37] / (norme) = ...
#   ... si tot asa

def convolution_1d(document_nums, filter_nums):
    """
    Aplica convolutia 1D intre un document numeric si un filtru.

    document_nums: vector numeric al documentului
    filter_nums: vector numeric al filtrului (3-gram)

    Returneaza: vector de similaritati cosinus (lungime = L - n + 1)
    """
    L = len(document_nums)
    n = len(filter_nums)

    if L < n:
        return np.array([])

    result = np.zeros(L - n + 1)
    filter_norm = np.linalg.norm(filter_nums)  # norma filtrului (constanta)

    # Daca filtrul e zero, returnam zerouri
    if filter_norm == 0:
        return result

    for i in range(L - n + 1):
        # Extragem subvectorul de la pozitia i
        subvector = document_nums[i: i + n]

        # Calculam norma subvectorului
        sub_norm = np.linalg.norm(subvector)

        # Produsul scalar normalizat (cosinus similarity)
        if sub_norm > 0:
            result[i] = np.dot(subvector, filter_nums) / (sub_norm * filter_norm)
        else:
            result[i] = 0

    return result


def extract_convolution_features(sentences, trigrams, mapping, threshold=0.9):
    """
    Extrage features de convolutie pentru o lista de propozitii.

    Pentru fiecare propozitie si fiecare 3-gram:
      1. Converteste ambele in numere
      2. Aplica convolutia
      3. Numara cate valori depasesc pragul threshold

    Returneaza: matrice (num_propozitii x num_trigrams)
                unde features[i][k] = cate valori > threshold
                la convolutia propozitiei i cu 3-gram-ul k
    """
    num_samples = len(sentences)
    num_filters = len(trigrams)
    features = np.zeros((num_samples, num_filters))

    # Pre-convertim toate 3-grams in vectori numerici (se refolosesc)
    trigram_nums = [text_to_numbers(tg, mapping) for tg in trigrams]

    for i, sentence in enumerate(sentences):
        # Convertim propozitia in numere
        doc_nums = text_to_numbers(sentence, mapping)

        for k, tg_nums in enumerate(trigram_nums):
            # Aplicam convolutia
            conv_result = convolution_1d(doc_nums, tg_nums)

            # Numaram valorile care depasesc pragul
            if len(conv_result) > 0:
                features[i][k] = np.sum(conv_result > threshold)

        # Progres
        if (i + 1) % 100 == 0:
            print(f"  Procesate {i + 1}/{num_samples} propozitii...")

    return features


# ================================================================
# PASUL 4: CALCULAM FEATURES DE CONVOLUTIE
# ================================================================
print("\nCalculam features de convolutie pentru train...")
train_conv_features = extract_convolution_features(
    train_sentences, trigrams, char_to_num, threshold=0.9
)

print("Calculam features de convolutie pentru test...")
test_conv_features = extract_convolution_features(
    test_sentences, trigrams, char_to_num, threshold=0.9
)

print(f"\nTrain convolution features shape: {train_conv_features.shape}")
print(f"Test convolution features shape: {test_conv_features.shape}")

# Salvam features pentru a le refolosi in exercitiile 3 si 4
np.save('train_conv_features.npy', train_conv_features)
np.save('test_conv_features.npy', test_conv_features)
print("Features salvate in train_conv_features.npy si test_conv_features.npy")
