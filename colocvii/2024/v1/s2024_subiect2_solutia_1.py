import numpy as np
import os

# ================================================================
# EXERCITIUL 2 - Matrice de tranzitie Markov
# ================================================================
# IDEEA GENERALA:
#   Un semnal accelerometru are valori continue (ex: 5.2, 4.1, ...).
#   Vrem sa capturez "pattern-ul" de miscare, adica cum se schimba
#   valorile in timp.
#
#   Pasii:
#   1. DISCRETIZAM: impartim range-ul valorilor in k intervale egale
#      si inlocuim fiecare valoare cu indexul intervalului sau.
#      Ex: [5.2, 4.1, 3.1] cu 4 intervale -> [2, 2, 1]
#
#   2. CONSTRUIM MATRICEA DE TRANZITIE: numarim de cate ori se trece
#      din intervalul i in intervalul j (valori consecutive).
#      Ex: [2, 2, 1, 3] -> tranzitii: 2->2, 2->1, 1->3
#          Matricea[2][2] += 1, Matricea[2][1] += 1, Matricea[1][3] += 1
#
#   3. NORMALIZAM: impartim fiecare rand la suma lui -> probabilitati.
#      Matricea[i][j] = probabilitatea de a trece din starea i in starea j.
#
#   4. LINIARIZAM: facem matricea un vector si concatenam cele 3 axe.
#      Vector final = [matrice_x_flatten, matrice_y_flatten, matrice_z_flatten]
#
#   Rezultat: un vector de 3 * k * k features per semnal.


# ================================================================
# PASUL 1: INCARCAREA DATELOR
# ================================================================

def load_signal(filepath):
    """Incarca un semnal din fisier (matrice num_timestamps x 3)."""
    return np.loadtxt(filepath)


def load_dataset(data_dir, labels_file=None):
    """Incarca semnalele si etichetele."""
    signals = []
    filenames = []
    labels = []

    if labels_file is not None:
        with open(labels_file, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:  # sarim header
            line = line.strip()
            if line:
                parts = line.split(',')
                fname = parts[0]
                label = int(parts[1])
                signal = load_signal(os.path.join(data_dir, fname))
                signals.append(signal)
                filenames.append(fname)
                labels.append(label)
        return signals, filenames, np.array(labels)
    else:
        with open(os.path.join('data', 'test.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                signal = load_signal(os.path.join(data_dir, line))
                signals.append(signal)
                filenames.append(line)
        return signals, filenames, None


print("Incarcam datele...")
train_signals, train_files, train_labels = load_dataset('data/train', 'data/train.txt')
test_signals, test_files, _ = load_dataset('data/test')
print(f"Train: {len(train_signals)}, Test: {len(test_signals)}")


# ================================================================
# PASUL 2: GASIREA INTERVALELOR PENTRU DISCRETIZARE
# ================================================================
# Trebuie sa stim min si max pe FIECARE AXA din TOATE semnalele de train.
# Apoi impartim [min, max] in k intervale egale.
# IMPORTANT: folosim doar datele de train pentru a calcula intervalele!

def find_axis_ranges(signals):
    """
    Gaseste min si max pe fiecare axa din toate semnalele.
    Returneaza: lista de (min, max) per axa.
    """
    all_values = np.vstack(signals)  # concatenam toate semnalele
    # all_values are shape (total_timestamps x 3)

    ranges = []
    for axis in range(3):
        axis_min = all_values[:, axis].min()
        axis_max = all_values[:, axis].max()
        ranges.append((axis_min, axis_max))

    return ranges

axis_ranges = find_axis_ranges(train_signals)
for i, (a_min, a_max) in enumerate(axis_ranges):
    print(f"Axa {['x','y','z'][i]}: min={a_min:.4f}, max={a_max:.4f}")


# ================================================================
# PASUL 3: DISCRETIZAREA
# ================================================================

def discretize_signal(signal, axis_ranges, k):
    """
    Discretizeaza un semnal: inlocuieste fiecare valoare cu indexul
    intervalului in care cade.

    signal: matrice (num_timestamps x 3)
    axis_ranges: lista de (min, max) per axa
    k: numarul de intervale

    Returneaza: matrice (num_timestamps x 3) cu valori intre 0 si k-1
    """
    discretized = np.zeros_like(signal, dtype=int)

    for axis in range(3):
        a_min, a_max = axis_ranges[axis]

        # Cream k intervale egale intre min si max
        # np.linspace creeaza k+1 capete -> k intervale
        bins = np.linspace(a_min, a_max, k + 1)

        # np.digitize: gaseste in ce interval cade fiecare valoare
        # Returneaza indici de la 1 la k, scadem 1 sa fie de la 0 la k-1
        indices = np.digitize(signal[:, axis], bins) - 1

        # Clipuim la [0, k-1] (pentru valorile exact pe margine)
        indices = np.clip(indices, 0, k - 1)

        discretized[:, axis] = indices

    return discretized


# ================================================================
# PASUL 4: CALCULUL MATRICEI DE TRANZITIE
# ================================================================

def compute_transition_matrix(discretized_axis, k):
    """
    Calculeaza matricea de tranzitie pentru o singura axa.

    discretized_axis: vector de indici (0 la k-1)
    k: numarul de intervale (stari)

    Returneaza: matrice k x k normalizata (probabilitati pe rand)

    Exemplu:
      signal = [2, 2, 1, 3, 3, 0, 2, 0, 2, 2]
      Tranzitii: 2->2, 2->1, 1->3, 3->3, 3->0, 0->2, 2->0, 0->2, 2->2
      Matricea[2][2] += 1, Matricea[2][1] += 1, etc.
    """
    # Matricea nenormalizata (numaratoare)
    A = np.zeros((k, k))

    # Parcurgem perechi consecutive
    for t in range(len(discretized_axis) - 1):
        stare_curenta = discretized_axis[t]
        stare_urmatoare = discretized_axis[t + 1]
        A[stare_curenta][stare_urmatoare] += 1

    # Normalizam: fiecare rand se imparte la suma lui -> probabilitati
    row_sums = A.sum(axis=1, keepdims=True)
    # Evitam impartirea la 0 (daca o stare nu apare niciodata)
    row_sums[row_sums == 0] = 1
    A_normalized = A / row_sums

    return A_normalized


# ================================================================
# PASUL 5: EXTRAGEREA FEATURES (matrici de tranzitie concatenate)
# ================================================================

def extract_markov_features(signal, axis_ranges, k):
    """
    Extrage features Markov dintr-un semnal.

    Pasii:
    1. Discretizeaza semnalul pe fiecare axa
    2. Calculeaza matricea de tranzitie per axa
    3. Liniarizeaza si concateneaza cele 3 matrici

    Returneaza: vector de 3 * k * k features
    """
    discretized = discretize_signal(signal, axis_ranges, k)

    features = []
    for axis in range(3):
        # Matricea de tranzitie pentru axa curenta
        A = compute_transition_matrix(discretized[:, axis], k)
        # Liniarizam (flatten) si adaugam
        features.extend(A.flatten())

    return np.array(features)


# ================================================================
# PASUL 6: CALCULAM FEATURES PENTRU TOATE SEMNALELE
# ================================================================
k = 6  # numarul de intervale (cerut in enunt)
print(f"\nCalculam features Markov cu k={k}...")

train_markov = np.array([extract_markov_features(s, axis_ranges, k)
                         for s in train_signals])
test_markov = np.array([extract_markov_features(s, axis_ranges, k)
                        for s in test_signals])

print(f"Train Markov shape: {train_markov.shape}")  # (1000, 3*6*6) = (1000, 108)
print(f"Test Markov shape: {test_markov.shape}")

# Salvam pentru exercitiile 3 si 4
np.save('train_markov_features.npy', train_markov)
np.save('test_markov_features.npy', test_markov)
np.save('train_labels_saved.npy', train_labels)

# Salvam si filenames pentru formatul de output
with open('train_filenames.txt', 'w') as f:
    for fname in train_files:
        f.write(fname + '\n')
with open('test_filenames.txt', 'w') as f:
    for fname in test_files:
        f.write(fname + '\n')

print("Features Markov salvate.")
