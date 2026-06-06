import numpy as np
import os

# ================================================================
# EXERCITIUL 2 (2024 V3) - Matrice de tranzitie Markov cu k=7
# ================================================================
# V3 specific: k=7 -> 3 axe * 7 * 7 = 147 features per semnal

def load_signal(filepath):
    return np.loadtxt(filepath)

def load_dataset(data_dir, labels_file=None):
    signals, filenames, labels = [], [], []
    if labels_file:
        with open(labels_file, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if line:
                parts = line.split(',')
                signals.append(load_signal(os.path.join(data_dir, parts[0])))
                filenames.append(parts[0])
                labels.append(int(parts[1]))
        return signals, filenames, np.array(labels)
    else:
        with open(os.path.join('data', 'test.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                signals.append(load_signal(os.path.join(data_dir, line)))
                filenames.append(line)
        return signals, filenames, None

train_signals, train_files, train_labels = load_dataset('data/train', 'data/train.txt')
test_signals, test_files, _ = load_dataset('data/test')

# Range per axa (doar din train)
all_values = np.vstack(train_signals)
axis_ranges = [(all_values[:, i].min(), all_values[:, i].max()) for i in range(3)]


def discretize_signal(signal, axis_ranges, k):
    """Inlocuieste valori continue cu indici de interval (0..k-1)."""
    discretized = np.zeros_like(signal, dtype=int)
    for axis in range(3):
        bins = np.linspace(axis_ranges[axis][0], axis_ranges[axis][1], k + 1)
        discretized[:, axis] = np.clip(np.digitize(signal[:, axis], bins) - 1, 0, k - 1)
    return discretized


def compute_transition_matrix(axis_values, k):
    """Matricea de tranzitie normalizata: A[i][j] = P(stare_j | stare_i)."""
    A = np.zeros((k, k))
    for t in range(len(axis_values) - 1):
        A[axis_values[t]][axis_values[t + 1]] += 1
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return A / row_sums


def extract_markov_features(signal, axis_ranges, k):
    """3 matrici Markov (una per axa) -> liniarizate si concatenate."""
    discretized = discretize_signal(signal, axis_ranges, k)
    features = []
    for axis in range(3):
        A = compute_transition_matrix(discretized[:, axis], k)
        features.extend(A.flatten())
    return np.array(features)


# k=7 conform cerintei V3
k = 7
print(f"Calculam features Markov cu k={k}...")

train_markov = np.array([extract_markov_features(s, axis_ranges, k) for s in train_signals])
test_markov = np.array([extract_markov_features(s, axis_ranges, k) for s in test_signals])

print(f"Train: {train_markov.shape}")  # (1000, 147)
print(f"Test: {test_markov.shape}")

np.save('train_markov_features.npy', train_markov)
np.save('test_markov_features.npy', test_markov)
np.save('train_labels_saved.npy', train_labels)

with open('test_filenames.txt', 'w') as f:
    for fname in test_files:
        f.write(fname + '\n')
print("Features salvate.")
