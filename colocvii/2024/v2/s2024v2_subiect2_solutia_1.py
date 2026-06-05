import numpy as np
import os

# ================================================================
# EXERCITIUL 2 (2024 V2) - Matrice de tranzitie Markov cu k=4
# ================================================================
# Identic cu V1, dar k=4 (nu k=6).
# Vector final: 3 axe * 4 * 4 = 48 features per semnal.

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

print("Incarcam datele...")
train_signals, train_files, train_labels = load_dataset('data/train', 'data/train.txt')
test_signals, test_files, _ = load_dataset('data/test')

# Gasim range-ul pe fiecare axa (doar din train!)
all_values = np.vstack(train_signals)
axis_ranges = [(all_values[:, i].min(), all_values[:, i].max()) for i in range(3)]

for i, (a_min, a_max) in enumerate(axis_ranges):
    print(f"Axa {['x','y','z'][i]}: [{a_min:.4f}, {a_max:.4f}]")


def discretize_signal(signal, axis_ranges, k):
    """Inlocuieste fiecare valoare cu indexul intervalului (0 la k-1)."""
    discretized = np.zeros_like(signal, dtype=int)
    for axis in range(3):
        a_min, a_max = axis_ranges[axis]
        bins = np.linspace(a_min, a_max, k + 1)
        indices = np.clip(np.digitize(signal[:, axis], bins) - 1, 0, k - 1)
        discretized[:, axis] = indices
    return discretized


def compute_transition_matrix(discretized_axis, k):
    """Calculeaza matricea de tranzitie normalizata k x k."""
    A = np.zeros((k, k))
    for t in range(len(discretized_axis) - 1):
        A[discretized_axis[t]][discretized_axis[t + 1]] += 1
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return A / row_sums


def extract_markov_features(signal, axis_ranges, k):
    """Extrage features: 3 matrici de tranzitie liniarizate si concatenate."""
    discretized = discretize_signal(signal, axis_ranges, k)
    features = []
    for axis in range(3):
        A = compute_transition_matrix(discretized[:, axis], k)
        features.extend(A.flatten())
    return np.array(features)


# k=4 conform cerintei V2
k = 4
print(f"\nCalculam features Markov cu k={k}...")

train_markov = np.array([extract_markov_features(s, axis_ranges, k) for s in train_signals])
test_markov = np.array([extract_markov_features(s, axis_ranges, k) for s in test_signals])

print(f"Train Markov: {train_markov.shape}")  # (1000, 48)
print(f"Test Markov: {test_markov.shape}")

np.save('train_markov_features.npy', train_markov)
np.save('test_markov_features.npy', test_markov)
np.save('train_labels_saved.npy', train_labels)

with open('test_filenames.txt', 'w') as f:
    for fname in test_files:
        f.write(fname + '\n')

print("Features salvate.")
