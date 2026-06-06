import numpy as np
import os

# ================================================================
# EXERCITIUL 2 (2024 V4) - Matrice de tranzitie Markov cu k=5
# ================================================================
# V4 specific: k=5 -> 3 axe * 5 * 5 = 75 features per semnal

def load_signal(fp): return np.loadtxt(fp)

def load_dataset(data_dir, labels_file=None):
    signals, fnames, labels = [], [], []
    if labels_file:
        with open(labels_file) as f: lines = f.readlines()
        for l in lines[1:]:
            l = l.strip()
            if l:
                p = l.split(',')
                signals.append(load_signal(os.path.join(data_dir, p[0])))
                fnames.append(p[0]); labels.append(int(p[1]))
        return signals, fnames, np.array(labels)
    else:
        with open(os.path.join('data','test.txt')) as f: lines = f.readlines()
        for l in lines:
            l = l.strip()
            if l:
                signals.append(load_signal(os.path.join(data_dir, l)))
                fnames.append(l)
        return signals, fnames, None

train_signals, train_files, train_labels = load_dataset('data/train', 'data/train.txt')
test_signals, test_files, _ = load_dataset('data/test')

# Range per axa din train
all_vals = np.vstack(train_signals)
axis_ranges = [(all_vals[:, i].min(), all_vals[:, i].max()) for i in range(3)]

def discretize(signal, axis_ranges, k):
    disc = np.zeros_like(signal, dtype=int)
    for ax in range(3):
        bins = np.linspace(axis_ranges[ax][0], axis_ranges[ax][1], k + 1)
        disc[:, ax] = np.clip(np.digitize(signal[:, ax], bins) - 1, 0, k - 1)
    return disc

def transition_matrix(axis_vals, k):
    A = np.zeros((k, k))
    for t in range(len(axis_vals) - 1):
        A[axis_vals[t]][axis_vals[t+1]] += 1
    sums = A.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    return A / sums

def markov_features(signal, axis_ranges, k):
    disc = discretize(signal, axis_ranges, k)
    feats = []
    for ax in range(3):
        feats.extend(transition_matrix(disc[:, ax], k).flatten())
    return np.array(feats)

# k=5 conform cerintei V4
k = 5
print(f"Calculam Markov features cu k={k}...")
train_markov = np.array([markov_features(s, axis_ranges, k) for s in train_signals])
test_markov = np.array([markov_features(s, axis_ranges, k) for s in test_signals])
print(f"Train: {train_markov.shape}, Test: {test_markov.shape}")  # (1000, 75)

np.save('train_markov_features.npy', train_markov)
np.save('test_markov_features.npy', test_markov)
np.save('train_labels_saved.npy', train_labels)
with open('test_filenames.txt', 'w') as f:
    for fn in test_files: f.write(fn + '\n')
print("Features salvate.")
