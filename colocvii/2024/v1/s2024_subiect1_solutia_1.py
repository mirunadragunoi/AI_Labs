import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# ================================================================
# EXERCITIUL 1 - Retea neuronala feed-forward cu Adam
# ================================================================
# IDEEA:
#   Citim semnalele accelerometru (3 axe: x, y, z).
#   Le aducem la aceeasi lungime (padding/truncare).
#   Le dam la o retea neuronala cu max 3 straturi.
#   Optimizator: Adam (mai bun decat SGD, adapteaza rata de invatare).


# ================================================================
# PASUL 1: INCARCAREA DATELOR
# ================================================================
# train.txt contine: filename,label (prima linie e header)
# Fiecare fisier .txt din data/train/ are 3 coloane: x, y, z

def load_signal(filepath):
    """Incarca un semnal din fisier. Returneaza matrice (num_timestamps x 3)."""
    return np.loadtxt(filepath)


def load_dataset(data_dir, labels_file=None):
    """
    Incarca toate semnalele dintr-un director.

    Returneaza:
      signals: lista de matrice (fiecare de dimensiune variabila)
      filenames: lista de nume de fisiere
      labels: array de etichete (sau None pentru test)
    """
    signals = []
    filenames = []
    labels = []

    if labels_file is not None:
        # Citim train.txt: header pe prima linie, apoi filename,label
        with open(labels_file, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:  # sarim header-ul
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
        # Citim test.txt: doar filenames
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

print(f"Train: {len(train_signals)} semnale")
print(f"Test: {len(test_signals)} semnale")
print(f"Clase: {np.unique(train_labels)}")


# ================================================================
# PASUL 2: NORMALIZAREA LUNGIMII SEMNALELOR
# ================================================================
# Semnalele au lungimi diferite (frecventa de raportare variabila).
# Trebuie sa le aducem la aceeasi lungime:
#   - Semnale prea lungi: taiem de la final (truncare)
#   - Semnale prea scurte: adaugam zerouri la final (padding)
#
# Alegem o lungime fixa (ex: mediana lungimilor sau o valoare rotunda).

lengths = [len(s) for s in train_signals]
print(f"Lungimi semnale: min={min(lengths)}, max={max(lengths)}, "
      f"medie={np.mean(lengths):.0f}, mediana={np.median(lengths):.0f}")

# Folosim o lungime fixa
FIXED_LEN = int(np.median(lengths))
print(f"Lungime fixa aleasa: {FIXED_LEN}")


def normalize_length(signal, target_len):
    """
    Aduce un semnal la lungimea dorita.
    Prea lung -> taiem. Prea scurt -> adaugam zerouri.
    """
    current_len = len(signal)
    if current_len >= target_len:
        return signal[:target_len]  # truncam
    else:
        # Padding cu zerouri
        padding = np.zeros((target_len - current_len, signal.shape[1]))
        return np.vstack([signal, padding])


# Aplicam pe toate semnalele si le facem vectori (flatten)
# Fiecare semnal: (FIXED_LEN x 3) -> flatten -> vector de FIXED_LEN * 3
train_data = np.array([normalize_length(s, FIXED_LEN).flatten() for s in train_signals])
test_data = np.array([normalize_length(s, FIXED_LEN).flatten() for s in test_signals])

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")


# ================================================================
# PASUL 3: STANDARDIZARE (z-score)
# ================================================================
# Centram datele (medie=0, deviatie=1) pentru antrenare mai buna
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
std[std == 0] = 1  # evitam impartirea la 0

train_scaled = (train_data - mean) / std
test_scaled = (test_data - mean) / std


# ================================================================
# PASUL 4: DEFINIREA RETELEI NEURONALE (PyTorch)
# ================================================================
# Retea feed-forward cu maxim 3 straturi ascunse.
# Activare: ReLU (simpla si eficienta).
# Stratul de iesire: 4 neuroni (4 clase), fara activare (CrossEntropy include softmax).

num_features = train_scaled.shape[1]
num_classes = len(np.unique(train_labels))


class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),         # previne overfitting
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ================================================================
# PASUL 5: ANTRENAREA RETELEI
# ================================================================
# Convertim datele in tensori PyTorch
X_train = torch.FloatTensor(train_scaled)
y_train = torch.LongTensor(train_labels)
X_test = torch.FloatTensor(test_scaled)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Cream modelul
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FeedForwardNet(num_features, 256, 128, num_classes).to(device)

# Adam: optimizer adaptiv, mai bun decat SGD pentru majoritatea cazurilor
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Antrenare
NUM_EPOCHS = 50
print(f"\nAntrenam reteaua ({NUM_EPOCHS} epoci)...")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        # Calculam acuratete pe train
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train.to(device)).argmax(1).cpu().numpy()
            train_acc = (train_pred == train_labels).mean()
        print(f"  Epoca {epoch+1:3d} -> Loss: {total_loss/len(train_loader):.4f}, "
              f"Acc train: {train_acc*100:.1f}%")


# ================================================================
# PASUL 6: PREDICTII PE TEST
# ================================================================
model.eval()
with torch.no_grad():
    test_pred = model(X_test.to(device)).argmax(1).cpu().numpy()

print(f"\nPredictii generate: {len(test_pred)}")

# Salvam in formatul cerut: filename,label
with open('subiect1_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fname, pred in zip(test_files, test_pred):
        f.write(f"{fname},{pred}\n")

print("Predictii salvate in subiect1_solutia_1.txt")
