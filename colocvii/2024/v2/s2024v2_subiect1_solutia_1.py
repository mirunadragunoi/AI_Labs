import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# ================================================================
# EXERCITIUL 1 (2024 V2) - Retea neuronala, lr constant = 1e-3
# ================================================================
# Diferenta fata de V1: rata de invatare CONSTANTA = 10^-3
# (V1 folosea Adam care adapteaza automat lr-ul)
# Aici folosim SGD cu lr fix = 0.001


# ================================================================
# INCARCAREA DATELOR
# ================================================================

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
print(f"Train: {len(train_signals)}, Test: {len(test_signals)}")


# ================================================================
# NORMALIZARE LUNGIME + STANDARDIZARE
# ================================================================
lengths = [len(s) for s in train_signals]
FIXED_LEN = int(np.median(lengths))

def normalize_length(signal, target_len):
    if len(signal) >= target_len:
        return signal[:target_len]
    padding = np.zeros((target_len - len(signal), signal.shape[1]))
    return np.vstack([signal, padding])

train_data = np.array([normalize_length(s, FIXED_LEN).flatten() for s in train_signals])
test_data = np.array([normalize_length(s, FIXED_LEN).flatten() for s in test_signals])

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
std[std == 0] = 1
train_scaled = (train_data - mean) / std
test_scaled = (test_data - mean) / std

print(f"Features shape: {train_scaled.shape}")


# ================================================================
# RETEAUA NEURONALA
# ================================================================
num_features = train_scaled.shape[1]
num_classes = len(np.unique(train_labels))

class FeedForwardNet(nn.Module):
    def __init__(self, input_size, h1, h2, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h2, num_classes)
        )
    def forward(self, x):
        return self.net(x)


# ================================================================
# ANTRENARE CU LR CONSTANT = 1e-3
# ================================================================
# Cerinta: rata de invatare CONSTANTA 10^-3
# Folosim SGD (nu Adam!) ca sa fie clar ca lr-ul e fix

X_train = torch.FloatTensor(train_scaled)
y_train = torch.LongTensor(train_labels)
X_test = torch.FloatTensor(test_scaled)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FeedForwardNet(num_features, 256, 128, num_classes).to(device)

# SGD cu lr constant = 0.001 (10^-3)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

NUM_EPOCHS = 100  # mai multe epoci pt SGD (converge mai lent decat Adam)
print(f"\nAntrenam cu SGD, lr=1e-3, {NUM_EPOCHS} epoci...")

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

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train.to(device)).argmax(1).cpu().numpy()
            acc = (train_pred == train_labels).mean()
        print(f"  Epoca {epoch+1:3d} -> Loss: {total_loss/len(train_loader):.4f}, Acc: {acc*100:.1f}%")


# ================================================================
# PREDICTII
# ================================================================
model.eval()
with torch.no_grad():
    test_pred = model(X_test.to(device)).argmax(1).cpu().numpy()

with open('subiect1_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fname, pred in zip(test_files, test_pred):
        f.write(f"{fname},{pred}\n")

print(f"\nPredictii salvate: {len(test_pred)} etichete")
