import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# ================================================================
# EXERCITIUL 1 (2024 V3) - Retea neuronala, SGD lr constant = 1e-3
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

def normalize_length(signal, target_len):
    if len(signal) >= target_len:
        return signal[:target_len]
    padding = np.zeros((target_len - len(signal), signal.shape[1]))
    return np.vstack([signal, padding])

# Incarcare
train_signals, train_files, train_labels = load_dataset('data/train', 'data/train.txt')
test_signals, test_files, _ = load_dataset('data/test')

# Normalizare lungime + flatten + standardizare
FIXED_LEN = int(np.median([len(s) for s in train_signals]))
train_data = np.array([normalize_length(s, FIXED_LEN).flatten() for s in train_signals])
test_data = np.array([normalize_length(s, FIXED_LEN).flatten() for s in test_signals])

mean, std = train_data.mean(axis=0), train_data.std(axis=0)
std[std == 0] = 1
train_scaled = (train_data - mean) / std
test_scaled = (test_data - mean) / std

# Retea
num_features = train_scaled.shape[1]
num_classes = len(np.unique(train_labels))

class FeedForwardNet(nn.Module):
    def __init__(self, inp, h1, h2, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, h1), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h2, out)
        )
    def forward(self, x):
        return self.net(x)

X_train = torch.FloatTensor(train_scaled)
y_train = torch.LongTensor(train_labels)
X_test = torch.FloatTensor(test_scaled)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FeedForwardNet(num_features, 256, 128, num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # lr constant 10^-3
loss_fn = nn.CrossEntropyLoss()

# Antrenare
for epoch in range(100):
    model.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        loss = loss_fn(model(bx), by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(X_train.to(device)).argmax(1).cpu().numpy() == train_labels).mean()
        print(f"Epoca {epoch+1}: acc={acc*100:.1f}%")

# Predictii
model.eval()
with torch.no_grad():
    test_pred = model(X_test.to(device)).argmax(1).cpu().numpy()

with open('subiect1_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fname, pred in zip(test_files, test_pred):
        f.write(f"{fname},{pred}\n")
print(f"Salvat: {len(test_pred)} predictii")
