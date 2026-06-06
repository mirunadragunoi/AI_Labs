import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# ================================================================
# EXERCITIUL 1 (2024 V4) - Retea neuronala feed-forward cu SGD
# ================================================================
# V4: optimizator SGD (fara lr specificat -> folosim un lr rezonabil)

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

def normalize_length(sig, tgt):
    if len(sig) >= tgt: return sig[:tgt]
    return np.vstack([sig, np.zeros((tgt - len(sig), sig.shape[1]))])

train_signals, train_files, train_labels = load_dataset('data/train', 'data/train.txt')
test_signals, test_files, _ = load_dataset('data/test')

FIXED_LEN = int(np.median([len(s) for s in train_signals]))
train_data = np.array([normalize_length(s, FIXED_LEN).flatten() for s in train_signals])
test_data = np.array([normalize_length(s, FIXED_LEN).flatten() for s in test_signals])

mean, std = train_data.mean(0), train_data.std(0)
std[std == 0] = 1
train_sc, test_sc = (train_data - mean) / std, (test_data - mean) / std

class Net(nn.Module):
    def __init__(self, inp, h1, h2, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, h1), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h2, out))
    def forward(self, x): return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
X_tr = torch.FloatTensor(train_sc)
y_tr = torch.LongTensor(train_labels)
X_te = torch.FloatTensor(test_sc)
loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

model = Net(train_sc.shape[1], 256, 128, len(np.unique(train_labels))).to(device)
# SGD conform cerintei
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        loss = loss_fn(model(bx), by)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (epoch+1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(X_tr.to(device)).argmax(1).cpu().numpy() == train_labels).mean()
        print(f"Epoca {epoch+1}: acc={acc*100:.1f}%")

model.eval()
with torch.no_grad():
    preds = model(X_te.to(device)).argmax(1).cpu().numpy()

with open('subiect1_solutia_1.txt', 'w') as f:
    f.write('filename,label\n')
    for fn, p in zip(test_files, preds): f.write(f"{fn},{p}\n")
print(f"Salvat: {len(preds)} predictii")
