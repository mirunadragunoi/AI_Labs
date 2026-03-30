import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

train_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# a) 1 strat ascuns, 1 neuron, tanh, lr=1e-2
class Net_a(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28 * 28, 1)      # 1 singur neuron
        self.output = nn.Linear(1, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = (x - 0.1307) / 0.3081                # normalizare
        x = torch.tanh(self.hidden(x))            # tanh
        x = self.output(x)
        return x


# b) 1 strat ascuns, 10 neuroni, tanh, lr=1e-2
class Net_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28 * 28, 10)
        self.output = nn.Linear(10, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = (x - 0.1307) / 0.3081
        x = torch.tanh(self.hidden(x))
        x = self.output(x)
        return x


# c) 1 strat ascuns, 10 neuroni, tanh, lr=1e-5
Net_c = Net_b


# d) 1 strat ascuns, 10 neuroni, tanh, lr=10
Net_d = Net_b


# e) 2 straturi ascunse, 10 neuroni fiecare, tanh, lr=1e-2
class Net_e(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(28 * 28, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = (x - 0.1307) / 0.3081
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = self.output(x)
        return x


# f) 2 straturi ascunse, 10 neuroni fiecare, relu, lr=1e-2
class Net_f(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(28 * 28, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = (x - 0.1307) / 0.3081
        x = F.relu(self.hidden1(x))               # relu in loc de tanh
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x


# g) 2 straturi ascunse, 100 neuroni fiecare, relu, lr=1e-2
class Net_g(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(28 * 28, 100)    # 100 neuroni
        self.hidden2 = nn.Linear(100, 100)
        self.output = nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = (x - 0.1307) / 0.3081
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x


# h) 2 straturi ascunse, 100 neuroni fiecare, relu, lr=1e-2, momentum=0.9
Net_h = Net_g

def train_model(model, optimizer, train_dataloader, num_epochs=5):
    """ antreneaza modelul """
    loss_function = nn.CrossEntropyLoss()
    model = model.to(device)
    model.train(True)

    for i in range(num_epochs):
        running_loss = 0.0
        for batch, (image_batch, labels_batch) in enumerate(train_dataloader):
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)

            pred = model(image_batch)
            loss = loss_function(pred, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch % 100 == 0:
                print(f"  Epoch {i + 1}, Batch {batch}, loss: {loss.item():.4f}")

        print(f"  Epoch {i + 1} terminata, loss mediu: {running_loss / len(train_dataloader):.4f}")


def test_model(model, test_dataloader):
    """ testeaza modelul"""
    loss_function = nn.CrossEntropyLoss()
    correct = 0.0
    test_loss = 0.0
    size = len(test_dataloader.dataset)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for image_batch, labels_batch in test_dataloader:
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)

            pred = model(image_batch)
            test_loss += loss_function(pred, labels_batch).item()
            correct += (pred.argmax(1) == labels_batch).type(torch.float).sum().item()

    correct /= size
    test_loss /= size
    print(f"  Accuracy: {(100 * correct):.1f}%, Loss: {test_loss:.6f}")
    return correct

# fiecare configuratie: creez reteaua, creez optimizatorul, antrenez, testez

configs = [
    # (nume, clasa_retea, lr, momentum)
    ("a) 1 strat, 1 neuron, tanh, lr=1e-2", Net_a, 1e-2, 0),
    ("b) 1 strat, 10 neuroni, tanh, lr=1e-2", Net_b, 1e-2, 0),
    ("c) 1 strat, 10 neuroni, tanh, lr=1e-5", Net_c, 1e-5, 0),
    ("d) 1 strat, 10 neuroni, tanh, lr=10", Net_d, 10,   0),
    ("e) 2 straturi, 10 neuroni, tanh, lr=1e-2", Net_e, 1e-2, 0),
    ("f) 2 straturi, 10 neuroni, relu, lr=1e-2", Net_f, 1e-2, 0),
    ("g) 2 straturi, 100 neuroni, relu, lr=1e-2", Net_g, 1e-2, 0),
    ("h) 2 straturi, 100 neuroni, relu, lr=1e-2, momentum=0.9", Net_h, 1e-2, 0.9),
]

rezultate = []

for nume, net_class, lr, momentum in configs:
    print(f"configuratia {nume}")

    model = net_class()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    train_model(model, optimizer, train_dataloader, num_epochs=5)
    acc = test_model(model, test_dataloader)
    rezultate.append((nume, acc))

print("SUMAR REZULTATE")
for nume, acc in rezultate:
    print(f"  {nume:55s} -> {acc * 100:.1f}%")