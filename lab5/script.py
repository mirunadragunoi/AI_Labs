import numpy as np
from skimage.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle

# load training data
training_data = np.load('data/training_data.npy')
prices = np.load('data/prices.npy')
# print the first 4 samples
print('The first 4 samples are:\n ', training_data[:4])
print('The first 4 prices are:\n ', prices[:4])
# shuffle
training_data, prices = shuffle(training_data, prices, random_state=0)


# ex 1 --->> normalizare
def normalizare(train_date, test_date):
    media = np.mean(train_date, axis=0)
    deviatia = np.std(train_date, axis=0)

    deviatia[deviatia == 0] = 1

    train_normalizat = (train_date - media) / deviatia
    test_normalizat = (test_date - media) / deviatia

    return train_normalizat, test_normalizat


# ex 2 -->> model regresie liniara cu validarea incrucisata cu 3 flod uri
# valoarea medie a functiilor MSE su MAE

# regresie liniara cu cross-validation cu 3 flod uri

nr_folds = 3
lg_fold = len(training_data) // nr_folds

mse = []
mae = []

for fold in range(nr_folds):
    # tre sa facem split in train si validation
    start = fold * lg_fold
    end = (fold + 1) * lg_fold

    # pt validare -->> fold curent
    validare_data = training_data[start:end]
    validare_preturi = prices[start:end]

    # pt train --->> restul
    train_date = np.concatenate([training_data[:start], training_data[end:]])
    train_preturi = np.concatenate([prices[:start], prices[end:]])

    # normalizam
    date_normalizate, preturi_normalizate = normalizare(train_date, validare_data)

    # antrenare
    model = LinearRegression()
    model.fit(date_normalizate, train_preturi)

    # predictie
    pred = model.predict(preturi_normalizate)

    # metrici
    mse_val = mean_squared_error(validare_preturi, pred)
    mae_val = mean_absolute_error(validare_preturi, pred)
    mse.append(mse_val)
    mae.append(mae_val)
    print(f"fold {fold + 1} -->> MSE = {mse_val:.4f}, MAE = {mae_val:.4f}")

print(f"media MSE -->> {np.mean(mse):.4f}")
print(f"media MAE -->> {np.mean(mae):.4f}")

# ex 3 -->> regresie ridge tot cu cross validation cu 3 flod uri
# valoarea medie a fct MSE + MAE
# a 1 10 100 1000 care val are performanta mai buna

valori_a = [1, 10, 100, 1000]
best_a = 0
best_mse = float('inf')

for a in valori_a:
    mse_lista = []
    mae_lista = []

    for fold in range(nr_folds):
        # tre sa facem split in train si validation
        start = fold * lg_fold
        end = (fold + 1) * lg_fold

        # pt validare -->> fold curent
        validare_data = training_data[start:end]
        validare_preturi = prices[start:end]

        # pt train --->> restul
        train_date = np.concatenate([training_data[:start], training_data[end:]])
        train_preturi = np.concatenate([prices[:start], prices[end:]])

        # normalizam
        date_normalizate, preturi_normalizate = normalizare(train_date, validare_data)

        # antrenare
        model = Ridge(alpha=a)
        model.fit(date_normalizate, train_preturi)

        # predictie
        pred = model.predict(preturi_normalizate)

        # metrici
        mse_val = mean_squared_error(validare_preturi, pred)
        mae_val = mean_absolute_error(validare_preturi, pred)
        mse_lista.append(mse_val)
        mae_lista.append(mae_val)

    medie_mse = np.mean(mse_lista)
    medie_mae = np.mean(mae_lista)
    print(f"Alpha = {a:4d} -> MSE mediu: {medie_mse:.4f}, MAE mediu: {medie_mae:.4f}")

    if medie_mse < best_mse:
        best_mse = medie_mse
        best_a = a

print(f"cel mai bun alpha: {best_a}")

# ex 4 --->> antrenati un model de regresie ridge pe intreaga multime de antrenare
# afisati coef si bias regresie
# cel mai semnificativ atribut??? al doilea cel mai semn atribut??? cel mai putin semnificativ??

# normalizez pe tot setul train si test adica
media = np.mean(training_data, axis=0)
deviatie = np.std(training_data, axis=0)
deviatie[deviatie == 0] = 1
data_normalizata = (training_data - media) / deviatie

model_final = Ridge(alpha=best_a)
model_final.fit(data_normalizata, prices)

atribute = [
    'anul fabricatiei', 'nr kilometri', 'mileage', 'motor', 'putere',
    'nr locuri', 'nr proprietari',
    'combustibil_1', 'combustibil_2', 'combustibil_3', 'combustibil_4', 'combustibil_5',
    'transmisie_1', 'transmisie_2'
]

print(f"bias: {model_final.intercept_:.4f}")
print("coeficienti:")

for i, (nume, coeficient) in enumerate(zip(atribute, model_final.coef_)):
    print(f"   --->>> {i + 1:2d}. {nume:20s} --->> {coeficient:.4f}")

# cel mai semnificativ coeficient ->> cel cu valoarea absoluta cea mai mare
coeficient_abs = np.abs(model_final.coef_)
indice_maxim = np.argmax(coeficient_abs)
indice_minim = np.argmin(coeficient_abs)

# al doilea cel mai semnificativ
coeficient_abs_2 = coeficient_abs.copy()
coeficient_abs_2[indice_maxim] = -1
index_2 = np.argmax(coeficient_abs_2)

print(f"cel mai semnificativ atribut --->> {atribute[indice_maxim]} (coef={model_final.coef_[indice_maxim]:.4f})")
print(f"al doilea cel mai semnificativ --->> {atribute[index_2]} (coef={model_final.coef_[index_2]:.4f})")
print(f"cel mai putin semnificativ atribut: {atribute[indice_minim]} (coef={model_final.coef_[indice_minim]:.4f})")