import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

# incarcare date
training_data = np.load('data/training_data.npy')
prices = np.load('data/prices.npy')

print('Primele 4 exemple:\n', training_data[:4])
print('Primele 4 preturi:\n', prices[:4])

# shuffle
training_data, prices = shuffle(training_data, prices, random_state=0)


# ============================================================
# EX 1 - Normalizare
# ============================================================
# Folosim standardizare (z-score): (x - media) / deviatie_standard
# Asta face ca fiecare atribut sa aiba media 0 si deviatia 1
# Important: calculam media si deviatia DOAR pe train, le aplicam si pe test

def normalize(train_data, test_data):
    media = np.mean(train_data, axis=0)
    deviatie = np.std(train_data, axis=0)
    # evitam impartirea la 0
    deviatie[deviatie == 0] = 1
    train_norm = (train_data - media) / deviatie
    test_norm = (test_data - media) / deviatie
    return train_norm, test_norm


# ============================================================
# EX 2 - Regresie liniara cu cross-validation 3 fold-uri
# ============================================================
print("\n" + "=" * 60)
print("EX 2 - Regresie Liniara, 3-fold CV")
print("=" * 60)

num_folds = 3
fold_size = len(training_data) // num_folds

mse_list = []
mae_list = []

for fold in range(num_folds):
    # split in train si validation
    start = fold * fold_size
    end = (fold + 1) * fold_size

    # validation = fold-ul curent
    val_data = training_data[start:end]
    val_prices = prices[start:end]

    # train = restul
    train_data = np.concatenate([training_data[:start], training_data[end:]])
    train_prices = np.concatenate([prices[:start], prices[end:]])

    # normalizare
    train_norm, val_norm = normalize(train_data, val_data)

    # antrenare
    model = LinearRegression()
    model.fit(train_norm, train_prices)

    # predictie
    pred = model.predict(val_norm)

    # metrici
    mse = mean_squared_error(val_prices, pred)
    mae = mean_absolute_error(val_prices, pred)
    mse_list.append(mse)
    mae_list.append(mae)
    print(f"Fold {fold + 1}: MSE = {mse:.4f}, MAE = {mae:.4f}")

print(f"Media MSE: {np.mean(mse_list):.4f}")
print(f"Media MAE: {np.mean(mae_list):.4f}")


# ============================================================
# EX 3 - Regresie Ridge cu cross-validation, alpha in {1,10,100,1000}
# ============================================================
print("\n" + "=" * 60)
print("EX 3 - Regresie Ridge, 3-fold CV")
print("=" * 60)

alphas = [1, 10, 100, 1000]
best_alpha = 0
best_mse = float('inf')

for alpha in alphas:
    mse_list = []
    mae_list = []

    for fold in range(num_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size

        val_data = training_data[start:end]
        val_prices = prices[start:end]
        train_data = np.concatenate([training_data[:start], training_data[end:]])
        train_prices = np.concatenate([prices[:start], prices[end:]])

        train_norm, val_norm = normalize(train_data, val_data)

        model = Ridge(alpha=alpha)
        model.fit(train_norm, train_prices)
        pred = model.predict(val_norm)

        mse_list.append(mean_squared_error(val_prices, pred))
        mae_list.append(mean_absolute_error(val_prices, pred))

    mean_mse = np.mean(mse_list)
    mean_mae = np.mean(mae_list)
    print(f"Alpha = {alpha:4d} -> MSE mediu: {mean_mse:.4f}, MAE mediu: {mean_mae:.4f}")

    if mean_mse < best_mse:
        best_mse = mean_mse
        best_alpha = alpha

print(f"\nCel mai bun alpha: {best_alpha}")


# ============================================================
# EX 4 - Ridge pe toata multimea, afisare coeficienti
# ============================================================
print("\n" + "=" * 60)
print(f"EX 4 - Ridge (alpha={best_alpha}) pe tot setul")
print("=" * 60)

# normalizare pe tot setul (train = test = tot)
media = np.mean(training_data, axis=0)
deviatie = np.std(training_data, axis=0)
deviatie[deviatie == 0] = 1
data_norm = (training_data - media) / deviatie

model_final = Ridge(alpha=best_alpha)
model_final.fit(data_norm, prices)

atribute = [
    'anul fabricatiei', 'nr kilometri', 'mileage', 'motor', 'putere',
    'nr locuri', 'nr proprietari',
    'combustibil_1', 'combustibil_2', 'combustibil_3', 'combustibil_4', 'combustibil_5',
    'transmisie_1', 'transmisie_2'
]

print(f"Bias: {model_final.intercept_:.4f}")
print("\nCoeficienti:")
for i, (nume, coef) in enumerate(zip(atribute, model_final.coef_)):
    print(f"  {i + 1:2d}. {nume:20s} -> {coef:.4f}")

# cel mai semnificativ = coeficientul cu valoarea absoluta cea mai mare
coef_abs = np.abs(model_final.coef_)
idx_max = np.argmax(coef_abs)
idx_min = np.argmin(coef_abs)

# al doilea cel mai semnificativ
coef_abs_copy = coef_abs.copy()
coef_abs_copy[idx_max] = -1
idx_second = np.argmax(coef_abs_copy)

print(f"\nCel mai semnificativ atribut:       {atribute[idx_max]} (coef={model_final.coef_[idx_max]:.4f})")
print(f"Al doilea cel mai semnificativ:     {atribute[idx_second]} (coef={model_final.coef_[idx_second]:.4f})")
print(f"Cel mai putin semnificativ atribut: {atribute[idx_min]} (coef={model_final.coef_[idx_min]:.4f})")