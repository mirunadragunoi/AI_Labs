import numpy as np

# citire imagini si salvare in np.array
imagini = []
for i in range(9):
    img = np.load(f"images/car_{i}.npy")
    imagini.append(img)
imagini = np.array(imagini)

print(imagini.shape)

# suma valorilor pixelilor tuturor imaginilor
suma_pixeli = np.sum(imagini)
print("suma pixeli toate imaginile:", suma_pixeli)

# suma valorilor pixelilor pentru fiecare imagine in parte???
# axis in care sa iau fiecare imagine specifica cu axis
suma_pe_imagine = np.sum(imagini, axis = (1, 2))
print("sume imagini pe imagine:", suma_pe_imagine)

# indexul imaginii cu suma maxima ->> iau direct din suma pe imagine
# valoarea cea mai mare si ii iau cirect indexul
index_imagine_maxima = np.argmax(suma_pe_imagine)
print("indexul sumei imaginii maxime:", index_imagine_maxima)

# imaginea medie --- adica tre sa fac media
from skimage import io

# fac media imaginilor cu mean
medie_imagine = np.mean(imagini, axis = 0)

io.imshow(medie_imagine.astype(np.uint8))
io.show()

# deviatia standard a imaginilor cu std aparent
deviatie = np.std(imagini, axis=0)

print("deviatia imaginilor este:", deviatie)

# normalizarea imaginilor
# scad imagine medie si impart rezultatul la deviatia standard
imagini_normalizate = (imagini - medie_imagine) / deviatie

io.imshow(imagini_normalizate[3].astype(np.uint8))
io.show()

# decupare imagini
# linii intre 200 si 300 si coloane intre 280 si 400

imagini_decupate = imagini[:, 200:300, 280:400]

io.imshow(imagini_decupate[3].astype(np.uint8))
io.show()

# deci nu trebuia grid, trebuie sa fac un np.array mare care sa cuprinda gen
# toate imaginile, adica cum am acum toate imaginile separate, trebuie sa
# fac o imagine mare care sa le cuprinda pe toate
# deci sa fie o imagine mare in care dau paste la imaginile mici

# imagine are dimensiunea 400 X 600
# grid de 3 ori 3
# deci 3 randuri * 400 = 1200
# 3 coloane * 600 = 1800

# imagine mare
grid = np.zeros((1200, 1800))

# punem imaginile in grid
for i in range(9):
  rand = i // 3 # ar trebui sa fie 0 0 0 1 1 1 2 2 2
  coloana = i % 3 # 0 1 2 0 1 2 0 1 2

  # coordonate imagine in grid ul mare
  start_rand = rand * 400
  start_coloana = coloana * 600

  # lipim imaginea adica paste ul
  grid[start_rand : start_rand + 400, start_coloana : start_coloana + 600] = imagini[i]

io.imshow(grid.astype(np.uint8))
io.show()

# fac fiecare rand concatenand 2 imagini pe orizontala adica tre sa le lipesc din start
# 2 imagini 400 x 600 deci o sa am acum 400 1200
# np.rot90 cu k=-1 deci -90, cu k=1 deci 90

rand_1 = np.concatenate([np.rot90(imagini[0], k=1), np.rot90(imagini[2], k=-1)], axis=1)
rand_2 = np.concatenate([np.rot90(imagini[3], k=1), np.rot90(imagini[5], k=-1)], axis=1)
rand_3 = np.concatenate([np.rot90(imagini[6], k=1), np.rot90(imagini[8], k=-1)], axis=1)

# concatenez cele 3 randuri pe verticala adica randuri
grid = np.concatenate([rand_1, rand_2, rand_3], axis=0)

io.imshow(grid.astype(np.uint8))
io.show()