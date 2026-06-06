# Teorie — Inteligență Artificială / Învățare Automată

---

## 1. Concepte Generale

### Paradigme de învățare
- **Supervizată** — date etichetate `(x, y)`, scopul e să înveți `f(x) → y`
- **Nesupervizată** — date fără etichete, scopul e să găsești structură (clustering, reducere dimensiuni)
- **Semi-supervizată** — mix de date etichetate și neetichetate
- **Ranforsată** — agent care primește recompense/penalizări pentru acțiuni

### Clasificare vs. Regresie
- **Clasificare**: `y` e o clasă discretă (0, 1, 2, 3...)
- **Regresie**: `y` e un număr continuu

### Erori și Generalizare
- **Eroare empirică** — eroarea pe datele de antrenare (ce măsuram)
- **Eroare de generalizare** — eroarea pe date noi (ce vrem de fapt)
- **Underfitting** (bias mare) — modelul e prea simplu, nu învață nici antrenarea
- **Overfitting** (variance mare) — modelul a memorat antrenarea, nu generalizează

**Bias-Variance Trade-off**: un model mai complex are bias mai mic dar variance mai mare.  
Soluție la overfitting: mai multe date, regularizare, scăderea complexității modelului.

### Train / Validare / Test
- **Train** — date pentru antrenarea modelului
- **Validare** — date pentru alegerea hiperparametrilor *(nu pentru antrenare, nu pentru evaluarea finală)*
- **Test** — evaluare finală, **o singură dată**

> Dacă tunezi hiperparametrii pe setul de test, faci overfitting în spațiul hiperparametrilor.

**Cross-validation (k-fold)**: împarți datele în k parți, antrenezi pe k-1, testezi pe 1, repeți de k ori și faci media. Util când ai puține date.

### Măsurarea performanței
- **Acuratețe** = (predicții corecte) / (total) — simplu, dar înșelător pe date dezechilibrate
- **Matricea de confuzie** — tabel care arată câte sample-uri din clasa i au fost clasificate ca j
- **Precision** = TP / (TP + FP) — din ce am prezis pozitiv, cât e corect
- **Recall** = TP / (TP + FN) — din ce era pozitiv, cât am găsit
- **F1** = 2 · Precision · Recall / (Precision + Recall) — medie armonică
- **MSE** = media pătratelor erorilor — pentru regresie

---

## 2. Naive Bayes

### Ideea de bază
Clasificator probabilistic. Alege clasa cu probabilitate maximă a posteriori:

```
ŷ = argmax_c  P(c) · ∏ P(x_i | c)
```

Folosim **Regula Bayes**: `P(c | x) ∝ P(x | c) · P(c)`

**Ipoteza "naivă"**: trăsăturile `x_i` sunt **independente** dat clasa `c`.  
În realitate rareori e adevărat, dar în practică funcționează surprinzător de bine.

### Estimarea parametrilor
- `P(c)` — frecvența clasei c în datele de antrenare
- `P(x_i | c)` — frecvența trăsăturii i pentru clasa c

**Laplace smoothing** (parametru `alpha`): adaugă `alpha` la toate numărătorile pentru a evita probabilități zero. Default: `alpha=1`.

### MultinomialNB vs. GaussianNB
- **MultinomialNB** — pentru frecvențe / numere de numarat (Bag of Words). Necesită valori ≥ 0.
- **GaussianNB** — pentru valori continue reale (modelează fiecare trăsătură cu o Gaussiană).

### Când funcționează bine
Text classification, features independente sau aproape independente, date puține (parametri puțini de estimat).

---

## 3. K-Nearest Neighbors (KNN)

### Ideea de bază
Clasifică un exemplu de test pe baza votului majoritar al celor mai apropiați `k` vecini din train.

**Algoritm**:
1. Calculează distanța de la exemplul de test la **toate** exemplele din train
2. Ia primii `k` vecini (cei cu distanță minimă)
3. Votul majoritar al etichetelor lor = predicția

La egalitate de voturi: alege eticheta primului vecin mai apropiat din cele la egalitate.

### Distanțe Minkowski
```
d(a, b) = ( Σ |a_i - b_i|^p )^(1/p)
```
- `p=1` — **Manhattan** (L1): sumă de valori absolute. Robustă la outlieri.
- `p=2` — **Euclidiană** (L2): distanța "normală". Cea mai folosită.
- `p→∞` — **Chebyshev**: maximul diferențelor absolute.

### Efectul parametrului k
- `k=1` — granița de decizie e neliniară (Diagrama Voronoi), overfit posibil
- `k` mare — granița se netezește, mai multă regularizare, dar poate underfit
- Valori impare evită egalitățile la clasificare binară

### Avantaje / Dezavantaje
**+** Simplu, neparametric, funcționează cu mai multe clase, granița neliniară  
**−** Lent la testare (trebuie parcurs tot trainul), sensibil la scale, suferă de blestemul dimensionalității

### Blestemul dimensionalității
Pe măsură ce numărul de dimensiuni crește, distanțele devin similare între ele — vecinii "apropiați" nu mai sunt cu adevărat apropiați. Sunt necesare exponențial mai multe date pentru a acoperi spațiul.

---

## 4. Metode Kernel

### Ideea de bază
Lucrezi cu **produse scalare** în loc de coordonate explicite.  
"Kernel trick": înlocuiești produsul scalar `x·z` cu o funcție kernel `k(x, z)` care calculează implicit produsul scalar într-un spațiu de dimensiune mai mare (posibil infinită).

```
k(x, z) = φ(x) · φ(z)   — φ e funcția de scufundare implicită
```

### Funcții kernel comune

| Kernel | Formula | Când se folosește |
|--------|---------|-------------------|
| **Linear** | `k(x,z) = x·z` | Punct de start, date deja bune |
| **Polinomial** | `k(x,z) = (x·z + c)^d` | Relații polinomiale |
| **RBF/Gaussian** | `k(x,z) = exp(-‖x-z‖²/2σ²)` | General, cel mai popular |
| **Hellinger** | `k(x,z) = Σ √(x_i · z_i)` | Probabilități / histograme |
| **Intersecție** | `k(x,z) = Σ min(x_i, z_i)` | Histograme / frecvențe |

O funcție e **kernel valid** dacă și numai dacă matricea Gram este pozitiv semi-definită.

### Matricea Kernel (Gram)
```
K_train[i,j] = k(x_train_i, x_train_j)   # (N_train × N_train)
K_test[i,j]  = k(x_test_i,  x_train_j)   # (N_test  × N_train)
```
> Al doilea argument e mereu față de **train**, nu față de test.

### Normalizarea kernelului
```
K_norm[i,j] = K[i,j] / sqrt(K[i,i] · K[j,j])
```

---

## 5. Regresia Ridge Kernel (KRR)

### Regresia Ridge (forma primală)
Minimizează:
```
‖Xw - y‖² + λ‖w‖²
```
Soluție: `w = (X^T X + λI)^{-1} X^T y`

Regularizarea `λ` previne inversarea unei matrice singulare și previne overfitting-ul.

### Forma duală + Kernel Trick
Predicția devine:
```
ŷ = K_test · (K_train + λI)^{-1} · y_train
```
unde `K` e matricea kernel. Astfel se poate folosi orice funcție kernel.

### Parametri
- `alpha` (`λ`) — regularizare: mic = mai flexibil, mare = mai regularizat
- kernelul ales

### Diferența față de SVM
- KRR produce valori **continue** → trebuie rotunjit la clasa cea mai apropiată
- KRR are soluție analitică exactă (mai rapid la fit pentru date mici-medii)
- SVM e mai robust la outlieri, de obicei mai bun la clasificare

---

## 6. SVM — Support Vector Machine

### Ideea de bază
Găsește **hiperplanul de separare cu margine maximă** între clase.  
"Vectorii suport" sunt exemplele cele mai apropiate de graniță — acestea determină hiperplanul.

### SVM Hard Margin
Presupune date liniar separabile. Maximizează marja `2/‖w‖`.

### SVM Soft Margin
Date care nu sunt liniar separabile. Permite "violări" ale marginii, penalizate cu parametrul `C`:
```
min  ½‖w‖² + C · Σ ξ_i
```
- `C` mare — penalizează mai mult erorile, granița mai rigidă, risc overfitting
- `C` mic — tolerează mai multe erori, granița mai lată, mai multă regularizare

### SVM cu Kernel (forma duală)
Înlocuiești produsul scalar cu kernelul ales → granița de decizie devine **neliniară** în spațiul original.

```python
model = SVC(C=3, kernel="precomputed")
model.fit(K_train, y_train)      # K_train: (N_train, N_train)
preds = model.predict(K_test)    # K_test:  (N_test, N_train)
```

### Multi-clasă
- **One-vs-One** — un classifier per pereche de clase, vot majoritar
- **One-vs-All** — un classifier per clasă (clasa vs. restul), alege scorul maxim
- **sklearn SVC** — one-vs-one implicit

---

## 7. Rețele Neuronale Feed-Forward

### Arhitectura
Straturi de neuroni conectați complet (*fully connected*):
```
input → [Linear → Activare → Dropout] × n_straturi → Linear(num_classes)
```
- Ultimul strat: **fără activare** — CrossEntropyLoss include softmax intern
- Adâncime mai mare = capacitate mai mare, dar necesită regularizare mai puternică

### Funcții de activare

| Funcție | Formula | Avantaje | Dezavantaje |
|---------|---------|----------|-------------|
| **Sigmoid** | `1/(1+e^{-x})` | Output în [0,1] | Saturație, gradient dispare |
| **Tanh** | `(e^x - e^{-x})/(e^x + e^{-x})` | Centrat în 0 | Saturație |
| **ReLU** | `max(0, x)` | Rapid, nu se saturează (pt x>0) | Neuroni "morți" (x<0) |
| **Leaky ReLU** | `max(0.1x, x)` | Nu are neuroni morți | — |
| **ELU** | `x if x>0 else α(e^x-1)` | Output aproape de medie 0 | Calcul exp() |

**În practică: folosește ReLU** (sau Leaky ReLU). Evită sigmoid.

### Funcții de pierdere
- **CrossEntropyLoss** — clasificare multi-clasă, combina softmax + log-loss
  - Input: scoruri brute `(N, num_classes)` + etichete întregi `(N,)`
  - `Li = -log( e^{s_yi} / Σ e^{s_j} )`
- **MSE** — regresie

### Optimizatori

| Optimizer | Comportament |
|-----------|-------------|
| **SGD** | Simplu, actualizare `w ← w - lr · grad` |
| **SGD + Momentum** | Acumulează viteză în direcțiile constante, amortizează zig-zag-ul. `mu ≈ 0.9` |
| **Adam** | Adaptiv per parametru, combină momentum + RMSProp. `lr=1e-3` default |

- `lr` (learning rate) prea mare → diverge; prea mic → convergență lentă
- Adam converge mai rapid în general; SGD poate generaliza uneori mai bine

### Backpropagation
Calculul gradienților prin **regula de înlănțuire** (*chain rule*) de-a lungul grafului computațional, de la ieșire spre intrare.

```
∂L/∂w = ∂L/∂z · ∂z/∂w   (regula de înlănțuire)
```

- **Forward pass**: calculează activările și salvează ce e necesar pentru backward
- **Backward pass**: calculează gradienții de la ieșire spre intrare

### Regularizare

**Dropout**:
- Dezactivează aleator `p%` din neuroni la fiecare forward pass **în antrenare**
- La testare: **toți neuronii activi**, dar scalați cu `(1-p)` (sau invers la train)
- Efect: forțează reprezentări redundante, similar cu un ansamblu de modele
- `model.train()` → Dropout activ; `model.eval()` → Dropout dezactivat

**L2 (Weight Decay)**: adaugă `λ‖w‖²` la funcția de pierdere → penalizează ponderi mari

**Early stopping**: oprești antrenarea când eroarea pe validare începe să crească

### Preprocesarea datelor pentru NN
Normalizarea inputului e **importantă** — NN sunt sensibile la scală:
```python
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
X_train = (X_train - mean) / (std + 1e-9)
X_test  = (X_test  - mean) / (std + 1e-9)   # același mean/std din train!
```

### Inițializarea ponderilor
- `W=0` → toți neuronii calculează același lucru, gradienții identici (rău)
- Aleator mic (`N(0, 0.01)`) → poate satura neuronii pentru rețele adânci
- **Xavier**: `W ~ N(0, 1/n_in)` — menține variația semnalului consistentă prin straturi

---

## 8. Bag of Words

Reprezintă un document ca vector de frecvențe ale tokenilor din vocabular.

**Token** = caracter sau cuvânt, în funcție de cerință.

**Pipeline**:
1. Construiește vocabularul **doar din train**
2. Transformă train și test cu același vocabular (tokenuri din test absente → ignorate)
3. Opțional: normalizare L1 sau L2

**Normalizare L1**: `x / Σ|x_i|` → suma devine 1  
**Normalizare L2**: `x / ‖x‖` → norma euclidiană devine 1

Vectorii BoW sunt **nonneg** → compatibili cu MultinomialNB, Hellinger, Intersection.

---

## 9. Matricea de Tranziție Markov

Captureaza structura secvențială a unui semnal: probabilitatea de a trece din intervalul `i` în intervalul `j`.

**Pipeline pentru o axă**:
1. Calculează `vmin`, `vmax` **global** (pe train + test combinat)
2. Definește `k` intervale egale: `np.linspace(vmin, vmax+ε, k+1)`
3. Discretizează: fiecare valoare → indexul intervalului în care cade (`np.digitize - 1`)
4. Construieste `A[k,k]`: `A[disc[t]][disc[t+1]] += 1` pentru fiecare pas `t`
5. Normalizează fiecare linie la sumă 1 (liniilie cu sumă 0 rămân 0)
6. Flatten `A` → vector `(k²,)`

Pentru semnale multidimensionale (ex: 3 axe): aplică pentru fiecare axă și concatenează → `(3k²,)`.

**De ce vmin/vmax global?** Ca intervalele de discretizare să fie identice la train și test. Dacă folosești doar trainul, valorile din test pot cădea în afara intervalelor.

Vectorii Markov sunt probabilități → **nonneg** → compatibili cu kernelele Hellinger și Intersection.

---

## 10. Convoluție 1D pe Text cu N-grams

Aplică fiecare filtru (n-gram) pe document printr-un **sliding window**.  
La fiecare poziție calculează **cosine similarity locală** între fereastră și filtru.  
Numără câte poziții depășesc un prag → un singur număr per (document, filtru).

**Formula** pentru fereastra de la poziția `i`:
```
score(i) = (Σ_j map[doc[i+j]] · map[gram[j]]) / (‖map(window)‖ · ‖map(gram)‖ + ε)
```

**Rezultat per document**: vector de lungime nr_filtre (ex: 500 valori întregi nonneg).

**Mapping**: transformă fiecare caracter într-un număr din `mapping.txt`. Caractere absente din mapping → 0.

Vectorii de convoluție sunt **nonneg** → compatibili cu kernelele Hellinger, Intersection, Linear.

---

## 11. Rezumat — Când să Folosești Ce

| Tip date | Metodă | Model |
|----------|--------|-------|
| Text + clasificare rapidă | BoW caractere/cuvinte | MultinomialNB |
| Text + performanță mai bună | Convoluție n-grams | SVM / KRR cu Hellinger |
| Semnale secvențiale | Markov (k din subiect) | KNN / SVM / KRR |
| Orice + NN cerut | Flatten + normalizare | PyTorch feed-forward |

| Features | Kernel recomandat |
|----------|-------------------|
| Probabilități / histograme nonneg | Hellinger sau Intersection |
| Vectori generali | Linear |

| Hiperparametru | Ce face | Valori tipice |
|----------------|---------|---------------|
| `k` KNN | nr vecini; mare = mai neted | 1, 3, 5, 7 |
| `p` Minkowski | ordinul distanței | 1, 2, 3, 5 |
| `C` SVM | regularizare inversă | 0.1, 1, 3, 10, 100 |
| `alpha` KRR | regularizare | 0.001, 0.01, 0.1, 1, 10 |
| `lr` NN | rata de învățare | 1e-4, 1e-3, 1e-2 |
| `k` Markov | nr intervale discretizare | 4, 5, 6, 7, 8 |
