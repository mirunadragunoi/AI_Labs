# =============================================================================
# HOW TO — Ghid de abordare pentru exercitiile posibile la laborator IA
# =============================================================================
# Focusat pe logica de rezolvare, ordine de pasi, capcane.
# Referinte la functions.py marcate cu (-> f:L<nr>)
# =============================================================================


# =============================================================================
# 1. PREGATIREA DATELOR
# =============================================================================
#
# DATE SECVENTIALE (semnale, time-series):
#   Problema principala: semnalele au lungimi diferite, modelele vor input fix.
#   Solutie: padding cu zerouri la final pana la lungimea maxima.
#   REGULA: calculeaza max_len pe TRAIN+TEST impreuna, niciodata doar pe train.
#   Altfel un sample din test mai lung decat orice din train va fi trunchiat gresit.
#   (-> f:L316  pad_to_max)
#
# DATE TEXT:
#   Fiecare document e un string. Trebuie transformat in vector numeric.
#   Alege metoda in functie de ce cere subiectul:
#     - "Bag of Words" -> BagOfWords  (-> f:L249)
#     - "convolutie cu n-grams" -> char_convolution  (-> f:L288)
#
# NORMALIZARE:
#   Calculeaza mean/std/min/max DOAR pe train, aplica si pe test cu aceleasi valori.
#   (-> f:L270  l1_normalize,  f:L274  l2_normalize)


# =============================================================================
# 2. BAG OF WORDS
# =============================================================================
#
# Transforma un document in vector de frecvente ale tokenilor.
# Token = caracter (daca subiectul zice "la nivel de caracter") sau cuvant.
# Iterand un string in Python dai caracter cu caracter; iterand lista de cuvinte dai cuvinte.
#
# Ordine obligatorie:
#   1. fit pe TRAIN    (-> f:L253  BagOfWords.fit)
#   2. transform train (-> f:L260  BagOfWords.transform)
#   3. transform test cu ACELASI obiect (vocabularul ramas din fit)
#
# Ce se intampla cu tokeni din test absenti din vocabular? Sunt ignorati, nu crasha.
#
# Cand normalizeaza:
#   - pt MultinomialNB: poti da frecventele brute direct, nu e nevoie de normalizare
#   - pt SVM/KRR pe text: normalizeaza L1 sau L2 inainte de a construi kernelul
#   Atentie: MultinomialNB cere valori >= 0, normalizarea L1/L2 poate produce
#   valori negative din floating point -> clip sau nu normaliza deloc pt NB.


# =============================================================================
# 3. NAIVE BAYES
# =============================================================================
#
# Merge direct dupa BagOfWords, fara pasi intermediari complicati.
# MultinomialNB pentru frecvente (nonneg).   (-> f:L100)
# GaussianNB pentru date continue reale.     (-> f:L104)
#
# Singurul lucru de acordat: alpha (smoothing Laplace, default=1.0).
# Daca acuratetea e slaba, incearca alpha mai mic (0.1, 0.01).


# =============================================================================
# 4. MATRICE DE TRANZITIE MARKOV
# =============================================================================
#
# De ce: capteaza structura secventiala a semnalului ca vector de probabilitati.
# Rezultatul e un vector nonneg -> compatibil direct cu kernel_hellinger/intersection.
#
# Logica pasilor:
#   1. Calculeaza vmin/vmax GLOBAL per axa pe train+test combinat.
#      De ce global: daca folosesti doar train, valorile din test pot cadea
#      in afara intervalelor definite -> index gresit la discretizare.
#
#   2. Discretizeaza: imparte [vmin, vmax] in k intervale egale,
#      inlocuieste fiecare valoare cu indexul intervalului in care cade.
#      np.linspace + np.digitize - 1  (-> f:L64)
#      Clip la k-1 pt valoarea exact egala cu vmax  (-> f:L222)
#
#   3. Construieste A[k,k]: parcurge secventa discretizata,
#      la fiecare pas t incrementeaza A[disc[t]][disc[t+1]].
#
#   4. Normalizeaza fiecare rand la suma 1. Randurile cu suma 0 raman 0.
#
#   5. Flatten A -> vector (k^2,). Daca ai n axe, concateneaza n vectori -> (n*k^2,).
#
#   Implementare: (-> f:L212  markovize)
#   Apeleaza markovize() per axa, concateneaza cu np.concatenate.
#
# Parametru de tunat: k (numarul de intervale). Tipic 4-10.
# k mare = mai detaliat dar matricea devine mai sparsa (multe zerouri).


# =============================================================================
# 5. KNN
# =============================================================================
#
# Dupa ce ai vectorii de features (ex: dupa Markov), KNN e simplu:
# pentru fiecare sample din test, calculeaza distanta la toate sample-urile din train,
# ia primii K si fa vot majoritar.
#
# Distanta Minkowski de ordin p:  d = (sum |a_i - b_i|^p)^(1/p)   (-> f:L56  argsort)
#   p=1 Manhattan, p=2 Euclidiana, p>2 mai sensibil la diferente mari.
#
# La egalitate de voturi: alege eticheta primului vecin mai apropiat din cei la egalitate.
# (nu alege random, nu alege prima clasa numerica)
#
# Implementare: (-> f:L316  nu e in functions, scrie clasa direct)
# vezi solutiile din subiecte_anterioare pt implementare completa.
#
# Vectorizare cu numpy (mai rapida decat for loop):
#   diff = train[np.newaxis,:,:] - test[:,np.newaxis,:]   # (N_test, N_train, d)
#   dist = np.sum(np.abs(diff)**p, axis=2)**(1/p)         # (N_test, N_train)
#
# Parametri de tunat: K (1, 3, 5, 7), p (1, 2, 3, 5).


# =============================================================================
# 6. KERNEL FUNCTIONS SI KERNEL MATRIX
# =============================================================================
#
# Ce kernel sa alegi:
#   Hellinger    -> cand vectorii sunt probabilitati sau frecvente nonneg  (-> f:L121)
#   Intersection -> la fel, mai simplu de calculat, performanta similara   (-> f:L130)
#   Linear       -> produs scalar, functioneaza cu orice vectori           (-> f:L139)
#
# Constructia matricilor - cel mai frecvent punct de confuzie:
#   K_train = kernel(train, train)   # (N_train, N_train)
#   K_test  = kernel(test,  train)   # (N_test,  N_train)  <- AL DOILEA ARGUMENT E MEREU TRAIN
#
# De ce al doilea argument e train si la K_test?
#   Modelul a invatat o combinatie de sample-uri din train.
#   La predictie masuram cat de similar e fiecare sample de test cu FIECARE sample din train.
#   Dimensiunea rezultata trebuie sa fie (N_test, N_train), nu (N_test, N_test).
#
# Capcana clasica: kernel(test, test) -> shape gresit -> eroare sau rezultat silentios gresit.


# =============================================================================
# 7. SVM CU KERNEL PRECOMPUTED
# =============================================================================
#
# Ordine: features -> K_train, K_test -> SVC.fit -> SVC.predict
#   (-> f:L80  SVC,  f:L121/130/139  kernelele)
#
# C mic = mai multa regularizare (frontiera mai simpla, mai generalizabila).
# C mare = se adapteaza mai bine la train, risc de overfit.
# Valori de incercat: 0.1, 1, 3, 10, 100.
#
# Nu uita: K_test = kernel(test_features, train_features), nu (test, test).


# =============================================================================
# 8. KERNEL RIDGE REGRESSION (KRR)
# =============================================================================
#
# Ca SVM cu kernel precomputed, dar produce valori continue.
# Dupa .predict() trebuie rotunjit la cea mai apropiata clasa valida.  (-> f:L94)
#   classes = np.unique(y_train)
#   preds = [classes[np.argmin(np.abs(classes - p))] for p in raw]
#
# (-> f:L91  KernelRidge,  f:L121/130/139  kernelele)
#
# Parametrul alpha: mic (0.001, 0.01) = mai flexibil; mare (1, 10) = mai regularizat.
# Mai usor de tunat decat C la SVM (un singur parametru, comportament mai previzibil).


# =============================================================================
# 9. RETEA NEURONALA FEED-FORWARD (PyTorch)
# =============================================================================
#
# Ordine de pasi:
#   1. Flatten datele la (N, d) si normalizeaza  (-> f:L27  reshape)
#      mean/std calculate pe train, aplicate si pe test.
#
#   2. Converteste la tensori  (-> f:L164)
#      X: dtype=torch.float32,  y: dtype=torch.long  (obligatoriu pentru CrossEntropyLoss)
#
#   3. Defineste arhitectura  (-> f:L149  Net)
#      Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear(num_classes)
#      Ultimul strat: fara activare. CrossEntropyLoss include softmax intern.
#
#   4. Loop de antrenare  (-> f:L176)
#      model.train() la inceputul fiecarei epoci (activeaza Dropout).
#      Ordinea obligatorie: zero_grad -> forward -> loss -> backward -> step.
#
#   5. Predictii  (-> f:L187)
#      model.eval() inainte (dezactiveaza Dropout).
#      torch.no_grad() inainte (nu pastreaza graful, economiseste memorie).
#      torch.max(outputs, 1) returneaza (valori, indici); vrei indicii = clasele.
#
# Capcana: uiti model.eval() -> Dropout activ la predictii -> rezultate diferite la fiecare run.
#
# Optimizer:
#   (-> f:L168  Adam,  f:L169  SGD cu momentum,  f:L170  SGD simplu)
#   Adam e default bun. SGD cu momentum=0.9 uneori generalizeaza mai bine.
#
# Debugging: overfit intentionat pe 10 sample-uri.
#   Daca loss nu scade la aproape 0 -> bug in model sau in date.


# =============================================================================
# 10. CONVOLUTIE 1D PE TEXT CU N-GRAMS
# =============================================================================
#
# Logica: pentru fiecare filtru (n-gram) si fiecare pozitie din document,
# calculeaza cosine similarity intre fereastra de n caractere si filtru.
# Numara cate pozitii depasesc un prag (ex: 0.9) -> un int per (document, filtru).
# Rezultat per document: vector de lungime nr_filtre.
#
# Implementare: (-> f:L288  char_convolution)
#   vec = [char_convolution(doc, gram, mapping) for gram in grams]
#
# Ordinea de construit mapping din mapping.txt:
#   - initializeaza toti cei 256 ascii la 0
#   - citeste perechile caracter,numar linie cu linie
#   - trateaza cazuri speciale: virgula (linia 9 din fisier) si spatiu (linia 17)
#   - adauga si varianta uppercase a fiecarui caracter cu acelasi numar
#
# Performanta: lent cu for loop triplu. Optimizare simpla:
#   calculeaza norm_gram o singura data per gram, nu la fiecare document.
#   (deja facut in char_convolution  -> f:L296)
#
# Vectorii rezultati sunt nonneg -> compatibili cu kernel_hellinger/intersection.  (-> f:L121/130)


# =============================================================================
# 11. VALIDARE HIPERPARAMETRI (Ex 5)
# =============================================================================
#
# Separa un subset de validare din TRAIN (nu atinge testul).  (-> f:L112  train_test_split)
# Antreneaza pe train_redus, evalueaza pe validare, repeta pentru fiecare combinatie.
# Capcana: cand construiesti K_val folosesti kernel(val_features, train_features),
#   NU kernel(val_features, val_features).  (-> aceeasi regula din sectiunea 6)
#
# Ce sa raportezi: tabel cu model, hiperparametri, acuratete pe validare.
# Valori tipice de explorat:
#   SVM:  C in [0.1, 1, 3, 10, 100]
#   KRR:  alpha in [0.001, 0.01, 0.1, 1, 10]
#   KNN:  K in [1, 3, 5, 7, 11],  p in [1, 2, 3, 5]
#   NN:   lr in [1e-4, 1e-3, 1e-2],  dropout in [0.1, 0.3, 0.5]


# =============================================================================
# DECIZII RAPIDE
# =============================================================================
#
# Ce features sa construiesti:
#   Date secventiale                 -> Markov  (-> f:L212)
#   Date text + BoW cerut            -> BagOfWords  (-> f:L249)
#   Date text + convolutie ceruta    -> char_convolution  (-> f:L288)
#
# Ce model:
#   NB cerut                         -> MultinomialNB pe BoW  (-> f:L100)
#   KNN cerut                        -> KnnClassifier cu p din subiect
#   SVM cerut                        -> SVC precomputed  (-> f:L80)
#   KRR cerut                        -> KernelRidge + rotunjire  (-> f:L91)
#   NN cerut                         -> Net + CrossEntropyLoss  (-> f:L149)
#
# Ce kernel:
#   Features sunt probabilitati/frecvente nonneg  -> hellinger sau intersection  (-> f:L121/130)
#   Features sunt vectori generali                -> linear  (-> f:L139)
#
# Salvare rezultate:
#   Format .txt (un label pe linie)    -> save_txt   (-> f:L322)
#   Format .npy                        -> save_npy   (-> f:L327)
