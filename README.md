# Predviđanje troškova GCP resursa pomoću neuronskih mreža

## 1. Opis problema

Cilj projekta je da se, na osnovu podataka o korišćenju Google Cloud Platform resursa, predvidi trošak korišćenja primenom neuronskih mreža implementiranih u PyTorch-u.

---

## 2. Podaci

Podaci su preuzeti sa Hugging Face Hub-a, iz [gcp-cloud-billing-cost](https://huggingface.co/datasets/sairamn/gcp-cloud-billing-cost) skupa podataka.

Kolone u skupu podataka:

| Kolona | Opis |
|---|---|
| Resource ID | Identifikator resursa (uklonjen — nema prediktivnu vrednost) |
| Service Name | Naziv GCP servisa (npr. Compute Engine, BigQuery, Cloud Storage...) |
| Usage Quantity | Količina iskorišćenog resursa |
| Usage Unit | Jedinica mere korišćenja (uklonjena) |
| Region / Zone | Region/zona u kojoj je resurs korišćen |
| CPU Utilization (%) | Iskorišćenost CPU-a |
| Memory Utilization (%) | Iskorišćenost memorije |
| Network Inbound Data (Bytes) | Dolazni mrežni saobraćaj |
| Network Outbound Data (Bytes) | Odlazni mrežni saobraćaj |
| Usage Start Date / Usage End Date | Vreme početka/kraja korišćenja |
| Cost per Quantity ($) | Cena po jedinici korišćenja |
| Unrounded Cost ($) | **Ciljna promenljiva** — tačan (nezaokružen) trošak |
| Rounded Cost ($) | Zaokružen trošak (uklonjen — redundantan sa ciljnom promenljivom) |
| Total Cost (INR) | Trošak u indijskim rupijama (uklonjen — redundantan, druga valuta) |

---

## 3. Arhitektura modela

Ukupno je definisano i trenirano 7 varijanti eedforward neuronske mreže.

| Model | Arhitektura (skriveni slojevi) | Aktivacija | Funkcija gubitka | Optimizator |
|---|---|---|---|---|
| **CostPredictor1** | 64 → 32 → 1 | Sigmoid | MSELoss | SGD (lr=0.01) |
| **CostPredictor2** | 64 → 32 → 1 | Sigmoid | HuberLoss (delta=1) | SGD (lr=0.01) |
| **CostPredictor3** | 64 → 32 → 1 | ReLU | MSELoss | SGD (lr=0.01) |
| **CostPredictor4** | 64 → 32 → 1 | Tanh | MSELoss | SGD (lr=0.01) |
| **CostPredictor5** | 64 → 32 → 1 | Tanh | MSELoss | Adam (lr=0.001) |
| **CostPredictor6** | 128 → 32 → 1 | Tanh | MSELoss | Adam (lr=0.001) |
| **CostPredictor7** | 64 → 32 → 1 | Tanh | MSELoss | SGD (lr=0.01) + `ReduceLROnPlateau` scheduler |



---

## 4. Trening

Trening se satoji iz narednih koraka:

- Petlja kroz epohe
- Mini-batch trening
- Praćenje trening i validacione greške po epohi.
- Early stopping: ako se validaciona greška ne poboljša tokom N uzastopnih epoha (definisano u promenjivoj `patience`), trening se prekida i vraća se stanje modela sa najboljom validacionom greškom.
- Opcioni LR scheduler: kod Modela 7 korišćen je cheduler koji automatski smanjuje stopu učenja kada validaciona greška stagnira.


---

## 5. Analiza osetljivosti i hiperparametarska optimizacija


U ranijoj verziji projekta, kolone `Usage Quantity` i `Cost per Quantity ($)` su bile uklonjene iz skupa obeležja — `Usage Quantity` zbog multikolinearnosti sa drugim obeležjima uočene na korelacionoj matrici, a `Cost per Quantity ($)` iz opreza da se izbegne data leakage.

Problem je bio u tome što su `Usage Quantity` i `Cost per Quantity ($)` upravo dve veličine koje direktno i skoro deterministički određuju trošak. Njihovim uklanjanjem, preostala obeležja nose mnogo slabiju vezu sa ciljnom promenljivom.

Posledica je bila ta sto su se arhitekture, bez obzira na aktivacionu funkciju, optimizator, dubinu mreže ili regularizaciju, zaglavljivale se na približno istoj validacionoj grešci, sa R² u rasponu 0.14–0.27 i MAPE preko 200%. Činjenica da promena arhitekture nije uopšte pomerala rezultat ukazivala je da uzrok problema nije kapacitet mreže, već nedostatak signala u ulaznim podacima.

Nakon što su `Usage Quantity` i `Cost per Quantity ($)` vraćeni u skup obeležja (uz odgovarajuće skaliranje `StandardScaler`-om), performanse su se drastično poboljšale kod svih modela.

## 6. Hiperparametarska optimizacija

Hiperparametarska optimizacija sprovedena je inkrementalno, kroz sedam varijanti modela, gde je u svakom koraku menjan po jedan ključni hiperparametar u odnosu na prethodnu varijantu, kako bi se izolovao njegov pojedinačni uticaj na performanse. Polazna tačka bio je Model 1 (Sigmoid aktivacija, SGD optimizator sa lr=0.01, MSE gubitak). U Modelu 2 ispitan je uticaj funkcije gubitka zamenom MSE sa HuberLoss, uz iste ostale hiperparametre - rezultat je bio blago lošiji, što ukazuje da HuberLoss, manje osetljiv na velika odstupanja, ovde usporava konvergenciju ka preciznoj rekonstrukciji cene. Model 3 je testirao uticaj aktivacione funkcije, zamenom Sigmoid-a sa ReLU (uz nepromenjen SGD i MSE), što je dalo poboljšanje. Model 4 je zatim testirao Tanh aktivaciju (uz isti SGD optimizator i MSE gubitak) i pokazao vrlo dobre rezultate, uporedive sa ostalim varijantama. U Modelu 5, uz identičnu Tanh arhitekturu, ispitan je uticaj optimizatora zamenom SGD-a Adam optimizatorom, što je takođe dalo visoku tačnost. Model 6 je ispitao uticaj kapaciteta mreže, povećanjem širine prvog skrivenog sloja sa 64 na 128 neurona (uz Tanh i Adam), bez značajnog dodatnog poboljšanja, što sugeriše da dodatni kapacitet mreže nije bio neophodan za ovaj problem. Konačno, Model 7 je testirao uticaj adaptivnog opadanja stope učenja uvođenjem u najbolji model do sada, tj. Model 4, u kombinaciji sa dužim treningom (150 umesto 50 epoha) i većim patience parametrom (10 umesto 8) — ova kombinacija dala je najbolje rezultate od svih testiranih varijanti.

---

## 7. Rezultati evaluacije

Rezultati na **validacionom skupu** za svih 7 modela (nakon vraćanja `Usage Quantity` i `Cost per Quantity ($)` u skup obeležja):

| Model | MSE | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|---|
| Model 1 (Sigmoid, SGD, MSE) | 9 207.35 | 95.95 | 45.37 | 0.9974 | 6.86% |
| Model 2 (Sigmoid, SGD, Huber) | 34 531.66 | 185.83 | 112.62 | 0.9902 | 17.24% |
| Model 3 (ReLU, SGD, MSE) | 4 370.79 | 66.11 | 30.94 | 0.9988 | 2.70% |
| Model 4 (Tanh, SGD, MSE) | 873.74 | 29.55| 14.43 | 0.9998| 2.01%|
| Model 5 (Tanh, Adam, MSE) | 3 098.78 | 55.67 | 23.63 | 0.9991 | 2.23% |
| Model 6 (Tanh 128→32, Adam, MSE) | 2 475.35 | 49.75 | 25.75 | 0.9993 | 2.72% |
| Model 7 (Tanh, SGD+scheduler, MSE) | 736.37 | 27.14 | 13.51 | 0.9998 | 1.77% |

Finalna evaluacija najboljeg modela (Model 7) na test skupu (podaci koje model nikada nije video, ni tokom treninga ni tokom podešavanja):

| Metrika | Vrednost |
|---|---|
| MSE | 970.93 |
| RMSE | 31.16 |
| MAE | 13.76 |
| R² | 0.9997 |
| MAPE | 1.72% |

Model 7 je izabran kao finalni model, jer konzistentno postiže najniže greške i na validacionom i na test skupu, uz najmanju razliku između trening i validacione greške (znak dobre generalizacije bez prekomernog prilagođavanja).

---

## 8. Diskusija

Nakon uključivanja `Usage Quantity` i `Cost per Quantity ($)`, model suštinski uči (skoro) determinističku relaciju `trošak ≈ količina × cena po jedinici`, uz manje doprinose ostalih obeležja. To objašnjava zašto gotovo sve arhitekture dostižu R² > 0.99 - mreža ne mora da otkriva skrivene, složene obrasce, već pretežno aproksimira jednu poznatu aritmetičku operaciju nad dva ulazna broja.

Ovo je važno ograničenje na koje treba skrenuti pažnju. `Usage Quantity` i `Cost per Quantity ($)` su, u praksi, već delovi konačnog obračuna troška - dostupni su tek nakon što je trošak izračunat/naplaćen. Ako je stvarni cilj projekta da se trošak unapred proceni na osnovu telemetrije korišćenja resursa (CPU/memorija/mreža/trajanje/tip servisa/region), pre nego što je trošak zaista obračunat, onda ovaj model nije prikladan za tu namenu.

---

## 8. Zaključak

Projekat je pokazao da je izbor ulaznih obeležja presudan faktor za uspeh modela, čak i više od arhitekture same neuronske mreže. Takođe, pokazalo se da je, za unapređenje performansi modela, uvek prvo potrebno izvršiti hiperparametarsku optimizaciju, pre nego što se odlučimo za dodavanje novih slojeva unutar mreže ili proširivanje starih. 