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
## 3. Priprema i analiza podataka
Priprema i analiza podataka se sastoji iz sledecih faza:
- Ucitavanje učitavanje i čišćenje dataset-a
- Uklanjanje obeležja koja logički ne doprinose otkrivanju obrazaca u podacima
- Ispitivanje korelacije obeležja i izbacivanje onih koja jedinstveno ne doprinose opisu varijabiliteta izlaznog obeležja
- Testiranje simetričnosti raspodela obeležja, radi uvođenja pravilnog standardnog skaliranja  i ujednačenog gradijenta
- Logaritamska transformacija obeležja sa asimetričnom raspodelom
- Konverzija kategoričkih promenljivi u numeričke one-hot encoding-om

---
## 4. Arhitektura modela

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

## 5. Trening

Trening se satoji iz narednih koraka:

- Petlja kroz epohe
- Mini-batch trening
- Praćenje trening i validacione greške po epohi.
- Early stopping: ako se validaciona greška ne poboljša tokom N uzastopnih epoha (definisano u promenjivoj `patience`), trening se prekida i vraća se stanje modela sa najboljom validacionom greškom.
- Opcioni LR scheduler: kod Modela 7 korišćen je cheduler koji automatski smanjuje stopu učenja kada validaciona greška stagnira.


---

## 6. Analiza osetljivosti i hiperparametarska optimizacija


U ranijoj verziji projekta, kolone `Usage Quantity` i `Cost per Quantity ($)` su bile uklonjene iz skupa obeležja — `Usage Quantity` zbog multikolinearnosti sa drugim obeležjima uočene na korelacionoj matrici, a `Cost per Quantity ($)` iz opreza da se izbegne data leakage.

Problem je bio u tome što su `Usage Quantity` i `Cost per Quantity ($)` upravo dve veličine koje direktno i skoro deterministički određuju trošak. Njihovim uklanjanjem, preostala obeležja nose mnogo slabiju vezu sa ciljnom promenljivom.

Posledica je bila ta sto su se arhitekture, bez obzira na aktivacionu funkciju, optimizator, dubinu mreže ili regularizaciju, zaglavljivale se na približno istoj validacionoj grešci, sa R² u rasponu 0.14–0.27 i MAPE preko 200%. Činjenica da promena arhitekture nije uopšte pomerala rezultat ukazivala je da uzrok problema nije kapacitet mreže, već nedostatak signala u ulaznim podacima.

Nakon što su `Usage Quantity` i `Cost per Quantity ($)` vraćeni u skup obeležja (uz odgovarajuće skaliranje `StandardScaler`-om), performanse su se drastično poboljšale kod svih modela.

## 7. Hiperparametarska optimizacija

Hiperparametarska optimizacija sprovedena je inkrementalno, kroz sedam varijanti modela, gde je u svakom koraku menjan po jedan ključni hiperparametar u odnosu na prethodnu varijantu, kako bi se izolovao njegov pojedinačni uticaj na performanse. Polazna tačka bio je Model 1 (Sigmoid aktivacija, SGD optimizator sa lr=0.01, MSE gubitak). U Modelu 2 ispitan je uticaj funkcije gubitka zamenom MSE sa HuberLoss, uz iste ostale hiperparametre - rezultat je bio blago lošiji, što ukazuje da HuberLoss, manje osetljiv na velika odstupanja, ovde usporava konvergenciju ka preciznoj rekonstrukciji cene. Model 3 je testirao uticaj aktivacione funkcije, zamenom Sigmoid-a sa ReLU (uz nepromenjen SGD i MSE), što je dalo poboljšanje. Model 4 je zatim testirao Tanh aktivaciju (uz isti SGD optimizator i MSE gubitak) i pokazao vrlo dobre rezultate, uporedive sa ostalim varijantama. U Modelu 5, uz identičnu Tanh arhitekturu, ispitan je uticaj optimizatora zamenom SGD-a Adam optimizatorom, što je takođe dalo visoku tačnost. Model 6 je ispitao uticaj kapaciteta mreže, povećanjem širine prvog skrivenog sloja sa 64 na 128 neurona (uz Tanh i Adam), bez značajnog dodatnog poboljšanja, što sugeriše da dodatni kapacitet mreže nije bio neophodan za ovaj problem. Konačno, Model 7 je testirao uticaj adaptivnog opadanja stope učenja uvođenjem u najbolji model do sada, tj. Model 4, u kombinaciji sa dužim treningom (150 umesto 50 epoha) i većim patience parametrom (10 umesto 8) — ova kombinacija dala je najbolje rezultate od svih testiranih varijanti.

---

## 8. Rezultati evaluacije

Rezultati na **validacionom skupu** za svih 7 modela (nakon vraćanja `Usage Quantity` i `Cost per Quantity ($)` u skup obeležja):

| Model | MSE | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|---|
| Model 1 (Sigmoid, SGD, MSE) | 8008.42 | 89.49 | 37.96 | 0.9977 | 4.01% |
| Model 2 (Sigmoid, SGD, Huber) | 21321.35 | 146.02 | 62.99 | 0.9940 | 6.43% |
| Model 3 (ReLU, SGD, MSE) | 899.33 | 29.99 | 15.01 | 0.9997 | 1.48% |
| Model 4 (Tanh, SGD, MSE) | 665.28 | 25.79 | 12.23 | 0.9998 | 1.38% |
| Model 5 (Tanh, Adam, MSE) | 900.41 | 30.01 | 13.13 | 0.9997 | 1.20% |
| Model 6 (Tanh 128→32, Adam, MSE) | 1153.11 | 33.96 | 14.60 | 0.9997 | 1.27% |
| Model 7 (Tanh, SGD+scheduler, MSE) | 283.88 | 16.85 | 7.83 | 0.9999 | 0.87% |

Finalna evaluacija najboljeg modela (Model 7) na test skupu:

| Metrika | Vrednost |
|---|---|
| MSE | 303.11 |
| RMSE | 17.41 |
| MAE | 7.78 |
| R² | 0.9999 |
| MAPE | 0.85% |

Model 7 je izabran kao finalni model, jer konzistentno postiže najniže greške i na validacionom i na test skupu, uz najmanju razliku između trening i validacione greške (znak dobre generalizacije bez prekomernog prilagođavanja).

---

## 9. Diskusija

Nakon uključivanja `Usage Quantity` i `Cost per Quantity ($)`, model suštinski uči (skoro) determinističku relaciju `trošak ≈ količina × cena po jedinici`, uz manje doprinose ostalih obeležja. To objašnjava zašto gotovo sve arhitekture dostižu R² > 0.99 - mreža ne mora da otkriva skrivene, složene obrasce, već pretežno aproksimira jednu poznatu aritmetičku operaciju nad dva ulazna broja.

Ako je stvarni cilj projekta da se trošak unapred proceni na osnovu telemetrije korišćenja resursa (CPU/memorija/mreža/trajanje/tip servisa/region), pre nego što je trošak zaista obračunat, onda ovaj model nije prikladan za tu namenu. 

---

## 10. Zaključak

Projekat je pokazao da je skup podataka i izbor ulaznih obeležja presudan faktor za uspeh modela, čak i više od arhitekture same neuronske mreže. Takođe, pokazalo se da je, za unapređenje performansi modela, uvek prvo potrebno izvršiti hiperparametarsku optimizaciju, pre nego što se odlučimo za dodavanje novih slojeva unutar mreže ili proširivanje starih. 