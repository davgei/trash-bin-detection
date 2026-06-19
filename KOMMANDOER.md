# Kommandoer — slik kjører du prosjektet

Kort oversikt over alle scriptene, hva de gjør og hvordan de kjøres.
Alt kjøres fra prosjektroten med `py -3.14` (pakkene ligger i global Python 3.14, **ikke** i `.venv`).

---

## Hurtigstart — vanlige kommandoer

Kopier og lim inn. Sett API-nøkkelen først (én gang per terminal-økt):

```powershell
$env:GOOGLE_MAPS_API_KEY = "din-nøkkel"
```

**Hente bilder:**
```powershell
# Test: bare geometri, henter ingen bilder (gratis)
py -3.14 -m src.fetch_streetview_from_csv --dry-run --limit 5

# Test: hent de 5 første kassene
py -3.14 -m src.fetch_streetview_from_csv --limit 5

# Hent alle aktive kasser fra CSV
py -3.14 -m src.fetch_streetview_from_csv

# Hent fra adresse-/koordinatfil
py -3.14 -m src.fetch_streetview
```

**Annotere → splitte → trene → evaluere:**
```powershell
# Annoter med YOLO-forslag
py -3.14 -m src.labeling.annotate --mode assisted --model models/trained/trash_bin_yolo11n_best.pt

# Bygg om train/val/test (kjør IKKE under trening)
py -3.14 -m src.prepare_dataset

# Tren lokalt
py -3.14 -m src.train --name run4

# Evaluer beste modell på test-splitten
py -3.14 -m src.evaluate --model models/trained/trash_bin_yolo11n_best.pt
```

Detaljer og alle flagg står lenger ned.

---

## 0. Forutsetninger

API-nøkkel for Google Maps (kun nødvendig for Street View-henting):
```powershell
$env:GOOGLE_MAPS_API_KEY = "din-nøkkel"
```
Nøkkelen gjelder bare for den terminal-økten. Åpner du en ny terminal, må den settes på nytt.

---

## 1. Hente Street View-bilder fra kasse-CSV (anbefalt)

Leser eksakte kassekoordinater fra `data/Uttrekk_products(result).csv`, finner nærmeste
panorama og sikter kameraet rett mot kassen. Bildene lagres i `data/to_annotate/`.

```powershell
# Tørrkjøring: bare geometri + gratis metadata, henter INGEN bilder
py -3.14 -m src.fetch_streetview_from_csv --dry-run --limit 5

# Hent de 5 første for å se om sikting ser riktig ut
py -3.14 -m src.fetch_streetview_from_csv --limit 5

# Hent alle aktive kasser
py -3.14 -m src.fetch_streetview_from_csv
```

Nyttige flagg:
| Flagg | Hva det gjør |
|---|---|
| `--dry-run` | Beregner heading/pitch og sjekker metadata, men laster ikke ned bilder (gratis) |
| `--limit N` | Behandler bare de N første stedene (for testing) |
| `--include-inactive` | Tar også med kasser merket `Active=USANN` |
| `--reverse-geocode` | Slår opp gateadresse for hvert sted (ekstra API-kall) |
| `--size 640x480` | Bildestørrelse (maks 640x640 med standard nøkkel) |
| `--fov 120` | Synsvinkel i grader |
| `--radius 50` | Søkeradius i meter for nærmeste panorama |

Henter på nytt? Den hopper over bilder som allerede finnes — ingen dobbelthenting.
Logg over alt som hentes skrives til `data/streetview_log.csv`.

---

## 2. Hente Street View-bilder fra adresse-/koordinatfil

Leser `data/streetview_addresses.txt` — én oppføring per linje:
- **Gateadresse** → geokodes, snappes til nærmeste kasse i CSV-en (innen 30 m), sikter mot den
- **Koordinat** (`59.987806, 11.110444`) → sikter rett mot punktet (trenger ikke CSV)

```powershell
py -3.14 -m src.fetch_streetview
```

Nyttige flagg:
| Flagg | Hva det gjør |
|---|---|
| `--addresses <fil>` | Annen inputfil (default `data/streetview_addresses.txt`) |
| `--max-snap-distance 30` | Maks meter en adresse kan være fra en kasse før den hoppes over |
| `--csv <fil>` | Kasse-CSV brukt til adresse-snapping |
| `--size`, `--fov`, `--radius` | Som over |

Oppføringer slettes fra fila etter vellykket henting (adresser uten kasse i nærheten beholdes).

---

## 3. Annotere bilder

OpenCV-verktøy for å tegne bounding-bokser. Leser uannoterte bilder i `data/to_annotate/`,
eksporterer til poolen (`data/annotated_backup/`) og bygger om splittene ved sesjonsslutt.

```powershell
# Manuell modus — tegn alle bokser fra bunnen
py -3.14 -m src.labeling.annotate --mode manual

# Assistert modus — YOLO foreslår bokser du godkjenner/retter
py -3.14 -m src.labeling.annotate --mode assisted --model models/trained/trash_bin_yolo11n_best.pt
```

Taster: `a/Enter` godkjenn · `b` ingen kasse · `r` tegn på nytt · `s` hopp over · `q` avslutt.

| Flagg | Hva det gjør |
|---|---|
| `--mode manual\|assisted` | Tegn selv, eller la YOLO foreslå |
| `--model <vekter>` | Påkrevd i assistert modus |
| `--conf 0.25` | Konfidensgrense for YOLO-forslag |
| `--dir <mappe>` | Annen bildemappe (default `data/to_annotate/`) |

---

## 4. Bygge om train/val/test-splitt

Deler datasettet på nytt fra poolen med innholds-hash (hindrer at dupliserte
hard-example-bilder lekker til val/test). Kjør dette etter manuelle endringer i poolen.

```powershell
py -3.14 -m src.prepare_dataset
py -3.14 -m src.prepare_dataset --val-frac 0.15 --test-frac 0.15 --seed 1
```

⚠️ Ikke kjør dette mens en trening pågår — det endrer `data/annotated/` under trening.

---

## 5. Trene modellen

Finjusterer en YOLO-modell på `data/annotated/`. Vekter lagres under
`runs/detect/models/trained/<navn>/weights/best.pt`.

```powershell
py -3.14 -m src.train
py -3.14 -m src.train --epochs 100 --name run4
```

| Flagg | Hva det gjør |
|---|---|
| `--model yolo11n.pt` | Basismodell å finjustere fra |
| `--epochs 50` | Antall epoker |
| `--imgsz 640` | Bildestørrelse i piksler |
| `--batch 8` | Batch-størrelse |
| `--name run` | Navn på treningskjøringen |
| `--patience` | Early stopping: epoker uten forbedring før stopp |

(Tung trening gjøres normalt på Colab med GPU; beste modell så langt er `models/trained/trash_bin_yolo11n_best.pt`.)

---

## 6. Evaluere modellen

Kjører modellen på den holdte test-splitten og skriver ut Precision/Recall/mAP.

```powershell
py -3.14 -m src.evaluate
py -3.14 -m src.evaluate --model models/trained/trash_bin_yolo11n_best.pt
py -3.14 -m src.evaluate --split val
```

| Flagg | Hva det gjør |
|---|---|
| `--model <vekter>` | Hvilke vekter som evalueres (default run3) |
| `--split train\|val\|test` | Hvilken splitt (default test) |

---

## Typisk arbeidsflyt (aktiv læring)

```
1. Hent bilder        →  fetch_streetview_from_csv  (eller fetch_streetview)
2. Annoter            →  annotate --mode assisted --model <beste vekter>
3. (Splittes om automatisk ved slutten av annoteringsøkten)
4. Tren              →  train   (gjerne på Colab med GPU)
5. Evaluer           →  evaluate
6. Gjenta — hvert steg gir bedre forslag i annoteringen
```
