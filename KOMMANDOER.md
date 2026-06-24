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
# Del hovedfila (data/hentesteder.csv) i biter à 1000 hentesteder (kjør én gang)
py -3.14 -m src.split_hentesteder

# Test: bare geometri, henter ingen bilder (gratis)
py -3.14 -m src.fetch_streetview_from_csv --csv data/hentesteder_chunks/hentesteder_001.csv --dry-run --limit 5

# Hent ÉN bit av gangen (~1000 hentesteder) — med deteksjon + retry
py -3.14 -m src.fetch_streetview_from_csv --csv data/hentesteder_chunks/hentesteder_001.csv

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

Hovedfila `data/hentesteder.csv` har ~196 000 rader (én per beholder) men bare ~55 000
unike hentesteder. Del den derfor **først** i biter à 1000 unike steder, og hent **én bit
av gangen** — det er sikkerheten mot å bruke opp API-kvoten ved et uhell. Skriptet finner
nærmeste panorama og sikter kameraet rett mot stedet. Bildene lagres i `data/to_annotate/`.

**Del opp først (kjør én gang, gratis):**
```powershell
py -3.14 -m src.split_hentesteder      # -> data/hentesteder_chunks/hentesteder_001.csv, _002 ...
```

**Deteksjon + retry:** etter at det nærmeste bildet er hentet, kjøres YOLO på det. Ser
modellen en kasse, går vi videre. Ser den ingen, hentes også det *nest nærmeste* panoramaet
(med ny heading/pitch). Under annotering beholdes **begge** bilder (det nest nærmeste får
`_p2.jpg`), siden YOLO kan bomme — du avgjør selv ved annotering. `--no-detect` gir gammel
oppførsel (bare nærmeste, ingen YOLO).

Koster opptil **2 bildehentinger per sted** (1 hvis YOLO ser kassen med en gang).
Metadata-søk er gratis. Krever at `--model`-vektene finnes.

```powershell
# Tørrkjøring på én bit: bare geometri + gratis metadata, ingen bilder, ingen YOLO
py -3.14 -m src.fetch_streetview_from_csv --csv data/hentesteder_chunks/hentesteder_001.csv --dry-run --limit 5

# Hent ~5 nye bilder fra biten for å se om sikting ser riktig ut
py -3.14 -m src.fetch_streetview_from_csv --csv data/hentesteder_chunks/hentesteder_001.csv --limit 5

# Hent hele biten (~1000 steder, med deteksjon + retry)
py -3.14 -m src.fetch_streetview_from_csv --csv data/hentesteder_chunks/hentesteder_001.csv

# Gammel oppførsel: bare nærmeste panorama, ingen YOLO
py -3.14 -m src.fetch_streetview_from_csv --csv data/hentesteder_chunks/hentesteder_001.csv --no-detect
```

Nyttige flagg:
| Flagg | Hva det gjør |
|---|---|
| `--csv <fil>` | Hvilken bit som hentes, f.eks. `data/hentesteder_chunks/hentesteder_001.csv` |
| `--no-detect` | Skru av YOLO + retry — hent bare nærmeste panorama (gammel oppførsel) |
| `--model <vekter>` | YOLO-vekter for deteksjon (default `models/trained/trash_bin_yolo11n_best.pt`) |
| `--conf 0.25` | Konfidensgrense for «ser YOLO en kasse?» |
| `--max-attempts 2` | Panorama per sted: 1 = bare nærmeste, 2 = også nest nærmeste ved bom |
| `--ring-radius 18` | Søkeradius (m) for å finne det nest nærmeste panoramaet |
| `--dry-run` | Beregner heading/pitch + metadata, laster ikke ned bilder, ingen YOLO (gratis) |
| `--limit N` | Stopp etter ~N nye bilder (et sted kan gi 2 ved retry) |
| `--include-inactive` | Tar med `Active=USANN` (kun Uttrekk-formatet; hentesteder har ingen Active-kolonne) |
| `--reverse-geocode` | Slår opp gateadresse for hvert sted (ekstra API-kall) |
| `--size 640x480` | Bildestørrelse (maks 640x640 med standard nøkkel) |
| `--fov 120` | Synsvinkel i grader |
| `--radius 50` | Søkeradius i meter for nærmeste panorama |

Henter på nytt? Den hopper over bilder som allerede finnes — ingen dobbelthenting.
Logg over alt som hentes skrives til `data/streetview_log.csv` (nye kolonner:
`attempts`, `detected`, `detection_conf`, `pano_rank`, `pair`).

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

## 3.5 Lage segmenteringsdatasett (SAM2 + ADE20K)

Bygger et YOLO-seg-datasett (`0 trash_bin`, `1 ground`) fra det eksisterende
deteksjonsdatasettet — uten å tegne masker manuelt. For hvert bilde i
`data/annotated/` brukes den eksisterende bounding-boksen som prompt til **SAM2**
(lager kassemaske), og en ferdigtrent **ADE20K-modell** slår sammen
road/sidewalk/earth/grass til én `ground`-maske. Usikre kassemasker flagges til
`data/annotated_seg/review_flags.csv`, og hvert bilde får en forhåndsvisning med
maskene tegnet på.

Leser kun fra `data/annotated/` (skriver aldri dit, rører aldri
`annotated_backup`). Det nye datasettet speiler train/val/test-splitten fra kilden.
Modellvektene lastes ned én gang og kjører på GPU hvis tilgjengelig, ellers CPU.

```powershell
# Røyktest: behandle 3 bilder (rask, liten modell)
py -3.14 -m src.labeling.sam2_seg_autolabel --limit 3 --semantic-model nvidia/segformer-b0-finetuned-ade-512-512

# Full kjøring (default B5-modell, helst på GPU — Colab eller 4070)
py -3.14 -m src.labeling.sam2_seg_autolabel
```

| Flagg | Hva det gjør |
|---|---|
| `--source <mappe>` | Deteksjonsdatasett å lese bilder + bokser fra (default `data/annotated`) |
| `--output <mappe>` | Hvor seg-datasettet skrives (default `data/annotated_seg`) |
| `--sam-model <vekter>` | SAM2-vekter for kassemasker (default `sam2.1_b.pt`) |
| `--semantic-model <navn>` | HF ADE20K-modell for ground (default `nvidia/segformer-b5-finetuned-ade-640-640`) |
| `--splits train val test` | Hvilke splitter som behandles |
| `--limit N` | Stopp etter N nye bilder (ferdige hoppes over) |
| `--overwrite` | Behandle bilder på nytt selv om seg-label finnes |
| `--device cpu\|cuda` | Tving enhet (default: auto) |
| `--clip-margin 0.15` | Hvor mye boksen utvides før kassemasken klippes (hindrer at SAM2 tar med gjerder ved siden av) |

Kjør på nytt? Bilder som allerede har en seg-label hoppes over. Etter kjøring:
sjekk forhåndsvisningene i `data/annotated_seg/previews/` og de flaggede bildene i
`review_flags.csv` før du trener. Treningsskriptet for YOLO-seg legges til etter
at du har kontrollert annoteringene.

Dette steget kjører hele datasettet i én batch. Vil du i stedet godkjenne ett og
ett bilde mens SAM2 kjører fortløpende, bruk den interaktive gjennomgangen under.

---

## 3.6 Interaktiv gjennomgang (godkjenn ett bilde om gangen)

Samme pipeline som 3.5, men SAM2 + ground-modellen kjøres på **ett bilde om gangen**
og maskene vises i et OpenCV-vindu der du godkjenner eller hopper over hvert bilde.
Et godkjent bilde gir nøyaktig samme label som batch-kjøringen. Mens du ser på ett
bilde, segmenteres de neste i bakgrunnen (look-ahead-buffer på 5 bilder), så det
føles raskt etter det første — også når du klikker raskt gjennom flere på rad.

> Vinduet trenger en lokal skjerm — kjør dette på din egen maskin, **ikke** på en
> headless Colab.

```powershell
# Gå gjennom alt som mangler en seg-label
py -3.14 -m src.labeling.sam2_seg_review

# Bare én split (f.eks. val)
py -3.14 -m src.labeling.sam2_seg_review --splits val
```

Taster i vinduet:

| Tast | Handling |
|---|---|
| `a` / Enter / mellomrom | **Godkjenn** — skriver bilde + label + forhåndsvisning til seg-datasettet |
| `s` | **Hopp over** — skriver en `.skip`-markør; bildet dukker **ikke** opp igjen (bruk `--overwrite` for å se det på nytt) |
| `f` | **Flagg** — logg til `review_flags.csv` + `.skip`-markør (manuell-fiks-bunken, kommer ikke igjen) |
| `o` | Slå maske-overlegget av/på (sammenlign mot originalbildet) |
| `q` / Esc | **Avslutt** — fremdrift er lagret; fortsetter der du slapp |

Samme flagg som 3.5 (`--source`, `--output`, `--sam-model`, `--semantic-model`,
`--splits`, `--overwrite`, `--device`, `--clip-margin`), pluss `--limit N` som
stopper etter N gjennomgåtte bilder denne økten, og `--prefetch N` som styrer hvor
mange bilder som segmenteres i forkant (default 5).

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

## 5.5 Trene segmenteringsmodell (YOLO-seg)

Trener en YOLO-segmenteringsmodell på seg-datasettet (`data/annotated_seg/`, klasser
`0 trash_bin` + `1 ground`) bygget i 3.5/3.6. Speiler `src.train`, men bruker
`configs/data_seg.yaml` og en `-seg`-basemodell. Ultralytics kjenner igjen
segmenteringsoppgaven og lagrer under `runs/segment/...`, så deteksjonsvektene i
`runs/detect/...` røres aldri.

```powershell
py -3.14 -m src.train_seg
py -3.14 -m src.train_seg --epochs 100 --name seg2
```

| Flagg | Hva det gjør |
|---|---|
| `--model yolo11n-seg.pt` | Basis-segmenteringsmodell å finjustere fra |
| `--epochs 50` | Antall epoker |
| `--imgsz 640` | Bildestørrelse i piksler |
| `--batch 8` | Batch-størrelse |
| `--name seg` | Navn på treningskjøringen |
| `--patience 20` | Early stopping: epoker uten forbedring før stopp |

Vekter lagres under `runs/segment/models/trained/<navn>/weights/best.pt`.
Krever at seg-datasettet er bygget (3.5/3.6) og at `data/annotated_seg/` har
bilder i train/val. Tung trening gjøres normalt på Colab med GPU.

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

## 7. Statistikk: treff på andre forsøk

Leser `data/streetview_log.csv` og viser hvor ofte YOLO fant en kasse på det
*nest nærmeste* panoramaet, blant stedene der det nærmeste bommet (retry).

```powershell
py -3.14 -m src.streetview_stats
py -3.14 -m src.streetview_stats --log data/streetview_log.csv
```

Skriver ut antall andreforsøk, antall treff og prosenten. (Krever at du har kjørt
`fetch_streetview_from_csv` med deteksjon på, så loggen har `pano_rank`-kolonnen.)

---

## 8. Estimere GPS-posisjon til kasser

Kjører seg-modellen på Street View-bilder og regner ut hvor hver detekterte kasse
står (lat/lng), ved å projisere kassen tilbake gjennom et pinhole-kameramodell.
Kameradata (pano-posisjon, heading, pitch) hentes fra `data/streetview_log.csv`
per filnavn; FOV er ikke logget og settes med `--fov` (hentingen bruker 80).
Resultatet skrives til `outputs/bin_positions/bin_positions.csv`.

```powershell
# Bakke-geometri (standard, mest nøyaktig): stråle gjennom kassens bunn ∩ bakkeplan
py -3.14 -m src.estimate_bin_positions --images data/annotated_seg/images

# Bakke-geometri med dybde-fallback for kasser som ikke står på flat bakke
py -3.14 -m src.estimate_bin_positions --images data/annotated_seg/images --method both

# Kun DAV2-dybde (laster Depth Anything V2 fra HuggingFace)
py -3.14 -m src.estimate_bin_positions --method depth

# Ett enkelt bilde
py -3.14 -m src.estimate_bin_positions --images data/to_annotate/streetview_10000001.jpg

# Se maskene + back-projection-punktene (og dybdekart for depth/both)
py -3.14 -m src.estimate_bin_positions --method both --save-vis --limit 20
```

Med `--save-vis` skrives ett bilde per kasse-bilde til `outputs/bin_positions/previews/`:
maskene tegnes som farget overlegg (grønn=bakke, oransje=dybde), og punktet som
projiseres tilbake markeres med en prikk + etikett (`metode avstand avvik`). For
`depth`/`both` lagres også et dybdekart `<navn>_depth.jpg` (blå=nær, rød=fjern).

| Flagg | Hva det gjør |
|---|---|
| `--images <fil\|mappe>` | Bilde eller mappe (søkes rekursivt) (default `data/annotated_seg/images`) |
| `--method ground\|depth\|both` | Estimeringsmetode (default `ground`) |
| `--fov <grader>` | Horisontal FOV, må matche hentingen (default 80) |
| `--conf <terskel>` | Konfidensterskel for seg-deteksjon (default 0.25) |
| `--camera-height <m>` | Kamerahøyde for bakke-geometri (default 2.5) |
| `--max-ground-dist <m>` | Forkast bakke-estimat over denne avstanden (default 60) |
| `--seg-weights <vekter>` | YOLO-seg vekter (default colab_seg) |
| `--save-vis` | Lagre maske-overlegg + dybdekart til `<output>/previews/` |
| `--device cuda\|cpu` | Enhet (default auto) |
| `--limit <N>` | Stopp etter N bilder |

Når manifestet har kassens egen koordinat (`bin_lat`/`bin_lng`) skrives den ut som
fasit, og scriptet rapporterer median/snitt avvik. På testkjøring ga `ground`
~2.3 m median ved FOV 80 (nær forventningsrett) mot ~22 m for `depth` — derfor er
`ground` standard. Bilder uten manifestrad (hentet via `fetch_streetview`) hoppes over.

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
