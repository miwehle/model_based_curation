# Model-Based Curation

## Big Picture

`model_based_curation` dient der modell-basierten Kuration von Trainingsdaten.

Im Gesamtworkflow gibt es zwei zusammenhaengende Ablaeufe:

1. DL-Pipeline:

   `Hugging Face dataset -> [data_preprocessor] -> train dataset -> [translator training] -> checkpoint`

2. Curation-Pipeline:

   `train dataset -> [split: teacher-forced loss scoring with translator] -> loss buckets -> [manual review with keep] -> reviewed buckets -> [filter] -> curated train dataset`

Fuer die Curation bewertet der `translator` Beispiele aus dem Trainingsdataset
ueber ihren Loss. `split` schreibt diese Beispiele in Loss-Buckets als
CSV-Dateien. Ein Mensch sichtet vor allem die High-Loss-Buckets und markiert
fachlich korrekte Beispiele ueber die Spalte `keep`. `filter` entfernt
anschliessend aus dem urspruenglichen Dataset nur noch die Bucket-Beispiele
ohne gesetztes `keep`.

So dient `model_based_curation` nicht nur zum Entfernen problematischer Daten,
sondern auch zum Sichtbarmachen von schwierigen, aber korrekten Beispielen.
Wiederkehrende Befunde aus den Buckets koennen spaeter genutzt werden, um die
fruehe Filterung im `data_preprocessor` gezielt zu verbessern.
Aus den Inhalten der Buckets lassen sich also schrittweise neue oder praezisere
Filter fuer den `data_preprocessor` finden. So wird manuelle Kuration nach und
nach in fruehere, automatische Qualitaetssicherung ueberfuehrt.

## Struktur

- `src/model_based_curation/`
- `tests/`
- `scripts/split.py`

## Startpunkt

Die oeffentliche Paketoberflaeche liegt in `src/model_based_curation/__init__.py`.
Der praktische Einstieg fuer das Triggern des Splits ist `scripts/split.py`.
