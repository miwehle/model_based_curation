# Model-Based Curation

## Das Big Picture

`model_based_curation` unterstuetzt die manuelle und modellgestuetzte Kuration
bestehender Trainingsdatensaetze.

Im Gesamtworkflow gibt es zwei zusammenhaengende Ablaeufe:

1. DL-Pipeline:
   `Hugging-Face-Dataset -> data_preprocessor -> Trainingsdataset -> Training des translators`

2. Curations-Pipeline:
   `Trainingsdataset -> Bewertung mit dem translator -> manuelle Sichtung der Buckets -> kuratiertes Trainingsdataset`

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

## Struktur

- `src/model_based_curation/`
- `tests/`
- `scripts/split.py`

## Startpunkt

Die oeffentliche Paketoberflaeche liegt in `src/model_based_curation/__init__.py`.
Der praktische Einstieg fuer das Triggern des Splits ist `scripts/split.py`.
