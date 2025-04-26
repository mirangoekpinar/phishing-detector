# Phishing Mail Detector

Dieses Projekt verwendet Machine Learning zur Erkennung von Phishing-Mails.  
Das Modell basiert auf TF-IDF-Vektorisierung und einem Naive Bayes-Klassifikator.

## Struktur
- `data/`: enthält Rohdaten (CSV) – lokal vorhanden, aber nicht im Repository
- `src/`: Preprocessing, Modelltraining, Evaluation
- `notebooks/`: explorative Analyse
- `ui/`: Streamlit-App
- `models/`: gespeicherte Modelle (pkl)
- `reports/`: Plots, Auswertungen

## Setup
```bash
pip install -r requirements.txt
