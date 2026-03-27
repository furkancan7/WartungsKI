# WartungsKI
### KI-gestützte Fehlererkennung für Maschinenanlagen

WartungsKI ist ein integriertes System zur Vorhersage von Maschinenausfällen (Predictive Maintenance). Es kombiniert eine Java Spring Boot-Webanwendung für die Benutzeroberfläche mit einem Python FastAPI-Service für Machine Learning-Vorhersagen.

---

## Datensätze & Ressourcen
Aufgrund der Dateigrößenbeschränkungen von GitHub sind die Rohdaten extern gespeichert:
* **Externer Speicher:** Google Drive
* **Link:** [Zum Datensatz-Ordner](https://drive.google.com/drive/folders/1MTvfTTlmT50STuFRAzwW-UDQxF5nvlgA?usp=sharing)

---

## Systemarchitektur
Das Projekt besteht aus zwei Hauptkomponenten:

1. **Frontend & Backend (Java Spring Boot):** Verwaltung der Benutzeroberfläche, Formularverarbeitung und Kommunikation mit dem API-Service.
2. **Prediction Service (Python FastAPI):** Ein spezialisierter Service, der trainierte RandomForest-Modelle nutzt, um Ausfallwahrscheinlichkeiten basierend auf Sensordaten zu berechnen.

---

## Installation und Ausführung
Um das System lokal zu starten, müssen beide Dienste gleichzeitig ausgeführt werden.

### 1. Python API-Service starten
Installieren Sie zunächst die erforderlichen Bibliotheken und starten Sie den FastAPI-Server:

```bash
# Erforderliche Bibliotheken
pip install fastapi uvicorn joblib numpy scikit-learn

# Service starten
uvicorn modelAPI:app --reload
