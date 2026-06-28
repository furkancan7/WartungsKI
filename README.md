# 🛠️ WartungsKI  
### 🚀 Akıllı Makine Sağlığı & Kestirimci Bakım Platformu | Intelligente Plattform für Maschinenzustand & Predictive Maintenance

WartungsKI, endüstriyel makinelerden gelen sensör verilerini yalnızca izlemekle kalmayıp, onları analiz eden, anlamlandıran ve gelecekte oluşabilecek arızaları tahmin eden yapay zeka tabanlı bir kestirimci bakım sistemidir.  
WartungsKI ist ein KI-basiertes Predictive-Maintenance-System, das industrielle Sensordaten nicht nur überwacht, sondern analysiert, interpretiert und zukünftige Ausfälle vorhersagt.

Amaç, arızaları oluşmadan önce tespit ederek üretim kayıplarını ve bakım maliyetlerini minimuma indirmektir.  
Ziel ist es, Ausfälle frühzeitig zu erkennen und Produktionsverluste sowie Wartungskosten zu minimieren.

---

## ⚡ Temel Fikir | Kernidee

> “Makine verisini geleceğe dönüştürmek.”  
> „Maschinendaten in Zukunftsprognosen verwandeln.“

WartungsKI, sensör verilerindeki gizli desenleri öğrenerek olası arızaları önceden tahmin eder ve makine sağlığını sürekli olarak değerlendirir.  
WartungsKI erkennt versteckte Muster in Sensordaten, prognostiziert mögliche Ausfälle und bewertet kontinuierlich den Maschinenzustand.

---

## 🧠 Öne Çıkan Özellikler | Hauptfunktionen

- 🔍 Erken arıza tespiti | Frühzeitige Fehlererkennung  
- 📉 Arıza olasılığı tahmini | Ausfallwahrscheinlichkeitsprognose  
- 📡 Sensör verisi analizi | Sensordatenanalyse  
- ⚙️ FastAPI ile gerçek zamanlı ML tahmini | Echtzeit-ML-Inferenz über FastAPI  
- 🧩 Java Spring Boot tabanlı modüler mimari | Modulare Architektur mit Java Spring Boot  
- 📊 Ölçeklenebilir endüstriyel sistem tasarımı | Skalierbares industrielles Systemdesign  

---

## 🏗️ Sistem Mimarisi | Systemarchitektur

**Java Spring Boot (Backend & UI)**  
Kullanıcı arayüzünü yönetir, veri akışını kontrol eder ve sistem mantığını çalıştırır.  
Verwaltet die Benutzeroberfläche, steuert Datenflüsse und Systemlogik.

**Python FastAPI (Yapay Zeka Servisi)**  
Sensör verilerini alır ve eğitilmiş RandomForest modelleri ile arıza olasılığı tahmini yapar.  
Nimmt Sensordaten entgegen und berechnet Ausfallwahrscheinlichkeiten mit trainierten RandomForest-Modellen.

---

## ⚙️ Teknoloji Yığını | Tech Stack

- Java (Spring Boot)  
- Python (FastAPI)  
- Scikit-learn  
- RandomForest algoritmaları | RandomForest-Modelle  
- NumPy / Pandas  
- REST API mimarisi | REST-API-Architektur  
- Git & GitHub  

---

## 📊 Veri Stratejisi | Datenstrategie

Büyük veri setleri nedeniyle eğitim verileri harici olarak saklanmaktadır ve gerektiğinde bulut üzerinden erişilmektedir.  
Aufgrund großer Datensätze werden Trainingsdaten extern gespeichert und bei Bedarf über die Cloud abgerufen.

🔗 Veri seti | Datensatz: Google Drive (https://drive.google.com/drive/u/3/folders/1MTvfTTlmT50STuFRAzwW-UDQxF5nvlgA)

---

## 🚀 Neden Önemli? | Warum ist das wichtig?

Endüstriyel arızalar ciddi maliyetlere yol açar. WartungsKI, bakım yaklaşımını reaktif sistemlerden proaktif ve akıllı sistemlere dönüştürmeyi hedefler.  
Industrielle Ausfälle verursachen hohe Kosten. WartungsKI transformiert Wartung von reaktiv zu proaktiv und intelligent.

---

## 📦 Kurulum | Installation

```bash
# Gerekli kütüphaneler | Required dependencies
pip install fastapi uvicorn joblib numpy scikit-learn

# API servisini başlat | Start API service
uvicorn modelAPI:app --reload
