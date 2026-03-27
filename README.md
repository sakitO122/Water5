# Water5 - Système d'Irrigation Intelligent

> **Irrigation intelligente par IA - Yamoussoukro, Côte d'Ivoire**  
> INP-HB · Année académique 2025–2026 · Finale Aquatech 2026 · 2ème place

---

## Table des matières

- [Présentation du projet](#présentation-du-projet)
- [Contexte et problématique](#contexte-et-problématique)
- [Architecture du système](#architecture-du-système)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Le modèle d'IA](#le-modèle-dia)
- [L'API météo Open-Meteo](#lapi-météo-open-meteo)
- [L'application web Water5](#lapplication-web-water5)
- [Méthode agronomique FAO-56](#méthode-agronomique-fao-56)
- [Performances](#performances)
- [Équipe](#équipe)

---

## Présentation du projet

Water5 est un système d'irrigation intelligente conçu pour les petites parcelles maraîchères de Côte d'Ivoire. Il détermine automatiquement **quand irriguer** et **quelle quantité d'eau utiliser** en combinant des données météorologiques temps réel, un calcul agronomique basé sur la norme FAO-56, et un modèle de Machine Learning entraîné sur 3 ans de données locales.

Le système produit deux résultats à chaque analyse :
- Une **décision binaire** : ARROSER (1) ou NE PAS ARROSER (0)
- Un **volume d'eau en litres** calculé au litre près pour une parcelle de 200 m²

---

## Contexte et problématique

Dans plusieurs régions de Côte d'Ivoire, notamment autour de Yamoussoukro, la culture maraîchère de la tomate représente une activité économique importante. L'irrigation y est réalisée manuellement, basée sur l'expérience de l'agriculteur, sans données météorologiques ni mesure d'humidité du sol.

Cette pratique entraîne un **gaspillage de 40 à 60% de l'eau** utilisée et peut provoquer un stress hydrique en période de forte chaleur.

**Question centrale :** Comment concevoir un système capable d'analyser les conditions météorologiques et l'humidité du sol afin de décider automatiquement quand irriguer et quelle quantité d'eau utiliser ?

---

## Architecture du système

```
┌─────────────────────────────────────────────────────────────┐
│                      SYSTÈME WATER5                         │
├──────────────┬──────────────────────────┬───────────────────┤
│   LE TERRAIN │       LE CERVEAU IA      │    L'INTERFACE    │
│              │                          │                   │
│  Capteur sol │  Python                  │  App Web Water5   │
│  DHT22       │  Random Forest (ML)      │  Android / iOS    │
│  Arduino Uno │  API Open-Meteo          │  Navigateur web   │
│  Relais      │  Calcul ET0 FAO-56       │                   │
│  Pompe 12V   │  FastAPI (serveur)       │  M. Koffi         │
└──────────────┴──────────────────────────┴───────────────────┘
```

**Flux de données :**

```
Open-Meteo API
      ↓
06_api_openmeteo.py  ←── Modèles ML (.joblib)
      ↓
api.py (FastAPI :8000)
      ↓
App Web (index.html / app.js)
      ↓
M. Koffi (smartphone)
```

---

## Structure du projet

```
irrigation_ml/
│
├── src/
│   ├── config.py                  # Configuration centralisée (seuils, coordonnées, features)
│   ├── agronomie.py               # Penman-Monteith FAO-56 + Kc dynamique tomate
│   ├── 01_preparation_donnees.py  # Préparation du dataset ML 2022-2024
│   ├── 02_entrainement_ml.py      # Entraînement Random Forest + XGBoost
│   ├── 04_backtesting.py          # Simulation sur le dataset historique
│   ├── 06_api_openmeteo.py        # Prédiction J+3 via API Open-Meteo (CLI)
│   └── api.py                     # Serveur FastAPI (endpoints /analyser, /decision)
│
├── app web/
│   ├── index.html                 # Application mobile Water5
│   ├── app.js                     # Logique frontend (appels API, affichage)
│   ├── style.css                  # Styles de l'interface
│   └── manifest.json              # Configuration PWA
│
├── data/
│   ├── open_meteo_brut.csv        # Données brutes Open-Meteo (2022-2024)
│   └── yamoussoukro_dataset_ML.csv # Dataset nettoyé prêt pour le ML
│
├── models/
│   ├── modele_classification.joblib  # Random Forest — décision OUI/NON
│   └── modele_regression.joblib      # Random Forest — volume en litres
│
├── outputs/
│   ├── rapport_entrainement.txt      # Métriques d'entraînement
│   ├── importance_classification.png # Importance des variables
│   ├── matrice_confusion.png         # Matrice de confusion (jeu de test)
│   ├── pred_vs_reel_regression.png   # Prédit vs Réel (régression)
│   └── historique_decisions.csv      # Log des décisions quotidiennes
│
└── README.md
```

---

## Prérequis

- **Python** 3.10 ou supérieur
- **pip** (gestionnaire de paquets Python)
- Connexion Internet (pour l'API Open-Meteo)

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-repo/irrigation_ml.git
cd irrigation_ml
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Dépendances principales :**

```
pandas
numpy
scikit-learn
xgboost
joblib
fastapi
uvicorn
openmeteo-requests
requests-cache
retry-requests
matplotlib
```

### 3. Préparer le dataset

```bash
python src/01_preparation_donnees.py
```

Cette commande lit le fichier `data/open_meteo_brut.csv`, nettoie les données, calcule ET0 Penman-Monteith, les coefficients Kc dynamiques, et génère `data/yamoussoukro_dataset_ML.csv`.

### 4. Entraîner les modèles

```bash
python src/02_entrainement_ml.py
```

Entraîne le Random Forest Classifier (décision) et le Random Forest Regressor (volume). Les modèles sont sauvegardés dans `models/`.

---

## Configuration

Tous les paramètres sont centralisés dans **`src/config.py`**. Aucune autre modification de fichier n'est nécessaire.

```python
# Localisation
LATITUDE_DEG  = 6.8205       # Yamoussoukro
LONGITUDE_DEG = -5.2767
ALTITUDE_M    = 212.0
TIMEZONE      = "Africa/Abidjan"

# Parcelle
SURFACE_M2    = 200          # Surface en m²
EFFICACITE    = 0.90         # Efficacité système goutte-à-goutte

# Seuils agronomiques
SOL_HUMIDE_SEUIL   = 70.0   # % : pas d'irrigation au-dessus
SOL_MOYEN_SEUIL    = 50.0   # % : combiné avec pluie modérée
PLUIE_FORTE_SEUIL  = 10.0   # mm : irrigation annulée
PLUIE_MODERE_SEUIL =  5.0   # mm : pluie modérée
PLUIE_EFFECTIVE_PCT = 0.80  # 80% des précipitations utiles (FAO)
```

**Pour adapter à une autre culture**, modifiez uniquement :

```python
STADES_JOURS = { ... }   # Durées des stades phénologiques
KC_FAO_BASE  = { ... }   # Coefficients Kc (Tableau 12 FAO-56)
HAUTEUR_CULTURE_M = ...  # Hauteur de la culture en mi-saison
```

---

## Utilisation

### Mode 1 — Prédiction en ligne de commande (sans serveur)

```bash
python src/06_api_openmeteo.py
```

Interroge Open-Meteo en temps réel, calcule la décision pour aujourd'hui et les 3 prochains jours, et affiche :

```
============================================================
  WATER5 - IRRIGATION INTELLIGENTE - M. KOFFI
  Yamoussoukro, CI | Tomate | 200m2
============================================================

  AUJOURD'HUI (J)    (2026-03-22)
  Temp : 34.5/24.1C | Pluie : 0.0mm | Sol : 40.0%
  ET0 : 4.58mm | ETc : 5.45mm | Déficit : 5.45mm | Kc : 1.1917
  DECISION : ARROSER
  Volume  : 1303L (parcelle 200m2)
```

**Options disponibles :**

```bash
python src/06_api_openmeteo.py --jour-cycle 60   # Jour depuis plantation
python src/06_api_openmeteo.py --no-capteur      # Sans capteur Arduino
python src/06_api_openmeteo.py --test            # Affiche données brutes API
```

### Mode 2 — Serveur FastAPI + Application web

**Lancer le serveur :**

```bash
cd irrigation_ml/src
python api.py
```

Le serveur démarre sur `http://localhost:8000`.

**Accéder à l'application web :**

Ouvrez `http://localhost:8000` dans votre navigateur ou sur votre smartphone (même réseau Wi-Fi).

**Endpoints disponibles :**

| Endpoint | Méthode | Description |
|---|---|---|
| `/analyser` | GET | Décision complète J+3 (Open-Meteo côté serveur) |
| `/decision` | POST | Décision depuis données fournies par le client |
| `/health` | GET | Vérification état des modèles |
| `/` | GET | Application web Water5 |

**Exemple d'appel API :**

```bash
curl http://localhost:8000/analyser?jour_cycle=60
```

**Réponse JSON :**

```json
{
  "timestamp": "2026-03-22T08:15:00",
  "source_meteo": "Open-Meteo (serveur)",
  "aujourd_hui": {
    "date": "2026-03-22",
    "irriguer": true,
    "volume_L": 1303.0,
    "confiance_pct": 100.0,
    "source": "Random Forest (Water5 2022-2024)",
    "stade": "mi_saison",
    "kc": 1.1917,
    "ETc_mm": 5.45,
    "deficit_mm": 5.45,
    "ET0_mm": 4.58,
    "pluie_mm": 0.0,
    "humidite_sol": 40.0,
    "temp_max_C": 34.5,
    "temp_min_C": 24.1
  },
  "previsions": [ ... ]
}
```

### Mode 3 — Backtesting sur données historiques

```bash
python src/04_backtesting.py 2024-04-23
python src/04_backtesting.py 2023-11-01 full   # + rapport annuel
```

Compare les prédictions du modèle aux décisions réelles connues sur le dataset 2022-2024.

---

## Le modèle d'IA

### Algorithme

Le système utilise deux modèles **Random Forest** distincts :

| Modèle | Type | Sortie |
|---|---|---|
| Modèle 1 | Classification binaire | ARROSER (1) ou NE PAS ARROSER (0) |
| Modèle 2 | Régression numérique | Volume d'eau en litres |

### Données d'entraînement

- **Période** : 2022–2024 (1 096 jours)
- **Source** : API Open-Meteo — Yamoussoukro (6.82°N 5.28°W, 212m)
- **Distribution** : 822 jours irrigués (75%) · 274 jours non irrigués (25%)

### Features du modèle (16 variables)

```python
FEATURES = [
    "humidite_sol_moy_pct",    # Humidité sol moyenne (%)
    "humidite_sol_min_pct",    # Humidité sol minimale (%)
    "humidite_sol_0_7_moy",    # Humidité sol 0-7cm (%)
    "temp_max_C",              # Température maximale (°C)
    "temp_min_C",              # Température minimale (°C)
    "temp_moy_C",              # Température moyenne (°C)
    "humidite_air_moy_pct",    # Humidité air moyenne (%)
    "vent_u2_ms",              # Vent à 2m (m/s)
    "rayonnement_Rs_MJ",       # Rayonnement solaire (MJ/m²)
    "ET0_reference_mm",        # ET0 Open-Meteo (mm/j)
    "ETc_mm",                  # ETc = Kc × ET0 (mm/j)
    "deficit_hydrique_mm",     # ETc - Pluie effective (mm)
    "pluie_totale_mm",         # Précipitations (mm)
    "pluie_effective_mm",      # Pluie × 0.80 FAO (mm)
    "jour_annee",              # Jour julien (1-365)
    "mois",                    # Mois (1-12)
]
```

### Split d'entraînement

Le split est **chronologique 80/20** (pas aléatoire) pour éviter le leakage temporel :
- Train : 2022-01-01 → 2023-09-14 (876 jours)
- Test  : 2023-09-15 → 2024-12-31 (220 jours)

La validation croisée utilise **TimeSeriesSplit(5)** pour respecter l'ordre temporel des données.

---

## L'API météo Open-Meteo

Le système interroge [Open-Meteo](https://open-meteo.com/) pour obtenir les données météo en temps réel et les prévisions sur 4 jours.

**Paramètres horaires récupérés :**
- Température 2m, pluie, vent 10m
- Humidité sol 7-28cm et 0-7cm
- Humidité relative

**Paramètres quotidiens récupérés :**
- Températures min/max/moyenne
- Rayonnement solaire (MJ/m²)
- ET0 FAO (évapotranspiration de référence)
- Précipitations totales

**Pas de clé API requise** — Open-Meteo est gratuit et sans authentification.

---

## L'application web Water5

L'application est une **Progressive Web App (PWA)** développée en HTML/CSS/JavaScript pur. Elle fonctionne sur Android, iOS et navigateur web.

**Fonctionnalités :**
- Analyse en temps réel via le serveur FastAPI
- Décision du jour + prévisions 4 jours
- Jauge d'humidité du sol
- Historique des 60 dernières analyses
- Aperçu SMS formaté pour M. Koffi
- Alertes et notifications

**Pour utiliser l'app sans serveur Python**, ouvrez directement `app web/index.html` dans un navigateur, elle appellera Open-Meteo directement et calculera ET0/ETc/Kc en JavaScript avec les mêmes formules que le Python.

---

## Méthode agronomique FAO-56

Le calcul des besoins hydriques suit la norme **FAO Irrigation and Drainage Paper No. 56** (Allen et al., 1998).

### Évapotranspiration de référence ET0

La méthode **Penman-Monteith FAO-56** (équation 6) :

```
ET0 = [0.408·Δ·(Rn-G) + γ·(900/(T+273))·u2·(es-ea)] / [Δ + γ·(1+0.34·u2)]
```

Où :
- `Δ` : pente de la courbe de vapeur saturante (kPa/°C)
- `Rn` : rayonnement net (MJ/m²/j)
- `γ` : constante psychrométrique (kPa/°C)
- `u2` : vent à 2m (m/s) - converti depuis 10m via FAO-56 eq.47
- `es` : pression de vapeur saturante (kPa)
- `ea` : pression de vapeur réelle (kPa)

### Évapotranspiration de la culture ETc

```
ETc = Kc × ET0
```

Le coefficient cultural **Kc** est dynamique selon le stade phénologique de la tomate :

| Stade | Durée | Kc |
|---|---|---|
| Initial | 0–25 jours | 0.45 |
| Croissance | 25–60 jours | interpolé linéairement |
| Mi-saison | 60–110 jours | 1.15 |
| Fin de saison | 110–135 jours | 0.80 |

### Décision d'irrigation - Règles agronomiques

Avant le modèle IA, quatre règles sont appliquées en priorité :

```python
if humidite_sol > 70%          → NE PAS ARROSER (sol humide)
if pluie > 10mm                → NE PAS ARROSER (forte pluie)
if pluie > 5mm AND sol > 50%   → NE PAS ARROSER (pluie modérée + sol ok)
if deficit <= 0                → NE PAS ARROSER (pas de déficit)
else                           → MODÈLE IA décide
```

### Calcul du volume d'eau

```
Volume (L) = Déficit (mm) × Facteur_sol × Surface (m²) / Efficacité
```

Avec `Facteur_sol = max(0, (65 - humidite_sol) / 25)`

---

## Performances

Évaluation sur le jeu de **TEST chronologique** (jamais vu pendant l'entraînement) :

| Indicateur | Résultat | Seuil idéal |
|---|---|---|
| Accuracy (classification) | **98%** | > 95% |
| Précision | 100% | > 90% |
| Rappel | 98.39% | > 80% |
| F1-Score | **99.55%** | > 90% |
| AUC-ROC | **1.0000** | > 0.99 |
| R² régression (volume) | **0.9954** | > 0.90 |
| MAE régression | **18.4 L** | < 100 L |
| RMSE régression | 30.7 L | < 150 L |
| Stabilité cross-validation | **99.63% ± 0.34%** | — |

---

## Équipe

| Nom | Filière | Rôle |
|---|---|---|
| **Saih Piely Uriel Loic** |ING STIC2 INFO | Chef du groupe et Responsable Machine Learning et Modèles IA|
| Nebie Ange-Michel | ING STIC1 | Responsable Données, Documentation et Support Technique|
| YAO Milikédan Erica | Ts2 GAE GSC  | Responsable Agronomie |
| Traore Christ-Ismael | ING STGI2 | Responsable Matériel, Simulation & Arduino
| Zoukou Kenny Larry | ING STIC2 INFO | Responsable Développement Web et Interface |

**Institut National Polytechnique Félix Houphouët-Boigny (INP-HB)**  
Yamoussoukro, Côte d'Ivoire  
Année académique 2025–2026

---

*Water5 - Finale Aquatech 2026 · 2ème place*