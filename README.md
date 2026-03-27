# Water5 - Irrigation Intelligente

## Description

Water5 est un système d'irrigation intelligente utilisant l'intelligence artificielle pour aider les maraîchers à irriguer au bon moment et avec le bon volume d'eau. Le système analyse les données météorologiques en temps réel, applique des modèles de machine learning, et fournit des recommandations d'irrigation via une application web.

## Fonctionnalités

- **Analyse météorologique** : Récupération des données d'Open-Meteo
- **Modèles ML** : Classification (oui/non irrigation) et régression (volume d'eau)
- **Application web** : Interface PWA pour consulter les décisions
- **API** : Serveur FastAPI pour l'intégration
- **Évaluation** : Scripts pour backtesting et évaluation des modèles

## Installation

1. Cloner le dépôt :

   ```bash
   git clone https://github.com/sakitO122/Water5.git
   cd Water5
   ```

2. Installer les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

3. Préparer les données et entraîner les modèles :
   ```bash
   python src/01_preparation_donnees.py
   python src/02_entrainement_ml.py
   ```

## Utilisation

### Lancer l'application web

```bash
cd src
python api.py
```

Ouvrez votre navigateur à l'adresse `http://localhost:8000` pour accéder à l'application.

### Utilisation en ligne de commande

- Obtenir une décision d'irrigation :

  ```bash
  python src/06_api_openmeteo.py
  ```

- Évaluer le modèle :
  ```bash
  python src/05_evaluation.py
  ```

## Structure du projet

- `app web/` : Application web PWA
- `src/` : Scripts Python (préparation, entraînement, API)
- `models/` : Modèles ML entraînés
- `data/` : Données météorologiques et dataset
- `outputs/` : Résultats et rapports

## Technologies

- Python 3.13
- FastAPI
- Scikit-learn (Random Forest)
- Open-Meteo API

## Licence

MIT
