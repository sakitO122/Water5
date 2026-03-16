"""
config.py
Configuration centralisee du systeme Water5.
Toute modification de parametre se fait uniquement dans ce fichier.
"""

import os
import math

# ── Chemins ───────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
SRC_DIR   = os.path.join(BASE_DIR, "src")

for _d in (DATA_DIR, MODEL_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)

CSV_BRUT  = os.path.join(DATA_DIR, "open_meteo_brut.csv")
CSV_CLEAN = os.path.join(DATA_DIR, "yamoussoukro_dataset_ML.csv")
CLF_PATH  = os.path.join(MODEL_DIR, "modele_classification.joblib")
REG_PATH  = os.path.join(MODEL_DIR, "modele_regression.joblib")

# ── Site : Yamoussoukro ───────────────────────────────────────────────────
LATITUDE_DEG  = 6.8205
LONGITUDE_DEG = -5.2767
ALTITUDE_M    = 212.0
TIMEZONE      = "Africa/Abidjan"
LATITUDE_RAD  = math.radians(LATITUDE_DEG)

# ── Culture : Tomate (Solanum lycopersicum) ───────────────────────────────
SURFACE_M2        = 200
EFFICACITE        = 0.90    # efficacite systeme goutte-a-goutte
ALBEDO            = 0.23    # albedo gazon de reference FAO-56
HAUTEUR_CULTURE_M = 0.8     # hauteur tomate en mi-saison (m)

# Stades phenologiques FAO-56 (durees medianes en jours depuis plantation)
STADES_JOURS = {
    "initial"    : (0,   25),
    "croissance" : (25,  60),
    "mi_saison"  : (60,  110),
    "fin_saison" : (110, 135),
}

# Kc de base FAO-56 (Tableau 12, Allen et al. 1998)
KC_FAO_BASE = {
    "initial"    : 0.45,
    "mi_saison"  : 1.15,
    "fin_saison" : 0.80,
    # "croissance" : interpole lineairement entre initial et mi_saison
}

# ── Physique FAO-56 ───────────────────────────────────────────────────────
SIGMA        = 4.903e-9   # Stefan-Boltzmann (MJ m-2 j-1 K-4)
PRESSION_KPA = 101.3 * ((293.0 - 0.0065 * ALTITUDE_M) / 293.0) ** 5.26
GAMMA        = 0.000665 * PRESSION_KPA   # constante psychrometrique (kPa/C)

# ── Seuils agronomiques ───────────────────────────────────────────────────
SOL_HUMIDE_SEUIL    = 70.0   # % humidite sol : pas d'irrigation au-dessus
SOL_MOYEN_SEUIL     = 50.0   # % : combine avec pluie moderee
PLUIE_FORTE_SEUIL   = 10.0   # mm : forte pluie, irrigation annulee
PLUIE_MODERE_SEUIL  =  5.0   # mm : pluie moderee (combinee avec sol)
PLUIE_EFFECTIVE_PCT = 0.80   # 80% des precipitations sont utiles (FAO)

# ── Features du modele ML ─────────────────────────────────────────────────
FEATURES = [
    "humidite_sol_moy_pct",
    "humidite_sol_min_pct",
    "humidite_sol_0_7_moy",
    "temp_max_C",
    "temp_min_C",
    "temp_moy_C",
    "humidite_air_moy_pct",
    "vent_u2_ms",
    "rayonnement_Rs_MJ",
    "ET0_reference_mm",
    "ETc_mm",
    "deficit_hydrique_mm",
    "pluie_totale_mm",
    "pluie_effective_mm",
    "jour_annee",
    "mois",
]

TARGET_CLF = "irriguer"
TARGET_REG = "volume_litres"
