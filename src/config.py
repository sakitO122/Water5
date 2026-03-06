"""
=============================================================
 SYSTÈME D'IRRIGATION INTELLIGENTE — CÔTE D'IVOIRE
 config.py : Constantes et configuration centralisées
=============================================================
 Toute modification de paramètre se fait ICI uniquement.
 Importé par tous les autres modules.
=============================================================
"""

import os
import math

# ══════════════════════════════════════════════════════════════
# CHEMINS
# ══════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════
# SITE — YAMOUSSOUKRO
# ══════════════════════════════════════════════════════════════
LATITUDE_DEG  = 6.8205
LONGITUDE_DEG = -5.2767
ALTITUDE_M    = 212.0
TIMEZONE      = "Africa/Abidjan"
LATITUDE_RAD  = math.radians(LATITUDE_DEG)

# ══════════════════════════════════════════════════════════════
# CULTURE — TOMATE (Solanum lycopersicum)
# ══════════════════════════════════════════════════════════════
SURFACE_M2 = 200        # Surface de la parcelle de M. Koffi
EFFICACITE = 0.90       # Efficacité du système goutte-à-goutte
ALBEDO     = 0.23       # Albédo gazon de référence FAO-56

# ── Stades phénologiques FAO-56 (durées médianes en jours) ──
STADES_JOURS = {
    "initial"    : (0,  25),   # Jeunes plants
    "croissance" : (25, 60),   # Développement végétatif
    "mi_saison"  : (60, 110),  # Floraison / fructification
    "fin_saison" : (110, 135), # Maturation / sénescence
}

# ── Kc FAO-56 base (Tableau 12, Allen et al. 1998) ──────────
KC_FAO_BASE = {
    "initial"    : 0.45,
    "mi_saison"  : 1.15,
    "fin_saison" : 0.80,
    # "croissance" : interpolé linéairement entre initial et mi_saison
}

# Hauteur culture tomate en mi-saison (pour correction FAO-56 éq. 62)
HAUTEUR_CULTURE_M = 0.8

# ══════════════════════════════════════════════════════════════
# PHYSIQUE — FAO-56
# ══════════════════════════════════════════════════════════════
# Constante de Stefan-Boltzmann (MJ m⁻² j⁻¹ K⁻⁴)
SIGMA = 4.903e-9

# Pression atmosphérique à ALTITUDE_M (FAO-56 éq. 7)
PRESSION_KPA = 101.3 * ((293.0 - 0.0065 * ALTITUDE_M) / 293.0) ** 5.26

# Constante psychrométrique γ à ALTITUDE_M (FAO-56 éq. 8)
GAMMA = 0.000665 * PRESSION_KPA   # ≈ 0.06571 kPa/°C à 212m

# ══════════════════════════════════════════════════════════════
# RÈGLES AGRONOMIQUES (seuils de décision)
# ══════════════════════════════════════════════════════════════
SOL_HUMIDE_SEUIL    = 70.0   # % humidité sol — pas d'irrigation si > seuil
SOL_MOYEN_SEUIL     = 50.0   # % — combiné avec pluie modérée
PLUIE_FORTE_SEUIL   = 10.0   # mm — forte pluie, pas d'irrigation
PLUIE_MODERE_SEUIL  =  5.0   # mm — pluie modérée (combinée avec sol)
PLUIE_EFFECTIVE_PCT = 0.80   # 80% des précipitations utiles (FAO)

# ══════════════════════════════════════════════════════════════
# ML — FEATURES
# ══════════════════════════════════════════════════════════════
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