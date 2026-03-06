"""
=============================================================
 SYSTÈME D'IRRIGATION INTELLIGENTE — CÔTE D'IVOIRE
 Module 0 : Noyau agronomique partagé
=============================================================
 Référence : FAO Irrigation and Drainage Paper No. 56
             Allen et al. (1998)
 Culture   : Tomate (Solanum lycopersicum)
 Site      : Yamoussoukro — 6.82°N, 5.28°W, altitude 212 m
=============================================================
 Contenu :
   - Penman-Monteith FAO-56 exact (avec RH_max/RH_min, Ra, Rso)
   - Kc dynamique par stade phénologique + corrections CI
   - Bilan hydrique et décision d'irrigation
=============================================================
"""

import math
from datetime import date
from typing import Optional


# ══════════════════════════════════════════════════════════════
# CONSTANTES DU SITE — YAMOUSSOUKRO
# ══════════════════════════════════════════════════════════════

ALTITUDE_M    = 212.0
LATITUDE_DEG  = 6.8205
LATITUDE_RAD  = math.radians(LATITUDE_DEG)
SURFACE_M2    = 200.0       # champ M. Koffi
EFFICACITE    = 0.90        # goutte-à-goutte
ALBEDO        = 0.23        # gazon de référence FAO
SIGMA         = 4.903e-9    # Stefan-Boltzmann (MJ m⁻² j⁻¹ K⁻⁴)

# Pression atmosphérique à 212 m — FAO-56 éq. 7
PRESSION_KPA  = 101.3 * ((293.0 - 0.0065 * ALTITUDE_M) / 293.0) ** 5.26

# Constante psychrométrique γ — FAO-56 éq. 8  (kPa/°C)
GAMMA         = 0.000665 * PRESSION_KPA   # ≈ 0.06571  (et non 0.0665 fixe)


# ══════════════════════════════════════════════════════════════
# STADES PHÉNOLOGIQUES TOMATE (FAO-56 Tableau 11)
# ══════════════════════════════════════════════════════════════
#
#  Stade          | Durée médiane | Kc FAO  | Bornes en jours
#  ───────────────┼───────────────┼─────────┼────────────────
#  Initial        |  25 j         | 0.45    | [0 – 25[
#  Croissance     |  35 j         | linéaire| [25 – 60[
#  Mi-saison      |  50 j         | 1.15    | [60 – 110[
#  Fin de saison  |  25 j         | 0.80    | [110 – 130[
#
J_INITIAL    = 25
J_CROISSANCE = 60
J_MI_SAISON  = 110
J_FIN        = 130

KC_INITIAL   = 0.45
KC_MI_SAISON = 1.15
KC_FIN       = 0.80


# ══════════════════════════════════════════════════════════════
# 1. FONCTIONS AUXILIAIRES PENMAN-MONTEITH
# ══════════════════════════════════════════════════════════════

def _esat(T: float) -> float:
    """Pression de vapeur saturante (kPa) — FAO-56 éq. 11."""
    return 0.6108 * math.exp(17.27 * T / (T + 237.3))


def _delta(T_moy: float) -> float:
    """Pente Δ de la courbe de vapeur saturante (kPa/°C) — FAO-56 éq. 13."""
    return 4098.0 * _esat(T_moy) / (T_moy + 237.3) ** 2


def _Ra(J: int) -> float:
    """
    Rayonnement extraterrestre Ra (MJ/m²/j) — FAO-56 éq. 21.
    Calculé pour Yamoussoukro (6.82°N).
    """
    dr      = 1.0 + 0.033 * math.cos(2 * math.pi * J / 365.0)
    delta_s = 0.409 * math.sin(2 * math.pi * J / 365.0 - 1.39)
    cos_ws  = max(-1.0, min(1.0, -math.tan(LATITUDE_RAD) * math.tan(delta_s)))
    ws      = math.acos(cos_ws)
    return (
        (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
            ws * math.sin(LATITUDE_RAD) * math.sin(delta_s)
            + math.cos(LATITUDE_RAD) * math.cos(delta_s) * math.sin(ws)
        )
    )


def _Rso(J: int) -> float:
    """Rayonnement sous ciel clair Rso (MJ/m²/j) — FAO-56 éq. 37."""
    return (0.75 + 2e-5 * ALTITUDE_M) * _Ra(J)


# ══════════════════════════════════════════════════════════════
# 2. PENMAN-MONTEITH FAO-56 EXACT
# ══════════════════════════════════════════════════════════════

def penman_monteith_fao56(
    T_max  : float,
    T_min  : float,
    T_moy  : float,
    RH_max : float,
    RH_min : float,
    u2     : float,
    Rs     : float,
    J      : Optional[int] = None,
) -> Optional[float]:
    """
    Évapotranspiration de référence ET₀ (mm/j) — FAO-56 éq. 6.

    Parameters
    ----------
    T_max  : Température maximale journalière (°C)
    T_min  : Température minimale journalière (°C)
    T_moy  : Température moyenne journalière (°C)
    RH_max : Humidité relative maximale journalière (%)
    RH_min : Humidité relative minimale journalière (%)
    u2     : Vitesse du vent à 2 m — MOYENNE journalière (m/s)
             ⚠️  Ne jamais utiliser le vent maximum
    Rs     : Rayonnement solaire mesuré ou estimé (MJ/m²/j)
    J      : Jour julien (1–365). Si None, utilise la date du jour.

    Returns
    -------
    ET₀ en mm/j (≥ 0), arrondi à 2 décimales.
    Retourne None en cas de données invalides.

    Notes
    -----
    - ea calculé selon FAO-56 éq. 17 (RH_max + RH_min) — plus précis
      que l'approximation par HR_moy seule.
    - γ calculé à partir de la pression atmosphérique locale (212 m).
    - Rnl utilise le terme de nébulosité dynamique (Rs/Rso), clampé
      à [0.05, 1.00] pour éviter les valeurs physiquement impossibles.
    """
    if J is None:
        J = date.today().timetuple().tm_yday

    # ── Validation ───────────────────────────────────────────
    if T_max <= T_min:
        return None
    if not (0.0 <= RH_max <= 100.0 and 0.0 <= RH_min <= 100.0):
        return None
    if u2 < 0.0 or Rs < 0.0:
        return None

    try:
        # ── Pressions de vapeur (kPa) ──────────────────────
        es_max = _esat(T_max)
        es_min = _esat(T_min)
        es     = (es_max + es_min) / 2.0            # FAO-56 éq. 12

        # ea — méthode FAO-56 éq. 17 (RH_max + RH_min)
        ea = (es_min * RH_max / 100.0
              + es_max * RH_min / 100.0) / 2.0

        # ── Pente Δ ────────────────────────────────────────
        Delta = _delta(T_moy)

        # ── Rayonnements ──────────────────────────────────
        Rns = (1.0 - ALBEDO) * Rs                   # FAO-56 éq. 38

        rso_j        = _Rso(J)
        ratio_nuages = min(Rs / rso_j, 1.0) if rso_j > 0 else 0.5
        # Clampage FAO recommandé : évite Rnl négatif (ciel très couvert)
        facteur_neb  = max(0.05, 1.35 * ratio_nuages - 0.35)

        Rnl = (
            SIGMA
            * ((T_max + 273.16) ** 4 + (T_min + 273.16) ** 4) / 2.0
            * (0.34 - 0.14 * math.sqrt(max(ea, 1e-6)))
            * facteur_neb
        )                                             # FAO-56 éq. 39

        Rn  = Rns - Rnl                              # G = 0 (journalier)

        # ── Équation PM-FAO56 (éq. 6) ─────────────────────
        num = (
            0.408 * Delta * Rn
            + GAMMA * (900.0 / (T_moy + 273.0)) * u2 * (es - ea)
        )
        den = Delta + GAMMA * (1.0 + 0.34 * u2)

        ET0 = num / den
        return round(max(ET0, 0.0), 2)

    except (ZeroDivisionError, OverflowError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════
# 3. Kc DYNAMIQUE TOMATE
# ══════════════════════════════════════════════════════════════

def kc_tomate(
    jour_cycle : int,
    saison     : str,
    hr_min     : float = 45.0,
    u2         : float = 2.0,
) -> float:
    """
    Kc dynamique de la tomate selon le stade phénologique.

    Parameters
    ----------
    jour_cycle : Jours depuis la plantation (0 = jour de plantation)
    saison     : 'seche' | 'grande_pluie' | 'petite_pluie'
    hr_min     : HR minimale journalière (%) — pour correction FAO-56 éq. 62
    u2         : Vent moyen à 2 m (m/s) — pour correction FAO-56 éq. 62

    Returns
    -------
    Kc final (float), clampé dans [0.30, 1.40]

    Méthode
    -------
    1. Kc FAO de base selon le stade
    2. Interpolation LINÉAIRE en phase de croissance (FAO-56 §6.2.2)
       → évite les sauts brutaux entre stades
    3. Correction vent/HR pour mi-saison et fin de saison (FAO-56 éq. 62)
    4. Correction saison ivoirienne (+0.05 sec / -0.05 humide)
    """
    # ── 1. Kc de base selon stade ─────────────────────────
    if jour_cycle < J_INITIAL:
        kc_base  = KC_INITIAL
        en_mi    = False

    elif jour_cycle < J_CROISSANCE:
        # Interpolation linéaire FAO-56 §6.2.2
        t        = (jour_cycle - J_INITIAL) / (J_CROISSANCE - J_INITIAL)
        kc_base  = KC_INITIAL + t * (KC_MI_SAISON - KC_INITIAL)
        en_mi    = False

    elif jour_cycle < J_MI_SAISON:
        kc_base  = KC_MI_SAISON
        en_mi    = True

    else:
        kc_base  = KC_FIN
        en_mi    = True   # correction vent/HR applicable aussi en fin

    # ── 2. Correction vent/HR (FAO-56 éq. 62) ────────────
    #    Applicable uniquement mi-saison et fin de saison
    #    h ≈ 0.8 m (hauteur tomate)
    if en_mi:
        h         = 0.8
        delta_kc  = (
            (0.04 * (u2 - 2.0) - 0.004 * (hr_min - 45.0))
            * (h / 3.0) ** 0.3
        )
        kc_base  += delta_kc

    # ── 3. Correction saison CI ───────────────────────────
    corrections_saison = {
        "seche"        : +0.05,
        "petite_pluie" :  0.00,
        "grande_pluie" : -0.05,
    }
    kc_base += corrections_saison.get(saison, 0.0)

    # ── 4. Clampage agronomique ───────────────────────────
    return round(max(0.30, min(1.40, kc_base)), 4)


def stade_tomate(jour_cycle: int) -> str:
    """Retourne le nom du stade phénologique."""
    if jour_cycle < J_INITIAL:
        return "Initial (jeunes plants)"
    elif jour_cycle < J_CROISSANCE:
        return "Croissance végétative"
    elif jour_cycle < J_MI_SAISON:
        return "Mi-saison (floraison/fructification)"
    else:
        return "Fin de saison (maturation)"


# ══════════════════════════════════════════════════════════════
# 4. BILAN HYDRIQUE
# ══════════════════════════════════════════════════════════════

def bilan_hydrique(
    ET0       : float,
    pluie_mm  : float,
    kc        : float,
) -> dict:
    """
    Calcule ETc, pluie effective, déficit hydrique et volume brut.

    Parameters
    ----------
    ET0      : Évapotranspiration de référence (mm/j)
    pluie_mm : Précipitations totales journalières (mm)
    kc       : Coefficient cultural (issu de kc_tomate())

    Returns
    -------
    dict avec ETc_mm, pluie_eff_mm, deficit_mm, volume_brut_L
    """
    ETc        = round(ET0 * kc, 3)
    pluie_eff  = round(pluie_mm * 0.80, 3)          # FAO-56 : 80 % efficace
    deficit    = round(max(ETc - pluie_eff, 0.0), 3)
    volume     = round(deficit * SURFACE_M2 / EFFICACITE, 1)

    return {
        "ETc_mm"        : ETc,
        "pluie_eff_mm"  : pluie_eff,
        "deficit_mm"    : deficit,
        "volume_brut_L" : volume,
    }


# ══════════════════════════════════════════════════════════════
# 5. DÉCISION D'IRRIGATION
# ══════════════════════════════════════════════════════════════

def decision_irrigation(
    hum_sol_pct : float,
    pluie_mm    : float,
    deficit_mm  : float,
) -> tuple[int, str]:
    """
    Règles agronomiques prioritaires (court-circuit avant ML).

    Returns
    -------
    (décision, raison)  —  décision : 0 = non, 1 = oui
    """
    if hum_sol_pct > 70.0:
        return 0, f"Sol suffisamment humide ({hum_sol_pct:.1f}% > 70%)"
    if pluie_mm > 10.0:
        return 0, f"Forte pluie prévue ({pluie_mm:.1f} mm)"
    if pluie_mm > 5.0 and hum_sol_pct > 50.0:
        return 0, f"Pluie modérée ({pluie_mm:.1f} mm) + sol ok ({hum_sol_pct:.1f}%)"
    if deficit_mm <= 0.0:
        return 0, f"Aucun déficit hydrique ({deficit_mm:.2f} mm)"
    return 1, "Analyse ML requise"


# ══════════════════════════════════════════════════════════════
# 6. UTILITAIRE SAISON
# ══════════════════════════════════════════════════════════════

def saison_ci(mois: int) -> str:
    """
    Saison climatique ivoirienne selon le mois (1–12).
    Adapté au régime bimodal de Yamoussoukro.
    """
    if mois in (11, 12, 1, 2, 3):
        return "seche"
    elif mois in (6, 7, 8, 9):
        return "grande_pluie"
    else:
        return "petite_pluie"