"""
agronomie.py
Noyau agronomique : Penman-Monteith FAO-56 et Kc dynamique tomate.

Reference : FAO Irrigation and Drainage Paper No. 56
            Allen et al. (1998)
"""

import math
from datetime import date
from typing import Optional

import os, sys
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR, os.path.join(_PARENT_DIR, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    LATITUDE_RAD, ALTITUDE_M, SIGMA, PRESSION_KPA, GAMMA,
    ALBEDO, HAUTEUR_CULTURE_M,
    KC_FAO_BASE, STADES_JOURS,
    PLUIE_EFFECTIVE_PCT, EFFICACITE, SURFACE_M2,
)


# ── Fonctions auxiliaires thermodynamiques ────────────────────────────────

def pression_vapeur_saturante(T: float) -> float:
    """Pression de vapeur saturante e(T) en kPa. FAO-56 eq. 11."""
    return 0.6108 * math.exp(17.27 * T / (T + 237.3))


def pente_courbe_vapeur(T_moy: float) -> float:
    """Pente Delta de la courbe de vapeur saturante (kPa/C). FAO-56 eq. 13."""
    return 4098.0 * pression_vapeur_saturante(T_moy) / (T_moy + 237.3) ** 2


# ── Rayonnements ──────────────────────────────────────────────────────────

def rayonnement_extraterrestre(J: int) -> float:
    """
    Rayonnement extraterrestre Ra (MJ/m2/j). FAO-56 eq. 21.

    Parameters
    ----------
    J : jour julien (1-365)
    """
    dr      = 1.0 + 0.033 * math.cos(2 * math.pi * J / 365.0)
    delta_s = 0.409 * math.sin(2 * math.pi * J / 365.0 - 1.39)
    cos_ws  = -math.tan(LATITUDE_RAD) * math.tan(delta_s)
    ws      = math.acos(max(-1.0, min(1.0, cos_ws)))
    return (
        (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
            ws * math.sin(LATITUDE_RAD) * math.sin(delta_s)
            + math.cos(LATITUDE_RAD) * math.cos(delta_s) * math.sin(ws)
        )
    )


def rayonnement_ciel_clair(J: int) -> float:
    """Rayonnement sous ciel clair Rso (MJ/m2/j). FAO-56 eq. 37."""
    return (0.75 + 2e-5 * ALTITUDE_M) * rayonnement_extraterrestre(J)


def rayonnement_net_courtes_ondes(Rs: float) -> float:
    """Rns (MJ/m2/j). FAO-56 eq. 38."""
    return (1.0 - ALBEDO) * Rs


def rayonnement_net_grandes_ondes(
    T_max : float,
    T_min : float,
    ea    : float,
    Rs    : float,
    Rso   : float,
) -> float:
    """
    Rnl (MJ/m2/j). FAO-56 eq. 39.
    Le facteur de nebulosity (1.35*Rs/Rso - 0.35) est dynamique,
    clampe a [0.05, 1.00].
    """
    terme_temp     = SIGMA * ((T_max + 273.16) ** 4 + (T_min + 273.16) ** 4) / 2.0
    terme_humid    = 0.34 - 0.14 * math.sqrt(max(ea, 0.001))
    ratio_nuages   = min(Rs / Rso, 1.0) if Rso > 0 else 0.5
    facteur_nuages = max(0.05, 1.35 * ratio_nuages - 0.35)
    return terme_temp * terme_humid * facteur_nuages


# ── Penman-Monteith FAO-56 ────────────────────────────────────────────────

def penman_monteith_fao56(
    T_max  : float,
    T_min  : float,
    T_moy  : float,
    RH_max : float,
    RH_min : float,
    u2     : float,
    Rs     : float,
    J      : Optional[int] = None,
    G      : float = 0.0,
) -> Optional[float]:
    """
    Evapotranspiration de reference ET0 (mm/j). FAO-56 eq. 6.

    Parameters
    ----------
    T_max  : Temperature maximale journaliere (C)
    T_min  : Temperature minimale journaliere (C)
    T_moy  : Temperature moyenne journaliere (C)
    RH_max : Humidite relative maximale (%)
    RH_min : Humidite relative minimale (%)
    u2     : Vitesse du vent a 2m, MOYENNE journaliere (m/s)
    Rs     : Rayonnement solaire incident (MJ/m2/j)
    J      : Jour julien 1-365 (auto-detecte si None)
    G      : Flux de chaleur du sol (MJ/m2/j), 0 en base journaliere

    Returns
    -------
    ET0 en mm/j (>= 0), arrondi a 2 decimales. None si donnees invalides.
    """
    if J is None:
        J = date.today().timetuple().tm_yday

    if T_max <= T_min:
        return None
    if not (0.0 <= RH_max <= 100.0 and 0.0 <= RH_min <= 100.0):
        return None
    if RH_min > RH_max or u2 < 0.0 or Rs < 0.0:
        return None
    if not (1 <= J <= 366):
        return None

    try:
        es_max = pression_vapeur_saturante(T_max)
        es_min = pression_vapeur_saturante(T_min)
        es     = (es_max + es_min) / 2.0
        ea     = (es_min * RH_max / 100.0 + es_max * RH_min / 100.0) / 2.0
        delta  = pente_courbe_vapeur(T_moy)

        Rso = rayonnement_ciel_clair(J)
        Rns = rayonnement_net_courtes_ondes(Rs)
        Rnl = rayonnement_net_grandes_ondes(T_max, T_min, ea, Rs, Rso)
        Rn  = Rns - Rnl

        num = (
            0.408 * delta * (Rn - G)
            + GAMMA * (900.0 / (T_moy + 273.0)) * u2 * (es - ea)
        )
        den = delta + GAMMA * (1.0 + 0.34 * u2)

        return round(max(num / den, 0.0), 2)

    except (ZeroDivisionError, OverflowError, ValueError):
        return None


# ── Kc dynamique tomate ───────────────────────────────────────────────────

def get_stade(jour_cycle: int) -> str:
    """Retourne le stade phenologique selon l'age de la plante."""
    for stade, (debut, fin) in STADES_JOURS.items():
        if debut <= jour_cycle < fin:
            return stade
    return "fin_saison"


def kc_tomate(
    jour_cycle : int,
    saison     : str,
    hr_min     : float = 45.0,
    u2         : float = 2.0,
) -> float:
    """
    Coefficient cultural Kc dynamique de la tomate. FAO-56 sec. 6.2.2.

    Parameters
    ----------
    jour_cycle : Jours depuis la plantation
    saison     : 'seche' | 'grande_pluie' | 'petite_pluie'
    hr_min     : HR minimale journaliere (%) pour correction FAO-56 eq. 62
    u2         : Vent moyen a 2m (m/s) pour correction FAO-56 eq. 62

    Returns
    -------
    Kc final corrige, clampe dans [0.30, 1.40]
    """
    stade = get_stade(jour_cycle)
    J1    = STADES_JOURS["initial"][1]
    J2    = STADES_JOURS["croissance"][1]

    if stade == "initial":
        kc_base = KC_FAO_BASE["initial"]

    elif stade == "croissance":
        t       = (jour_cycle - J1) / (J2 - J1)
        t       = max(0.0, min(1.0, t))
        kc_base = KC_FAO_BASE["initial"] + t * (
            KC_FAO_BASE["mi_saison"] - KC_FAO_BASE["initial"]
        )

    elif stade == "mi_saison":
        kc_base = KC_FAO_BASE["mi_saison"] + (
            0.04 * (u2 - 2.0) - 0.004 * (hr_min - 45.0)
        ) * (HAUTEUR_CULTURE_M / 3.0) ** 0.3

    else:
        kc_base = KC_FAO_BASE["fin_saison"]

    corrections = {
        "seche"        : +0.05,
        "petite_pluie" :  0.00,
        "grande_pluie" : -0.05,
    }
    kc_base += corrections.get(saison, 0.0)

    return round(max(0.30, min(1.40, kc_base)), 4)


# ── Bilan hydrique ────────────────────────────────────────────────────────

def bilan_hydrique(
    ET0         : float,
    pluie_mm    : float,
    hum_sol_pct : float,
    jour_cycle  : int,
    saison      : str,
    hr_min      : float = 45.0,
    u2          : float = 2.0,
) -> dict:
    """
    Calcule le bilan hydrique complet de la journee.

    Returns
    -------
    dict : stade, kc, ETc_mm, pluie_eff_mm, deficit_mm,
           irriguer, volume_litres, raison
    """
    from config import (
        SOL_HUMIDE_SEUIL, SOL_MOYEN_SEUIL,
        PLUIE_FORTE_SEUIL, PLUIE_MODERE_SEUIL,
    )

    stade     = get_stade(jour_cycle)
    kc        = kc_tomate(jour_cycle, saison, hr_min, u2)
    ETc       = round(ET0 * kc, 2)
    pluie_eff = round(pluie_mm * PLUIE_EFFECTIVE_PCT, 2)
    deficit   = round(max(ETc - pluie_eff, 0.0), 2)

    irriguer = 1
    raison   = "ML requis"

    if hum_sol_pct > SOL_HUMIDE_SEUIL:
        irriguer, raison = 0, f"Sol humide ({hum_sol_pct:.1f}% > {SOL_HUMIDE_SEUIL}%)"
    elif pluie_mm > PLUIE_FORTE_SEUIL:
        irriguer, raison = 0, f"Forte pluie ({pluie_mm:.1f}mm)"
    elif pluie_mm > PLUIE_MODERE_SEUIL and hum_sol_pct > SOL_MOYEN_SEUIL:
        irriguer, raison = 0, f"Pluie moderee + sol ok ({hum_sol_pct:.1f}%)"
    elif deficit <= 0.0:
        irriguer, raison = 0, f"Pas de deficit ({deficit:.2f}mm)"

    volume = 0.0
    if irriguer == 1:
        facteur = max(0.0, (65.0 - hum_sol_pct) / 25.0)
        volume  = round(deficit * facteur * SURFACE_M2 / EFFICACITE, 1)
        raison  = f"Deficit {deficit:.2f}mm | Sol {hum_sol_pct:.1f}% | Kc {kc:.4f}"

    return {
        "stade"         : stade,
        "kc"            : kc,
        "ETc_mm"        : ETc,
        "pluie_eff_mm"  : pluie_eff,
        "deficit_mm"    : deficit,
        "irriguer"      : irriguer,
        "volume_litres" : volume,
        "raison"        : raison,
    }


# ── Utilitaire saison ─────────────────────────────────────────────────────

def get_saison(mois: int) -> str:
    """Saison climatique ivoirienne selon le mois (1-12)."""
    if mois in (11, 12, 1, 2, 3):
        return "seche"
    elif mois in (6, 7, 8, 9):
        return "grande_pluie"
    return "petite_pluie"
