"""
=============================================================
 SYSTÈME D'IRRIGATION INTELLIGENTE — CÔTE D'IVOIRE
 agronomie.py : Penman-Monteith FAO-56 + Kc dynamique tomate
=============================================================
 Référence : FAO Irrigation and Drainage Paper No. 56
             Allen et al. (1998) — Chapters 2, 4, 5, 6
=============================================================
"""

import os as _os, sys as _sys
# ── Résolution robuste du chemin src/ ─────────────────────────────────────
# Fonctionne depuis : python src/fichier.py  |  python fichier.py  |  chemin absolu
_THIS_DIR   = _os.path.dirname(_os.path.abspath(__file__))
_PARENT_DIR = _os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR, _os.path.join(_PARENT_DIR, 'src')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ──────────────────────────────────────────────────────────────────────────

import math
from datetime import date
from typing import Optional
from config import (
    LATITUDE_RAD, ALTITUDE_M, SIGMA, PRESSION_KPA, GAMMA,
    ALBEDO, HAUTEUR_CULTURE_M,
    KC_FAO_BASE, STADES_JOURS,
    PLUIE_EFFECTIVE_PCT, EFFICACITE, SURFACE_M2,
)


# ══════════════════════════════════════════════════════════════
# SECTION 1 — FONCTIONS AUXILIAIRES THERMODYNAMIQUES
# ══════════════════════════════════════════════════════════════

def pression_vapeur_saturante(T: float) -> float:
    """
    Pression de vapeur saturante e°(T) en kPa.
    FAO-56 éq. 11 : e°(T) = 0.6108 × exp(17.27T / (T + 237.3))
    """
    return 0.6108 * math.exp(17.27 * T / (T + 237.3))


def pente_courbe_vapeur(T_moy: float) -> float:
    """
    Pente Δ de la courbe de pression de vapeur saturante (kPa/°C).
    FAO-56 éq. 13 : Δ = 4098 × e°(T) / (T + 237.3)²
    """
    return 4098.0 * pression_vapeur_saturante(T_moy) / (T_moy + 237.3) ** 2


# ══════════════════════════════════════════════════════════════
# SECTION 2 — RAYONNEMENTS
# ══════════════════════════════════════════════════════════════

def rayonnement_extraterrestre(J: int) -> float:
    """
    Rayonnement extraterrestre Ra (MJ/m²/j) — FAO-56 éq. 21.
    Calculé pour la latitude de Yamoussoukro.

    Parameters
    ----------
    J : jour julien (1–365)
    """
    dr      = 1.0 + 0.033 * math.cos(2 * math.pi * J / 365.0)
    delta_s = 0.409 * math.sin(2 * math.pi * J / 365.0 - 1.39)
    cos_ws  = -math.tan(LATITUDE_RAD) * math.tan(delta_s)
    ws      = math.acos(max(-1.0, min(1.0, cos_ws)))
    Ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(LATITUDE_RAD) * math.sin(delta_s)
        + math.cos(LATITUDE_RAD) * math.cos(delta_s) * math.sin(ws)
    )
    return Ra


def rayonnement_ciel_clair(J: int) -> float:
    """
    Rayonnement sous ciel clair Rso (MJ/m²/j) — FAO-56 éq. 37.
    Rso = (0.75 + 2×10⁻⁵ × z) × Ra
    """
    return (0.75 + 2e-5 * ALTITUDE_M) * rayonnement_extraterrestre(J)


def rayonnement_net_courtes_ondes(Rs: float) -> float:
    """
    Rns (MJ/m²/j) — FAO-56 éq. 38.
    Rns = (1 − α) × Rs   avec α = 0.23 (gazon FAO)
    """
    return (1.0 - ALBEDO) * Rs


def rayonnement_net_grandes_ondes(
    T_max : float,
    T_min : float,
    ea    : float,
    Rs    : float,
    Rso   : float,
) -> float:
    """
    Rnl (MJ/m²/j) — FAO-56 éq. 39.

    Rnl = σ × (T_max_K⁴ + T_min_K⁴)/2 × (0.34 − 0.14√ea)
            × (1.35 × Rs/Rso − 0.35)

    Le facteur de nébulosité (1.35×Rs/Rso − 0.35) est DYNAMIQUE.
    Clampé à [0.05, 1.00] : physiquement Rnl > 0 toujours.
    """
    terme_temp  = SIGMA * ((T_max + 273.16) ** 4 + (T_min + 273.16) ** 4) / 2.0
    terme_humid = 0.34 - 0.14 * math.sqrt(max(ea, 0.001))

    ratio_nuages   = min(Rs / Rso, 1.0) if Rso > 0 else 0.5
    facteur_nuages = max(0.05, 1.35 * ratio_nuages - 0.35)

    return terme_temp * terme_humid * facteur_nuages


# ══════════════════════════════════════════════════════════════
# SECTION 3 — PENMAN-MONTEITH FAO-56 COMPLET
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
    G      : float = 0.0,
) -> Optional[float]:
    """
    Évapotranspiration de référence ET₀ (mm/j) — FAO-56 éq. 6.

    ET₀ = [0.408×Δ×(Rn−G) + γ×(900/(T+273))×u2×(es−ea)]
           ─────────────────────────────────────────────────
                      Δ + γ×(1 + 0.34×u2)

    Parameters
    ----------
    T_max  : Température maximale journalière (°C)
    T_min  : Température minimale journalière (°C)
    T_moy  : Température moyenne journalière (°C)
    RH_max : Humidité relative maximale journalière (%)
    RH_min : Humidité relative minimale journalière (%)
    u2     : Vitesse du vent à 2m — MOYENNE journalière (m/s)
             ⚠ Ne pas utiliser le vent maximal
    Rs     : Rayonnement solaire incident (MJ/m²/j)
    J      : Jour julien 1–365 (auto-détecté si None)
    G      : Flux de chaleur du sol (MJ/m²/j) ≈ 0 en base journalière

    Returns
    -------
    ET₀ en mm/j (≥ 0), arrondi à 2 décimales. None si données invalides.

    Notes
    -----
    γ calculé depuis la pression atm. réelle à 212m (0.06571 kPa/°C),
    pas codé en dur à 0.0665.
    ea calculé avec RH_max et RH_min (méthode FAO-56 éq. 17, plus
    précise que HR_moy seul).
    """
    # ── Valeur par défaut pour J ───────────────────────────
    if J is None:
        J = date.today().timetuple().tm_yday

    # ── Validation des entrées ─────────────────────────────
    try:
        if T_max <= T_min:
            raise ValueError(f"T_max ({T_max}°C) doit être > T_min ({T_min}°C)")
        if not (0.0 <= RH_max <= 100.0) or not (0.0 <= RH_min <= 100.0):
            raise ValueError(f"HR hors plage [0, 100] : max={RH_max}, min={RH_min}")
        if RH_min > RH_max:
            raise ValueError(f"RH_min ({RH_min}) > RH_max ({RH_max})")
        if u2 < 0.0:
            raise ValueError(f"u2 ({u2} m/s) ne peut pas être négatif")
        if Rs < 0.0:
            raise ValueError(f"Rs ({Rs} MJ/m²) ne peut pas être négatif")
        if not (1 <= J <= 366):
            raise ValueError(f"Jour julien J={J} hors plage [1, 366]")
    except ValueError as e:
        print(f"  [ERREUR PM-FAO56] Données invalides : {e}")
        return None

    try:
        # ── Pressions de vapeur (kPa) ──────────────────────
        # es = moyenne e°(Tmax) et e°(Tmin) — FAO-56 éq. 12
        es_max = pression_vapeur_saturante(T_max)
        es_min = pression_vapeur_saturante(T_min)
        es     = (es_max + es_min) / 2.0

        # ea avec RH_max et RH_min — FAO-56 éq. 17 (méthode exacte)
        ea = (es_min * RH_max / 100.0 + es_max * RH_min / 100.0) / 2.0

        # ── Pente courbe vapeur Δ (kPa/°C) ────────────────
        delta = pente_courbe_vapeur(T_moy)

        # ── Rayonnements (MJ/m²/j) ────────────────────────
        Rso = rayonnement_ciel_clair(J)
        Rns = rayonnement_net_courtes_ondes(Rs)
        Rnl = rayonnement_net_grandes_ondes(T_max, T_min, ea, Rs, Rso)
        Rn  = Rns - Rnl

        # ── Équation Penman-Monteith (FAO-56 éq. 6) ───────
        numerateur   = (0.408 * delta * (Rn - G)
                        + GAMMA * (900.0 / (T_moy + 273.0)) * u2 * (es - ea))
        denominateur = delta + GAMMA * (1.0 + 0.34 * u2)

        ET0 = numerateur / denominateur
        return round(max(ET0, 0.0), 2)

    except (ZeroDivisionError, OverflowError, ValueError) as e:
        print(f"  [ERREUR PM-FAO56] Calcul impossible : {e}")
        return None


# ══════════════════════════════════════════════════════════════
# SECTION 4 — Kc DYNAMIQUE TOMATE (FAO-56 §6.2.2)
# ══════════════════════════════════════════════════════════════

def get_stade(jour_cycle: int) -> str:
    """Retourne le stade phénologique selon l'âge de la plante."""
    for stade, (debut, fin) in STADES_JOURS.items():
        if debut <= jour_cycle < fin:
            return stade
    return "fin_saison"   # au-delà de 135 jours


def kc_tomate(
    jour_cycle : int,
    saison     : str,
    hr_min     : float = 45.0,
    u2         : float = 2.0,
) -> float:
    """
    Coefficient cultural Kc de la tomate — FAO-56 dynamique.

    Calcul :
      1. Kc FAO de base selon le stade phénologique
      2. Interpolation linéaire durant la croissance (FAO-56 §6.2.2)
      3. Correction vent/HR pour mi-saison (FAO-56 éq. 62)
      4. Correction saison ivoirienne

    Parameters
    ----------
    jour_cycle : Jours depuis la plantation (0 = jour de plantation)
    saison     : 'seche' | 'grande_pluie' | 'petite_pluie'
    hr_min     : HR minimale journalière (%) — pour correction FAO éq.62
    u2         : Vitesse vent à 2m (m/s) — pour correction FAO éq.62

    Returns
    -------
    Kc final corrigé, clampé dans [0.30, 1.40]
    """
    stade = get_stade(jour_cycle)
    J1    = STADES_JOURS["initial"][1]     # 25
    J2    = STADES_JOURS["croissance"][1]  # 60

    # ── 1. Kc FAO de base ─────────────────────────────────
    if stade == "initial":
        kc_base = KC_FAO_BASE["initial"]

    elif stade == "croissance":
        # Interpolation linéaire : Kc_ini → Kc_mid (FAO-56 §6.2.2)
        t      = (jour_cycle - J1) / (J2 - J1)
        t      = max(0.0, min(1.0, t))
        kc_base = KC_FAO_BASE["initial"] + t * (KC_FAO_BASE["mi_saison"] - KC_FAO_BASE["initial"])

    elif stade == "mi_saison":
        # Correction vent/HR — FAO-56 éq. 62
        # Kc_mid_adj = Kc_mid + [0.04(u2−2) − 0.004(HRmin−45)] × (h/3)^0.3
        kc_base = KC_FAO_BASE["mi_saison"] + (
            0.04 * (u2 - 2.0) - 0.004 * (hr_min - 45.0)
        ) * (HAUTEUR_CULTURE_M / 3.0) ** 0.3

    else:  # fin_saison
        kc_base = KC_FAO_BASE["fin_saison"]

    # ── 2. Correction saison CI ────────────────────────────
    corrections_saison = {
        "seche"        : +0.05,   # chaud, sec, vent → plus d'évaporation
        "petite_pluie" :  0.00,   # transition, neutre
        "grande_pluie" : -0.05,   # humide, couvert → moins d'évaporation
    }
    delta_saison = corrections_saison.get(saison, 0.0)

    kc_final = kc_base + delta_saison
    return round(max(0.30, min(1.40, kc_final)), 4)


# ══════════════════════════════════════════════════════════════
# SECTION 5 — BILAN HYDRIQUE COMPLET
# ══════════════════════════════════════════════════════════════

def bilan_hydrique(
    ET0           : float,
    pluie_mm      : float,
    hum_sol_pct   : float,
    jour_cycle    : int,
    saison        : str,
    hr_min        : float = 45.0,
    u2            : float = 2.0,
) -> dict:
    """
    Calcule le bilan hydrique complet de la journée.

    Intègre le Kc dynamique (stade + saison) et la décision
    d'irrigation selon les règles agronomiques FAO.

    Returns
    -------
    dict avec : kc, stade, ETc_mm, pluie_eff_mm,
                deficit_mm, irriguer, volume_litres
    """
    from config import (SOL_HUMIDE_SEUIL, SOL_MOYEN_SEUIL,
                        PLUIE_FORTE_SEUIL, PLUIE_MODERE_SEUIL)

    stade        = get_stade(jour_cycle)
    kc           = kc_tomate(jour_cycle, saison, hr_min, u2)
    ETc          = round(ET0 * kc, 2)
    pluie_eff    = round(pluie_mm * PLUIE_EFFECTIVE_PCT, 2)
    deficit      = round(max(ETc - pluie_eff, 0.0), 2)

    # ── Règles agronomiques prioritaires ──────────────────
    irriguer = 1
    raison   = "ML requis"

    if hum_sol_pct > SOL_HUMIDE_SEUIL:
        irriguer, raison = 0, f"Sol humide ({hum_sol_pct:.1f}% > {SOL_HUMIDE_SEUIL}%)"
    elif pluie_mm > PLUIE_FORTE_SEUIL:
        irriguer, raison = 0, f"Forte pluie ({pluie_mm:.1f}mm > {PLUIE_FORTE_SEUIL}mm)"
    elif pluie_mm > PLUIE_MODERE_SEUIL and hum_sol_pct > SOL_MOYEN_SEUIL:
        irriguer, raison = 0, (f"Pluie modérée ({pluie_mm:.1f}mm) + "
                                f"sol ok ({hum_sol_pct:.1f}%)")
    elif deficit <= 0.0:
        irriguer, raison = 0, f"Pas de déficit hydrique ({deficit:.2f}mm)"

    # ── Volume si irrigation déclenchée ───────────────────
    volume = 0.0
    if irriguer == 1:
        facteur = max(0.0, (65.0 - hum_sol_pct) / 25.0)
        volume  = round(deficit * facteur * SURFACE_M2 / EFFICACITE, 1)
        raison  = f"Déficit {deficit:.2f}mm | Sol {hum_sol_pct:.1f}% | Kc {kc:.4f}"

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