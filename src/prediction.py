"""
=============================================================
 SYSTÈME D'IRRIGATION INTELLIGENTE — CÔTE D'IVOIRE
 03_prediction.py
=============================================================
 Moteur de décision : règles agronomiques + modèles ML
 Modes :
   --mode sim  → simulation avec données exemple (défaut)
   --mode api  → données réelles Open-Meteo (internet requis)

 Corrections apportées :
   ✔ Règle pluie modérée + sol humide réintégrée
   ✔ Import api_openmeteo corrigé (construire_features, pas preparer_features_jour)
   ✔ Kc dynamique intégré dans le bilan hydrique
   ✔ Volume calculé uniquement si décision = OUI
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

import os, sys, joblib, argparse
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from typing import Optional

from config import (
    CLF_PATH, REG_PATH, FEATURES,
    LATITUDE_DEG, LONGITUDE_DEG, TIMEZONE,
    SURFACE_M2, EFFICACITE,
    SOL_HUMIDE_SEUIL, SOL_MOYEN_SEUIL,
    PLUIE_FORTE_SEUIL, PLUIE_MODERE_SEUIL,
    PLUIE_EFFECTIVE_PCT, OUT_DIR,
)
from agronomie import penman_monteith_fao56, kc_tomate, get_stade, bilan_hydrique


# ══════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES MODÈLES
# ══════════════════════════════════════════════════════════════

def charger_modeles() -> tuple:
    if not os.path.exists(CLF_PATH) or not os.path.exists(REG_PATH):
        print("❌ Modèles introuvables → lancez d'abord : 02_entrainement_ml.py")
        sys.exit(1)
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    print("✅ Modèles ML chargés (entraînés sur 2022-2024)")
    return clf, reg


# ══════════════════════════════════════════════════════════════
# 2. RÈGLES AGRONOMIQUES (couche prioritaire)
# ══════════════════════════════════════════════════════════════

def appliquer_regles(hs: float, pluie: float, deficit: float) -> tuple:
    """
    Retourne (décision, raison, court_circuit).
    CORRECTION : règle pluie modérée + sol humide réintégrée.
    """
    if hs > SOL_HUMIDE_SEUIL:
        return 0, f"Sol humide ({hs:.1f}% > {SOL_HUMIDE_SEUIL}%)", True
    if pluie > PLUIE_FORTE_SEUIL:
        return 0, f"Forte pluie ({pluie:.1f}mm > {PLUIE_FORTE_SEUIL}mm)", True
    if pluie > PLUIE_MODERE_SEUIL and hs > SOL_MOYEN_SEUIL:
        return 0, (f"Pluie modérée ({pluie:.1f}mm) + "
                   f"sol ok ({hs:.1f}% > {SOL_MOYEN_SEUIL}%)"), True
    if deficit <= 0.0:
        return 0, f"Pas de déficit hydrique ({deficit:.2f}mm)", True
    return 1, "Délégué au modèle ML", False


# ══════════════════════════════════════════════════════════════
# 3. MOTEUR DE DÉCISION COMPLET
# ══════════════════════════════════════════════════════════════

def decider(clf, reg, donnees: dict) -> dict:
    """
    Décision complète : règles agronomiques prioritaires → ML.

    Parameters
    ----------
    donnees : dict contenant toutes les clés de FEATURES
              + 'pluie_totale_mm', 'humidite_sol_moy_pct', 'saison'
    """
    # Calculs dérivés
    saison    = donnees.get("saison", "seche")
    jour_cycle= donnees.get("jour_cycle", 60)
    hr_min    = donnees.get("RH_min", 45.0)
    u2        = donnees.get("vent_u2_ms", 2.0)

    kc        = kc_tomate(jour_cycle, saison, hr_min, u2)
    ETc       = round(donnees["ET0_reference_mm"] * kc, 2)
    pluie_e   = round(donnees["pluie_totale_mm"] * PLUIE_EFFECTIVE_PCT, 2)
    deficit   = round(ETc - pluie_e, 2)

    donnees.update({
        "ETc_mm"              : ETc,
        "pluie_effective_mm"  : pluie_e,
        "deficit_hydrique_mm" : deficit,
    })

    hs    = donnees["humidite_sol_moy_pct"]
    pluie = donnees["pluie_totale_mm"]

    # ── Couche 1 : règles agronomiques ────────────────────
    dec_r, raison, court = appliquer_regles(hs, pluie, deficit)
    if court:
        return {
            "irriguer"  : dec_r,
            "volume_L"  : 0.0,
            "raison"    : raison,
            "source"    : "Règle agronomique",
            "confiance" : 100.0,
            "ETc_mm"    : ETc,
            "deficit_mm": deficit,
            "kc"        : kc,
            "stade"     : get_stade(jour_cycle),
        }

    # ── Couche 2 : modèle ML ──────────────────────────────
    X         = pd.DataFrame([donnees])[FEATURES]
    proba     = clf.predict_proba(X)[0]
    decision  = int(clf.predict(X)[0])
    confiance = round(proba[decision] * 100, 1)

    volume = 0.0
    if decision == 1:
        vol_ml  = float(reg.predict(X)[0])
        facteur = max(0.0, (65.0 - hs) / 25.0)
        vol_fo  = deficit * facteur * SURFACE_M2 / EFFICACITE
        volume  = round(max(0.60 * vol_ml + 0.40 * vol_fo, 0.0), 1)

    raison = (f"ML {confiance}% — Déficit {deficit:.2f}mm | "
              f"Sol {hs:.1f}% | Kc {kc:.4f} ({get_stade(jour_cycle)})")

    return {
        "irriguer"  : decision,
        "volume_L"  : volume,
        "raison"    : raison,
        "source"    : "Modèle ML (RF 2022-2024)",
        "confiance" : confiance,
        "ETc_mm"    : ETc,
        "deficit_mm": deficit,
        "kc"        : kc,
        "stade"     : get_stade(jour_cycle),
    }


# ══════════════════════════════════════════════════════════════
# 4. PRÉVISION 3 JOURS (simulation)
# ══════════════════════════════════════════════════════════════

def prevoir_3_jours(clf, reg, donnees_j0: dict, previsions: list) -> list:
    """Simule l'évolution du sol et les décisions pour J+1 à J+3."""
    resultats    = []
    humidite_sol = donnees_j0["humidite_sol_moy_pct"]

    for i, meteo in enumerate(previsions, 1):
        # Mise à jour humidité sol selon irrigation J-1
        if i > 1 and resultats[i - 2]["irriguer"]:
            apport_mm    = resultats[i - 2]["volume_L"] * EFFICACITE / SURFACE_M2
            humidite_sol = min(80.0, humidite_sol + apport_mm * 2.0)

        humidite_sol += meteo.get("pluie_totale_mm", 0.0) * PLUIE_EFFECTIVE_PCT * 0.1
        humidite_sol  = min(90.0, max(10.0, humidite_sol))

        d_j = date.today() + timedelta(days=i)
        donnees_j = {
            **meteo,
            "humidite_sol_moy_pct" : round(humidite_sol, 1),
            "humidite_sol_min_pct" : round(humidite_sol - 3.0, 1),
            "humidite_sol_0_7_moy" : round(humidite_sol - 2.0, 1),
            "jour_annee"           : d_j.timetuple().tm_yday,
            "mois"                 : d_j.month,
            "jour_cycle"           : donnees_j0.get("jour_cycle", 60) + i,
            "saison"               : donnees_j0.get("saison", "seche"),
        }

        res = decider(clf, reg, donnees_j)
        res["jour"]         = f"J+{i} ({d_j.strftime('%d/%m')})"
        res["humidite_sol"] = round(humidite_sol, 1)
        resultats.append(res)

        ETc_j        = meteo.get("ET0_reference_mm", 4.0) * res["kc"]
        humidite_sol = max(10.0, humidite_sol - ETc_j * 0.5)

    return resultats


# ══════════════════════════════════════════════════════════════
# 5. AFFICHAGE CONSOLE
# ══════════════════════════════════════════════════════════════

def afficher(label: str, res: dict, date_str: str = ""):
    icone = "✅ ARROSER       " if res["irriguer"] else "❌ NE PAS ARROSER"
    print(f"\n  ┌{'─'*60}┐")
    print(f"  │  {label:<56}  │")
    if date_str:
        print(f"  │  Date     : {date_str:<47}│")
    print(f"  ├{'─'*60}┤")
    print(f"  │  Décision : {icone:<47}│")
    if res["irriguer"]:
        print(f"  │  Volume   : {res['volume_L']:.0f} L (parcelle {SURFACE_M2}m²)"
              f"{'':>28}│")
    print(f"  │  Stade    : {res['stade']:<47}│")
    print(f"  │  Kc       : {res['kc']:.4f}{'':>43}│")
    print(f"  │  Source   : {res['source']:<47}│")
    raison = res["raison"]
    print(f"  │  Raison   : {raison[:47]:<47}│")
    if len(raison) > 47:
        print(f"  │             {raison[47:94]:<47}│")
    print(f"  │  ETc      : {res['ETc_mm']:.2f}mm/j  |  "
          f"Déficit : {res['deficit_mm']:.2f}mm{'':>22}│")
    print(f"  └{'─'*60}┘")


# ══════════════════════════════════════════════════════════════
# 6. GÉNÉRATION SMS
# ══════════════════════════════════════════════════════════════

def generer_sms(res_auj: dict, previsions: list, donnees_auj: dict) -> str:
    d     = date.today().strftime("%d/%m/%Y")
    dec   = "ARROSER ✅" if res_auj["irriguer"] else "PAS D'ARROSAGE ❌"
    vol   = f"{res_auj['volume_L']:.0f}L" if res_auj["irriguer"] else "0L"
    stade = res_auj.get("stade", "—").upper()

    sms  = f"🌱 AgroIrri CI — {d}\n"
    sms += f"Stade    : {stade}\n"
    sms += f"Kc       : {res_auj.get('kc', '—')}\n"
    sms += f"Décision : {dec}\n"
    sms += f"Volume   : {vol}\n"
    sms += f"Sol      : {donnees_auj['humidite_sol_moy_pct']:.1f}% | "
    sms += f"Pluie : {donnees_auj['pluie_totale_mm']:.1f}mm | "
    sms += f"ET₀ : {donnees_auj['ET0_reference_mm']:.2f}mm\n\n"
    sms += "Prévisions :\n"
    for i, p in enumerate(previsions, 1):
        # p est le dict retourné par prevoir_3_jours, qui contient la clé "jour"
        label = p.get("jour", f"J+{i}")
        e     = "✅" if p["irriguer"] else "❌"
        v     = f"{p['volume_L']:.0f}L" if p["irriguer"] else "0L"
        sms  += f"  {label} : {e} {v}\n"
    sms += f"\nIA entraînée 2022-2024 | {datetime.now():%d/%m/%Y %H:%M}"
    return sms


# ══════════════════════════════════════════════════════════════
# 7. DONNÉES EXEMPLE (mode simulation)
# ══════════════════════════════════════════════════════════════

def donnees_exemple_auj() -> dict:
    """Données représentatives d'une journée de saison sèche (jan–mars)."""
    d = date.today()
    return {
        "humidite_sol_moy_pct" : 38.0,
        "humidite_sol_min_pct" : 35.0,
        "humidite_sol_0_7_moy" : 36.0,
        "temp_max_C"           : 34.5,
        "temp_min_C"           : 22.0,
        "temp_moy_C"           : 28.0,
        "humidite_air_moy_pct" : 60.0,
        "vent_u2_ms"           : 2.0,
        "rayonnement_Rs_MJ"    : 18.0,
        "ET0_reference_mm"     : 5.0,
        "pluie_totale_mm"      : 0.0,
        "jour_annee"           : d.timetuple().tm_yday,
        "mois"                 : d.month,
        "jour_cycle"           : 60,     # floraison/mi-saison
        "saison"               : "seche",
        "RH_min"               : 45.0,
    }


def previsions_exemple() -> list:
    return [
        {"temp_max_C": 35.0, "temp_min_C": 22.5, "temp_moy_C": 28.5,
         "humidite_air_moy_pct": 58.0, "vent_u2_ms": 2.1,
         "rayonnement_Rs_MJ": 19.0, "ET0_reference_mm": 5.3,
         "pluie_totale_mm": 0.0, "RH_min": 42.0},
        {"temp_max_C": 29.0, "temp_min_C": 21.0, "temp_moy_C": 25.0,
         "humidite_air_moy_pct": 80.0, "vent_u2_ms": 1.5,
         "rayonnement_Rs_MJ": 11.0, "ET0_reference_mm": 3.5,
         "pluie_totale_mm": 14.0, "RH_min": 68.0},
        {"temp_max_C": 31.5, "temp_min_C": 22.0, "temp_moy_C": 26.5,
         "humidite_air_moy_pct": 65.0, "vent_u2_ms": 1.8,
         "rayonnement_Rs_MJ": 16.5, "ET0_reference_mm": 4.5,
         "pluie_totale_mm": 1.0, "RH_min": 50.0},
    ]


# ══════════════════════════════════════════════════════════════
# EXÉCUTION
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Irrigation intelligente CI")
    parser.add_argument("--mode", choices=["sim", "api"], default="sim",
                        help="sim=données exemples | api=Open-Meteo temps réel")
    parser.add_argument("--port", default="COM3",
                        help="Port série Arduino (COM3 ou /dev/ttyUSB0)")
    args = parser.parse_args()

    clf, reg = charger_modeles()

    print("\n" + "═" * 64)
    print("   🌱 SYSTÈME D'IRRIGATION INTELLIGENTE — M. KOFFI")
    print(f"   📍 Yamoussoukro, CI  |  🍅 Tomate  |  {SURFACE_M2}m²")
    print(f"   📅 {date.today():%d/%m/%Y}  |  Mode : {args.mode.upper()}")
    print(f"   🧠 Modèles entraînés sur 2022-2024 (1096 jours)")
    print("═" * 64)

    if args.mode == "api":
        # ✅ CORRECTION : nom de fonction corrigé (construire_features)
        from api_openmeteo import appeler_api, construire_features, lire_capteur_arduino
        df_h, df_q       = appeler_api()
        humidite_capteur = lire_capteur_arduino(args.port)
        donnees_auj      = construire_features(df_h, df_q, 0, humidite_capteur)
        previsions_3j    = [construire_features(df_h, df_q, i) for i in range(1, 4)]
    else:
        print("   ℹ  Mode simulation — données exemples utilisées")
        print("      (--mode api pour les données météo réelles)\n")
        donnees_auj   = donnees_exemple_auj()
        previsions_3j = previsions_exemple()

    # ── Décision du jour ──────────────────────────────────
    res_auj = decider(clf, reg, donnees_auj.copy())
    afficher("AUJOURD'HUI", res_auj, date.today().strftime("%d/%m/%Y"))

    # ── Prévisions 3 jours ────────────────────────────────
    print("\n" + "─" * 64)
    print("   PRÉVISIONS 3 JOURS :")
    previsions_res = prevoir_3_jours(clf, reg, donnees_auj, previsions_3j)
    for p in previsions_res:
        afficher(p["jour"], p)

    # ── SMS ───────────────────────────────────────────────
    sms = generer_sms(res_auj, previsions_res, donnees_auj)
    print("\n" + "─" * 64)
    print("   📱 MESSAGE SMS POUR M. KOFFI :")
    print("   " + "─" * 48)
    for ligne in sms.split("\n"):
        print(f"   {ligne}")
    print("   " + "─" * 48)

    # ── Log CSV — schéma fixe pour éviter ParserError ────
    # CORRECTION : colonnes toujours dans le même ordre, peu importe
    # l'ordre d'initialisation du dict. Un CSV en append échoue si les
    # colonnes varient d'une exécution à l'autre (pandas.errors.ParserError).
    COLONNES_LOG = [
        "timestamp", "date", "mode", "irriguer", "volume_L",
        "source", "stade", "kc", "humidite_sol",
        "ET0_mm", "pluie_mm", "deficit_mm",
    ]
    log = {
        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "date"        : str(date.today()),
        "mode"        : args.mode,
        "irriguer"    : res_auj["irriguer"],
        "volume_L"    : res_auj["volume_L"],
        "source"      : res_auj["source"],
        "stade"       : res_auj.get("stade", "—"),
        "kc"          : res_auj.get("kc", "—"),
        "humidite_sol": donnees_auj["humidite_sol_moy_pct"],
        "ET0_mm"      : donnees_auj["ET0_reference_mm"],
        "pluie_mm"    : donnees_auj["pluie_totale_mm"],
        "deficit_mm"  : res_auj["deficit_mm"],
    }
    lp        = os.path.join(OUT_DIR, "historique_decisions.csv")
    existe    = os.path.exists(lp)
    df_log    = pd.DataFrame([log])[COLONNES_LOG]   # ordre garanti

    if existe:
        # Vérifier que le schéma existant est compatible
        try:
            df_existant = pd.read_csv(lp, nrows=0)
            if list(df_existant.columns) != COLONNES_LOG:
                # Schéma incompatible : renommer l'ancien et repartir proprement
                import shutil
                shutil.move(lp, lp.replace(".csv", "_archive.csv"))
                existe = False
                print("   ⚠  Ancien historique archivé (schéma incompatible) "
                      "→ historique_decisions_archive.csv")
        except Exception:
            existe = False   # Fichier corrompu : on recrée

    df_log.to_csv(lp, mode="a", header=not existe, index=False)
    print(f"\n   💾 Décision enregistrée → historique_decisions.csv")
    print("═" * 64)