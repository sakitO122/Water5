"""
=============================================================
 SYSTÈME D'IRRIGATION INTELLIGENTE — CÔTE D'IVOIRE
 04_backtesting.py
=============================================================
 Simule le système sur n'importe quelle date du dataset.
 Compare la prédiction ML à la réalité connue.

 Usage :
   python src/04_backtesting.py                        ← 23/04/2024
   python src/04_backtesting.py 2022-07-15             ← date choisie
   python src/04_backtesting.py 2023-11-01 full        ← + rapport annuel
   python src/04_backtesting.py 2022-01-01 full        ← rapport 2022

 Corrections apportées :
   ✔ Virgule parasite supprimée dans recuperer_jours()
   ✔ Évaluation uniquement sur données de TEST (jeu chronologique)
   ✔ Kc dynamique intégré dans predire_jour()
   ✔ Règle pluie modérée + sol humide réintégrée
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

import os, sys, joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from config import (
    CSV_CLEAN, CLF_PATH, REG_PATH,
    FEATURES, PLUIE_EFFECTIVE_PCT, EFFICACITE, SURFACE_M2,
    SOL_HUMIDE_SEUIL, SOL_MOYEN_SEUIL,
    PLUIE_FORTE_SEUIL, PLUIE_MODERE_SEUIL,
)
from agronomie import kc_tomate, get_stade


# ══════════════════════════════════════════════════════════════
# 1. RÈGLES AGRONOMIQUES
# ══════════════════════════════════════════════════════════════

def appliquer_regles(hs: float, pluie: float, deficit: float) -> tuple:
    """Règles agronomiques prioritaires avec toutes les conditions."""
    if hs > SOL_HUMIDE_SEUIL:
        return 0, f"Sol humide ({hs:.1f}%)", True
    if pluie > PLUIE_FORTE_SEUIL:
        return 0, f"Forte pluie ({pluie:.1f}mm)", True
    if pluie > PLUIE_MODERE_SEUIL and hs > SOL_MOYEN_SEUIL:
        return 0, f"Pluie modérée + sol ok ({hs:.1f}%)", True
    if deficit <= 0.0:
        return 0, f"Pas de déficit ({deficit:.2f}mm)", True
    return 1, "Délégué au ML", False


def predire_jour(clf, reg, row: pd.Series) -> dict:
    """
    Prédit la décision d'irrigation pour un jour donné.
    Intègre le Kc dynamique selon le stade et la saison.
    """
    saison     = row.get("saison", "seche")
    jour_cycle = int(row.get("jour_cycle", 60))
    hr_min     = float(row.get("RH_min", row.get("humidite_air_min_pct", 45.0)))
    u2         = float(row.get("vent_u2_ms", 2.0))

    kc      = kc_tomate(jour_cycle, saison, hr_min, u2)
    stade   = get_stade(jour_cycle)
    ETc     = round(float(row["ET0_reference_mm"]) * kc, 2)
    pluie_e = round(float(row["pluie_totale_mm"]) * PLUIE_EFFECTIVE_PCT, 2)
    deficit = round(ETc - pluie_e, 2)

    hs    = float(row["humidite_sol_moy_pct"])
    pluie = float(row["pluie_totale_mm"])

    dec_r, raison, court = appliquer_regles(hs, pluie, deficit)
    if court:
        return {
            "irriguer_predit": dec_r, "volume_predit_L": 0.0,
            "raison": raison, "source": "Règle agronomique",
            "confiance": 100.0, "ETc_mm": ETc,
            "deficit_mm": deficit, "kc": kc, "stade": stade,
        }

    # Mise à jour des colonnes dérivées dans row
    row = row.copy()
    row["ETc_mm"]              = ETc
    row["pluie_effective_mm"]  = pluie_e
    row["deficit_hydrique_mm"] = deficit

    X         = pd.DataFrame([row])[FEATURES]
    proba     = clf.predict_proba(X)[0]
    decision  = int(clf.predict(X)[0])
    confiance = round(proba[decision] * 100, 1)

    volume = 0.0
    if decision == 1:
        vol_ml  = float(reg.predict(X)[0])
        facteur = max(0.0, (65.0 - hs) / 25.0)
        vol_fo  = deficit * facteur * SURFACE_M2 / EFFICACITE
        volume  = round(max(0.60 * vol_ml + 0.40 * vol_fo, 0.0), 1)

    return {
        "irriguer_predit": decision, "volume_predit_L": volume,
        "raison": f"ML confiance {confiance}%",
        "source": "Modèle ML", "confiance": confiance,
        "ETc_mm": ETc, "deficit_mm": deficit, "kc": kc, "stade": stade,
    }


# ══════════════════════════════════════════════════════════════
# 2. RÉCUPÉRATION DES JOURS
# ══════════════════════════════════════════════════════════════

def recuperer_jours(date_str: str, df: pd.DataFrame) -> list:
    """
    Retourne les 4 lignes du dataset à partir de la date cible.
    CORRECTION : virgule parasite supprimée.
    """
    df["date_dt"] = pd.to_datetime(df["date"]).dt.date
    dates         = df["date_dt"].tolist()
    date_cible    = pd.to_datetime(date_str).date()

    if date_cible not in dates:
        debut = str(min(dates)); fin = str(max(dates))
        print(f"❌ Date {date_str} non trouvée.")
        print(f"   Période disponible : {debut} → {fin}")
        sys.exit(1)

    jours = []
    for i in range(4):
        d = date_cible + timedelta(days=i)
        if d in dates:
            jours.append(df[df["date_dt"] == d].iloc[0])
    return jours


# ══════════════════════════════════════════════════════════════
# 3. AFFICHAGE DES RÉSULTATS
# ══════════════════════════════════════════════════════════════

def afficher_resultats(jours: list, clf, reg, date_str: str) -> tuple:
    print("\n" + "═" * 66)
    print(f"  🌱 BACKTESTING — Simulation à partir du {date_str}")
    print(f"  📍 Yamoussoukro, CI | 🍅 Tomate | Dataset : 2022-2024")
    print("═" * 66)

    labels = [
        "AUJOURD'HUI (J)  ",
        "DEMAIN    (J+1)  ",
        "APRÈS-DEM (J+2)  ",
        "J+3              ",
    ]
    preds, reels = [], []

    for i, row in enumerate(jours):
        lbl      = labels[i] if i < len(labels) else f"J+{i}"
        date_aff = str(row["date"])[:10]
        res      = predire_jour(clf, reg, row)
        vrai_irr = int(row["irriguer"])
        vrai_vol = float(row["volume_litres"])
        correct  = (res["irriguer_predit"] == vrai_irr)
        ip       = "✅" if res["irriguer_predit"] else "❌"
        ir       = "✅" if vrai_irr else "❌"
        iv       = "🎯" if correct else "⚠️ "

        print(f"\n  ┌{'─'*62}┐")
        print(f"  │  {lbl} — {date_aff}  (saison : {row['saison']}){'':>14}│")
        print(f"  ├{'─'*62}┤")
        print(f"  │  📊 DONNÉES RÉELLES DU JOUR :{'':>33}│")
        print(f"  │     Humidité sol   : {row['humidite_sol_moy_pct']:.1f}%{'':>41}│")
        print(f"  │     Température    : {row['temp_max_C']:.1f}°C max / "
              f"{row['temp_min_C']:.1f}°C min{'':>26}│")
        print(f"  │     Pluie          : {row['pluie_totale_mm']:.1f}mm{'':>40}│")
        print(f"  │     ET₀ référence  : {row['ET0_reference_mm']:.2f}mm/j{'':>38}│")
        print(f"  ├{'─'*62}┤")
        print(f"  │  🤖 PRÉDICTION MODÈLE ML (Kc dynamique) :{'':>21}│")
        dec_txt = "ARROSER        " if res["irriguer_predit"] else "NE PAS ARROSER "
        print(f"  │     {ip} {dec_txt}  (Kc={res['kc']:.4f}, {res['stade']}){'':>17}│")
        if res["irriguer_predit"]:
            print(f"  │        Volume prédit   : {res['volume_predit_L']:.0f}L{'':>35}│")
        print(f"  │        ETc             : {res['ETc_mm']:.2f}mm/j{'':>33}│")
        print(f"  │        Déficit         : {res['deficit_mm']:.2f}mm{'':>34}│")
        print(f"  │        Raison          : {res['raison'][:37]:<37}{'':>1}│")
        print(f"  ├{'─'*62}┤")
        print(f"  │  📋 RÉALITÉ (dataset 2022-2024) :{'':>29}│")
        reel_txt = "ARROSER        " if vrai_irr else "NE PAS ARROSER "
        print(f"  │     {ir} {reel_txt}{'':>42}│")
        if vrai_irr:
            print(f"  │        Volume réel     : {vrai_vol:.0f}L{'':>36}│")
        print(f"  ├{'─'*62}┤")
        verdict = "CORRECT   ✅" if correct else "INCORRECT  ⚠️ "
        print(f"  │  {iv} VERDICT : {verdict:<52}│")
        if correct and res["irriguer_predit"] and vrai_vol > 0:
            ecart     = abs(res["volume_predit_L"] - vrai_vol)
            ecart_pct = ecart / vrai_vol * 100
            print(f"  │     Écart volume : {ecart:.0f}L ({ecart_pct:.1f}%){'':>34}│")
        print(f"  └{'─'*62}┘")

        preds.append(res["irriguer_predit"])
        reels.append(vrai_irr)

    n_ok  = sum(p == r for p, r in zip(preds, reels))
    score = n_ok / len(preds) * 100
    print(f"\n  {'═'*62}")
    print(f"  📈 SCORE : {n_ok}/{len(preds)} correct ({score:.0f}%)")
    print(f"  {'═'*62}")
    return preds, reels


# ══════════════════════════════════════════════════════════════
# 4. RAPPORT COMPLET PAR ANNÉE (mode full)
# ══════════════════════════════════════════════════════════════

def rapport_complet(df: pd.DataFrame, clf, reg, annee: Optional[int] = None):
    """
    Teste le modèle sur les données de TEST (20% chronologiques).
    CORRECTION : évaluation sur jeu de test uniquement, pas sur train.
    """
    # ── Utiliser uniquement les 20% finaux (jeu de test) ──
    df_sorted = df.sort_values("date").reset_index(drop=True)
    cut       = int(len(df_sorted) * 0.80)
    df_test   = df_sorted.iloc[cut:].copy()

    if annee:
        df_test = df_test[df_test["annee"] == annee].copy()
        titre   = f"Backtesting TEST — année {annee}"
    else:
        titre = "Backtesting TEST — 20% chronologiques"

    print("\n" + "═" * 66)
    print(f"  📊 RAPPORT COMPLET — {titre}")
    print(f"  ⚠  Évaluation sur jeu de TEST uniquement (pas de leakage)")
    print("═" * 66)

    if len(df_test) == 0:
        print(f"  ℹ  Aucune donnée pour l'année {annee} dans le jeu de test.")
        return

    preds, reels, ecarts_vol = [], [], []

    for _, row in df_test.iterrows():
        res      = predire_jour(clf, reg, row)
        vrai_irr = int(row["irriguer"])
        vrai_vol = float(row["volume_litres"])
        preds.append(res["irriguer_predit"])
        reels.append(vrai_irr)
        if res["irriguer_predit"] == 1 and vrai_irr == 1:
            ecarts_vol.append(abs(res["volume_predit_L"] - vrai_vol))

    preds = np.array(preds); reels = np.array(reels)
    acc   = (preds == reels).mean() * 100
    VP    = ((preds == 1) & (reels == 1)).sum()
    VN    = ((preds == 0) & (reels == 0)).sum()
    FP    = ((preds == 1) & (reels == 0)).sum()
    FN    = ((preds == 0) & (reels == 1)).sum()
    prec  = VP / (VP + FP) * 100 if (VP + FP) > 0 else 0.0
    rec   = VP / (VP + FN) * 100 if (VP + FN) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"\n  Période testée : {df_test['date'].min()} → {df_test['date'].max()}")
    print(f"  Observations  : {len(df_test)} jours\n")
    print(f"  CLASSIFICATION :")
    print(f"  {'─'*47}")
    print(f"  Accuracy   : {acc:.2f}%")
    print(f"  Précision  : {prec:.2f}%")
    print(f"  Rappel     : {rec:.2f}%")
    print(f"  F1-score   : {f1:.2f}%")
    print(f"\n  Matrice de confusion :")
    print(f"               Prédit NON   Prédit OUI")
    print(f"  Réel NON  :      {VN:>4}         {FP:>4}")
    print(f"  Réel OUI  :      {FN:>4}         {VP:>4}")

    if ecarts_vol:
        print(f"\n  RÉGRESSION (volume en litres) :")
        print(f"  {'─'*47}")
        print(f"  MAE    : {np.mean(ecarts_vol):.0f}L")
        print(f"  Médiane: {np.median(ecarts_vol):.0f}L")
        print(f"  Max    : {np.max(ecarts_vol):.0f}L")

    print(f"\n  🔴 Oublis irrigation (FN) : {FN} jours")
    print(f"  ⚠  Irrigations inutiles (FP) : {FP} jours")

    print(f"\n  PAR SAISON :")
    print(f"  {'─'*52}")
    df_test["predit"] = preds
    df_test["reel"]   = reels
    for sais, grp in df_test.groupby("saison"):
        acc_s = (grp["predit"] == grp["reel"]).mean() * 100
        print(f"  {sais:<22} : {len(grp):>4}j | "
              f"irrig={( grp.reel == 1).sum():>3} | acc={acc_s:.1f}%")
    print("═" * 66)


# ══════════════════════════════════════════════════════════════
# EXÉCUTION
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    date_str = sys.argv[1] if len(sys.argv) > 1 else "2024-04-23"
    mode     = sys.argv[2] if len(sys.argv) > 2 else "normal"

    print("📂 Chargement...")
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    df  = pd.read_csv(CSV_CLEAN, parse_dates=["date"])
    df  = df.sort_values("date").reset_index(drop=True)

    # Colonnes dérivées
    if "annee"      not in df.columns: df["annee"]      = df["date"].dt.year
    if "mois"       not in df.columns: df["mois"]       = df["date"].dt.month
    if "jour_annee" not in df.columns: df["jour_annee"] = df["date"].dt.dayofyear
    if "saison"     not in df.columns:
        from agronomie import get_saison
        df["saison"] = df["mois"].map(get_saison)
    if "jour_cycle" not in df.columns: df["jour_cycle"] = df["jour_annee"] % 130
    if "kc_dynamique" not in df.columns: df["kc_dynamique"] = 1.05
    if "ETc_mm" not in df.columns:
        from config import PLUIE_EFFECTIVE_PCT
        df["ETc_mm"]              = (df["ET0_reference_mm"] * df["kc_dynamique"]).round(2)
        df["pluie_effective_mm"]  = (df["pluie_totale_mm"]  * PLUIE_EFFECTIVE_PCT).round(2)
        df["deficit_hydrique_mm"] = (df["ETc_mm"] - df["pluie_effective_mm"]).round(2)

    print(f"   ✅ Dataset 2022-2024 : {len(df)} jours")

    jours = recuperer_jours(date_str, df)
    afficher_resultats(jours, clf, reg, date_str)

    if mode == "full":
        annee = pd.to_datetime(date_str).year
        rapport_complet(df, clf, reg, annee=annee)
        rapport_complet(df, clf, reg, annee=None)

    print(f"\n💡 Autres tests :")
    print(f"   python src/04_backtesting.py 2022-07-15")
    print(f"   python src/04_backtesting.py 2023-01-10")
    print(f"   python src/04_backtesting.py 2024-09-05 full")