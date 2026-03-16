"""
01_preparation_donnees.py
Preparation du dataset ML 2022-2024.

Source      : Open-Meteo (Yamoussoukro, 6.82N 5.28W, 212m)
Culture     : Tomate (Solanum lycopersicum)
Surface     : 200 m2
Periode     : 3 ans (1096 jours)
"""

import os
import sys

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR, os.path.join(_PARENT_DIR, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import numpy as np

from config import (
    CSV_BRUT, CSV_CLEAN, OUT_DIR,
    PLUIE_EFFECTIVE_PCT, EFFICACITE, SURFACE_M2,
    SOL_HUMIDE_SEUIL, SOL_MOYEN_SEUIL,
    PLUIE_FORTE_SEUIL, PLUIE_MODERE_SEUIL,
    FEATURES,
)
from agronomie import penman_monteith_fao56, kc_tomate, get_stade, get_saison


# ── Lecture Open-Meteo ────────────────────────────────────────────────────

def detecter_separation(filepath: str) -> int:
    """Detecte la ligne de separation entre bloc horaire et quotidien."""
    with open(filepath, "r", encoding="utf-8") as f:
        for i, ligne in enumerate(f):
            if i > 5 and ligne.startswith("time,temperature_2m_mean"):
                return i
    raise ValueError("Separation horaire/quotidien introuvable dans le CSV.")


def lire_open_meteo(filepath: str) -> tuple:
    """Lit le fichier Open-Meteo et retourne (df_horaire, df_quotidien)."""
    print("Lecture du fichier brut Open-Meteo...")
    sep  = detecter_separation(filepath)
    n_h  = sep - 4
    df_h = pd.read_csv(filepath, skiprows=3, nrows=n_h,  parse_dates=["time"])
    df_q = pd.read_csv(filepath, skiprows=sep - 1,        parse_dates=["time"])
    print(f"  Horaires   : {len(df_h):,} lignes")
    print(f"  Quotidiens : {len(df_q):,} lignes")
    return df_h, df_q


# ── Nettoyage colonnes ────────────────────────────────────────────────────

def renommer(df_h: pd.DataFrame, df_q: pd.DataFrame) -> tuple:
    df_h.rename(columns={
        "time"                            : "datetime",
        "temperature_2m (°C)"             : "temp_C",
        "rain (mm)"                       : "pluie_mm",
        "wind_speed_10m (km/h)"           : "vent_10m_kmh",
        "soil_temperature_0_to_7cm (°C)"  : "temp_sol_0_7cm",
        "soil_moisture_7_to_28cm (m³/m³)" : "humidite_sol_7_28cm_m3",
        "relative_humidity_2m (%)"        : "humidite_air_pct",
        "soil_moisture_0_to_7cm (m³/m³)"  : "humidite_sol_0_7cm_m3",
    }, inplace=True)

    df_q.rename(columns={
        "time"                            : "date",
        "temperature_2m_mean (°C)"        : "temp_moy_C",
        "temperature_2m_max (°C)"         : "temp_max_C",
        "temperature_2m_min (°C)"         : "temp_min_C",
        "wind_speed_10m_max (km/h)"       : "vent_max_kmh",
        "shortwave_radiation_sum (MJ/m²)" : "rayonnement_Rs_MJ",
        "et0_fao_evapotranspiration (mm)" : "ET0_reference_mm",
        "precipitation_sum (mm)"          : "pluie_totale_mm",
        "relative_humidity_2m_max (%)"    : "RH_max",
        "relative_humidity_2m_min (%)"    : "RH_min",
    }, inplace=True)
    return df_h, df_q


def convertir_sol(df_h: pd.DataFrame) -> pd.DataFrame:
    """Convertit l'humidite du sol de m3/m3 en pourcentage."""
    df_h["humidite_sol_0_7cm_pct"]  = (df_h["humidite_sol_0_7cm_m3"]  * 100).round(1)
    df_h["humidite_sol_7_28cm_pct"] = (df_h["humidite_sol_7_28cm_m3"] * 100).round(1)
    return df_h


def agreger(df_h: pd.DataFrame) -> pd.DataFrame:
    """
    Agrege les donnees horaires en quotidien.
    Le vent est agrege en MOYENNE (FAO-56 utilise u2 moyen journalier).
    """
    df_h["date"] = df_h["datetime"].dt.date
    return df_h.groupby("date").agg(
        humidite_sol_moy_pct = ("humidite_sol_7_28cm_pct", "mean"),
        humidite_sol_min_pct = ("humidite_sol_7_28cm_pct", "min"),
        humidite_sol_max_pct = ("humidite_sol_7_28cm_pct", "max"),
        humidite_sol_0_7_moy = ("humidite_sol_0_7cm_pct",  "mean"),
        pluie_horaire_sum_mm = ("pluie_mm",                 "sum"),
        humidite_air_moy_pct = ("humidite_air_pct",         "mean"),
        humidite_air_max_pct = ("humidite_air_pct",         "max"),
        humidite_air_min_pct = ("humidite_air_pct",         "min"),
        vent_moy_kmh         = ("vent_10m_kmh",             "mean"),
    ).reset_index()


# ── Labels ────────────────────────────────────────────────────────────────

def decision(row: pd.Series) -> int:
    hs = row["humidite_sol_moy_pct"]
    p  = row["pluie_totale_mm"]
    d  = row["deficit_hydrique_mm"]
    if hs > SOL_HUMIDE_SEUIL:                                return 0
    if p  > PLUIE_FORTE_SEUIL:                               return 0
    if p  > PLUIE_MODERE_SEUIL and hs > SOL_MOYEN_SEUIL:    return 0
    if d  <= 0.0:                                            return 0
    return 1


def volume(row: pd.Series) -> float:
    if row["irriguer"] == 0:
        return 0.0
    facteur = max(0.0, (65.0 - row["humidite_sol_moy_pct"]) / 25.0)
    besoin  = row["deficit_hydrique_mm"] * facteur
    return round(besoin * SURFACE_M2 / EFFICACITE, 1) if besoin > 0 else 0.0


# ── Colonnes derivees ─────────────────────────────────────────────────────

def _recalculer_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    if "annee" not in df.columns:
        df["annee"] = pd.to_datetime(df["date"]).dt.year
    if "mois" not in df.columns:
        df["mois"] = pd.to_datetime(df["date"]).dt.month
    if "jour_annee" not in df.columns:
        df["jour_annee"] = pd.to_datetime(df["date"]).dt.dayofyear
    if "saison" not in df.columns:
        df["saison"] = df["mois"].map(get_saison)
    if "kc_dynamique" not in df.columns:
        df["kc_dynamique"] = 1.05
    if "ETc_mm" not in df.columns:
        df["ETc_mm"]              = (df["ET0_reference_mm"] * df["kc_dynamique"]).round(2)
        df["pluie_effective_mm"]  = (df["pluie_totale_mm"]  * PLUIE_EFFECTIVE_PCT).round(2)
        df["deficit_hydrique_mm"] = (df["ETc_mm"] - df["pluie_effective_mm"]).round(2)
    return df


# ── Pipeline principal ────────────────────────────────────────────────────

def preparer_dataset(source: str = "auto") -> pd.DataFrame:
    """
    Prepare le dataset ML 2022-2024.

    Parameters
    ----------
    source : 'auto'  -> charge le nettoye si existant, sinon relit le brut
             'brut'  -> force la relecture du CSV Open-Meteo
             'clean' -> charge uniquement le nettoye
    """
    if source != "brut" and os.path.exists(CSV_CLEAN):
        print("Dataset nettoye trouve, chargement...")
        df = pd.read_csv(CSV_CLEAN, parse_dates=["date"])
        df = _recalculer_colonnes(df)
        print(f"  {len(df)} jours | {df['date'].min().date()} -> {df['date'].max().date()}")
        return df

    df_h, df_q = lire_open_meteo(CSV_BRUT)
    df_h, df_q = renommer(df_h, df_q)
    df_h       = convertir_sol(df_h)
    df_sol     = agreger(df_h)

    df_q["date"] = pd.to_datetime(df_q["date"]).dt.date
    df = df_q.merge(df_sol, on="date", how="left")

    # Vent u2 : conversion 10m -> 2m, FAO-56 eq. 47
    df["vent_u2_ms"] = (df["vent_moy_kmh"] / 3.6 * 0.748).round(3)

    if "RH_max" not in df.columns:
        df["RH_max"] = df["humidite_air_max_pct"]
    if "RH_min" not in df.columns:
        df["RH_min"] = df["humidite_air_min_pct"]

    df["mois"]       = pd.to_datetime(df["date"]).dt.month
    df["jour_annee"] = pd.to_datetime(df["date"]).dt.dayofyear
    df["annee"]      = pd.to_datetime(df["date"]).dt.year
    df["saison"]     = df["mois"].map(get_saison)

    print("Calcul ET0 Penman-Monteith FAO-56...")
    df["ET0_calcule_mm"] = df.apply(lambda r: penman_monteith_fao56(
        T_max  = r.temp_max_C,
        T_min  = r.temp_min_C,
        T_moy  = r.temp_moy_C,
        RH_max = r.get("RH_max", r.humidite_air_max_pct),
        RH_min = r.get("RH_min", r.humidite_air_min_pct),
        u2     = r.vent_u2_ms,
        Rs     = r.rayonnement_Rs_MJ,
        J      = r.jour_annee,
    ), axis=1)

    valides = df["ET0_calcule_mm"].notna().sum()
    corr    = df[["ET0_reference_mm", "ET0_calcule_mm"]].dropna().corr().iloc[0, 1]
    print(f"  Valides : {valides}/{len(df)} | Correlation ET0 calculee/reference : {corr:.4f}")

    print("Calcul Kc dynamique...")
    df["jour_cycle"]    = df["jour_annee"] % 130
    df["kc_dynamique"]  = df.apply(lambda r: kc_tomate(
        jour_cycle = int(r.jour_cycle),
        saison     = r.saison,
        hr_min     = r.get("RH_min", 45.0),
        u2         = r.vent_u2_ms,
    ), axis=1)
    df["stade_culture"] = df["jour_cycle"].apply(get_stade)

    df["ETc_mm"]              = (df["ET0_reference_mm"] * df["kc_dynamique"]).round(2)
    df["pluie_effective_mm"]  = (df["pluie_totale_mm"]  * PLUIE_EFFECTIVE_PCT).round(2)
    df["deficit_hydrique_mm"] = (df["ETc_mm"] - df["pluie_effective_mm"]).round(2)

    df["irriguer"]      = df.apply(decision, axis=1)
    df["volume_litres"] = df.apply(volume,   axis=1)

    df.to_csv(CSV_CLEAN, index=False)
    print(f"Dataset sauvegarde -> {CSV_CLEAN}")
    return df


# ── Rapport de qualite ────────────────────────────────────────────────────

def rapport_qualite(df: pd.DataFrame):
    df     = _recalculer_colonnes(df)
    annees = sorted(df["annee"].unique().tolist())

    print("\n" + "=" * 56)
    print("  RAPPORT DE QUALITE DU DATASET")
    print("=" * 56)
    print(f"  Periode      : {df['date'].min()} -> {df['date'].max()}")
    print(f"  Observations : {len(df)} jours ({annees})")
    print(f"  Colonnes     : {df.shape[1]}")
    print(f"  Valeurs nulles : {df.isnull().sum().sum()}")

    d = df["irriguer"].value_counts()
    print(f"\n  Distribution decisions :")
    print(f"    OUI : {d.get(1,0):>5} jours ({d.get(1,0)/len(df)*100:.1f}%)")
    print(f"    NON : {d.get(0,0):>5} jours ({d.get(0,0)/len(df)*100:.1f}%)")

    print(f"\n  Par annee :")
    for yr, g in df.groupby("annee"):
        v = g[g.volume_litres > 0]["volume_litres"].mean()
        print(f"    {yr} : {len(g)} j | irrig={(g.irriguer==1).sum()} | "
              f"pluie={g.pluie_totale_mm.sum():.0f}mm | vol_moy={v:.0f}L")

    irr = df[df["volume_litres"] > 0]["volume_litres"]
    print(f"\n  Volumes (jours irrigues) : moy={irr.mean():.0f}L | "
          f"min={irr.min():.0f}L | max={irr.max():.0f}L")
    print(f"  ET0 moyen    : {df['ET0_reference_mm'].mean():.2f} mm/j")
    print(f"  Pluie totale : {df['pluie_totale_mm'].sum():.0f}mm "
          f"(~{df['pluie_totale_mm'].sum()/len(annees):.0f}mm/an)")
    print("=" * 56)


if __name__ == "__main__":
    df = preparer_dataset(source="auto")
    rapport_qualite(df)
    print("\nEtape 1 terminee -> lancez : 02_entrainement_ml.py")
