"""
06_api_openmeteo.py
Prediction du jour J et J+3 via donnees meteo Open-Meteo.
Integration capteur Arduino/DHT22 optionnelle.

Installation :
  pip install openmeteo-requests requests-cache retry-requests pyserial

Usage :
  python src/06_api_openmeteo.py                 # decision du jour
  python src/06_api_openmeteo.py --test          # donnees brutes
  python src/06_api_openmeteo.py --port COM4     # avec Arduino
  python src/06_api_openmeteo.py --no-capteur    # sans capteur
"""

import os
import sys
import math
import joblib
import argparse
from typing import Optional
from datetime import date, datetime, timedelta

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR, os.path.join(_PARENT_DIR, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

from config import (
    CLF_PATH, REG_PATH, FEATURES, OUT_DIR,
    LATITUDE_DEG, LONGITUDE_DEG, TIMEZONE,
    SURFACE_M2, EFFICACITE, PLUIE_EFFECTIVE_PCT,
    SOL_HUMIDE_SEUIL, SOL_MOYEN_SEUIL,
    PLUIE_FORTE_SEUIL, PLUIE_MODERE_SEUIL,
)
from agronomie import kc_tomate, get_stade


# ── API Open-Meteo ────────────────────────────────────────────────────────

def appeler_api() -> tuple:
    """Recupere les donnees meteo via openmeteo-requests (cache 1h)."""
    try:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry
    except ImportError:
        print("Bibliotheques manquantes :")
        print("  pip install openmeteo-requests requests-cache retry-requests")
        sys.exit(1)

    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo     = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude"      : LATITUDE_DEG,
        "longitude"     : LONGITUDE_DEG,
        "timezone"      : TIMEZONE,
        "forecast_days" : 4,
        "hourly": [
            "temperature_2m", "rain", "wind_speed_10m",
            "soil_temperature_0_to_7cm", "soil_moisture_7_to_28cm",
            "relative_humidity_2m", "soil_moisture_0_to_7cm",
        ],
        "daily": [
            "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "wind_speed_10m_max", "shortwave_radiation_sum",
            "et0_fao_evapotranspiration", "precipitation_sum",
        ],
    }

    responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    response  = responses[0]

    hourly  = response.Hourly()
    dates_h = pd.date_range(
        start     = pd.to_datetime(hourly.Time(),    unit="s", utc=True),
        end       = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq      = pd.Timedelta(seconds=hourly.Interval()),
        inclusive = "left",
    ).tz_convert(TIMEZONE)

    df_h = pd.DataFrame({
        "datetime"               : dates_h,
        "temp_C"                 : hourly.Variables(0).ValuesAsNumpy(),
        "pluie_mm"               : hourly.Variables(1).ValuesAsNumpy(),
        "vent_10m_kmh"           : hourly.Variables(2).ValuesAsNumpy(),
        "temp_sol_0_7cm"         : hourly.Variables(3).ValuesAsNumpy(),
        "humidite_sol_7_28cm_m3" : hourly.Variables(4).ValuesAsNumpy(),
        "humidite_air_pct"       : hourly.Variables(5).ValuesAsNumpy(),
        "humidite_sol_0_7cm_m3"  : hourly.Variables(6).ValuesAsNumpy(),
    })

    # Conversion m3/m3 -> % avec gestion NaN
    # Open-Meteo peut renvoyer NaN sur les heures futures
    df_h["humidite_sol_7_28cm_pct"] = (df_h["humidite_sol_7_28cm_m3"] * 100).round(1)
    df_h["humidite_sol_0_7cm_pct"]  = (df_h["humidite_sol_0_7cm_m3"]  * 100).round(1)
    df_h["date_only"] = pd.to_datetime(df_h["datetime"]).dt.date

    daily   = response.Daily()
    dates_d = pd.date_range(
        start     = pd.to_datetime(daily.Time(),    unit="s", utc=True),
        end       = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq      = pd.Timedelta(seconds=daily.Interval()),
        inclusive = "left",
    ).tz_convert(TIMEZONE)

    df_q = pd.DataFrame({
        "date"              : dates_d.date,
        "temp_moy_C"        : daily.Variables(0).ValuesAsNumpy(),
        "temp_max_C"        : daily.Variables(1).ValuesAsNumpy(),
        "temp_min_C"        : daily.Variables(2).ValuesAsNumpy(),
        "vent_max_kmh"      : daily.Variables(3).ValuesAsNumpy(),
        "rayonnement_Rs_MJ" : daily.Variables(4).ValuesAsNumpy(),
        "ET0_reference_mm"  : daily.Variables(5).ValuesAsNumpy(),
        "pluie_totale_mm"   : daily.Variables(6).ValuesAsNumpy(),
    })

    print(f"  API OK : {len(df_h)} heures | {len(df_q)} jours")
    return df_h, df_q


# ── Lecture capteur Arduino ───────────────────────────────────────────────

def lire_capteur_arduino(port: str = "COM3") -> Optional[float]:
    """
    Lit l'humidite du sol depuis l'Arduino.
    Format attendu : "HUMIDITE:45.3"
    Retourne None si le capteur n'est pas connecte.
    """
    try:
        import serial
        import time

        ser = serial.Serial(port, baudrate=9600, timeout=3)
        time.sleep(2.5)

        for _ in range(5):
            ligne = ser.readline().decode("utf-8", errors="ignore").strip()
            if "HUMIDITE:" in ligne:
                valeur = float(ligne.split(":")[1])
                ser.close()
                print(f"  Capteur sol : {valeur:.1f}%")
                return valeur

        ser.close()
        return None

    except ImportError:
        print("  pyserial absent -> pip install pyserial")
        return None
    except Exception as e:
        msg = str(e).lower()
        if "could not open port" in msg or "no such file" in msg:
            print(f"  Arduino non connecte sur {port} -> utilisation Open-Meteo")
        else:
            print(f"  Erreur capteur ({e}) -> utilisation Open-Meteo")
        return None


# ── Construction des features ─────────────────────────────────────────────

def _get_saison(mois: int) -> str:
    if mois in [11, 12, 1, 2, 3]:
        return "seche"
    elif mois in [6, 7, 8, 9]:
        return "grande_pluie"
    return "petite_pluie"


def _nan_safe(valeur: float, defaut: float) -> float:
    """Retourne defaut si valeur est NaN ou infini."""
    if valeur is None:
        return defaut
    try:
        if math.isnan(valeur) or math.isinf(valeur):
            return defaut
    except (TypeError, ValueError):
        return defaut
    return valeur


def construire_features(
    df_h             : pd.DataFrame,
    df_q             : pd.DataFrame,
    index_jour       : int,
    humidite_capteur : Optional[float] = None,
    jour_cycle       : int = 60,
) -> dict:
    row_q  = df_q.iloc[index_jour]
    date_j = row_q["date"]
    d_dt   = pd.to_datetime(str(date_j))
    df_jour = df_h[df_h["date_only"] == date_j]

    # Valeur de repli si Open-Meteo ne fournit pas l'humidite sol
    # (frequent sur les jours J+1 a J+3 en prevision)
    HS_DEFAUT = 40.0

    if len(df_jour) == 0:
        hs_raw       = df_h["humidite_sol_7_28cm_pct"].dropna()
        hs_moy       = float(hs_raw.mean()) if len(hs_raw) > 0 else HS_DEFAUT
        hs_min       = hs_moy - 3.0
        hs_07        = hs_moy - 1.5
        ha_moy       = float(df_h["humidite_air_pct"].mean())
        ha_max       = min(100.0, ha_moy + 15.0)
        ha_min       = max(0.0,   ha_moy - 15.0)
        vent_moy_kmh = float(df_h["vent_10m_kmh"].mean())
    else:
        hs_raw       = df_jour["humidite_sol_7_28cm_pct"].dropna()
        hs_moy       = float(hs_raw.mean()) if len(hs_raw) > 0 else HS_DEFAUT
        hs_min       = float(hs_raw.min())  if len(hs_raw) > 0 else HS_DEFAUT - 5.0
        hs_07_raw    = df_jour["humidite_sol_0_7cm_pct"].dropna()
        hs_07        = float(hs_07_raw.mean()) if len(hs_07_raw) > 0 else max(10.0, hs_moy - 2.0)
        ha_moy       = float(df_jour["humidite_air_pct"].mean())
        ha_max       = float(df_jour["humidite_air_pct"].max())
        ha_min       = float(df_jour["humidite_air_pct"].min())
        vent_moy_kmh = float(df_jour["vent_10m_kmh"].mean())

    # Protection finale contre les NaN residuels
    hs_moy = max(5.0, min(100.0, _nan_safe(hs_moy, HS_DEFAUT)))
    hs_min = max(5.0, min(100.0, _nan_safe(hs_min, HS_DEFAUT - 5.0)))
    hs_07  = max(5.0, min(100.0, _nan_safe(hs_07,  HS_DEFAUT - 2.0)))
    ha_moy = _nan_safe(ha_moy, 60.0)
    ha_max = _nan_safe(ha_max, 80.0)
    ha_min = _nan_safe(ha_min, 40.0)
    vent_moy_kmh = _nan_safe(vent_moy_kmh, 7.2)

    if humidite_capteur is not None and index_jour == 0:
        hs_moy = humidite_capteur
        hs_min = max(10.0, humidite_capteur - 3.0)
        hs_07  = max(10.0, humidite_capteur - 1.5)

    vent_u2_ms = round(vent_moy_kmh / 3.6 * 0.748, 3)
    ET0        = float(row_q["ET0_reference_mm"])
    saison     = _get_saison(int(d_dt.month))
    kc         = kc_tomate(jour_cycle, saison, ha_min, vent_u2_ms)
    ETc        = round(ET0 * kc, 2)
    pluie      = float(row_q["pluie_totale_mm"])
    pluie_e    = round(pluie * PLUIE_EFFECTIVE_PCT, 2)
    deficit    = round(max(ETc - pluie_e, 0.0), 2)
    J          = int(d_dt.dayofyear)

    return {
        "humidite_sol_moy_pct" : round(hs_moy, 1),
        "humidite_sol_min_pct" : round(hs_min, 1),
        "humidite_sol_0_7_moy" : round(hs_07,  1),
        "temp_max_C"           : float(row_q["temp_max_C"]),
        "temp_min_C"           : float(row_q["temp_min_C"]),
        "temp_moy_C"           : float(row_q["temp_moy_C"]),
        "humidite_air_moy_pct" : round(ha_moy, 1),
        "vent_u2_ms"           : vent_u2_ms,
        "rayonnement_Rs_MJ"    : float(row_q["rayonnement_Rs_MJ"]),
        "ET0_reference_mm"     : ET0,
        "ETc_mm"               : ETc,
        "pluie_totale_mm"      : pluie,
        "pluie_effective_mm"   : pluie_e,
        "deficit_hydrique_mm"  : deficit,
        "jour_annee"           : J,
        "mois"                 : int(d_dt.month),
        "_date"                : str(date_j),
        "_ET0"                 : ET0,
        "_pluie"               : pluie,
        "_deficit"             : deficit,
        "_humidite_sol"        : round(hs_moy, 1),
        "_source_sol"          : ("Capteur" if (humidite_capteur and index_jour == 0) else "Open-Meteo"),
        "_capteur"             : (humidite_capteur is not None and index_jour == 0),
        "_kc"                  : kc,
        "_stade"               : get_stade(jour_cycle),
        "saison"               : saison,
        "jour_cycle"           : jour_cycle,
        "RH_max"               : round(ha_max, 1),
        "RH_min"               : round(ha_min, 1),
    }


# ── Moteur de decision ────────────────────────────────────────────────────

def appliquer_regles(f: dict) -> tuple:
    hs, p, d = f["humidite_sol_moy_pct"], f["pluie_totale_mm"], f["deficit_hydrique_mm"]
    if hs > SOL_HUMIDE_SEUIL:
        return 0, f"Sol humide ({hs:.1f}%)", True
    if p > PLUIE_FORTE_SEUIL:
        return 0, f"Forte pluie ({p:.1f}mm)", True
    if p > PLUIE_MODERE_SEUIL and hs > SOL_MOYEN_SEUIL:
        return 0, f"Pluie moderee + sol ok ({hs:.1f}%)", True
    if d <= 0.0:
        return 0, f"Pas de deficit ({d:.2f}mm)", True
    return 1, "Analyse ML requise", False


def decider(clf, reg, f: dict) -> dict:
    dec_r, raison, court = appliquer_regles(f)
    if court:
        return {"irriguer": dec_r, "volume_L": 0.0, "raison": raison,
                "source": "Regle agronomique", "confiance": 100.0}

    X         = pd.DataFrame([f])[FEATURES]
    proba     = clf.predict_proba(X)[0]
    decision  = int(clf.predict(X)[0])
    confiance = round(proba[decision] * 100, 1)

    volume = 0.0
    if decision == 1:
        vol_ml  = float(reg.predict(X)[0])
        facteur = max(0.0, (65.0 - f["humidite_sol_moy_pct"]) / 25.0)
        vol_fo  = f["deficit_hydrique_mm"] * facteur * SURFACE_M2 / EFFICACITE
        volume  = round(max(0.60 * vol_ml + 0.40 * vol_fo, 0.0), 1)

    return {
        "irriguer"  : decision,
        "volume_L"  : volume,
        "raison"    : f"ML {confiance}% | Deficit {f['deficit_hydrique_mm']:.2f}mm | Sol {f['humidite_sol_moy_pct']:.1f}% | Kc {f.get('_kc', '?')}",
        "source"    : "Modele ML (RF 2022-2024)",
        "confiance" : confiance,
    }


# ── Affichage console ─────────────────────────────────────────────────────

def afficher(label: str, f: dict, res: dict):
    dec_txt = "ARROSER        " if res["irriguer"] else "NE PAS ARROSER "
    print(f"\n  {'-'*60}")
    print(f"  {label}  ({f['_date']}) | Sol : {f['_source_sol']}")
    print(f"  Temp : {f['temp_max_C']:.1f}/{f['temp_min_C']:.1f}C | "
          f"Pluie : {f['_pluie']:.1f}mm | Sol : {f['_humidite_sol']:.1f}%")
    print(f"  ET0 : {f['ET0_reference_mm']:.2f}mm | ETc : {f['ETc_mm']:.2f}mm | "
          f"Deficit : {f['_deficit']:.2f}mm | Kc : {f.get('_kc', '?')}")
    print(f"  Stade : {f.get('_stade', '?')}")
    print(f"  DECISION : {dec_txt}")
    if res["irriguer"]:
        print(f"  Volume  : {res['volume_L']:.0f}L (parcelle {SURFACE_M2}m2)")
    print(f"  Source  : {res['source']}")
    raison = res["raison"]
    if len(raison) > 55:
        print(f"  Raison  : {raison[:55]}")
        print(f"            {raison[55:]}")
    else:
        print(f"  Raison  : {raison}")
    print(f"  {'-'*60}")


# ── SMS et log ────────────────────────────────────────────────────────────

def generer_sms(resultats: list) -> str:
    auj = resultats[0]
    f0  = auj["features"]
    r0  = auj["res"]

    dec = "ARROSER" if r0["irriguer"] else "PAS D'ARROSAGE"
    vol = f"{r0['volume_L']:.0f}L" if r0["irriguer"] else "0L"

    sms  = f"Water5 CI - {f0['_date']}\n"
    sms += f"Stade    : {f0.get('_stade', '?').upper()}\n"
    sms += f"Kc       : {f0.get('_kc', '?')}\n"
    sms += f"Decision : {dec}\n"
    sms += f"Volume   : {vol}\n"
    sms += f"Sol      : {f0['_humidite_sol']:.1f}% ({f0['_source_sol']})\n"
    sms += f"Pluie    : {f0['_pluie']:.1f}mm | ET0 : {f0['_ET0']:.2f}mm\n\n"
    sms += "Previsions :\n"
    for r in resultats[1:]:
        fi = r["features"]; ri = r["res"]
        e  = "OUI" if ri["irriguer"] else "NON"
        v  = f"{ri['volume_L']:.0f}L" if ri["irriguer"] else "0L"
        sms += f"  {fi['_date']} : {e} {v}\n"
    sms += f"\nWater5 IA 2022-2024 | {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    return sms


def sauvegarder_log(resultats: list):
    COLONNES_LOG = [
        "timestamp", "date", "irriguer", "volume_L", "source",
        "humidite_sol", "source_sol", "stade", "kc",
        "ET0_mm", "pluie_mm", "deficit_mm", "confiance_pct",
    ]
    rows = []
    for r in resultats:
        f = r["features"]; res = r["res"]
        rows.append({
            "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M"),
            "date"         : f["_date"],
            "irriguer"     : res["irriguer"],
            "volume_L"     : res["volume_L"],
            "source"       : res["source"],
            "humidite_sol" : f["_humidite_sol"],
            "source_sol"   : f["_source_sol"],
            "stade"        : f.get("_stade", "?"),
            "kc"           : f.get("_kc", "?"),
            "ET0_mm"       : f["_ET0"],
            "pluie_mm"     : f["_pluie"],
            "deficit_mm"   : f["_deficit"],
            "confiance_pct": res["confiance"],
        })

    lp     = os.path.join(OUT_DIR, "historique_decisions.csv")
    existe = os.path.exists(lp)

    if existe:
        try:
            df_existant = pd.read_csv(lp, nrows=0)
            if list(df_existant.columns) != COLONNES_LOG:
                import shutil
                shutil.move(lp, lp.replace(".csv", "_archive.csv"))
                existe = False
        except Exception:
            existe = False

    pd.DataFrame(rows)[COLONNES_LOG].to_csv(lp, mode="a", header=not existe, index=False)
    print(f"\n  {len(rows)} decisions enregistrees -> historique_decisions.csv")


# ── Point d'entree ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water5 - Irrigation intelligente CI")
    parser.add_argument("--test",       action="store_true", help="Affiche donnees brutes")
    parser.add_argument("--port",       default="COM3",      help="Port serie Arduino")
    parser.add_argument("--no-capteur", action="store_true", help="Desactive l'Arduino")
    parser.add_argument("--jour-cycle", type=int, default=60, help="Jours depuis plantation")
    args = parser.parse_args()

    print("=" * 60)
    print("  WATER5 - IRRIGATION INTELLIGENTE - M. KOFFI")
    print(f"  Yamoussoukro, CI | Tomate | {SURFACE_M2}m2")
    print(f"  {datetime.now().strftime('%A %d/%m/%Y  %H:%M')}")
    print(f"  Modeles entraines sur 2022-2024 (1096 jours)")
    print("=" * 60)

    df_h, df_q = appeler_api()

    if args.test:
        print("\n  DONNEES QUOTIDIENNES :")
        cols_q = ["date", "temp_max_C", "temp_min_C", "ET0_reference_mm", "pluie_totale_mm"]
        print(df_q[cols_q].to_string(index=False))
        print("\nTest API reussi.")
        sys.exit(0)

    humidite_capteur = None
    if not args.no_capteur:
        humidite_capteur = lire_capteur_arduino(args.port)
    if humidite_capteur is None:
        print("  Humidite sol depuis Open-Meteo (pas de capteur)")

    if not os.path.exists(CLF_PATH):
        print("Modeles introuvables -> lancez : 02_entrainement_ml.py")
        sys.exit(1)
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    print("  Modeles ML charges")

    labels    = ["AUJOURD'HUI (J)  ", "DEMAIN      (J+1)", "APRES-DEMAIN(J+2)", "DANS 3 JOURS(J+3)"]
    resultats = []

    for i in range(min(4, len(df_q))):
        jc  = args.jour_cycle + i
        f   = construire_features(df_h, df_q, i, humidite_capteur, jour_cycle=jc)
        res = decider(clf, reg, f)
        afficher(labels[i], f, res)
        resultats.append({"features": f, "res": res})

    sms = generer_sms(resultats)
    print("\n  SMS - M. KOFFI :")
    print("  " + "-" * 48)
    for ligne in sms.split("\n"):
        print(f"  {ligne}")
    print("  " + "-" * 48)

    sauvegarder_log(resultats)
    print("=" * 60)