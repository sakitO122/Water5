"""
=============================================================
 SYSTÈME D'IRRIGATION INTELLIGENTE — CÔTE D'IVOIRE
 06_api_openmeteo.py
=============================================================
 Prédiction du jour + 3 jours via données météo réelles.
 Intégration capteur DHT22/Arduino (optionnelle).

 Installation :
   pip install openmeteo-requests requests-cache retry-requests pyserial

 Usage :
   python src/06_api_openmeteo.py                    ← décision du jour
   python src/06_api_openmeteo.py --test             ← données brutes
   python src/06_api_openmeteo.py --port COM4        ← avec Arduino
   python src/06_api_openmeteo.py --port /dev/ttyUSB0

 Corrections apportées :
   ✔ Type hint Optional[float] (compatible Python 3.8+)
   ✔ Vent : variable MOYENNE horaire agrégée (pas le max quotidien)
   ✔ RH_max et RH_min extraits pour ea FAO-56 éq.17
   ✔ Fallback J sécurisé dans construire_features()
   ✔ Nom de fonction unifié : construire_features()
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

import os, sys, math, joblib, argparse
from typing import Optional
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from config import (
    CLF_PATH, REG_PATH, FEATURES, OUT_DIR,
    LATITUDE_DEG, LONGITUDE_DEG, TIMEZONE,
    SURFACE_M2, EFFICACITE, PLUIE_EFFECTIVE_PCT,
    SOL_HUMIDE_SEUIL, SOL_MOYEN_SEUIL,
    PLUIE_FORTE_SEUIL, PLUIE_MODERE_SEUIL,
)
from agronomie import penman_monteith_fao56, kc_tomate, get_stade


# ══════════════════════════════════════════════════════════════
# 1. APPEL API OPEN-METEO
# ══════════════════════════════════════════════════════════════

def appeler_api() -> tuple:
    """
    Récupère les données météo via openmeteo-requests (cache 1h).
    Retourne (df_horaire, df_quotidien).
    """
    try:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry
    except ImportError:
        print("❌ Bibliothèques manquantes :")
        print("   pip install openmeteo-requests requests-cache retry-requests")
        sys.exit(1)

    print("🌐 Connexion à l'API Open-Meteo...")

    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo     = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude"      : LATITUDE_DEG,
        "longitude"     : LONGITUDE_DEG,
        "timezone"      : TIMEZONE,
        "forecast_days" : 4,
        "hourly": [
            "temperature_2m",
            "rain",
            "wind_speed_10m",              # agrégé en MOYENNE (correction vent)
            "soil_temperature_0_to_7cm",
            "soil_moisture_7_to_28cm",
            "relative_humidity_2m",        # max et min extraits par agrégation
            "soil_moisture_0_to_7cm",
        ],
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "wind_speed_10m_max",          # conservé pour info, non utilisé pour ET₀
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
            "precipitation_sum",
        ],
    }

    responses = openmeteo.weather_api(
        "https://api.open-meteo.com/v1/forecast", params=params
    )
    response = responses[0]

    print(f"   ✅ Station   : {response.Latitude():.4f}°N "
          f"{abs(response.Longitude()):.4f}°W")
    print(f"   ✅ Altitude  : {response.Elevation():.0f}m | "
          f"Timezone : {response.Timezone()}")

    # ── Données horaires ──────────────────────────────────
    hourly   = response.Hourly()
    dates_h  = pd.date_range(
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

    # Conversions
    df_h["humidite_sol_7_28cm_pct"] = (df_h["humidite_sol_7_28cm_m3"] * 100).round(1)
    df_h["humidite_sol_0_7cm_pct"]  = (df_h["humidite_sol_0_7cm_m3"]  * 100).round(1)
    df_h["date_only"] = pd.to_datetime(df_h["datetime"]).dt.date

    # ── Données quotidiennes ──────────────────────────────
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

    print(f"   ✅ Données   : {len(df_h)} heures | {len(df_q)} jours")
    return df_h, df_q


# ══════════════════════════════════════════════════════════════
# 2. LECTURE CAPTEUR DHT22 / ARDUINO
# ══════════════════════════════════════════════════════════════

def lire_capteur_arduino(port: str = "COM3") -> Optional[float]:
    """
    Lit l'humidité du sol depuis le capteur DHT22 via Arduino (Serial).
    L'Arduino doit envoyer le format : "HUMIDITE:45.3\\n"

    CORRECTION : type hint Optional[float] au lieu de float | None
    (compatible Python 3.8 et 3.9).

    Returns None si le capteur n'est pas connecté.
    """
    try:
        import serial
        import time

        print(f"   🔌 Connexion Arduino sur {port}...")
        ser = serial.Serial(port, baudrate=9600, timeout=3)
        time.sleep(2.5)

        for _ in range(5):
            ligne = ser.readline().decode("utf-8", errors="ignore").strip()
            if "HUMIDITE:" in ligne:
                valeur = float(ligne.split(":")[1])
                ser.close()
                print(f"   ✅ Capteur DHT22 : humidité sol = {valeur:.1f}%")
                return valeur

        ser.close()
        print("   ⚠  Capteur connecté mais pas de données valides")
        return None

    except ImportError:
        print("   ℹ  pyserial absent → pip install pyserial")
        return None
    except Exception as e:
        msg = str(e).lower()
        if "could not open port" in msg or "no such file" in msg:
            print(f"   ℹ  Arduino non connecté sur {port} → utilisation Open-Meteo")
        else:
            print(f"   ⚠  Erreur capteur ({e}) → utilisation Open-Meteo")
        return None


# ══════════════════════════════════════════════════════════════
# 3. CONSTRUCTION DES FEATURES PAR JOUR
#    Nom unifié : construire_features() (correction import prediction.py)
# ══════════════════════════════════════════════════════════════

def construire_features(
    df_h             : pd.DataFrame,
    df_q             : pd.DataFrame,
    index_jour       : int,
    humidite_capteur : Optional[float] = None,
    jour_cycle       : int = 60,
) -> dict:
    """
    Construit le vecteur de features pour le jour à l'index donné.

    CORRECTIONS :
      - Vent : MOYENNE horaire agrégée (pas vent_max_kmh quotidien)
      - RH_max et RH_min extraits depuis les données horaires
      - J (jour julien) calculé proprement
      - Fallback sécurisé si données horaires manquantes

    Parameters
    ----------
    df_h             : DataFrame horaire (depuis appeler_api)
    df_q             : DataFrame quotidien (depuis appeler_api)
    index_jour       : 0=aujourd'hui, 1=J+1, 2=J+2, 3=J+3
    humidite_capteur : Lecture DHT22 (% sol), None si non disponible
    jour_cycle       : Jours depuis la plantation (pour Kc dynamique)
    """
    row_q  = df_q.iloc[index_jour]
    date_j = row_q["date"]
    d_dt   = pd.to_datetime(str(date_j))

    # ── Données horaires du jour ──────────────────────────
    df_jour = df_h[df_h["date_only"] == date_j]

    if len(df_jour) == 0:
        # Fallback : moyennes globales
        hs_moy = float(df_h["humidite_sol_7_28cm_pct"].mean())
        hs_min = hs_moy - 3.0
        hs_07  = hs_moy - 1.5
        ha_moy = float(df_h["humidite_air_pct"].mean())
        ha_max = min(100.0, ha_moy + 15.0)
        ha_min = max(0.0,   ha_moy - 15.0)
        # ✅ CORRECTION : vent moyen (pas max)
        vent_moy_kmh = float(df_h["vent_10m_kmh"].mean())
    else:
        hs_moy = float(df_jour["humidite_sol_7_28cm_pct"].mean())
        hs_min = float(df_jour["humidite_sol_7_28cm_pct"].min())
        hs_07  = float(df_jour["humidite_sol_0_7cm_pct"].mean())
        ha_moy = float(df_jour["humidite_air_pct"].mean())
        ha_max = float(df_jour["humidite_air_pct"].max())
        ha_min = float(df_jour["humidite_air_pct"].min())
        # ✅ CORRECTION : vent MOYEN horaire → u2 moyen journalier
        vent_moy_kmh = float(df_jour["vent_10m_kmh"].mean())

    # Remplacement par capteur DHT22 si disponible (J=0 uniquement)
    if humidite_capteur is not None and index_jour == 0:
        hs_moy = humidite_capteur
        hs_min = max(10.0, humidite_capteur - 3.0)
        hs_07  = max(10.0, humidite_capteur - 1.5)

    # ── Conversion vent 10m → 2m (FAO-56 éq. 47) ─────────
    # u2 = u10 × (4.87 / ln(67.8×10 − 5.42)) ≈ u10 × 0.748
    # ✅ CORRECTION : depuis la MOYENNE (pas le max)
    vent_u2_ms = round(vent_moy_kmh / 3.6 * 0.748, 3)

    # ── Calculs agronomiques ──────────────────────────────
    ET0     = float(row_q["ET0_reference_mm"])
    saison  = _get_saison(int(d_dt.month))
    kc      = kc_tomate(jour_cycle, saison, ha_min, vent_u2_ms)
    ETc     = round(ET0 * kc, 2)
    pluie   = float(row_q["pluie_totale_mm"])
    pluie_e = round(pluie * PLUIE_EFFECTIVE_PCT, 2)
    deficit = round(ETc - pluie_e, 2)

    J       = int(d_dt.dayofyear)

    return {
        # ── Features ML ────────────────────────────────
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
        # ── Contexte (non utilisé par ML) ──────────────
        "_date"         : str(date_j),
        "_ET0"          : ET0,
        "_pluie"        : pluie,
        "_deficit"      : deficit,
        "_humidite_sol" : round(hs_moy, 1),
        "_source_sol"   : ("DHT22" if (humidite_capteur and index_jour == 0)
                           else "Open-Meteo"),
        "_capteur"      : (humidite_capteur is not None and index_jour == 0),
        "_kc"           : kc,
        "_stade"        : get_stade(jour_cycle),
        "saison"        : saison,
        "jour_cycle"    : jour_cycle,
        "RH_max"        : round(ha_max, 1),
        "RH_min"        : round(ha_min, 1),
    }


def _get_saison(mois: int) -> str:
    if mois in [11, 12, 1, 2, 3]:
        return "seche"
    elif mois in [6, 7, 8, 9]:
        return "grande_pluie"
    return "petite_pluie"


# ══════════════════════════════════════════════════════════════
# 4. MOTEUR DE DÉCISION
# ══════════════════════════════════════════════════════════════

def appliquer_regles(f: dict) -> tuple:
    hs, p, d = (f["humidite_sol_moy_pct"],
                f["pluie_totale_mm"],
                f["deficit_hydrique_mm"])
    if hs > SOL_HUMIDE_SEUIL:
        return 0, f"Sol humide ({hs:.1f}%)", True
    if p > PLUIE_FORTE_SEUIL:
        return 0, f"Forte pluie ({p:.1f}mm)", True
    if p > PLUIE_MODERE_SEUIL and hs > SOL_MOYEN_SEUIL:
        return 0, f"Pluie modérée + sol ok ({hs:.1f}%)", True
    if d <= 0.0:
        return 0, f"Pas de déficit ({d:.2f}mm)", True
    return 1, "Analyse ML requise", False


def decider(clf, reg, f: dict) -> dict:
    dec_r, raison, court = appliquer_regles(f)
    if court:
        return {
            "irriguer"  : dec_r, "volume_L": 0.0,
            "raison"    : raison, "source": "Règle agronomique",
            "confiance" : 100.0,
        }

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
        "irriguer"  : decision, "volume_L": volume,
        "raison"    : (f"ML {confiance}% — "
                       f"Déficit {f['deficit_hydrique_mm']:.2f}mm | "
                       f"Sol {f['humidite_sol_moy_pct']:.1f}% | "
                       f"Kc {f.get('_kc', '—')}"),
        "source"    : "Modèle ML (RF 2022-2024)",
        "confiance" : confiance,
    }


# ══════════════════════════════════════════════════════════════
# 5. AFFICHAGE CONSOLE
# ══════════════════════════════════════════════════════════════

def afficher(label: str, f: dict, res: dict):
    icone      = "✅ ARROSER       " if res["irriguer"] else "❌ NE PAS ARROSER"
    capteur_txt = "🔌 DHT22" if f["_capteur"] else "🌐 Open-Meteo"
    stade_txt  = f.get("_stade", "—")
    kc_txt     = f"{f.get('_kc', 0):.4f}"

    print(f"\n  ┌{'─'*64}┐")
    print(f"  │  {label:<60}  │")
    print(f"  │  📅 {f['_date']}  |  Sol : {capteur_txt:<42}│")
    print(f"  ├{'─'*64}┤")
    print(f"  │  📊 Météo :{' ':>53}│")
    print(f"  │     🌡  Temp  : {f['temp_max_C']:.1f}°C max / "
          f"{f['temp_min_C']:.1f}°C min / {f['temp_moy_C']:.1f}°C moy{'':>7}│")
    print(f"  │     🌧  Pluie : {f['_pluie']:.1f}mm{'':>47}│")
    print(f"  │     💧  Sol   : {f['_humidite_sol']:.1f}% ({f['_source_sol']}){'':>36}│")
    print(f"  │     ☁  HR    : {f['humidite_air_moy_pct']:.1f}% "
          f"(max={f.get('RH_max', '—')}, min={f.get('RH_min', '—')}){'':>23}│")
    print(f"  │     💨  Vent  : {f['vent_u2_ms']:.2f}m/s (u2 moy. journalier){'':>28}│")
    print(f"  │     ☀  Rs    : {f['rayonnement_Rs_MJ']:.2f}MJ/m²{'':>41}│")
    print(f"  ├{'─'*64}┤")
    print(f"  │  🌿 Agronomie (Kc dynamique) :{' ':>34}│")
    print(f"  │     Stade   : {stade_txt:<49}│")
    print(f"  │     Kc      : {kc_txt:<49}│")
    print(f"  │     ET₀     : {f['ET0_reference_mm']:.2f}mm/j  |  "
          f"ETc : {f['ETc_mm']:.2f}mm/j{'':>24}│")
    print(f"  │     Déficit : {f['_deficit']:.2f}mm  |  "
          f"Pluie eff. : {f['pluie_effective_mm']:.2f}mm{'':>22}│")
    print(f"  ├{'─'*64}┤")
    print(f"  │  🤖 DÉCISION : {icone:<49}│")
    if res["irriguer"]:
        print(f"  │     Volume   : {res['volume_L']:.0f}L (parcelle {SURFACE_M2}m²){'':>35}│")
    print(f"  │     Source   : {res['source']:<49}│")
    raison = res["raison"]
    print(f"  │     Raison   : {raison[:49]:<49}│")
    if len(raison) > 49:
        print(f"  │               {raison[49:98]:<49}│")
    print(f"  └{'─'*64}┘")


def afficher_donnees_brutes(df_h: pd.DataFrame, df_q: pd.DataFrame):
    print("\n" + "═" * 72)
    print("  📊 DONNÉES BRUTES OPEN-METEO (4 jours)")
    print("═" * 72)
    print("\n  QUOTIDIENNES :")
    cols_q = ["date", "temp_max_C", "temp_min_C", "temp_moy_C",
              "ET0_reference_mm", "pluie_totale_mm", "rayonnement_Rs_MJ"]
    print(df_q[cols_q].to_string(index=False))
    print("\n  HORAIRES — 8 premières heures :")
    cols_h = ["datetime", "temp_C", "humidite_air_pct",
              "pluie_mm", "humidite_sol_7_28cm_pct"]
    print(df_h[cols_h].head(8).to_string(index=False))
    print("═" * 72)


# ══════════════════════════════════════════════════════════════
# 6. GÉNÉRATION SMS + LOG
# ══════════════════════════════════════════════════════════════

def generer_sms(resultats: list) -> str:
    auj = resultats[0]
    f0  = auj["features"]
    r0  = auj["res"]

    dec = "ARROSER ✅" if r0["irriguer"] else "PAS D'ARROSAGE ❌"
    vol = f"{r0['volume_L']:.0f}L" if r0["irriguer"] else "0L"

    sms  = f"🌱 AgroIrri CI — {f0['_date']}\n"
    sms += f"Stade    : {f0.get('_stade', '—').upper()}\n"
    sms += f"Kc       : {f0.get('_kc', '—')}\n"
    sms += f"Décision : {dec}\n"
    sms += f"Volume   : {vol}\n"
    sms += f"Sol      : {f0['_humidite_sol']:.1f}% ({f0['_source_sol']})\n"
    sms += f"Pluie    : {f0['_pluie']:.1f}mm | ET₀ : {f0['_ET0']:.2f}mm\n\n"
    sms += "Prévisions :\n"
    for r in resultats[1:]:
        fi = r["features"]; ri = r["res"]
        e  = "✅" if ri["irriguer"] else "❌"
        v  = f"{ri['volume_L']:.0f}L" if ri["irriguer"] else "0L"
        sms += f"  {fi['_date']} : {e} {v}\n"
    sms += f"\nIA 2022-2024 | Open-Meteo | {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    return sms


def sauvegarder_log(resultats: list):
    """
    Enregistre les décisions dans l'historique CSV.
    CORRECTION : schéma de colonnes fixe et détection de corruption
    pour éviter pandas.errors.ParserError lors d'appends successifs.
    """
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
            "stade"        : f.get("_stade", "—"),
            "kc"           : f.get("_kc", "—"),
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
                print("   ⚠  Ancien historique archivé (schéma incompatible)")
        except Exception:
            existe = False

    df_out = pd.DataFrame(rows)[COLONNES_LOG]   # ordre garanti
    df_out.to_csv(lp, mode="a", header=not existe, index=False)
    print(f"\n  💾 {len(rows)} décisions enregistrées → historique_decisions.csv")


# ══════════════════════════════════════════════════════════════
# EXÉCUTION PRINCIPALE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Irrigation intelligente CI")
    parser.add_argument("--test",      action="store_true",
                        help="Affiche données brutes sans décision ML")
    parser.add_argument("--port",      default="COM3",
                        help="Port série Arduino")
    parser.add_argument("--no-capteur", action="store_true",
                        help="Désactive la lecture du capteur Arduino")
    parser.add_argument("--jour-cycle", type=int, default=60,
                        help="Jours depuis la plantation (défaut: 60)")
    args = parser.parse_args()

    print("\n" + "═" * 68)
    print("  🌱 SYSTÈME D'IRRIGATION INTELLIGENTE — M. KOFFI")
    print(f"  📍 Yamoussoukro, CI  |  🍅 Tomate  |  {SURFACE_M2}m²")
    print(f"  🕐 {datetime.now().strftime('%A %d/%m/%Y  %H:%M')}")
    print(f"  🧠 Modèles entraînés sur 2022-2024 (1096 jours)")
    print("=" * 68)

    df_h, df_q = appeler_api()

    if args.test:
        afficher_donnees_brutes(df_h, df_q)
        print("\n  ✅ Test API réussi. Lancez sans --test pour les décisions ML.")
        sys.exit(0)

    humidite_capteur = None
    if not args.no_capteur:
        humidite_capteur = lire_capteur_arduino(args.port)
    if humidite_capteur is None:
        print("   ℹ  Humidité sol depuis Open-Meteo (pas de capteur)")

    if not os.path.exists(CLF_PATH):
        print("\n❌ Modèles introuvables → lancez : 02_entrainement_ml.py")
        sys.exit(1)
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    print("   ✅ Modèles ML chargés")

    labels = [
        "AUJOURD'HUI  (J)  ",
        "DEMAIN       (J+1)",
        "APRÈS-DEMAIN (J+2)",
        "DANS 3 JOURS (J+3)",
    ]
    resultats = []

    print("\n" + "═" * 68)
    for i in range(min(4, len(df_q))):
        jc  = args.jour_cycle + i
        f   = construire_features(df_h, df_q, i, humidite_capteur, jour_cycle=jc)
        res = decider(clf, reg, f)
        afficher(labels[i], f, res)
        resultats.append({"features": f, "res": res})

    sms = generer_sms(resultats)
    print("\n" + "═" * 68)
    print("  📱 MESSAGE SMS — M. KOFFI :")
    print("  " + "─" * 54)
    for ligne in sms.split("\n"):
        print(f"  {ligne}")
    print("  " + "─" * 54)

    sauvegarder_log(resultats)
    print("=" * 68)