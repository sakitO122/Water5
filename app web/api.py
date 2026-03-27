"""
api.py v3.0
Serveur FastAPI Water5.
Open-Meteo appele cote serveur -> memes resultats que 06_api_openmeteo.py.

Lancement :
  cd C:\\irrigation_ml\\src
  python api.py

Endpoints :
  GET  /analyser   -> decision complete J+3 (identique a 06_api_openmeteo.py)
  POST /decision   -> decision depuis donnees fournies par le client
  GET  /health     -> verification modeles
  GET  /           -> app web
"""

import os
import sys
import math
from datetime import datetime, date
from typing import Optional, List

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR, os.path.join(_PARENT_DIR, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import (
    CLF_PATH, REG_PATH, FEATURES,
    LATITUDE_DEG, LONGITUDE_DEG, TIMEZONE,
    SURFACE_M2, EFFICACITE, PLUIE_EFFECTIVE_PCT,
    SOL_HUMIDE_SEUIL, SOL_MOYEN_SEUIL,
    PLUIE_FORTE_SEUIL, PLUIE_MODERE_SEUIL,
)
from agronomie import kc_tomate, get_stade, get_saison


# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Water5 API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

clf = None
reg = None

@app.on_event("startup")
def charger_modeles():
    global clf, reg
    if not os.path.exists(CLF_PATH):
        print(f"[ERREUR] Modele introuvable : {CLF_PATH}")
        return
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    print(f"[OK] Modeles ML charges")


# ── Dossier app web ───────────────────────────────────────────────────────

def _trouver_web_dir():
    for c in [
        os.path.join(_PARENT_DIR, "app web"),
        os.path.join(_PARENT_DIR, "app_web"),
        _PARENT_DIR, _THIS_DIR, os.getcwd(),
    ]:
        if os.path.isfile(os.path.join(c, "index.html")):
            return c
    return _PARENT_DIR

_WEB_DIR = _trouver_web_dir()


# ── Schemas ───────────────────────────────────────────────────────────────

class JourDecision(BaseModel):
    date          : str
    irriguer      : bool
    volume_L      : float
    confiance_pct : float
    source        : str
    raison        : str
    stade         : str
    kc            : float
    ETc_mm        : float
    deficit_mm    : float
    ET0_mm        : float
    pluie_mm      : float
    humidite_sol  : float
    temp_max_C    : float
    temp_min_C    : float
    vent_u2_ms    : float

class AnalyseComplete(BaseModel):
    timestamp    : str
    source_meteo : str
    aujourd_hui  : JourDecision
    previsions   : List[JourDecision]

class MeteoInput(BaseModel):
    temp_max_C           : float
    temp_min_C           : float
    temp_moy_C           : float
    pluie_totale_mm      : float = 0.0
    vent_u2_ms           : float = 2.0
    rayonnement_Rs_MJ    : float = 15.0
    ET0_reference_mm     : float
    humidite_sol_moy_pct : float
    humidite_sol_min_pct : Optional[float] = None
    humidite_sol_0_7_moy : Optional[float] = None
    humidite_air_moy_pct : float = 60.0
    RH_max               : float = 80.0
    RH_min               : float = 40.0
    jour_annee           : Optional[int] = None
    mois                 : Optional[int] = None
    jour_cycle           : int = 60
    source_sol           : str = "Open-Meteo"

class DecisionOutput(BaseModel):
    irriguer      : bool
    volume_L      : float
    confiance_pct : float
    source        : str
    raison        : str
    stade         : str
    kc            : float
    ETc_mm        : float
    deficit_mm    : float
    ET0_mm        : float
    timestamp     : str


# ── Utilitaires ───────────────────────────────────────────────────────────

def _nan_safe(v, defaut):
    if v is None: return defaut
    try:
        if math.isnan(v) or math.isinf(v): return defaut
    except Exception:
        return defaut
    return v

def _get_saison(mois):
    if mois in [11,12,1,2,3]: return "seche"
    if mois in [6,7,8,9]:     return "grande_pluie"
    return "petite_pluie"

def _appliquer_regles(hs, pluie, deficit):
    if hs > SOL_HUMIDE_SEUIL:
        return 0, f"Sol humide ({hs:.1f}%)", True
    if pluie > PLUIE_FORTE_SEUIL:
        return 0, f"Forte pluie ({pluie:.1f}mm)", True
    if pluie > PLUIE_MODERE_SEUIL and hs > SOL_MOYEN_SEUIL:
        return 0, f"Pluie moderee + sol ok ({hs:.1f}%)", True
    if deficit <= 0.0:
        return 0, f"Pas de deficit ({deficit:.2f}mm)", True
    return 1, "Analyse ML requise", False

def _ml_decision(features_dict, hs, deficit):
    X         = pd.DataFrame([features_dict])[FEATURES]
    proba     = clf.predict_proba(X)[0]
    decision  = int(clf.predict(X)[0])
    confiance = round(proba[decision] * 100, 1)
    volume    = 0.0
    if decision == 1:
        vol_ml  = float(reg.predict(X)[0])
        facteur = max(0.0, (65.0 - hs) / 25.0)
        vol_fo  = deficit * facteur * SURFACE_M2 / EFFICACITE
        volume  = round(max(0.60 * vol_ml + 0.40 * vol_fo, 0.0), 1)
    return decision, confiance, volume


# ── Open-Meteo cote serveur ───────────────────────────────────────────────

def _appeler_open_meteo():
    try:
        import openmeteo_requests, requests_cache
        from retry_requests import retry
    except ImportError:
        raise HTTPException(500, "pip install openmeteo-requests requests-cache retry-requests")

    cache_dir = os.path.join(_THIS_DIR, ".cache")
    session   = retry(requests_cache.CachedSession(cache_dir, expire_after=3600),
                      retries=3, backoff_factor=0.3)
    client    = openmeteo_requests.Client(session=session)

    params = {
        "latitude": LATITUDE_DEG, "longitude": LONGITUDE_DEG,
        "timezone": TIMEZONE, "forecast_days": 4,
        "hourly": [
            "temperature_2m","rain","wind_speed_10m",
            "soil_temperature_0_to_7cm","soil_moisture_7_to_28cm",
            "relative_humidity_2m","soil_moisture_0_to_7cm",
        ],
        "daily": [
            "temperature_2m_mean","temperature_2m_max","temperature_2m_min",
            "wind_speed_10m_max","shortwave_radiation_sum",
            "et0_fao_evapotranspiration","precipitation_sum",
        ],
    }

    try:
        r = client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
    except Exception as e:
        raise HTTPException(503, f"Open-Meteo indisponible : {e}")

    h = r.Hourly()
    dates_h = pd.date_range(
        start=pd.to_datetime(h.Time(), unit="s", utc=True),
        end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=h.Interval()), inclusive="left",
    ).tz_convert(TIMEZONE)

    df_h = pd.DataFrame({
        "datetime"               : dates_h,
        "temp_C"                 : h.Variables(0).ValuesAsNumpy(),
        "pluie_mm"               : h.Variables(1).ValuesAsNumpy(),
        "vent_10m_kmh"           : h.Variables(2).ValuesAsNumpy(),
        "temp_sol_0_7cm"         : h.Variables(3).ValuesAsNumpy(),
        "humidite_sol_7_28cm_m3" : h.Variables(4).ValuesAsNumpy(),
        "humidite_air_pct"       : h.Variables(5).ValuesAsNumpy(),
        "humidite_sol_0_7cm_m3"  : h.Variables(6).ValuesAsNumpy(),
    })
    df_h["humidite_sol_7_28cm_pct"] = (df_h["humidite_sol_7_28cm_m3"] * 100).round(1)
    df_h["humidite_sol_0_7cm_pct"]  = (df_h["humidite_sol_0_7cm_m3"]  * 100).round(1)
    df_h["date_only"] = pd.to_datetime(df_h["datetime"]).dt.date

    d = r.Daily()
    dates_d = pd.date_range(
        start=pd.to_datetime(d.Time(), unit="s", utc=True),
        end=pd.to_datetime(d.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=d.Interval()), inclusive="left",
    ).tz_convert(TIMEZONE)

    df_q = pd.DataFrame({
        "date"              : dates_d.date,
        "temp_moy_C"        : d.Variables(0).ValuesAsNumpy(),
        "temp_max_C"        : d.Variables(1).ValuesAsNumpy(),
        "temp_min_C"        : d.Variables(2).ValuesAsNumpy(),
        "vent_max_kmh"      : d.Variables(3).ValuesAsNumpy(),
        "rayonnement_Rs_MJ" : d.Variables(4).ValuesAsNumpy(),
        "ET0_reference_mm"  : d.Variables(5).ValuesAsNumpy(),
        "pluie_totale_mm"   : d.Variables(6).ValuesAsNumpy(),
    })
    return df_h, df_q


def _construire_features(df_h, df_q, index_jour, humidite_capteur=None, jour_cycle=60):
    """Identique a construire_features() dans 06_api_openmeteo.py."""
    HS_DEFAUT = 40.0
    row_q   = df_q.iloc[index_jour]
    date_j  = row_q["date"]
    d_dt    = pd.to_datetime(str(date_j))
    df_jour = df_h[df_h["date_only"] == date_j]

    if len(df_jour) == 0:
        hs_raw = df_h["humidite_sol_7_28cm_pct"].dropna()
        hs_moy = float(hs_raw.mean()) if len(hs_raw) > 0 else HS_DEFAUT
        hs_min = hs_moy - 3.0
        hs_07  = hs_moy - 1.5
        ha_moy = float(df_h["humidite_air_pct"].mean())
        ha_max = min(100.0, ha_moy + 15.0)
        ha_min = max(0.0,   ha_moy - 15.0)
        vent   = float(df_h["vent_10m_kmh"].mean())
    else:
        hs_raw = df_jour["humidite_sol_7_28cm_pct"].dropna()
        hs_moy = float(hs_raw.mean()) if len(hs_raw) > 0 else HS_DEFAUT
        hs_min = float(hs_raw.min())  if len(hs_raw) > 0 else HS_DEFAUT - 5.0
        hs_07r = df_jour["humidite_sol_0_7cm_pct"].dropna()
        hs_07  = float(hs_07r.mean()) if len(hs_07r) > 0 else max(10.0, hs_moy - 2.0)
        ha_moy = float(df_jour["humidite_air_pct"].mean())
        ha_max = float(df_jour["humidite_air_pct"].max())
        ha_min = float(df_jour["humidite_air_pct"].min())
        vent   = float(df_jour["vent_10m_kmh"].mean())

    hs_moy = max(5.0, min(100.0, _nan_safe(hs_moy, HS_DEFAUT)))
    hs_min = max(5.0, min(100.0, _nan_safe(hs_min, HS_DEFAUT - 5.0)))
    hs_07  = max(5.0, min(100.0, _nan_safe(hs_07,  HS_DEFAUT - 2.0)))
    ha_moy = _nan_safe(ha_moy, 60.0)
    ha_max = _nan_safe(ha_max, 80.0)
    ha_min = _nan_safe(ha_min, 40.0)
    vent   = _nan_safe(vent,   7.2)

    if humidite_capteur is not None and index_jour == 0:
        hs_moy = humidite_capteur
        hs_min = max(10.0, humidite_capteur - 3.0)
        hs_07  = max(10.0, humidite_capteur - 1.5)

    vent_u2 = round(vent / 3.6 * 0.748, 3)
    ET0     = float(row_q["ET0_reference_mm"])
    saison  = _get_saison(int(d_dt.month))
    kc      = kc_tomate(jour_cycle, saison, ha_min, vent_u2)
    ETc     = round(ET0 * kc, 2)
    pluie   = float(row_q["pluie_totale_mm"])
    pluie_e = round(pluie * PLUIE_EFFECTIVE_PCT, 2)
    deficit = round(max(ETc - pluie_e, 0.0), 2)

    return {
        "humidite_sol_moy_pct" : round(hs_moy, 1),
        "humidite_sol_min_pct" : round(hs_min, 1),
        "humidite_sol_0_7_moy" : round(hs_07,  1),
        "temp_max_C"           : float(row_q["temp_max_C"]),
        "temp_min_C"           : float(row_q["temp_min_C"]),
        "temp_moy_C"           : float(row_q["temp_moy_C"]),
        "humidite_air_moy_pct" : round(ha_moy, 1),
        "vent_u2_ms"           : vent_u2,
        "rayonnement_Rs_MJ"    : float(row_q["rayonnement_Rs_MJ"]),
        "ET0_reference_mm"     : ET0,
        "ETc_mm"               : ETc,
        "pluie_totale_mm"      : pluie,
        "pluie_effective_mm"   : pluie_e,
        "deficit_hydrique_mm"  : deficit,
        "jour_annee"           : int(d_dt.dayofyear),
        "mois"                 : int(d_dt.month),
        "_date"                : str(date_j),
        "_ET0"                 : ET0,
        "_pluie"               : pluie,
        "_deficit"             : deficit,
        "_humidite_sol"        : round(hs_moy, 1),
        "_kc"                  : kc,
        "_stade"               : get_stade(jour_cycle),
        "saison"               : saison,
        "jour_cycle"           : jour_cycle,
        "RH_max"               : round(ha_max, 1),
        "RH_min"               : round(ha_min, 1),
    }


# ── ENDPOINT /analyser (principal) ────────────────────────────────────────

@app.get("/analyser", response_model=AnalyseComplete)
def analyser(jour_cycle: int = 60, humidite_capteur: Optional[float] = None):
    """
    Analyse complete cote serveur.
    Open-Meteo appele ici -> memes donnees que 06_api_openmeteo.py.
    L'app web appelle uniquement cet endpoint.
    """
    if clf is None or reg is None:
        raise HTTPException(503, "Modeles ML non charges. Lancez 02_entrainement_ml.py")

    df_h, df_q = _appeler_open_meteo()
    jours = []

    for i in range(min(4, len(df_q))):
        jc  = jour_cycle + i
        cap = humidite_capteur if i == 0 else None
        f   = _construire_features(df_h, df_q, i, cap, jc)
        hs  = f["humidite_sol_moy_pct"]
        p   = f["pluie_totale_mm"]
        d   = f["deficit_hydrique_mm"]

        dec_r, raison, court = _appliquer_regles(hs, p, d)
        if court:
            irriguer, volume, confiance = bool(dec_r), 0.0, 100.0
            source = "Regle agronomique"
        else:
            dec_int, confiance, volume = _ml_decision(f, hs, d)
            irriguer = bool(dec_int)
            source   = "Random Forest (Water5 2022-2024)"
            raison   = f"ML {confiance}% | Deficit {d:.2f}mm | Sol {hs:.1f}% | Kc {f['_kc']:.4f}"

        jours.append(JourDecision(
            date=f["_date"], irriguer=irriguer, volume_L=volume,
            confiance_pct=confiance, source=source, raison=raison,
            stade=f["_stade"], kc=f["_kc"], ETc_mm=f["ETc_mm"],
            deficit_mm=d, ET0_mm=f["_ET0"], pluie_mm=f["_pluie"],
            humidite_sol=hs, temp_max_C=f["temp_max_C"],
            temp_min_C=f["temp_min_C"], vent_u2_ms=f["vent_u2_ms"],
        ))

    return AnalyseComplete(
        timestamp=datetime.now().isoformat(),
        source_meteo="Open-Meteo (serveur)",
        aujourd_hui=jours[0],
        previsions=jours[1:],
    )


# ── ENDPOINT POST /decision (compatibilite) ───────────────────────────────

@app.post("/decision", response_model=DecisionOutput)
def calculer_decision(data: MeteoInput):
    if clf is None or reg is None:
        raise HTTPException(503, "Modeles ML non charges.")

    hs       = _nan_safe(data.humidite_sol_moy_pct, 40.0)
    aujourd  = date.today()
    mois     = data.mois or aujourd.month
    saison   = get_saison(mois)
    kc       = kc_tomate(data.jour_cycle, saison, data.RH_min, data.vent_u2_ms)
    stade    = get_stade(data.jour_cycle)
    ETc      = round(data.ET0_reference_mm * kc, 2)
    pluie_e  = round(data.pluie_totale_mm * PLUIE_EFFECTIVE_PCT, 2)
    deficit  = round(max(ETc - pluie_e, 0.0), 2)
    hs_min   = _nan_safe(data.humidite_sol_min_pct, max(10.0, hs - 5.0))
    hs_07    = _nan_safe(data.humidite_sol_0_7_moy, max(10.0, hs - 2.0))
    j_annee  = data.jour_annee or aujourd.timetuple().tm_yday

    dec_r, raison, court = _appliquer_regles(hs, data.pluie_totale_mm, deficit)
    if court:
        return DecisionOutput(
            irriguer=bool(dec_r), volume_L=0.0, confiance_pct=100.0,
            source="Regle agronomique", raison=raison, stade=stade,
            kc=kc, ETc_mm=ETc, deficit_mm=deficit,
            ET0_mm=data.ET0_reference_mm, timestamp=datetime.now().isoformat(),
        )

    fd = {
        "humidite_sol_moy_pct": hs, "humidite_sol_min_pct": hs_min,
        "humidite_sol_0_7_moy": hs_07, "temp_max_C": data.temp_max_C,
        "temp_min_C": data.temp_min_C, "temp_moy_C": data.temp_moy_C,
        "humidite_air_moy_pct": data.humidite_air_moy_pct,
        "vent_u2_ms": data.vent_u2_ms, "rayonnement_Rs_MJ": data.rayonnement_Rs_MJ,
        "ET0_reference_mm": data.ET0_reference_mm, "ETc_mm": ETc,
        "deficit_hydrique_mm": deficit, "pluie_totale_mm": data.pluie_totale_mm,
        "pluie_effective_mm": pluie_e, "jour_annee": j_annee, "mois": mois,
    }
    decision, confiance, volume = _ml_decision(fd, hs, deficit)
    return DecisionOutput(
        irriguer=bool(decision), volume_L=volume, confiance_pct=confiance,
        source=f"Random Forest (Water5 2022-2024) | {data.source_sol}",
        raison=f"ML {confiance}% | Deficit {deficit:.2f}mm | Sol {hs:.1f}% | Kc {kc:.4f}",
        stade=stade, kc=kc, ETc_mm=ETc, deficit_mm=deficit,
        ET0_mm=data.ET0_reference_mm, timestamp=datetime.now().isoformat(),
    )


# ── Health + fichiers statiques ───────────────────────────────────────────

@app.get("/health")
def health():
    return {"clf_charge": clf is not None, "reg_charge": reg is not None,
            "features": FEATURES, "surface_m2": SURFACE_M2}


def _servir(nom: str, media_type: str):
    """Sert un fichier statique depuis _WEB_DIR avec log si absent."""
    chemin = os.path.join(_WEB_DIR, nom)
    if not os.path.exists(chemin):
        print(f"[404] Fichier introuvable : {chemin}")
        print(f"      Dossier web courant : {_WEB_DIR}")
        print(f"      Fichiers presents   : {os.listdir(_WEB_DIR) if os.path.isdir(_WEB_DIR) else 'DOSSIER ABSENT'}")
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": f"{nom} introuvable dans {_WEB_DIR}"}, status_code=404)
    return FileResponse(chemin, media_type=media_type)


@app.get("/app.js",        include_in_schema=False)
def js():       return _servir("app.js",        "application/javascript")

@app.get("/style.css",     include_in_schema=False)
def css():      return _servir("style.css",     "text/css")

@app.get("/sw.js",         include_in_schema=False)
def sw():       return _servir("sw.js",         "application/javascript")

@app.get("/manifest.json", include_in_schema=False)
def manifest(): return _servir("manifest.json", "application/json")

@app.get("/",            include_in_schema=False)
@app.get("/index.html",  include_in_schema=False)
def index():    return _servir("index.html",    "text/html")


# ── Lancement ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("=" * 56)
    print("  Water5 API v3.0")
    print(f"  App      : http://localhost:8000")
    print(f"  Analyser : http://localhost:8000/analyser")
    print(f"  Docs     : http://localhost:8000/docs")
    print(f"  Web dir  : {_WEB_DIR}")
    print("=" * 56)

    # Verifier que les fichiers web existent
    for f in ["index.html", "style.css", "app.js"]:
        p = os.path.join(_WEB_DIR, f)
        ok = os.path.exists(p)
        print(f"  {'[OK]' if ok else '[MANQUANT]'} {f} -> {p}")
    print("=" * 56)

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,   # reload=False evite les conflits de fichiers
    )
