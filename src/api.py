"""
api.py
Serveur FastAPI — pont entre l'application web Water5 et le modele ML.

Installation :
  pip install fastapi uvicorn joblib pandas numpy

Lancement :
  python src/api.py
  ou
  uvicorn src.api:app --reload --port 8000

Endpoints :
  GET  /              -> statut du serveur
  POST /decision      -> decision d'irrigation (modele ML complet)
  GET  /health        -> verification modeles charges
"""

import os
import sys
import math
from datetime import datetime, date
from typing import Optional

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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import (
    CLF_PATH, REG_PATH, FEATURES,
    SURFACE_M2, EFFICACITE, PLUIE_EFFECTIVE_PCT,
    SOL_HUMIDE_SEUIL, SOL_MOYEN_SEUIL,
    PLUIE_FORTE_SEUIL, PLUIE_MODERE_SEUIL,
)
from agronomie import kc_tomate, get_stade, get_saison


# ── Application FastAPI ───────────────────────────────────────────────────

app = FastAPI(
    title="Water5 API",
    description="Systeme d'irrigation intelligente — Yamoussoukro, Cote d'Ivoire",
    version="2.0.0",
)

# CORS : autorise l'app web (fichier local ou GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Chargement des modeles au demarrage ───────────────────────────────────

clf = None
reg = None

@app.on_event("startup")
def charger_modeles():
    global clf, reg
    if not os.path.exists(CLF_PATH):
        print(f"[ERREUR] Modele classification introuvable : {CLF_PATH}")
        print("  -> Lancez d'abord : python src/02_entrainement_ml.py")
        return
    if not os.path.exists(REG_PATH):
        print(f"[ERREUR] Modele regression introuvable : {REG_PATH}")
        return
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    print(f"[OK] Modeles ML charges")
    print(f"  Classification : {CLF_PATH}")
    print(f"  Regression     : {REG_PATH}")

# Localiser le dossier contenant index.html — sera monte apres les routes API
def _trouver_web_dir() -> str:
    candidats = [
        os.path.join(_PARENT_DIR, "app web"),
        os.path.join(_PARENT_DIR, "app_web"),
        os.path.join(_PARENT_DIR, "web"),
        _PARENT_DIR,
        _THIS_DIR,
        os.getcwd(),
    ]
    for c in candidats:
        if os.path.isfile(os.path.join(c, "index.html")):
            return c
    return _PARENT_DIR

_WEB_DIR = _trouver_web_dir()


# ── Schemas de donnees ────────────────────────────────────────────────────

class MeteoInput(BaseModel):
    """Donnees d'entree envoyees par l'application web."""

    # Meteo Open-Meteo (quotidien)
    temp_max_C         : float = Field(..., description="Temperature maximale (C)")
    temp_min_C         : float = Field(..., description="Temperature minimale (C)")
    temp_moy_C         : float = Field(..., description="Temperature moyenne (C)")
    pluie_totale_mm    : float = Field(0.0,  description="Precipitations (mm)")
    vent_u2_ms         : float = Field(2.0,  description="Vent moyen a 2m (m/s)")
    rayonnement_Rs_MJ  : float = Field(15.0, description="Rayonnement solaire (MJ/m2/j)")
    ET0_reference_mm   : float = Field(...,  description="ET0 Open-Meteo (mm/j)")

    # Humidite sol (Open-Meteo ou capteur Arduino)
    humidite_sol_moy_pct : float = Field(...,  description="Humidite sol moyenne (%)")
    humidite_sol_min_pct : float = Field(None, description="Humidite sol minimale (%)")
    humidite_sol_0_7_moy : float = Field(None, description="Humidite sol 0-7cm (%)")
    humidite_air_moy_pct : float = Field(60.0, description="Humidite air moyenne (%)")
    RH_max               : float = Field(80.0, description="Humidite air maximale (%)")
    RH_min               : float = Field(40.0, description="Humidite air minimale (%)")

    # Contexte agronomique
    jour_annee  : Optional[int] = Field(None, description="Jour julien (1-365)")
    mois        : Optional[int] = Field(None, description="Mois (1-12)")
    jour_cycle  : int            = Field(60,   description="Jours depuis plantation")

    # Source de l'humidite sol
    source_sol  : str = Field("Open-Meteo", description="'Arduino' ou 'Open-Meteo'")


class DecisionOutput(BaseModel):
    """Reponse du modele ML renvoyee a l'application web."""
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


# ── Regles agronomiques prioritaires ─────────────────────────────────────

def appliquer_regles(hs: float, pluie: float, deficit: float):
    if hs > SOL_HUMIDE_SEUIL:
        return 0, f"Sol humide ({hs:.1f}% > {SOL_HUMIDE_SEUIL}%)", True
    if pluie > PLUIE_FORTE_SEUIL:
        return 0, f"Forte pluie ({pluie:.1f}mm)", True
    if pluie > PLUIE_MODERE_SEUIL and hs > SOL_MOYEN_SEUIL:
        return 0, f"Pluie moderee + sol ok ({hs:.1f}%)", True
    if deficit <= 0.0:
        return 0, f"Pas de deficit ({deficit:.2f}mm)", True
    return 1, "Analyse ML requise", False


# ── Endpoint principal ────────────────────────────────────────────────────

@app.post("/decision", response_model=DecisionOutput)
def calculer_decision(data: MeteoInput):
    """
    Calcule la decision d'irrigation a partir des donnees meteorologiques.

    Ordre de priorite :
      1. Regles agronomiques (sol trop humide, forte pluie, pas de deficit)
      2. Modele Random Forest si aucune regle ne court-circuite
    """
    if clf is None or reg is None:
        raise HTTPException(
            status_code=503,
            detail="Modeles ML non charges. Lancez d'abord 02_entrainement_ml.py"
        )

    # Nettoyage des valeurs NaN eventuelles (humidite sol Open-Meteo parfois manquante)
    import math
    HS_DEFAUT = 40.0
    if math.isnan(data.humidite_sol_moy_pct) or data.humidite_sol_moy_pct is None:
        data.humidite_sol_moy_pct = HS_DEFAUT
        print(f"  [WARN] humidite_sol_moy_pct NaN -> valeur par defaut {HS_DEFAUT}%")

    # Jour julien et mois par defaut
    aujourd = date.today()
    jour_annee = data.jour_annee or aujourd.timetuple().tm_yday
    mois       = data.mois or aujourd.month

    # Kc dynamique
    saison = get_saison(mois)
    kc     = kc_tomate(data.jour_cycle, saison, data.RH_min, data.vent_u2_ms)
    stade  = get_stade(data.jour_cycle)

    # Bilan hydrique
    ETc       = round(data.ET0_reference_mm * kc, 2)
    pluie_eff = round(data.pluie_totale_mm * PLUIE_EFFECTIVE_PCT, 2)
    deficit   = round(max(ETc - pluie_eff, 0.0), 2)

    # Valeurs manquantes
    hs_min = data.humidite_sol_min_pct if data.humidite_sol_min_pct is not None \
             else max(10.0, data.humidite_sol_moy_pct - 5.0)
    hs_07  = data.humidite_sol_0_7_moy if data.humidite_sol_0_7_moy is not None \
             else max(10.0, data.humidite_sol_moy_pct - 2.0)

    # Regles agronomiques
    dec_r, raison, court = appliquer_regles(
        data.humidite_sol_moy_pct, data.pluie_totale_mm, deficit
    )

    if court:
        return DecisionOutput(
            irriguer      = bool(dec_r),
            volume_L      = 0.0,
            confiance_pct = 100.0,
            source        = "Regle agronomique",
            raison        = raison,
            stade         = stade,
            kc            = kc,
            ETc_mm        = ETc,
            deficit_mm    = deficit,
            ET0_mm        = data.ET0_reference_mm,
            timestamp     = datetime.now().isoformat(),
        )

    # Construction du vecteur de features
    features_dict = {
        "humidite_sol_moy_pct" : data.humidite_sol_moy_pct,
        "humidite_sol_min_pct" : hs_min,
        "humidite_sol_0_7_moy" : hs_07,
        "temp_max_C"           : data.temp_max_C,
        "temp_min_C"           : data.temp_min_C,
        "temp_moy_C"           : data.temp_moy_C,
        "humidite_air_moy_pct" : data.humidite_air_moy_pct,
        "vent_u2_ms"           : data.vent_u2_ms,
        "rayonnement_Rs_MJ"    : data.rayonnement_Rs_MJ,
        "ET0_reference_mm"     : data.ET0_reference_mm,
        "ETc_mm"               : ETc,
        "deficit_hydrique_mm"  : deficit,
        "pluie_totale_mm"      : data.pluie_totale_mm,
        "pluie_effective_mm"   : pluie_eff,
        "jour_annee"           : jour_annee,
        "mois"                 : mois,
    }

    X         = pd.DataFrame([features_dict])[FEATURES]
    proba     = clf.predict_proba(X)[0]
    decision  = int(clf.predict(X)[0])
    confiance = round(proba[decision] * 100, 1)

    volume = 0.0
    if decision == 1:
        vol_ml  = float(reg.predict(X)[0])
        facteur = max(0.0, (65.0 - data.humidite_sol_moy_pct) / 25.0)
        vol_fo  = deficit * facteur * SURFACE_M2 / EFFICACITE
        volume  = round(max(0.60 * vol_ml + 0.40 * vol_fo, 0.0), 1)

    raison = (
        f"ML {confiance}% | Deficit {deficit:.2f}mm | "
        f"Sol {data.humidite_sol_moy_pct:.1f}% | Kc {kc:.4f} | {data.source_sol}"
    )

    return DecisionOutput(
        irriguer      = bool(decision),
        volume_L      = volume,
        confiance_pct = confiance,
        source        = f"Random Forest (Water5 2022-2024) | {data.source_sol}",
        raison        = raison,
        stade         = stade,
        kc            = kc,
        ETc_mm        = ETc,
        deficit_mm    = deficit,
        ET0_mm        = data.ET0_reference_mm,
        timestamp     = datetime.now().isoformat(),
    )


# ── Endpoints utilitaires ─────────────────────────────────────────────────

@app.get("/api", include_in_schema=False)
def root():
    return {
        "service"  : "Water5 API",
        "version"  : "2.0.0",
        "status"   : "ok",
        "modeles"  : "charges" if clf is not None else "non charges",
        "endpoint" : "POST /decision",
    }


@app.get("/health")
def health():
    return {
        "clf_charge" : clf is not None,
        "reg_charge" : reg is not None,
        "features"   : FEATURES,
        "surface_m2" : SURFACE_M2,
        "efficacite" : EFFICACITE,
    }


# ── Fichiers statiques (app web) ──────────────────────────────────────────
# IMPORTANT : montage apres toutes les routes API sinon elles sont masquees

@app.get("/app.js", include_in_schema=False)
def servir_js():
    return FileResponse(os.path.join(_WEB_DIR, "app.js"),
                        media_type="application/javascript")

@app.get("/style.css", include_in_schema=False)
def servir_css():
    return FileResponse(os.path.join(_WEB_DIR, "style.css"),
                        media_type="text/css")

@app.get("/sw.js", include_in_schema=False)
def servir_sw():
    return FileResponse(os.path.join(_WEB_DIR, "sw.js"),
                        media_type="application/javascript")

@app.get("/manifest.json", include_in_schema=False)
def servir_manifest():
    return FileResponse(os.path.join(_WEB_DIR, "manifest.json"),
                        media_type="application/json")

@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def servir_index():
    chemin = os.path.join(_WEB_DIR, "index.html")
    if os.path.exists(chemin):
        print(f"[OK] App web servie depuis : {_WEB_DIR}")
        return FileResponse(chemin, media_type="text/html")
    return {"error": f"index.html introuvable. Dossier recherche : {_WEB_DIR}"}


# ── Lancement direct ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("=" * 56)
    print("  Water5 API + App Web")
    print(f"  App web  : http://localhost:8000        <-- ouvrir ICI")
    print(f"  Sante    : http://localhost:8000/health")
    print(f"  Docs API : http://localhost:8000/docs")
    print(f"  Dossier  : {_WEB_DIR}")
    print("=" * 56)
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)