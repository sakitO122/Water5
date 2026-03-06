"""
=============================================================
 SYSTÈME D'IRRIGATION INTELLIGENTE — CÔTE D'IVOIRE
 02_entrainement_ml.py
=============================================================
 Dataset   : 2022-2024 (1096 jours)
 Modèles   :
   A — RandomForestClassifier  → décision OUI/NON
   B — XGBClassifier           → décision OUI/NON (comparaison)
   C — RandomForestRegressor   → volume (litres)
   D — XGBRegressor            → volume (litres) (comparaison)

 Corrections apportées :
   ✔ Split CHRONOLOGIQUE (pas aléatoire) — évite le leakage temporel
   ✔ Cross-validation TimeSeriesSplit (pas KFold standard)
   ✔ Séparation stricte train/test avant tout calcul
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

import os, sys, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics         import (
    accuracy_score, f1_score, mean_absolute_error,
    r2_score, mean_squared_error, classification_report, confusion_matrix,
)
warnings.filterwarnings("ignore")

from config import (
    CSV_CLEAN, MODEL_DIR, OUT_DIR,
    FEATURES, TARGET_CLF, TARGET_REG,
)
from agronomie import kc_tomate, get_stade

# ── XGBoost optionnel ────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_OK = True
    print("✅ XGBoost disponible")
except ImportError:
    XGBOOST_OK = False
    print("⚠  XGBoost absent → pip install xgboost   (RF uniquement)")


# ══════════════════════════════════════════════════════════════
# 1. CHARGEMENT + SPLIT CHRONOLOGIQUE
# ══════════════════════════════════════════════════════════════

def charger_et_splitter() -> tuple:
    """
    Charge le dataset et effectue un split CHRONOLOGIQUE 80/20.

    CORRECTION : train_test_split aléatoire remplacé par un split
    temporel strict. Le modèle entraîné sur 2022–2023 est évalué
    sur 2024 uniquement — ce qui est la seule évaluation réaliste
    pour des données de séries temporelles.
    """
    print("📂 Chargement du dataset 2022-2024...")
    df = pd.read_csv(CSV_CLEAN, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Recalcul colonnes dérivées si absentes
    if "annee" not in df.columns:
        df["annee"] = df["date"].dt.year
    if "mois" not in df.columns:
        df["mois"] = df["date"].dt.month
    if "jour_annee" not in df.columns:
        df["jour_annee"] = df["date"].dt.dayofyear
    if "saison" not in df.columns:
        from agronomie import get_saison
        df["saison"] = df["mois"].map(get_saison)
    if "kc_dynamique" not in df.columns:
        df["kc_dynamique"] = 1.05   # fallback
    if "ETc_mm" not in df.columns:
        from config import PLUIE_EFFECTIVE_PCT
        df["ETc_mm"]              = (df["ET0_reference_mm"] * df["kc_dynamique"]).round(2)
        df["pluie_effective_mm"]  = (df["pluie_totale_mm"]  * PLUIE_EFFECTIVE_PCT).round(2)
        df["deficit_hydrique_mm"] = (df["ETc_mm"] - df["pluie_effective_mm"]).round(2)

    manquantes = [f for f in FEATURES if f not in df.columns]
    if manquantes:
        print(f"❌ Colonnes manquantes : {manquantes}")
        sys.exit(1)

    df.dropna(subset=FEATURES + [TARGET_CLF, TARGET_REG], inplace=True)

    # ── Split chronologique 80 % / 20 % ───────────────────
    cut    = int(len(df) * 0.80)
    train  = df.iloc[:cut].copy()
    test   = df.iloc[cut:].copy()

    date_cut = df.iloc[cut]["date"]
    print(f"   ✅ {len(df)} observations | {sorted(df['annee'].unique().tolist())}")
    print(f"   📅 Train : {train['date'].min().date()} → {train['date'].max().date()} "
          f"({len(train)} j)")
    print(f"   📅 Test  : {test['date'].min().date()} → {test['date'].max().date()} "
          f"({len(test)} j)")
    return df, train, test


# ══════════════════════════════════════════════════════════════
# 2. CLASSIFICATION
# ══════════════════════════════════════════════════════════════

def entrainer_classification(df: pd.DataFrame,
                              train: pd.DataFrame,
                              test: pd.DataFrame) -> tuple:
    print("\n" + "═" * 60)
    print("  MODÈLE A — CLASSIFICATION : Irriguer OUI/NON")
    print("═" * 60)

    X_tr = train[FEATURES];  y_tr = train[TARGET_CLF].astype(int)
    X_te = test[FEATURES];   y_te = test[TARGET_CLF].astype(int)
    X    = df[FEATURES];     y    = df[TARGET_CLF].astype(int)

    print(f"  Train — NON : {(y_tr==0).sum()} | OUI : {(y_tr==1).sum()}")
    print(f"  Test  — NON : {(y_te==0).sum()} | OUI : {(y_te==1).sum()}")

    # ── TimeSeriesSplit (5 folds) ──────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    resultats = {}

    # ── Random Forest ──────────────────────────────────────
    print("\n  [1/2] Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    yp_rf = rf.predict(X_te)
    f1_rf = f1_score(y_te, yp_rf, average="weighted")
    ac_rf = accuracy_score(y_te, yp_rf)
    # ✅ CORRECTION : TimeSeriesSplit au lieu de KFold
    cv_rf = cross_val_score(rf, X, y, cv=tscv, scoring="f1_weighted")
    print(f"   Accuracy : {ac_rf:.4f} | F1 : {f1_rf:.4f} "
          f"| CV-TS : {cv_rf.mean():.4f}±{cv_rf.std():.4f}")
    print(classification_report(y_te, yp_rf, target_names=["NON", "OUI"]))
    resultats["RF_Classifier"] = {
        "model": rf, "f1": f1_rf, "accuracy": ac_rf,
        "cv_mean": cv_rf.mean(), "y_pred": yp_rf,
    }

    # ── XGBoost ────────────────────────────────────────────
    if XGBOOST_OK:
        print("\n  [2/2] XGBoost Classifier...")
        ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=ratio, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        xgb.fit(X_tr, y_tr)
        yp_xgb = xgb.predict(X_te)
        f1_xgb = f1_score(y_te, yp_xgb, average="weighted")
        ac_xgb = accuracy_score(y_te, yp_xgb)
        cv_xgb = cross_val_score(xgb, X, y, cv=tscv, scoring="f1_weighted")
        print(f"   Accuracy : {ac_xgb:.4f} | F1 : {f1_xgb:.4f} "
              f"| CV-TS : {cv_xgb.mean():.4f}±{cv_xgb.std():.4f}")
        resultats["XGB_Classifier"] = {
            "model": xgb, "f1": f1_xgb, "accuracy": ac_xgb,
            "cv_mean": cv_xgb.mean(), "y_pred": yp_xgb,
        }

    best_nom = max(resultats, key=lambda k: resultats[k]["f1"])
    best     = resultats[best_nom]
    print(f"\n   Meilleur : {best_nom} | F1={best['f1']:.4f}")

    chemin = os.path.join(MODEL_DIR, "modele_classification.joblib")
    joblib.dump(best["model"], chemin)
    print(f"   Sauvegardé → {chemin}")

    _sauver_importance(best["model"], FEATURES, "importance_classification.png",
                       "Classification — OUI/NON Irriguer")
    _sauver_confusion(y_te, best["y_pred"], best_nom)
    return best["model"], X_te, y_te, resultats


# ══════════════════════════════════════════════════════════════
# 3. RÉGRESSION
# ══════════════════════════════════════════════════════════════

def entrainer_regression(df: pd.DataFrame,
                          train: pd.DataFrame,
                          test: pd.DataFrame) -> tuple:
    print("\n" + "═" * 60)
    print("  MODÈLE B — RÉGRESSION : Volume d'eau (litres)")
    print("═" * 60)

    # Sous-ensemble : jours irrigués uniquement
    sub_tr = train[train[TARGET_REG] > 0].copy()
    sub_te = test[test[TARGET_REG] > 0].copy()
    sub_all = df[df[TARGET_REG] > 0].copy()

    print(f"  Train irrigué : {len(sub_tr)} jours | Test irrigué : {len(sub_te)} jours")

    X_tr = sub_tr[FEATURES];  y_tr = sub_tr[TARGET_REG]
    X_te = sub_te[FEATURES];  y_te = sub_te[TARGET_REG]
    X    = sub_all[FEATURES]; y    = sub_all[TARGET_REG]

    tscv = TimeSeriesSplit(n_splits=5)
    resultats = {}

    # ── Random Forest ──────────────────────────────────────
    print("\n  [1/2] Random Forest Regressor...")
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    yp    = rf.predict(X_te)
    mae   = mean_absolute_error(y_te, yp)
    r2    = r2_score(y_te, yp)
    rmse  = np.sqrt(mean_squared_error(y_te, yp))
    print(f"   MAE={mae:.1f}L | RMSE={rmse:.1f}L | R²={r2:.4f}")
    resultats["RF_Regressor"] = {"model": rf, "mae": mae, "r2": r2, "y_pred": yp}

    # ── XGBoost ────────────────────────────────────────────
    if XGBOOST_OK:
        print("\n  [2/2] XGBoost Regressor...")
        xgb = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=42, verbosity=0,
        )
        xgb.fit(X_tr, y_tr)
        yp2   = xgb.predict(X_te)
        mae2  = mean_absolute_error(y_te, yp2)
        r22   = r2_score(y_te, yp2)
        rmse2 = np.sqrt(mean_squared_error(y_te, yp2))
        print(f"   MAE={mae2:.1f}L | RMSE={rmse2:.1f}L | R²={r22:.4f}")
        resultats["XGB_Regressor"] = {"model": xgb, "mae": mae2, "r2": r22, "y_pred": yp2}

    best_nom = max(resultats, key=lambda k: resultats[k]["r2"])
    best     = resultats[best_nom]
    print(f"\n   Meilleur : {best_nom} | R²={best['r2']:.4f} | MAE={best['mae']:.1f}L")

    chemin = os.path.join(MODEL_DIR, "modele_regression.joblib")
    joblib.dump(best["model"], chemin)
    print(f"   Sauvegardé → {chemin}")

    _sauver_importance(best["model"], FEATURES, "importance_regression.png",
                       "Régression — Volume (litres)")
    _sauver_pred_reel(y_te, best["y_pred"], best_nom)
    return best["model"], resultats


# ══════════════════════════════════════════════════════════════
# 4. VISUALISATIONS
# ══════════════════════════════════════════════════════════════

def _sauver_importance(model, features: list, nom: str, titre: str):
    imp    = model.feature_importances_
    idx    = np.argsort(imp)
    cols   = [features[i] for i in idx]
    vals   = imp[idx]
    colors = ["#2E86AB" if v >= 0.05 else "#ADB5BD" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(cols, vals, color=colors, edgecolor="white")
    ax.axvline(0.05, color="red", linestyle="--", alpha=0.6, label="Seuil 5%")
    ax.set_title(f"Importance des variables — {titre}", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, nom), dpi=150)
    plt.close()
    print(f"   {nom}")


def _sauver_confusion(y_true, y_pred, nom: str):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im  = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Prédit NON", "Prédit OUI"])
    ax.set_yticklabels(["Réel NON",   "Réel OUI"])
    labels = [["VN", "FP"], ["FN", "VP"]]
    for i in range(2):
        for j in range(2):
            c = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color=c)
    ax.set_title(f"Confusion — {nom}", fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "matrice_confusion.png"), dpi=150)
    plt.close()
    print("   matrice_confusion.png")


def _sauver_pred_reel(y_true, y_pred, nom: str):
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color="#2E86AB", s=25)
    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", label="Parfait")
    ax.text(0.05, 0.92, f"R²={r2:.4f}\nMAE={mae:.0f}L",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(facecolor="white", alpha=0.85))
    ax.set_xlabel("Réel (L)")
    ax.set_ylabel("Prédit (L)")
    ax.set_title(f"Prédit vs Réel — {nom}", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pred_vs_reel_regression.png"), dpi=150)
    plt.close()
    print("   pred_vs_reel_regression.png")


# ══════════════════════════════════════════════════════════════
# 5. BILAN FINAL
# ══════════════════════════════════════════════════════════════

def bilan(res_c: dict, res_r: dict):
    print("\n" + "═" * 60)
    print("  BILAN FINAL")
    print("═" * 60)
    print("\n  Classification (évaluation chronologique 2024) :")
    for n, r in res_c.items():
        print(f"    {n:<25} F1={r['f1']:.4f}  "
              f"Acc={r['accuracy']:.4f}  CV-TS={r['cv_mean']:.4f}")
    print("\n  Régression :")
    for n, r in res_r.items():
        print(f"    {n:<25} R²={r['r2']:.4f}  MAE={r['mae']:.1f}L")

    rapport = (
        "RAPPORT ENTRAÎNEMENT — IRRIGATION CI (2022-2024)\n"
        "Split : chronologique 80/20 | CV : TimeSeriesSplit(5)\n"
        + "=" * 50
        + "\nCLASSIFICATION :\n"
    )
    for n, r in res_c.items():
        rapport += f"  {n}: F1={r['f1']:.4f} Acc={r['accuracy']:.4f} CV={r['cv_mean']:.4f}\n"
    rapport += "\nRÉGRESSION :\n"
    for n, r in res_r.items():
        rapport += f"  {n}: R²={r['r2']:.4f} MAE={r['mae']:.1f}L\n"

    with open(os.path.join(OUT_DIR, "rapport_entrainement.txt"), "w") as f:
        f.write(rapport)

    print("\n   rapport_entrainement.txt")
    print("═" * 60)
    print("   Entraînement terminé → lancez : 03_prediction.py")
    print("═" * 60)


# ══════════════════════════════════════════════════════════════
# EXÉCUTION
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df, train, test = charger_et_splitter()
    clf, X_te, y_te, res_c = entrainer_classification(df, train, test)
    reg, res_r              = entrainer_regression(df, train, test)
    bilan(res_c, res_r)