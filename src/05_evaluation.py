"""
05_evaluation.py
Evaluation complete des modeles ML sur le dataset 2022-2024.

Produit :
  - Metriques classification et regression (jeu de test uniquement)
  - Analyse par annee et par saison
  - Analyse des erreurs critiques (FN et FP)
  - Courbes ROC / Precision-Rappel
  - Graphique complet -> outputs/evaluation_complete.png

Split : chronologique 80/20, coherent avec 02_entrainement_ml.py
"""

import os
import sys
import warnings
import joblib

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR, os.path.join(_PARENT_DIR, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score,
)
warnings.filterwarnings("ignore")

from config import (
    CSV_CLEAN, CLF_PATH, REG_PATH, OUT_DIR,
    FEATURES, TARGET_CLF, TARGET_REG,
    PLUIE_EFFECTIVE_PCT,
)

COULEURS = {
    "bleu"  : "#1D3557",
    "orange": "#E76F51",
    "vert"  : "#2D6A4F",
    "clair" : "#A8DADC",
    "gris"  : "#ADB5BD",
}


# ── Chargement et split ───────────────────────────────────────────────────

def charger() -> tuple:
    """
    Charge les donnees et applique le split chronologique 80/20.
    Les metriques sont calculees sur le jeu de TEST uniquement.
    """
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    df  = pd.read_csv(CSV_CLEAN, parse_dates=["date"])
    df  = df.sort_values("date").reset_index(drop=True)

    if "annee"      not in df.columns: df["annee"]      = df["date"].dt.year
    if "mois"       not in df.columns: df["mois"]       = df["date"].dt.month
    if "jour_annee" not in df.columns: df["jour_annee"] = df["date"].dt.dayofyear
    if "saison"     not in df.columns:
        from agronomie import get_saison
        df["saison"] = df["mois"].map(get_saison)
    if "kc_dynamique"    not in df.columns: df["kc_dynamique"] = 1.05
    if "ETc_mm"          not in df.columns:
        df["ETc_mm"]              = (df["ET0_reference_mm"] * df["kc_dynamique"]).round(2)
        df["pluie_effective_mm"]  = (df["pluie_totale_mm"]  * PLUIE_EFFECTIVE_PCT).round(2)
        df["deficit_hydrique_mm"] = (df["ETc_mm"] - df["pluie_effective_mm"]).round(2)

    cut   = int(len(df) * 0.80)
    train = df.iloc[:cut].copy()
    test  = df.iloc[cut:].copy()

    X_all = df[FEATURES];    y_all = df[TARGET_CLF].astype(int)
    X_te  = test[FEATURES];  y_te  = test[TARGET_CLF].astype(int)

    test_irr = test[test[TARGET_REG] > 0]
    X_te_r   = test_irr[FEATURES]
    y_te_r   = test_irr[TARGET_REG]

    annees = sorted(df["annee"].unique().tolist())
    print(f"Dataset : {len(df)} jours | annees : {annees}")
    print(f"Train   : {len(train)}j  |  Test : {len(test)}j "
          f"({test['date'].min().date()} -> {test['date'].max().date()})")
    print(f"Test clf : {len(X_te)} obs | Test reg : {len(X_te_r)} obs")
    return clf, reg, df, train, test, X_all, y_all, X_te, y_te, X_te_r, y_te_r


# ── Classification ────────────────────────────────────────────────────────

def evaluer_classification(clf, X_all, y_all, X_te, y_te) -> tuple:
    print("\n" + "=" * 56)
    print("  CLASSIFICATION - OUI/NON Irriguer (jeu de TEST)")
    print("=" * 56)

    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]
    acc     = accuracy_score(y_te, y_pred)
    prec    = precision_score(y_te, y_pred, zero_division=0)
    rec     = recall_score(y_te, y_pred, zero_division=0)
    f1      = f1_score(y_te, y_pred, average="weighted")
    f1b     = f1_score(y_te, y_pred, average="binary")
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    roc_auc     = auc(fpr, tpr)

    print(f"\n  Metriques TEST ({len(y_te)} obs) :")
    print(f"  {'─'*40}")
    print(f"  Accuracy            : {acc*100:.2f}%")
    print(f"  Precision           : {prec*100:.2f}%")
    print(f"  Rappel              : {rec*100:.2f}%")
    print(f"  F1-score (binaire)  : {f1b*100:.2f}%")
    print(f"  F1-score (pondere)  : {f1*100:.2f}%")
    print(f"  AUC-ROC             : {roc_auc:.4f}")

    print(f"\n  Validation croisee TimeSeriesSplit (5 folds, {len(y_all)} jours) :")
    print(f"  {'─'*40}")
    tscv = TimeSeriesSplit(n_splits=5)
    for m in ["accuracy", "f1_weighted", "precision", "recall"]:
        sc = cross_val_score(clf, X_all, y_all, cv=tscv, scoring=m)
        print(f"  {m:<20}: {sc.mean()*100:.2f}% +/- {sc.std()*100:.2f}%")

    cm = confusion_matrix(y_te, y_pred)
    VP, VN = cm[1, 1], cm[0, 0]
    FP, FN = cm[0, 1], cm[1, 0]
    print(f"\n  Matrice de confusion (TEST) :")
    print(f"               Predit NON   Predit OUI")
    print(f"  Reel NON  :      {VN:>5}        {FP:>5}")
    print(f"  Reel OUI  :      {FN:>5}        {VP:>5}")
    print(f"\n  VP={VP} | VN={VN} | FP={FP} | FN={FN}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['NON', 'OUI'])}")
    return y_pred, y_proba, cm, acc, prec, rec, f1, roc_auc, fpr, tpr


# ── Regression ────────────────────────────────────────────────────────────

def evaluer_regression(reg, X_te_r, y_te_r) -> tuple:
    print("\n" + "=" * 56)
    print("  REGRESSION - Volume en litres (jeu de TEST)")
    print("=" * 56)

    y_pred_r = reg.predict(X_te_r)
    mae      = mean_absolute_error(y_te_r, y_pred_r)
    rmse     = np.sqrt(mean_squared_error(y_te_r, y_pred_r))
    r2       = r2_score(y_te_r, y_pred_r)
    mape     = np.mean(np.abs((y_te_r - y_pred_r) / y_te_r.clip(lower=1))) * 100
    erreurs  = np.abs(y_te_r.values - y_pred_r)

    print(f"\n  Metriques TEST ({len(y_te_r)} obs irrigues) :")
    print(f"  {'─'*40}")
    print(f"  MAE  : {mae:.1f} L")
    print(f"  RMSE : {rmse:.1f} L")
    print(f"  R2   : {r2:.4f} ({r2*100:.2f}%)")
    print(f"  MAPE : {mape:.1f}%")
    print(f"\n  Distribution des erreurs absolues :")
    for seuil in [50, 100, 150, 200]:
        pct = (erreurs <= seuil).mean() * 100
        print(f"  Erreur <= {seuil:>3}L : {pct:.1f}%")

    interp = ("Excellent" if r2 >= 0.95 else "Bon" if r2 >= 0.85 else "A ameliorer")
    print(f"\n  R2 : {interp} ({r2*100:.1f}% variance expliquee)")
    return y_pred_r, mae, rmse, r2, mape


# ── Analyse par annee et saison ───────────────────────────────────────────

def analyse_annee_saison(clf, df: pd.DataFrame, test: pd.DataFrame):
    """Analyse des performances par annee et saison sur le jeu de TEST."""
    print("\n" + "=" * 56)
    print("  ANALYSE PAR ANNEE (jeu de TEST)")
    print("=" * 56)

    print(f"\n  {'Annee':<8} {'Jours':>6} {'Irrig%':>7} {'Acc%':>7} {'Kc moy':>8} {'Pluie/an':>10}")
    print(f"  {'─'*50}")
    for yr, grp in test.groupby("annee"):
        X_y   = grp[FEATURES]
        y_y   = grp[TARGET_CLF].astype(int)
        y_p   = clf.predict(X_y)
        acc   = accuracy_score(y_y, y_p) * 100
        irr   = y_y.mean() * 100
        kc_m  = grp.get("kc_dynamique", pd.Series([1.05])).mean()
        pluie = grp["pluie_totale_mm"].sum()
        print(f"  {yr:<8} {len(grp):>6} {irr:>6.1f}% {acc:>6.1f}% {kc_m:>7.4f} {pluie:>9.0f}mm")

    print(f"\n  ANALYSE PAR SAISON (jeu de TEST) :")
    print(f"  {'─'*50}")
    print(f"  {'Saison':<26} {'Jours':>6} {'Irrig%':>7} {'Acc%':>7} {'Vol moy':>9}")
    print(f"  {'─'*50}")
    noms = {
        "seche"        : "Seche (nov-mars)",
        "grande_pluie" : "Grande pluie (juin-sept)",
        "petite_pluie" : "Petite pluie (avr-mai, oct)",
    }
    for code, label in noms.items():
        grp = test[test["saison"] == code]
        if len(grp) == 0:
            continue
        X_s = grp[FEATURES]; y_s = grp[TARGET_CLF].astype(int)
        y_p = clf.predict(X_s)
        acc = accuracy_score(y_s, y_p) * 100
        irr = y_s.mean() * 100
        vol = grp[grp[TARGET_REG] > 0][TARGET_REG].mean()
        print(f"  {label:<26} {len(grp):>6} {irr:>6.1f}% {acc:>6.1f}% {vol:>8.0f}L")


# ── Erreurs critiques ─────────────────────────────────────────────────────

def analyser_erreurs(clf, test: pd.DataFrame) -> tuple:
    """
    Analyse des FP et FN sur le jeu de TEST uniquement.
    FN = oubli d'irrigation (risque agronomique).
    FP = irrigation inutile (gaspillage d'eau).
    """
    print("\n" + "=" * 56)
    print("  ERREURS CRITIQUES (jeu de TEST)")
    print("=" * 56)

    y_real   = test[TARGET_CLF].astype(int)
    y_pred   = clf.predict(test[FEATURES])
    faux_neg = test[(y_pred == 0) & (y_real == 1)]
    faux_pos = test[(y_pred == 1) & (y_real == 0)]

    print(f"\n  FAUX NEGATIFS - oublis d'irrigation : {len(faux_neg)} jours")
    if len(faux_neg) > 0:
        print(f"  {'Date':<12} {'Annee':>5} {'Sol%':>6} {'Pluie':>7} {'ET0':>6} {'Deficit':>9} {'Saison'}")
        print(f"  {'─'*56}")
        for _, r in faux_neg.iterrows():
            print(f"  {str(r['date'])[:10]:<12} {int(r['annee']):>5} "
                  f"{r['humidite_sol_moy_pct']:>5.1f}% "
                  f"{r['pluie_totale_mm']:>6.1f}mm "
                  f"{r['ET0_reference_mm']:>5.2f} "
                  f"{r['deficit_hydrique_mm']:>8.2f}mm  {r['saison']}")

    print(f"\n  FAUX POSITIFS - irrigations inutiles : {len(faux_pos)} jours")
    if len(faux_pos) > 0:
        print(f"  {'Date':<12} {'Annee':>5} {'Sol%':>6} {'Pluie':>7} {'ET0':>6} {'Deficit':>9} {'Saison'}")
        print(f"  {'─'*56}")
        for _, r in faux_pos.iterrows():
            print(f"  {str(r['date'])[:10]:<12} {int(r['annee']):>5} "
                  f"{r['humidite_sol_moy_pct']:>5.1f}% "
                  f"{r['pluie_totale_mm']:>6.1f}mm "
                  f"{r['ET0_reference_mm']:>5.2f} "
                  f"{r['deficit_hydrique_mm']:>8.2f}mm  {r['saison']}")
    else:
        print("  Aucun gaspillage d'eau detecte sur le jeu de test.")

    return faux_neg, faux_pos


# ── Graphiques ────────────────────────────────────────────────────────────

def generer_graphiques(
    clf, reg, df, test,
    y_te, y_pred_clf, y_proba,
    y_te_r, y_pred_r, cm,
    fpr, tpr, roc_auc,
) -> str:
    print("\nGeneration des graphiques...")

    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    C   = COULEURS

    # 1. Matrice de confusion
    ax1 = fig.add_subplot(gs[0, 0])
    im  = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Predit NON", "Predit OUI"], fontsize=9)
    ax1.set_yticklabels(["Reel NON", "Reel OUI"],    fontsize=9)
    labs = [["VN", "FP"], ["FN", "VP"]]
    for i in range(2):
        for j in range(2):
            col = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax1.text(j, i, f"{labs[i][j]}\n{cm[i, j]}",
                     ha="center", va="center", fontsize=13, fontweight="bold", color=col)
    ax1.set_title("Matrice de confusion\n(jeu de TEST)", fontweight="bold", fontsize=11)
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # 2. Courbe ROC
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr, tpr, color=C["bleu"], lw=2.5, label=f"AUC = {roc_auc:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatoire")
    ax2.fill_between(fpr, tpr, alpha=0.15, color=C["bleu"])
    ax2.set_xlabel("Taux Faux Positifs", fontsize=10)
    ax2.set_ylabel("Taux Vrais Positifs", fontsize=10)
    ax2.set_title("Courbe ROC", fontweight="bold", fontsize=11)
    ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

    # 3. Courbe Precision-Rappel
    ax3 = fig.add_subplot(gs[0, 2])
    prec_c, rec_c, _ = precision_recall_curve(y_te, y_proba)
    pr_auc = auc(rec_c, prec_c)
    ax3.plot(rec_c, prec_c, color=C["orange"], lw=2.5, label=f"AUC-PR = {pr_auc:.4f}")
    ax3.fill_between(rec_c, prec_c, alpha=0.15, color=C["orange"])
    ax3.set_xlabel("Rappel", fontsize=10)
    ax3.set_ylabel("Precision", fontsize=10)
    ax3.set_title("Courbe Precision-Rappel", fontweight="bold", fontsize=11)
    ax3.legend(fontsize=10); ax3.grid(alpha=0.3)

    # 4. Predit vs Reel (regression)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(y_te_r, y_pred_r, alpha=0.5, color=C["bleu"], s=25)
    lim = max(y_te_r.max(), y_pred_r.max()) * 1.05
    ax4.plot([0, lim], [0, lim], "r--", lw=1.5, label="Parfait")
    r2v  = r2_score(y_te_r, y_pred_r)
    maev = mean_absolute_error(y_te_r, y_pred_r)
    ax4.text(0.05, 0.92, f"R2={r2v:.4f}\nMAE={maev:.0f}L",
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(facecolor="white", alpha=0.85, edgecolor=C["bleu"]))
    ax4.set_xlabel("Reel (L)"); ax4.set_ylabel("Predit (L)")
    ax4.set_title("Predit vs Reel - Regression\n(jeu de TEST)", fontweight="bold", fontsize=11)
    ax4.legend(fontsize=10); ax4.grid(alpha=0.3)

    # 5. Distribution erreurs regression
    ax5 = fig.add_subplot(gs[1, 1])
    erreurs = y_pred_r - y_te_r.values
    ax5.hist(erreurs, bins=30, color=C["clair"], edgecolor=C["bleu"], lw=0.8)
    ax5.axvline(0, color="red", lw=2, linestyle="--", label="Erreur=0")
    ax5.axvline(erreurs.mean(), color=C["orange"], lw=2, linestyle="-.",
                label=f"Moy={erreurs.mean():.0f}L")
    ax5.set_xlabel("Erreur (predit-reel) en L", fontsize=10)
    ax5.set_ylabel("Frequence", fontsize=10)
    ax5.set_title("Distribution des erreurs", fontweight="bold", fontsize=11)
    ax5.legend(fontsize=10); ax5.grid(alpha=0.3)

    # 6. Importance features (classification)
    ax6 = fig.add_subplot(gs[1, 2])
    imp    = clf.feature_importances_
    idx    = np.argsort(imp)
    cols_f = [FEATURES[i] for i in idx]
    vals_f = imp[idx]
    colors6 = [C["vert"] if v >= 0.05 else C["gris"] for v in vals_f]
    ax6.barh(cols_f, vals_f, color=colors6, edgecolor="white")
    ax6.axvline(0.05, color="red", lw=1.5, linestyle="--", alpha=0.7, label="5%")
    ax6.set_title("Importance variables\n(Classification)", fontweight="bold", fontsize=11)
    ax6.legend(fontsize=9); ax6.grid(axis="x", alpha=0.3)

    # 7. Importance features (regression)
    ax7 = fig.add_subplot(gs[2, 0])
    imp2    = reg.feature_importances_
    idx2    = np.argsort(imp2)
    cols7   = [FEATURES[i] for i in idx2]
    vals7   = imp2[idx2]
    colors7 = [C["orange"] if v >= 0.05 else C["gris"] for v in vals7]
    ax7.barh(cols7, vals7, color=colors7, edgecolor="white")
    ax7.axvline(0.05, color="red", lw=1.5, linestyle="--", alpha=0.7, label="5%")
    ax7.set_title("Importance variables\n(Regression)", fontweight="bold", fontsize=11)
    ax7.legend(fontsize=9); ax7.grid(axis="x", alpha=0.3)

    # 8. Volume mensuel moyen (TEST)
    ax8 = fig.add_subplot(gs[2, 1])
    test["mois_num"] = pd.to_datetime(test["date"]).dt.month
    vol_mois  = test.groupby("mois_num")[TARGET_REG].mean()
    mois_noms = ["Jan", "Fev", "Mar", "Avr", "Mai", "Jun",
                 "Jul", "Aou", "Sep", "Oct", "Nov", "Dec"]
    bar_c = [C["orange"] if v > vol_mois.mean() else C["clair"] for v in vol_mois.values]
    ax8.bar(vol_mois.index, vol_mois.values, color=bar_c, edgecolor="white")
    ax8.axhline(vol_mois.mean(), color="red", lw=1.5, linestyle="--",
                label=f"Moy. {vol_mois.mean():.0f}L")
    ax8.set_xticks(range(1, 13))
    ax8.set_xticklabels(mois_noms, fontsize=9)
    ax8.set_ylabel("Volume moyen (L)", fontsize=10)
    ax8.set_title("Volume irrigation mensuel\n(jeu de TEST)", fontweight="bold", fontsize=11)
    ax8.legend(fontsize=10); ax8.grid(axis="y", alpha=0.3)

    # 9. Accuracy par annee (TEST)
    ax9 = fig.add_subplot(gs[2, 2])
    annees_test = sorted(test["annee"].unique())
    accs_yr = []
    for yr in annees_test:
        sub_yr = test[test["annee"] == yr]
        yp_yr  = clf.predict(sub_yr[FEATURES])
        accs_yr.append(accuracy_score(sub_yr[TARGET_CLF].astype(int), yp_yr) * 100)
    palette = [C["vert"], C["bleu"], C["orange"]][:len(annees_test)]
    bars9   = ax9.bar([str(y) for y in annees_test], accs_yr, color=palette, edgecolor="white")
    for bar, val in zip(bars9, accs_yr):
        ax9.text(bar.get_x() + bar.get_width() / 2, val - 1.5,
                 f"{val:.1f}%", ha="center", va="top",
                 fontsize=12, fontweight="bold", color="white")
    ax9.set_ylim(88, 101)
    ax9.set_ylabel("Accuracy (%)", fontsize=10)
    ax9.set_title("Accuracy par annee\n(jeu de TEST)", fontweight="bold", fontsize=11)
    ax9.grid(axis="y", alpha=0.3)

    # 10. Tableau recapitulatif
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis("off")
    acc_v  = accuracy_score(y_te, y_pred_clf) * 100
    prec_v = precision_score(y_te, y_pred_clf, zero_division=0) * 100
    rec_v  = recall_score(y_te, y_pred_clf, zero_division=0) * 100
    f1_v   = f1_score(y_te, y_pred_clf, average="weighted") * 100
    mae_v  = mean_absolute_error(y_te_r, y_pred_r)
    r2_v   = r2_score(y_te_r, y_pred_r)
    rmse_v = np.sqrt(mean_squared_error(y_te_r, y_pred_r))

    data_tab = [
        ["Metrique", "Valeur", "Signification", "Seuil ideal"],
        ["Accuracy",        f"{acc_v:.2f}%",    "Taux de decisions correctes",        "> 95%"],
        ["Precision",       f"{prec_v:.2f}%",   "OUI predits qui sont vrais",          "> 90%"],
        ["Rappel",          f"{rec_v:.2f}%",    "OUI reels correctement detectes",     "> 90%"],
        ["F1-score",        f"{f1_v:.2f}%",     "Equilibre precision/rappel",           "> 90%"],
        ["AUC-ROC",         f"{roc_auc:.4f}",   "Capacite discriminante (1=parfait)",  "> 0.95"],
        ["R2 regression",   f"{r2_v:.4f}",      "Variance du volume expliquee",         "> 0.90"],
        ["MAE regression",  f"{mae_v:.1f} L",   "Erreur absolue moyenne (litres)",      "< 100 L"],
        ["RMSE regression", f"{rmse_v:.1f} L",  "Sensibilite aux grandes erreurs",      "< 150 L"],
    ]
    table = ax10.table(cellText=data_tab[1:], colLabels=data_tab[0],
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(C["bleu"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF2FF")
        cell.set_edgecolor("white")
    ax10.set_title(
        "Tableau recapitulatif - Metriques jeu de TEST",
        fontweight="bold", fontsize=12, pad=20,
    )

    fig.suptitle(
        "Evaluation complete - Systeme d'Irrigation Intelligente CI\n"
        "Tomate | 200m2 | Yamoussoukro | Dataset 2022-2024 | "
        "Kc dynamique | Split chronologique",
        fontsize=13, fontweight="bold", y=0.98, color=C["bleu"],
    )

    chemin = os.path.join(OUT_DIR, "evaluation_complete.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  evaluation_complete.png -> {chemin}")
    return chemin


# ── Bilan final ───────────────────────────────────────────────────────────

def bilan_final(acc, prec, rec, f1, roc_auc, mae, rmse, r2, mape, faux_neg, faux_pos):

    def badge(v, ok, bon):
        return ("Excellent" if v >= bon else "Bon" if v >= ok else "A ameliorer")

    print("\n" + "=" * 56)
    print("  BILAN FINAL - METRIQUES (jeu de TEST chronologique)")
    print("=" * 56)
    print(f"\n  CLASSIFICATION :")
    print(f"  {'─'*46}")
    print(f"  Accuracy   : {acc*100:>6.2f}%  [{badge(acc,  0.90, 0.95)}]")
    print(f"  Precision  : {prec*100:>6.2f}%  [{badge(prec, 0.85, 0.95)}]")
    print(f"  Rappel     : {rec*100:>6.2f}%  [{badge(rec,  0.85, 0.95)}]")
    print(f"  F1-score   : {f1*100:>6.2f}%  [{badge(f1,   0.88, 0.95)}]")
    print(f"  AUC-ROC    : {roc_auc:>6.4f}   [{badge(roc_auc, 0.90, 0.97)}]")
    print(f"\n  REGRESSION :")
    print(f"  {'─'*46}")
    print(f"  R2         : {r2:>6.4f}   [{badge(r2, 0.85, 0.95)}]")
    print(f"  MAE        : {mae:>6.1f} L  [{'Excellent' if mae<80 else 'Bon' if mae<120 else 'A ameliorer'}]")
    print(f"  RMSE       : {rmse:>6.1f} L")
    print(f"  MAPE       : {mape:>6.1f}%")
    print(f"\n  SECURITE AGRONOMIQUE :")
    print(f"  {'─'*46}")
    print(f"  Oublis irrigation (FN) : {len(faux_neg):>3}j")
    print(f"  Gaspillages eau  (FP)  : {len(faux_pos):>3}j")
    print(f"\n  Fichiers : outputs/evaluation_complete.png")
    print("=" * 56)


# ── Point d'entree ────────────────────────────────────────────────────────

if __name__ == "__main__":
    (clf, reg, df, train, test,
     X_all, y_all, X_te, y_te,
     X_te_r, y_te_r) = charger()

    y_pred_clf, y_proba, cm, acc, prec, rec, f1, roc_auc, fpr, tpr = \
        evaluer_classification(clf, X_all, y_all, X_te, y_te)

    y_pred_r, mae, rmse, r2, mape = \
        evaluer_regression(reg, X_te_r, y_te_r)

    analyse_annee_saison(clf, df, test)

    faux_neg, faux_pos = analyser_erreurs(clf, test)

    generer_graphiques(clf, reg, df, test, y_te, y_pred_clf, y_proba,
                       y_te_r, y_pred_r, cm, fpr, tpr, roc_auc)

    bilan_final(acc, prec, rec, f1, roc_auc,
                mae, rmse, r2, mape, faux_neg, faux_pos)
