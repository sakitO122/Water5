// app.js — Water5 v2.0
// Connexion au modele ML via FastAPI + fallback calcul local FAO-56

// ── Configuration ──────────────────────────────────────────────────────────

const API_URL     = "http://localhost:8000";
const LAT         = 6.8276;
const LON         = -5.2893;
const SURFACE     = 200;
const SOL_SEUIL   = 65;
const PLUIE_SEUIL = 8;

// ── Etat ───────────────────────────────────────────────────────────────────

let state = {
  history              : [],
  alerts               : [],
  notificationsEnabled : false,
  lastAnalysis         : null,
  apiConnectee         : false,
};

// ── Initialisation ─────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  chargerDonnees();
  enregistrerServiceWorker();
  verifierAPI();
  mettreAJourHistorique();
  mettreAJourAlertes();
});

function chargerDonnees() {
  try {
    const saved = localStorage.getItem("water5-state");
    if (saved) {
      const parsed = JSON.parse(saved);
      state = Object.assign(state, parsed);
      if (!Array.isArray(state.alerts))  state.alerts  = [];
      if (!Array.isArray(state.history)) state.history = [];
    }
  } catch (e) {
    console.warn("Impossible de charger les donnees locales.");
  }
}

function sauvegarderDonnees() {
  try {
    localStorage.setItem("water5-state", JSON.stringify(state));
  } catch (e) {
    console.warn("Sauvegarde localStorage echouee.");
  }
}

function enregistrerServiceWorker() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }
}

// ── Verification API au demarrage ──────────────────────────────────────────

async function verifierAPI() {
  const dot   = document.getElementById("api-dot");
  const label = document.getElementById("api-label");
  const status = document.getElementById("api-status");

  try {
    const res  = await fetch(`${API_URL}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    const data = await res.json();
    state.apiConnectee = !!(data.clf_charge && data.reg_charge);

    if (state.apiConnectee) {
      if (dot)    { dot.className = "api-dot ok"; }
      if (label)  { label.textContent = "ML actif"; }
      if (status) { status.textContent = "Random Forest actif"; status.className = "settings-val green"; }
    } else {
      if (dot)    { dot.className = "api-dot local"; }
      if (label)  { label.textContent = "Modeles absents"; }
      if (status) { status.textContent = "Modeles non charges"; status.className = "settings-val red"; }
    }
  } catch (e) {
    state.apiConnectee = false;
    if (dot)    { dot.className = "api-dot error"; }
    if (label)  { label.textContent = "Local"; }
    if (status) { status.textContent = "API non connectee"; status.className = "settings-val red"; }
  }
}

// ── Navigation ─────────────────────────────────────────────────────────────

function goToDashboard() {
  document.getElementById("screen-splash").classList.remove("active");
  document.getElementById("screen-dashboard").classList.add("active");
}

function showTab(tab) {
  ["dashboard", "history", "alerts", "settings"].forEach(t => {
    document.getElementById("screen-" + t).classList.remove("active");
  });
  document.getElementById("screen-" + tab).classList.add("active");

  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle(
      "active",
      el.getAttribute("onclick") === `showTab('${tab}')`
    );
  });

  if (tab === "history") mettreAJourHistorique();
  if (tab === "alerts")  mettreAJourAlertes();
}

function goToAlerts() { showTab("alerts"); }

// ── Analyse principale ─────────────────────────────────────────────────────

async function lancerAnalyse() {
  const btn = document.getElementById("analyze-btn");
  btn.disabled = true;

  const overlay = document.getElementById("loading-overlay");
  overlay.classList.remove("hidden");
  ["step1", "step2", "step3", "step4"].forEach(id => {
    document.getElementById(id).className = "loading-step";
  });

  try {
    // Etape 1 : reverifier l'API a chaque analyse
    activerStep("step1");
    await verifierAPI();
    await attendre(350);

    // Etape 2 : recuperer meteo
    activerStep("step2");
    const meteo = await recupererMeteo();
    await attendre(350);

    // Etape 3 : decision ML ou locale
    activerStep("step3");
    let resultat;
    if (state.apiConnectee) {
      try {
        resultat = await appellerAPIML(meteo);
      } catch (errAPI) {
        console.warn("Appel API ML echoue, bascule calcul local :", errAPI.message);
        resultat = calculerDecisionLocale(meteo);
        ajouterAlerte("orange", "Bascule locale",
          "Le serveur ML a repondu une erreur. Calcul FAO-56 applique.", new Date());
      }
    } else {
      resultat = calculerDecisionLocale(meteo);
    }
    await attendre(350);

    // Etape 4 : sauvegarde
    activerStep("step4");
    sauvegarderAnalyse(resultat, meteo);
    await attendre(250);

    overlay.classList.add("hidden");
    afficherResultats(resultat, meteo);

  } catch (err) {
    console.error("Erreur analyse :", err);
    overlay.classList.add("hidden");

    const meteoSecours = valeursDSecours();
    const resultat     = calculerDecisionLocale(meteoSecours);
    sauvegarderAnalyse(resultat, meteoSecours);
    afficherResultats(resultat, meteoSecours);
    ajouterAlerte("red", "Hors-ligne",
      "Donnees meteo indisponibles. Valeurs de secours Yamoussoukro utilisees.", new Date());
  }

  btn.disabled = false;
}

function activerStep(id) {
  document.querySelectorAll(".loading-step").forEach(s => {
    if (s.classList.contains("active")) s.className = "loading-step done";
  });
  document.getElementById(id).classList.add("active");
}

const attendre = ms => new Promise(resolve => setTimeout(resolve, ms));

// ── API Open-Meteo ──────────────────────────────────────────────────────────

async function recupererMeteo() {
  const url =
    `https://api.open-meteo.com/v1/forecast` +
    `?latitude=${LAT}&longitude=${LON}` +
    `&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,` +
    `precipitation_sum,windspeed_10m_max,shortwave_radiation_sum,` +
    `et0_fao_evapotranspiration,soil_moisture_0_to_1cm,` +
    `relative_humidity_2m_max,relative_humidity_2m_min` +
    `&timezone=Africa%2FAbidjan&forecast_days=5`;

  const res = await fetch(url);
  if (!res.ok) throw new Error("API Open-Meteo indisponible");
  const data = await res.json();
  const d    = data.daily;
  const jours = [];

  for (let i = 0; i < Math.min(4, d.time.length); i++) {
    // Humidite sol : NaN possible sur les jours futurs -> fallback 40%
    const HS_DEFAUT = 40;
    let sm_raw = d.soil_moisture_0_to_1cm ? d.soil_moisture_0_to_1cm[i] : null;
    if (sm_raw === null || sm_raw === undefined || isNaN(sm_raw)) sm_raw = 0.18;
    const humSol  = Math.min(Math.max(Math.round((sm_raw / 0.45) * 100), 5), 100);

    const ventKmh = d.windspeed_10m_max[i] || 7.2;
    const ventU2  = parseFloat(((ventKmh / 3.6) * 0.748).toFixed(3));
    const RHmax   = parseFloat((d.relative_humidity_2m_max[i] || 80).toFixed(1));
    const RHmin   = parseFloat((d.relative_humidity_2m_min[i] || 40).toFixed(1));

    jours.push({
      date       : d.time[i],
      tMax       : parseFloat((d.temperature_2m_max[i]  || 32).toFixed(1)),
      tMin       : parseFloat((d.temperature_2m_min[i]  || 22).toFixed(1)),
      tMoy       : parseFloat((d.temperature_2m_mean[i] || 27).toFixed(1)),
      pluie      : parseFloat((d.precipitation_sum[i]   || 0).toFixed(1)),
      ventU2     : ventU2,
      rayonnement: parseFloat((d.shortwave_radiation_sum[i]    || 15).toFixed(1)),
      et0        : parseFloat((d.et0_fao_evapotranspiration[i] || 4.5).toFixed(2)),
      humSol     : humSol,
      humSolMin  : Math.max(10, humSol - 5),
      humSol07   : Math.max(10, humSol - 2),
      humAir     : parseFloat(((RHmax + RHmin) / 2).toFixed(1)),
      RHmax      : RHmax,
      RHmin      : RHmin,
    });
  }
  return jours;
}

// ── Appel API FastAPI (modele ML reel) ─────────────────────────────────────

async function appellerAPIML(meteo) {
  const j         = meteo[0];
  const aujourd   = new Date();
  const jourAnnee = Math.floor((aujourd - new Date(aujourd.getFullYear(), 0, 0)) / 86400000);

  const payload = {
    temp_max_C           : j.tMax,
    temp_min_C           : j.tMin,
    temp_moy_C           : j.tMoy,
    pluie_totale_mm      : j.pluie,
    vent_u2_ms           : j.ventU2,
    rayonnement_Rs_MJ    : j.rayonnement,
    ET0_reference_mm     : j.et0,
    humidite_sol_moy_pct : j.humSol,
    humidite_sol_min_pct : j.humSolMin,
    humidite_sol_0_7_moy : j.humSol07,
    humidite_air_moy_pct : j.humAir,
    RH_max               : j.RHmax,
    RH_min               : j.RHmin,
    jour_annee           : jourAnnee,
    mois                 : aujourd.getMonth() + 1,
    jour_cycle           : 60,
    source_sol           : "Open-Meteo",
  };

  const res = await fetch(`${API_URL}/decision`, {
    method  : "POST",
    headers : { "Content-Type": "application/json" },
    body    : JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(`API erreur ${res.status} : ${err.detail || "inconnue"}`);
  }

  const api = await res.json();

  // Previsions J+1 a J+3
  const previsions = await Promise.allSettled(
    meteo.slice(1).map((jour, i) => appellerAPIMLJour(jour, aujourd.getMonth() + 1, 60 + i + 1))
  ).then(results =>
    results.map((r, i) =>
      r.status === "fulfilled" ? r.value : previsionLocale(meteo[i + 1])
    )
  );

  return {
    doitIrriguer : api.irriguer,
    volume       : Math.round(api.volume_L),
    deficit      : parseFloat(api.deficit_mm).toFixed(1),
    et0          : api.ET0_mm,
    etc          : parseFloat(api.ETc_mm).toFixed(2),
    confiance    : parseFloat(api.confiance_pct).toFixed(1),
    humSol       : j.humSol,
    tMax         : j.tMax,
    pluie        : j.pluie,
    stade        : api.stade,
    kc           : api.kc,
    source       : api.source,
    raison       : api.raison,
    previsions   : previsions,
    timestamp    : api.timestamp,
    sourceML     : true,
  };
}

async function appellerAPIMLJour(jour, mois, jourCycle) {
  const payload = {
    temp_max_C           : jour.tMax,
    temp_min_C           : jour.tMin,
    temp_moy_C           : jour.tMoy,
    pluie_totale_mm      : jour.pluie,
    vent_u2_ms           : jour.ventU2,
    rayonnement_Rs_MJ    : jour.rayonnement,
    ET0_reference_mm     : jour.et0,
    humidite_sol_moy_pct : jour.humSol,
    humidite_sol_min_pct : jour.humSolMin,
    humidite_sol_0_7_moy : jour.humSol07,
    humidite_air_moy_pct : jour.humAir,
    RH_max               : jour.RHmax,
    RH_min               : jour.RHmin,
    mois                 : mois,
    jour_cycle           : jourCycle,
    source_sol           : "Open-Meteo",
  };

  const res = await fetch(`${API_URL}/decision`, {
    method  : "POST",
    headers : { "Content-Type": "application/json" },
    body    : JSON.stringify(payload),
  });
  const api = await res.json();

  return {
    date     : jour.date,
    irriguer : api.irriguer,
    volume   : Math.round(api.volume_L),
    pluie    : jour.pluie,
  };
}

// ── Calcul local FAO-56 (fallback sans API) ────────────────────────────────

function calculerDecisionLocale(meteo) {
  const KC         = 1.05;
  const EFFICIENCE = 0.90;
  const j          = meteo[0];

  const etc      = parseFloat((j.et0 * KC).toFixed(2));
  const pluieEff = j.pluie * 0.80;
  const deficit  = Math.max(0, etc - pluieEff);

  const doitIrriguer = j.pluie < PLUIE_SEUIL && j.humSol < SOL_SEUIL && deficit > 0.5;

  let volume = 0;
  if (doitIrriguer) {
    const facteur = Math.max(0, (65 - j.humSol) / 25);
    volume = Math.round(deficit * facteur * SURFACE / EFFICIENCE);
    volume = Math.max(volume, 50);
  }

  return {
    doitIrriguer,
    volume,
    deficit   : deficit.toFixed(1),
    et0       : j.et0,
    etc       : etc.toFixed(2),
    confiance : null,
    humSol    : j.humSol,
    tMax      : j.tMax,
    pluie     : j.pluie,
    stade     : "Mi-saison",
    kc        : KC,
    source    : "Calcul local FAO-56",
    raison    : `FAO-56 local | Deficit ${deficit.toFixed(1)}mm | Sol ${j.humSol}%`,
    previsions: meteo.slice(1).map(previsionLocale),
    timestamp : new Date().toISOString(),
    sourceML  : false,
  };
}

function previsionLocale(jour) {
  const KC         = 1.05;
  const EFFICIENCE = 0.90;
  const etcJ    = jour.et0 * KC;
  const deficitJ = Math.max(0, etcJ - jour.pluie * 0.80);
  const irriguerJ = jour.pluie < PLUIE_SEUIL && jour.humSol < SOL_SEUIL && deficitJ > 0.5;
  const facteur  = Math.max(0, (65 - jour.humSol) / 25);
  return {
    date     : jour.date,
    irriguer : irriguerJ,
    volume   : irriguerJ ? Math.round(deficitJ * facteur * SURFACE / EFFICIENCE) : 0,
    pluie    : jour.pluie,
  };
}

function valeursDSecours() {
  return [
    { date: dateOffset(0), tMax:33, tMin:23, tMoy:28, pluie:0,  ventU2:1.6, rayonnement:16, et0:5.0, humSol:38, humSolMin:33, humSol07:36, humAir:55, RHmax:75, RHmin:35 },
    { date: dateOffset(1), tMax:34, tMin:24, tMoy:29, pluie:0,  ventU2:1.3, rayonnement:15, et0:4.8, humSol:32, humSolMin:27, humSol07:30, humAir:50, RHmax:70, RHmin:30 },
    { date: dateOffset(2), tMax:31, tMin:22, tMoy:26, pluie:8,  ventU2:1.9, rayonnement:12, et0:3.5, humSol:55, humSolMin:50, humSol07:53, humAir:72, RHmax:88, RHmin:56 },
    { date: dateOffset(3), tMax:32, tMin:23, tMoy:27, pluie:2,  ventU2:1.5, rayonnement:14, et0:4.2, humSol:45, humSolMin:40, humSol07:43, humAir:62, RHmax:80, RHmin:44 },
  ];
}

// ── Affichage resultats ─────────────────────────────────────────────────────

function afficherResultats(r, meteo) {
  const j = meteo[0];

  // Carte decision
  const card = document.getElementById("decision-card");
  card.className = "decision-card " + (r.doitIrriguer ? "irrigate" : "no-irrigate");

  const iconWrap = document.getElementById("dec-icon-wrap");
  iconWrap.className = "dec-icon-wrap " + (r.doitIrriguer ? "oui" : "non");
  document.getElementById("dec-icon").textContent =
    r.doitIrriguer ? "OUI" : "NON";

  document.getElementById("dec-text").textContent =
    r.doitIrriguer ? "ARROSER\nAUJOURD'HUI" : "PAS D'ARROSAGE\nAUJOURD'HUI";

  document.getElementById("dec-vol").textContent = r.doitIrriguer
    ? `Volume recommande : ${r.volume} L`
    : "Sol suffisamment humide ou pluie prevue";

  // Badge confiance
  const confEl = document.getElementById("dec-confidence");
  if (r.sourceML && r.confiance !== null) {
    confEl.textContent = `ML ${r.confiance}%`;
    confEl.className   = "dec-confidence";
    confEl.title       = r.raison;
  } else {
    confEl.textContent = "Local";
    confEl.className   = "dec-confidence local";
    confEl.title       = r.raison;
  }

  // Chips
  document.getElementById("dec-chips").innerHTML = `
    <div class="dec-chip">Deficit ${r.deficit}mm</div>
    <div class="dec-chip">${r.tMax}°C</div>
    <div class="dec-chip">${r.stade || "Mi-saison"} · Kc ${parseFloat(r.kc).toFixed(2)}</div>
    <div class="dec-chip">${r.sourceML ? "ML actif" : "Calcul local"}</div>
  `;

  // Metriques
  document.getElementById("m-sol").textContent   = `${r.humSol}%`;
  document.getElementById("m-temp").textContent  = `${r.tMax}°`;
  document.getElementById("m-vent").textContent  = `${j.ventU2}`;
  document.getElementById("m-pluie").textContent = `${r.pluie}mm`;
  document.getElementById("m-et0").textContent   = `${r.et0}`;

  // Jauge
  document.getElementById("gauge-val").textContent  = `${r.humSol}%`;
  document.getElementById("gauge-fill").style.width = `${Math.min(r.humSol, 100)}%`;

  // Previsions
  const labels = ["Auj.", "Dem.", "J+2", "J+3"];
  const irrig  = [r.doitIrriguer, ...r.previsions.map(p => p.irriguer)];
  const vols   = [r.volume, ...r.previsions.map(p => p.volume)];

  document.getElementById("forecast-strip").innerHTML = labels.map((lbl, i) => `
    <div class="forecast-item ${i === 0 ? "today" : ""}">
      <div class="fc-day ${i === 0 ? "today-txt" : ""}">${lbl}</div>
      <div class="fc-dec ${irrig[i] ? "yes" : "no"}">${irrig[i] ? "OUI" : "NON"}</div>
      <div class="fc-vol">${irrig[i] ? vols[i] + "L" : "0L"}</div>
    </div>
  `).join("");

  document.getElementById("notif-badge").classList.add("visible");
  mettreAJourApercuSMS(r);
}

// ── Apercu SMS ─────────────────────────────────────────────────────────────

function mettreAJourApercuSMS(r) {
  const dateFr = new Date().toLocaleDateString("fr-FR");
  const heure  = new Date().toLocaleTimeString("fr-FR", { hour:"2-digit", minute:"2-digit" });
  const prev   = r.previsions;

  const lignesPrev = [0, 1, 2].map(i =>
    prev[i]
      ? `  J+${i+1} : ${prev[i].irriguer ? "OUI " + prev[i].volume + "L" : "NON 0L"}`
      : `  J+${i+1} : --`
  ).join("\n");

  document.getElementById("sms-preview").innerHTML = `
    <div class="sms-header">
      <div class="sms-avatar">MK</div>
      <div>
        <div class="sms-sender-name">M. Koffi</div>
        <div class="sms-time">Aujourd'hui · ${heure}</div>
      </div>
    </div>
    <div class="sms-bubble">Water5 CI - ${dateFr}
Stade    : ${(r.stade || "MI-SAISON").toUpperCase()}
Kc       : ${parseFloat(r.kc).toFixed(4)}
Decision : <strong>${r.doitIrriguer ? "ARROSER" : "NE PAS ARROSER"}</strong>
Volume   : <strong>${r.volume}L</strong>
Sol      : ${r.humSol}% | Pluie : ${r.pluie}mm
ET0      : ${r.et0}mm
Source   : ${r.sourceML ? "Modele ML" : "Calcul local"}

Previsions :
${lignesPrev}

Water5 v2.0 | ${dateFr}</div>
  `;
}

// ── Sauvegarde analyse ──────────────────────────────────────────────────────

function sauvegarderAnalyse(r, meteo) {
  const entry = {
    id       : Date.now(),
    date     : new Date().toISOString(),
    irriguer : r.doitIrriguer,
    volume   : r.volume,
    confiance: r.sourceML ? r.confiance : "local",
    humSol   : r.humSol,
    tMax     : r.tMax,
    pluie    : r.pluie,
    et0      : r.et0,
    deficit  : r.deficit,
    stade    : r.stade,
    kc       : r.kc,
    sourceML : r.sourceML,
  };

  state.history.unshift(entry);
  if (state.history.length > 50) state.history.pop();
  state.lastAnalysis = entry;

  ajouterAlerte(
    r.doitIrriguer ? "green" : "orange",
    "Decision disponible",
    r.doitIrriguer
      ? `Irrigation recommandee · ${r.volume}L${r.sourceML ? ` · ML ${r.confiance}%` : ""}`
      : `Pas d'irrigation · Pluie ${r.pluie}mm · Sol ${r.humSol}%`,
    new Date()
  );

  if (r.humSol < 35) {
    ajouterAlerte("orange", "Sol sec detecte",
      `Humidite sol a ${r.humSol}% — sous le seuil optimal (${SOL_SEUIL}%)`, new Date());
  }
  if (r.previsions && r.previsions.some(p => p.pluie > 8)) {
    ajouterAlerte("blue", "Pluie prevue",
      "Forte pluie dans les prochains jours · Irrigation ajustee automatiquement", new Date());
  }

  sauvegarderDonnees();
  mettreAJourHistorique();
  mettreAJourAlertes();
}

function ajouterAlerte(couleur, titre, msg, date) {
  state.alerts.unshift({ couleur, titre, msg, date: date.toISOString(), nouveau: true });
  if (state.alerts.length > 30) state.alerts.pop();
}

// ── Historique ──────────────────────────────────────────────────────────────

function mettreAJourHistorique() {
  const hist = state.history;

  const nbIrrig     = hist.filter(h => h.irriguer).length;
  const totalLitres = hist.filter(h => h.irriguer).reduce((s, h) => s + (h.volume || 0), 0);
  const mlEntries   = hist.filter(h => h.sourceML && h.confiance !== "local");
  const precision   = mlEntries.length > 0
    ? (mlEntries.reduce((s, h) => s + parseFloat(h.confiance), 0) / mlEntries.length).toFixed(1) + "%"
    : "--";

  document.getElementById("stat-irrig").textContent  = nbIrrig;
  document.getElementById("stat-litres").textContent = totalLitres > 999
    ? (totalLitres / 1000).toFixed(1) + "k"
    : totalLitres;
  document.getElementById("stat-precision").textContent = precision;

  const mois = new Date().toLocaleDateString("fr-FR", { month: "long", year: "numeric" });
  document.getElementById("filter-label").textContent =
    mois.charAt(0).toUpperCase() + mois.slice(1);

  const container = document.getElementById("hist-list");
  if (!hist.length) {
    container.innerHTML = `<div class="hist-empty">Aucune analyse effectuee.<br>Lancez votre premiere analyse depuis l'accueil.</div>`;
    return;
  }

  let html     = "";
  let lastDate = "";

  hist.forEach(entry => {
    const d       = new Date(entry.date);
    const dateStr = d.toLocaleDateString("fr-FR", { weekday:"long", day:"numeric", month:"long" });
    const heure   = d.toLocaleTimeString("fr-FR", { hour:"2-digit", minute:"2-digit" });
    const auj     = new Date().toLocaleDateString("fr-FR", { weekday:"long", day:"numeric", month:"long" });

    if (dateStr !== lastDate) {
      html += `<div class="hist-date-sep">${dateStr === auj ? "Aujourd'hui" : dateStr}</div>`;
      lastDate = dateStr;
    }

    const confTxt = entry.sourceML && entry.confiance !== "local"
      ? `ML ${entry.confiance}%`
      : "Calcul local";

    html += `
      <div class="hist-item ${entry.irriguer ? "irrigated" : "not-irrigated"}">
        <div class="hist-icon-wrap ${entry.irriguer ? "green" : "red"}">
          ${entry.irriguer ? "OUI" : "NON"}
        </div>
        <div class="hist-content">
          <div class="hist-item-title">${entry.irriguer ? "Irrigation effectuee" : "Pas d'irrigation"}</div>
          <div class="hist-item-sub">${confTxt} · ${entry.irriguer ? "Deficit " + entry.deficit + "mm" : "Pluie " + entry.pluie + "mm"} · ${heure}</div>
          <div class="hist-item-meta">
            <span class="hist-tag green">${entry.stade || "Mi-saison"}</span>
            <span class="hist-tag orange">Kc ${entry.kc ? parseFloat(entry.kc).toFixed(2) : "1.05"}</span>
            <span class="hist-tag blue">Sol ${entry.humSol}%</span>
          </div>
        </div>
        <div class="hist-vol ${entry.irriguer ? "" : "red"}">${entry.irriguer ? entry.volume + "L" : "0L"}</div>
      </div>`;
  });

  container.innerHTML = html;
}

// ── Alertes ──────────────────────────────────────────────────────────────────

function mettreAJourAlertes() {
  const alertes   = state.alerts;
  const nouvelles = alertes.filter(a => a.nouveau).length;
  document.getElementById("alerts-count").textContent = nouvelles > 0 ? nouvelles + " nouv." : "0";

  const container = document.getElementById("alerts-list");
  if (!alertes.length) {
    container.innerHTML = `<div class="hist-empty">Aucune notification.</div>`;
    return;
  }

  container.innerHTML = alertes.map(a => {
    const heure   = new Date(a.date).toLocaleTimeString("fr-FR", { hour:"2-digit", minute:"2-digit" });
    const dateStr = new Date(a.date).toLocaleDateString("fr-FR") === new Date().toLocaleDateString("fr-FR")
      ? `Aujourd'hui · ${heure}`
      : `${new Date(a.date).toLocaleDateString("fr-FR", { day:"numeric", month:"long" })} · ${heure}`;

    return `
      <div class="alert-item">
        ${a.nouveau ? '<div class="alert-new-dot"></div>' : ""}
        <div class="alert-icon ${a.couleur}"></div>
        <div class="alert-content">
          <div class="alert-title">${a.titre}</div>
          <div class="alert-msg">${a.msg}</div>
          <div class="alert-time">${dateStr}</div>
        </div>
      </div>`;
  }).join("");

  state.alerts.forEach(a => (a.nouveau = false));
  sauvegarderDonnees();
}

// ── Notifications ────────────────────────────────────────────────────────────

function toggleNotifications() {
  const toggle = document.getElementById("notif-toggle");

  if (!state.notificationsEnabled) {
    if ("Notification" in window) {
      Notification.requestPermission().then(perm => {
        if (perm === "granted") {
          state.notificationsEnabled = true;
          toggle.classList.add("on");
          programmerNotification();
          sauvegarderDonnees();
        }
      });
    } else {
      state.notificationsEnabled = true;
      toggle.classList.add("on");
      sauvegarderDonnees();
    }
  } else {
    state.notificationsEnabled = false;
    toggle.classList.remove("on");
    sauvegarderDonnees();
  }
}

function programmerNotification() {
  const demain6h = new Date();
  demain6h.setDate(demain6h.getDate() + 1);
  demain6h.setHours(6, 0, 0, 0);

  setTimeout(() => {
    if (state.notificationsEnabled && Notification.permission === "granted") {
      new Notification("Water5", {
        body: "Votre analyse quotidienne est prete.",
        icon: "icons/icon.png",
      });
    }
    programmerNotification();
  }, demain6h - new Date());
}

// ── Reset ─────────────────────────────────────────────────────────────────────

function resetData() {
  if (!confirm("Effacer toutes les donnees ? Cette action est irreversible.")) return;

  state = {
    history              : [],
    alerts               : [],
    notificationsEnabled : false,
    lastAnalysis         : null,
    apiConnectee         : state.apiConnectee,
  };
  sauvegarderDonnees();
  mettreAJourHistorique();
  mettreAJourAlertes();

  const resets = {
    "dec-icon"       : "?",
    "dec-text"       : "En attente d'analyse",
    "dec-vol"        : "Appuyez sur Analyser",
    "dec-confidence" : "--",
    "m-sol"          : "--%",
    "m-temp"         : "--°",
    "m-vent"         : "--",
    "m-pluie"        : "--mm",
    "m-et0"          : "--",
    "gauge-val"      : "--%",
  };
  Object.entries(resets).forEach(([id, val]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  });
  document.getElementById("gauge-fill").style.width = "0%";
  document.getElementById("forecast-strip").innerHTML =
    '<div class="forecast-placeholder">Lancez une analyse pour voir les previsions</div>';
  document.getElementById("sms-preview").innerHTML =
    '<div class="hist-empty">Aucune analyse effectuee.</div>';
  document.getElementById("notif-badge").classList.remove("visible");
}

// ── Utilitaires ───────────────────────────────────────────────────────────────

function dateOffset(n) {
  const d = new Date();
  d.setDate(d.getDate() + n);
  return d.toISOString().split("T")[0];
}