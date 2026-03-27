// app.js — Water5 v3.0
// Appelle uniquement GET /analyser -> resultats identiques a 06_api_openmeteo.py

const API_URL = "http://localhost:8000";

// ── Etat ──────────────────────────────────────────────────────────────────

let state = {
  history              : [],
  alerts               : [],
  notificationsEnabled : false,
  apiOk                : false,
};

// ── Init ──────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  charger();
  verifierAPI();
  renderHistorique();
  renderAlertes();
});

function charger() {
  try {
    const s = localStorage.getItem("w5");
    if (s) {
      const p = JSON.parse(s);
      state.history = Array.isArray(p.history) ? p.history : [];
      state.alerts  = Array.isArray(p.alerts)  ? p.alerts  : [];
      state.notificationsEnabled = !!p.notificationsEnabled;
    }
  } catch (e) {}
}

function sauver() {
  try { localStorage.setItem("w5", JSON.stringify(state)); } catch (e) {}
}

// ── Verification API ──────────────────────────────────────────────────────

async function verifierAPI() {
  const dot    = document.getElementById("api-dot");
  const label  = document.getElementById("api-label");
  const sModele = document.getElementById("s-modele");

  try {
    const res  = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    state.apiOk = !!(data.clf_charge && data.reg_charge);

    if (state.apiOk) {
      setEl(dot,    "className", "api-dot ok");
      setEl(label,  "textContent", "ML actif");
      setEl(sModele, "textContent", "Random Forest actif");
      if (sModele) sModele.className = "val-green";
    } else {
      setEl(dot,    "className", "api-dot local");
      setEl(label,  "textContent", "Modeles absents");
      setEl(sModele, "textContent", "Modeles non charges");
    }
  } catch (e) {
    state.apiOk = false;
    setEl(dot,    "className", "api-dot error");
    setEl(label,  "textContent", "Hors-ligne");
    setEl(sModele, "textContent", "API non connectee");
  }
}

function setEl(el, prop, val) { if (el) el[prop] = val; }

// ── Navigation ─────────────────────────────────────────────────────────────

function goToDashboard() {
  document.getElementById("screen-splash").classList.remove("active");
  document.getElementById("screen-main").classList.add("active");
}

function showTab(tab) {
  const map = {
    dashboard : "screen-main",
    history   : "screen-history",
    alerts    : "screen-alerts",
    settings  : "screen-settings",
  };

  Object.values(map).forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove("active");
  });

  const target = document.getElementById(map[tab]);
  if (target) target.classList.add("active");

  // Nav actif dans chaque ecran
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.classList.toggle(
      "active",
      btn.getAttribute("onclick") === `showTab('${tab}')`
    );
  });

  if (tab === "history") renderHistorique();
  if (tab === "alerts")  renderAlertes();
}

// ── ANALYSE — appel unique a /analyser ────────────────────────────────────

async function lancerAnalyse() {
  const btn = document.getElementById("btn-analyze");
  if (btn) btn.disabled = true;

  const overlay = document.getElementById("overlay");
  if (overlay) overlay.classList.remove("hidden");
  ["step1","step2","step3","step4"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.className = "step";
  });

  try {
    // Etape 1 : verifier API
    activerStep("step1");
    await verifierAPI();
    await pause(300);

    if (!state.apiOk) {
      throw new Error("API non connectee. Lancez api.py dans le terminal.");
    }

    // Etape 2 : Open-Meteo (fait par le serveur)
    activerStep("step2");
    await pause(300);

    // Etape 3 : appel /analyser — une seule requete, tout est calcule cote serveur
    activerStep("step3");
    const analyse = await fetch(`${API_URL}/analyser?jour_cycle=60`, {
      signal: AbortSignal.timeout(20000),
    });

    if (!analyse.ok) {
      const err = await analyse.json().catch(() => ({}));
      throw new Error(err.detail || `Erreur serveur ${analyse.status}`);
    }

    const data = await analyse.json();
    await pause(300);

    // Etape 4 : sauvegarde et affichage
    activerStep("step4");
    sauverAnalyse(data);
    afficher(data);
    await pause(200);

    if (overlay) overlay.classList.add("hidden");

  } catch (err) {
    if (overlay) overlay.classList.add("hidden");
    console.error("Erreur analyse :", err);
    ajouterAlerte("red", "Erreur", err.message, new Date());
    renderAlertes();
    alert("Erreur : " + err.message);
  }

  if (btn) btn.disabled = false;
}

function activerStep(id) {
  document.querySelectorAll(".step").forEach(s => {
    if (s.classList.contains("active")) s.className = "step done";
  });
  const el = document.getElementById(id);
  if (el) el.classList.add("active");
}

const pause = ms => new Promise(r => setTimeout(r, ms));

// ── Affichage des resultats ───────────────────────────────────────────────

function afficher(data) {
  const a = data.aujourd_hui;

  // Carte decision
  const card = document.getElementById("card-decision");
  if (card) {
    card.className = "card-decision " + (a.irriguer ? "irrigate" : "no-irrigate");
  }

  const icon = document.getElementById("dec-icon");
  if (icon) {
    icon.textContent  = a.irriguer ? "OUI" : "NON";
    icon.className    = "decision-icon" + (a.irriguer ? "" : " non");
  }

  setEl(document.getElementById("dec-title"), "textContent",
    a.irriguer ? "ARROSER\nAUJOURD'HUI" : "PAS D'ARROSAGE\nAUJOURD'HUI");

  setEl(document.getElementById("dec-vol"), "textContent",
    a.irriguer
      ? `Volume recommande : ${Math.round(a.volume_L)} L`
      : "Sol suffisamment humide ou pluie prevue");

  // Badge ML
  const badge = document.getElementById("badge-ml");
  if (badge) {
    badge.textContent = `ML ${a.confiance_pct.toFixed(1)}%`;
  }

  // Chips
  const chips = document.getElementById("dec-chips");
  if (chips) {
    chips.innerHTML = [
      `Deficit ${a.deficit_mm.toFixed(1)}mm`,
      `${a.temp_max_C.toFixed(0)} C`,
      `${a.stade || "Mi-saison"} · Kc ${a.kc.toFixed(2)}`,
      a.source.includes("Random") ? "Modele ML" : "Regle agronomique",
    ].map(t => `<span class="chip">${t}</span>`).join("");
  }

  // Metriques
  setEl(document.getElementById("m-sol"),   "textContent", `${a.humidite_sol.toFixed(0)}%`);
  setEl(document.getElementById("m-temp"),  "textContent", `${a.temp_max_C.toFixed(0)}°`);
  setEl(document.getElementById("m-vent"),  "textContent", `${a.vent_u2_ms.toFixed(2)}`);
  setEl(document.getElementById("m-pluie"), "textContent", `${a.pluie_mm.toFixed(1)}mm`);
  setEl(document.getElementById("m-et0"),   "textContent", `${a.ET0_mm.toFixed(2)}`);
  setEl(document.getElementById("m-etc"),   "textContent", `${a.ETc_mm.toFixed(2)}`);

  // Jauge sol
  const pct = Math.min(Math.max(a.humidite_sol, 0), 100);
  setEl(document.getElementById("gauge-pct"),    "textContent", `${pct.toFixed(0)}%`);
  setEl(document.getElementById("gauge-source"), "textContent", data.source_meteo || "Open-Meteo (serveur)");
  const bar = document.getElementById("gauge-bar");
  if (bar) bar.style.width = `${pct}%`;

  setEl(document.getElementById("gauge-detail"), "textContent",
    `Deficit : ${a.deficit_mm.toFixed(2)} mm  |  Kc : ${a.kc.toFixed(4)}  |  Stade : ${a.stade || "mi_saison"}`);

  // Previsions
  const strip = document.getElementById("forecast-strip");
  if (strip) {
    const jours  = [
      { lbl: "Auj.",  today: true,  jour: a },
      ...data.previsions.map((p, i) => ({
        lbl: ["Dem.","J+2","J+3"][i] || `J+${i+2}`,
        today: false,
        jour: p,
      })),
    ];
    strip.innerHTML = jours.map(({ lbl, today, jour }) => `
      <div class="forecast-card ${today ? "today" : ""}">
        <div class="fc-day ${today ? "today" : ""}">${lbl}</div>
        <div class="fc-badge ${jour.irriguer ? "yes" : "no"}">${jour.irriguer ? "OUI" : "NON"}</div>
        <div class="fc-vol">${jour.irriguer ? Math.round(jour.volume_L) + "L" : "0L"}</div>
      </div>
    `).join("");
  }

  // Apercu SMS
  renderSMS(data);
}

// ── Apercu SMS ────────────────────────────────────────────────────────────

function renderSMS(data) {
  const a      = data.aujourd_hui;
  const dateFr = new Date().toLocaleDateString("fr-FR");
  const heure  = new Date().toLocaleTimeString("fr-FR", { hour:"2-digit", minute:"2-digit" });

  const prevLines = data.previsions.map((p, i) =>
    `  J+${i+1} : ${p.irriguer ? "OUI " + Math.round(p.volume_L) + "L" : "NON 0L"}`
  ).join("\n");

  const corps = [
    `Water5 CI - ${dateFr}`,
    `Stade    : ${(a.stade || "mi_saison").toUpperCase()}`,
    `Kc       : ${a.kc.toFixed(4)}`,
    `Decision : ${a.irriguer ? "ARROSER" : "NE PAS ARROSER"}`,
    `Volume   : ${Math.round(a.volume_L)}L`,
    `Sol      : ${a.humidite_sol.toFixed(1)}%`,
    `Pluie    : ${a.pluie_mm.toFixed(1)}mm  |  ET0 : ${a.ET0_mm.toFixed(2)}mm`,
    ``,
    `Previsions :`,
    prevLines,
    ``,
    `Water5 v3.0 | ${dateFr}`,
  ].join("\n");

  const box = document.getElementById("sms-preview");
  if (!box) return;
  box.innerHTML = `
    <div class="sms-header">
      <div class="sms-av">MK</div>
      <div>
        <div class="sms-name">M. Koffi</div>
        <div class="sms-time">Aujourd'hui · ${heure}</div>
      </div>
    </div>
    <div class="sms-body">${escHtml(corps)}</div>
  `;
}

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ── Sauvegarde ────────────────────────────────────────────────────────────

function sauverAnalyse(data) {
  const a = data.aujourd_hui;
  state.history.unshift({
    id        : Date.now(),
    date      : new Date().toISOString(),
    irriguer  : a.irriguer,
    volume    : Math.round(a.volume_L),
    confiance : a.confiance_pct,
    humSol    : a.humidite_sol,
    tMax      : a.temp_max_C,
    pluie     : a.pluie_mm,
    et0       : a.ET0_mm,
    deficit   : a.deficit_mm,
    stade     : a.stade,
    kc        : a.kc,
  });
  if (state.history.length > 60) state.history.pop();

  ajouterAlerte(
    a.irriguer ? "green" : "orange",
    "Decision disponible",
    a.irriguer
      ? `Irrigation recommandee · ${Math.round(a.volume_L)}L · ML ${a.confiance_pct.toFixed(1)}%`
      : `Pas d'irrigation · Pluie ${a.pluie_mm.toFixed(1)}mm · Sol ${a.humidite_sol.toFixed(0)}%`,
    new Date()
  );

  if (a.humidite_sol < 35) {
    ajouterAlerte("orange", "Sol sec", `Humidite a ${a.humidite_sol.toFixed(0)}% — sous le seuil de 65%`, new Date());
  }
  if (data.previsions.some(p => p.pluie_mm > 8)) {
    ajouterAlerte("blue", "Pluie prevue", "Forte pluie dans les prochains jours — irrigation ajustee", new Date());
  }

  sauver();
}

function ajouterAlerte(couleur, titre, msg, date) {
  state.alerts.unshift({ couleur, titre, msg, date: date.toISOString(), nouveau: true });
  if (state.alerts.length > 40) state.alerts.pop();
}

// ── Render historique ─────────────────────────────────────────────────────

function renderHistorique() {
  const hist = state.history;

  const nbI    = hist.filter(h => h.irriguer).length;
  const totL   = hist.filter(h => h.irriguer).reduce((s,h) => s + h.volume, 0);
  const mlEntr = hist.filter(h => h.confiance != null);
  const prec   = mlEntr.length > 0
    ? (mlEntr.reduce((s,h) => s + h.confiance, 0) / mlEntr.length).toFixed(1) + "%"
    : "--";

  setEl(document.getElementById("h-irrig"),  "textContent", nbI);
  setEl(document.getElementById("h-litres"), "textContent",
    totL > 999 ? (totL/1000).toFixed(1)+"k" : totL);
  setEl(document.getElementById("h-prec"),   "textContent", prec);

  const mois = new Date().toLocaleDateString("fr-FR", { month:"long", year:"numeric" });
  setEl(document.getElementById("hist-month"), "textContent",
    mois.charAt(0).toUpperCase() + mois.slice(1));

  const container = document.getElementById("hist-list");
  if (!container) return;

  if (!hist.length) {
    container.innerHTML = '<div class="empty-msg">Aucune analyse effectuee.<br>Lancez votre premiere analyse depuis l\'accueil.</div>';
    return;
  }

  let html = "";
  let lastDate = "";

  hist.forEach(e => {
    const d       = new Date(e.date);
    const dateStr = d.toLocaleDateString("fr-FR", { weekday:"long", day:"numeric", month:"long" });
    const heure   = d.toLocaleTimeString("fr-FR", { hour:"2-digit", minute:"2-digit" });
    const auj     = new Date().toLocaleDateString("fr-FR", { weekday:"long", day:"numeric", month:"long" });

    if (dateStr !== lastDate) {
      html += `<div class="hist-sep">${dateStr === auj ? "Aujourd'hui" : dateStr}</div>`;
      lastDate = dateStr;
    }

    html += `
      <div class="hist-item ${e.irriguer ? "irrig" : "no-irrig"}">
        <div class="hist-badge ${e.irriguer ? "green" : "red"}">${e.irriguer ? "OUI" : "NON"}</div>
        <div class="hist-info">
          <div class="hist-title">${e.irriguer ? "Irrigation effectuee" : "Pas d'irrigation"}</div>
          <div class="hist-sub">ML ${e.confiance ? e.confiance.toFixed(1)+"%" : "--"} · ${e.irriguer ? "Deficit "+e.deficit.toFixed(1)+"mm" : "Pluie "+e.pluie.toFixed(1)+"mm"} · ${heure}</div>
          <div class="hist-tags">
            <span class="hist-tag g">${e.stade || "mi_saison"}</span>
            <span class="hist-tag a">Kc ${e.kc ? e.kc.toFixed(2) : "--"}</span>
            <span class="hist-tag s">Sol ${e.humSol ? e.humSol.toFixed(0) : "--"}%</span>
          </div>
        </div>
        <div class="hist-vol ${e.irriguer ? "green" : "red"}">${e.irriguer ? e.volume+"L" : "0L"}</div>
      </div>`;
  });

  container.innerHTML = html;
}

// ── Render alertes ────────────────────────────────────────────────────────

function renderAlertes() {
  const alertes   = state.alerts;
  const nouvelles = alertes.filter(a => a.nouveau).length;

  const badge = document.getElementById("alerts-badge");
  if (badge) badge.textContent = nouvelles > 0 ? nouvelles + " nouv." : "0";

  const container = document.getElementById("alerts-list");
  if (!container) return;

  if (!alertes.length) {
    container.innerHTML = '<div class="empty-msg">Aucune notification.</div>';
    return;
  }

  container.innerHTML = alertes.map(a => {
    const heure   = new Date(a.date).toLocaleTimeString("fr-FR", { hour:"2-digit", minute:"2-digit" });
    const dateFr  = new Date(a.date).toLocaleDateString("fr-FR");
    const aujFr   = new Date().toLocaleDateString("fr-FR");
    const dateStr = dateFr === aujFr
      ? `Aujourd'hui · ${heure}`
      : `${new Date(a.date).toLocaleDateString("fr-FR", {day:"numeric",month:"long"})} · ${heure}`;

    // Icone textuelle selon la couleur
    const icons = { green:"OK", orange:"!", red:"X", blue:"i" };
    const ico   = icons[a.couleur] || "?";

    return `
      <div class="alert-item">
        ${a.nouveau ? '<div class="alert-new"></div>' : ""}
        <div class="alert-dot-icon ${a.couleur}">${ico}</div>
        <div class="alert-content">
          <div class="alert-title">${a.titre}</div>
          <div class="alert-msg">${a.msg}</div>
          <div class="alert-time">${dateStr}</div>
        </div>
      </div>`;
  }).join("");

  state.alerts.forEach(a => (a.nouveau = false));
  sauver();
}

// ── Notifications ─────────────────────────────────────────────────────────

function toggleNotif() {
  const btn = document.getElementById("notif-toggle");

  if (!state.notificationsEnabled) {
    if ("Notification" in window) {
      Notification.requestPermission().then(p => {
        if (p === "granted") {
          state.notificationsEnabled = true;
          if (btn) btn.classList.add("on");
          programmerNotif();
          sauver();
        }
      });
    } else {
      state.notificationsEnabled = true;
      if (btn) btn.classList.add("on");
      sauver();
    }
  } else {
    state.notificationsEnabled = false;
    if (btn) btn.classList.remove("on");
    sauver();
  }
}

function programmerNotif() {
  const d = new Date();
  d.setDate(d.getDate() + 1);
  d.setHours(6, 0, 0, 0);
  setTimeout(() => {
    if (state.notificationsEnabled && Notification.permission === "granted") {
      new Notification("Water5", { body: "Votre analyse quotidienne est prete." });
    }
    programmerNotif();
  }, d - new Date());
}

// ── Reset ─────────────────────────────────────────────────────────────────

function resetData() {
  if (!confirm("Effacer toutes les donnees ? Cette action est irreversible.")) return;

  state.history = [];
  state.alerts  = [];
  sauver();
  renderHistorique();
  renderAlertes();

  // Reset dashboard
  const card = document.getElementById("card-decision");
  if (card) card.className = "card-decision";

  const icon = document.getElementById("dec-icon");
  if (icon) { icon.textContent = "?"; icon.className = "decision-icon"; }

  setEl(document.getElementById("dec-title"),    "textContent", "En attente d'analyse");
  setEl(document.getElementById("dec-vol"),      "textContent", "Appuyez sur Analyser");
  setEl(document.getElementById("badge-ml"),     "textContent", "--");
  setEl(document.getElementById("m-sol"),        "textContent", "--%");
  setEl(document.getElementById("m-temp"),       "textContent", "--");
  setEl(document.getElementById("m-vent"),       "textContent", "--");
  setEl(document.getElementById("m-pluie"),      "textContent", "--");
  setEl(document.getElementById("m-et0"),        "textContent", "--");
  setEl(document.getElementById("m-etc"),        "textContent", "--");
  setEl(document.getElementById("gauge-pct"),    "textContent", "--%");
  setEl(document.getElementById("gauge-detail"), "textContent", "Deficit : -- mm  |  Kc : --  |  Stade : --");

  const bar = document.getElementById("gauge-bar");
  if (bar) bar.style.width = "0%";

  const strip = document.getElementById("forecast-strip");
  if (strip) strip.innerHTML = '<div class="forecast-empty">Lancez une analyse pour voir les previsions</div>';

  const sms = document.getElementById("sms-preview");
  if (sms) sms.innerHTML = '<div class="sms-empty">Aucune analyse effectuee.</div>';

  const chips = document.getElementById("dec-chips");
  if (chips) chips.innerHTML = `
    <span class="chip">Deficit --</span>
    <span class="chip">-- C</span>
    <span class="chip">Mi-saison</span>`;
}
