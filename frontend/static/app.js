
const map = L.map("map").setView([52.52, 13.405], 12);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap-Mitwirkende",
}).addTo(map);

const markers = L.featureGroup().addTo(map);
let deleteMode = false;
let addPointMode = false;
let measureMode = false;
let currentJob = null;
let rasterLayer = null;

const exportBtn = document.getElementById("exportGeoJSON");
if (exportBtn) {
  exportBtn.addEventListener("click", () => {
    const geojson = markers.toGeoJSON();
    const geojsonStr = JSON.stringify(geojson, null, 2);
    const blob = new Blob([geojsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "points.geojson";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}


function colorForDb(db) {
  if (db <= 60) return "#2ecc71";
  if (db <= 75) return "#f39c12";
  return "#e74c3c";
}

function iconForDb(db) {
  const color = colorForDb(db);
  return L.divIcon({
    className: "custom-marker",
    html: `<div style="width:14px;height:14px;border-radius:50%;background:${color};
            border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.3)"></div>`,
    iconAnchor: [7, 7],
  });
}


async function fetchPoints() {
  const username = sessionStorage.getItem("username");
  if (!username) throw new Error("Kein Benutzername gefunden – bitte einloggen.");

  const res = await fetch(`/api/points/${username}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Fehler beim Laden der Punkte");
  }

  const data = await res.json();
  return data.features.map((f) => ({
    id: f.properties.id,
    lat: f.geometry.coordinates[1],
    lon: f.geometry.coordinates[0],
    db_value: f.properties.db_value,
    note: f.properties.note,
  }));
}

async function savePointToAPI(lat, lon, db, note) {
  const username = sessionStorage.getItem("username");
  if (!username) throw new Error("Kein Benutzername im SessionStorage – bitte neu einloggen.");

  const res = await fetch("/api/points", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lat, lon, db_value: db, note, username }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Fehler beim Speichern des Punkts");
  }
  return res.json();
}

async function deletePointFromAPI(id) {
  const res = await fetch(`/api/points/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Fehler beim Löschen des Punkts");
}


function addPoint(latlng, db, note = "", id = null) {
  const m = L.marker(latlng, { icon: iconForDb(db) });
  m.db = db;
  m.note = note;
  m.id = id;
  m.bindPopup(`<b>${db} dB</b>${note ? `<br>${note}` : ""}`);

  m.on("click", async () => {
    if (!deleteMode) return;
    if (confirm("Punkt löschen?")) {
      markers.removeLayer(m);
      if (m.id) {
        try {
          await deletePointFromAPI(m.id);
        } catch (err) {
          alert("Fehler beim Löschen: " + err.message);
        }
      }
    }
  });

  markers.addLayer(m);
}


map.on("dblclick", async (e) => {
  if (measureMode) return;

  const lat = e.latlng.lat;
  const lon = e.latlng.lng;
  const dbStr = prompt("dB-Wert für diesen Punkt eingeben (z. B. 65):");
  if (dbStr === null) return;

  const db = Number(dbStr);
  if (!Number.isFinite(db)) {
    alert("Bitte eine Zahl eingeben.");
    return;
  }

  const note = prompt("Optionale Notiz (Straße, Quelle, etc.):") || "";

  try {
    const saved = await savePointToAPI(lat, lon, db, note);
    addPoint([saved.lat, saved.lon], saved.db_value, saved.note, saved.id);
  } catch (err) {
    alert("Fehler beim Speichern: " + err.message);
  }
});

const toggleBtn = document.getElementById("toggleDelete");
if (toggleBtn) {
  toggleBtn.addEventListener("click", () => {
    deleteMode = !deleteMode;
    toggleBtn.textContent = `Löschmodus: ${deleteMode ? "AN" : "AUS"}`;
    toggleBtn.classList.toggle("active", deleteMode);
  });
}

let measureStart = null;
let measureTempLine = null;
let measureLine = null;
let isMeasuring = false;

const measureBtn = document.getElementById("btnMeasure");

if (measureBtn) {
  measureBtn.addEventListener("click", () => {
    measureMode = !measureMode;
    measureBtn.classList.toggle("active", measureMode);
    measureBtn.textContent = measureMode ? "Messmodus: AN" : "Entfernung messen";
    if (!measureMode) resetMeasurement();
  });
}

function resetMeasurement() {
  isMeasuring = false;
  measureStart = null;
  if (measureTempLine) map.removeLayer(measureTempLine);
  if (measureLine) map.removeLayer(measureLine);
  measureTempLine = null;
  measureLine = null;
  map.closePopup();
}


map.on("click", (e) => {
  if (!measureMode) return;

  if (!isMeasuring) {
    
    isMeasuring = true;
    measureStart = e.latlng;

    
    measureTempLine = L.polyline([measureStart, measureStart], {
      color: "#81c784",
      weight: 3,
      dashArray: "6,4",
    }).addTo(map);
  } else {

    const measureEnd = e.latlng;
    if (measureTempLine) map.removeLayer(measureTempLine);
    if (measureLine) map.removeLayer(measureLine);

    measureLine = L.polyline([measureStart, measureEnd], {
      color: "#4caf50",
      weight: 3,
    }).addTo(map);

  
    const distance = measureStart.distanceTo(measureEnd);
    const unit = distance >= 1000 ? "km" : "m";
    const value = distance >= 1000 ? (distance / 1000).toFixed(2) : distance.toFixed(1);


    L.popup({ closeButton: false, offset: L.point(0, -10) })
      .setLatLng(measureEnd)
      .setContent(`<b>${value} ${unit}</b>`)
      .openOn(map);

    isMeasuring = false;
  }
});


map.on("mousemove", (e) => {
  if (measureMode && isMeasuring && measureTempLine) {
    measureTempLine.setLatLngs([measureStart, e.latlng]);
  }
});


map.on("contextmenu", () => {
  if (measureMode) resetMeasurement();
});



const processBtn = document.getElementById("btnProcess");

if (processBtn) {
  processBtn.addEventListener("click", async () => {
    const username = sessionStorage.getItem("username");

    if (!username) {
      alert("Kein Benutzername gefunden – bitte zuerst einloggen!");
      return;
    }

    processBtn.disabled = true;
    processBtn.textContent = "Berechnung läuft...";

    try {
      const resp = await fetch("/api/process/sound-raster", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: username,
          cell_size_m: 10,
          buffer_m: 1000,
          alpha_db_per_m: 0.005,
          max_distance_m: 1500,
        }),
      });

      if (!resp.ok) throw new Error("Fehler beim Starten des Prozesses");

      const { job_id } = await resp.json();
      currentJob = job_id;
      pollJobStatus(job_id);
    } catch (err) {
      alert("Fehler beim Starten: " + err.message);
      processBtn.disabled = false;
      processBtn.textContent = "Sound-Raster berechnen";
    }
  });
}

async function pollJobStatus(job_id) {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/process/status/${job_id}`);
      const data = await res.json();

      if (data.status === "done") {
        clearInterval(interval);
        processBtn.disabled = false;
        processBtn.textContent = "Sound-Raster berechnen";
        addWcsLayer(data.layer_name);
      } else if (data.status === "error" || data.status === "unknown") {
        clearInterval(interval);
        processBtn.disabled = false;
        processBtn.textContent = "Fehler – erneut versuchen";
        alert(data.message || "Unbekannter Fehler beim Processing");
      } else {
        console.log("Status:", data.status);
      }
    } catch (err) {
      clearInterval(interval);
      processBtn.disabled = false;
      processBtn.textContent = "Fehler beim Abrufen";
      alert("Fehler beim Statusabruf: " + err.message);
    }
  }, 1500);
}


function createLegendGradient(min, max) {
  const legendContainer = document.getElementById("wcsLegendContainer");
  const legendImg = document.getElementById("wcsLegendImg");
  if (!legendContainer || !legendImg) return;

  const canvas = document.createElement("canvas");
  canvas.width = 90;
  canvas.height = 220;
  const ctx = canvas.getContext("2d");

  const gradient = ctx.createLinearGradient(0, canvas.height - 10, 0, 10);
  for (let i = 0; i <= 1; i += 0.01) {
    const hue = 120 - 120 * i;
    const lightness = 50 - i * 15;
    gradient.addColorStop(i, `hsl(${hue}, 100%, ${lightness}%)`);
  }

  const barX = 30;
  const barWidth = 24;
  const barY = 10;
  const barHeight = canvas.height - 20;

  ctx.fillStyle = gradient;
  ctx.fillRect(barX, barY, barWidth, barHeight);
  ctx.strokeStyle = "rgba(255,255,255,0.3)";
  ctx.strokeRect(barX, barY, barWidth, barHeight);

  ctx.fillStyle = "#f8f8f8";
  ctx.font = "bold 14px 'Segoe UI', Roboto, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";

  const numTicks = 3;
  const step = (max - min) / (numTicks - 1);
  for (let i = 0; i < numTicks; i++) {
    const value = max - step * i;
    const y = barY + (barHeight / (numTicks - 1)) * i;
    ctx.fillText(value.toFixed(0), barX + barWidth + 10, y);
  }

  ctx.font = "12px 'Segoe UI', sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("dB", barX + barWidth / 2, canvas.height - 8);

  legendImg.src = canvas.toDataURL();
  legendImg.style.display = "block";
  legendContainer.style.display = "flex";
}


async function addWcsLayer(layerName) {
  if (rasterLayer) map.removeLayer(rasterLayer);

  const wcsUrl =
    `/geoserver/distgis/ows?service=WCS&version=2.0.1&request=GetCoverage` +
    `&coverageId=distgis:${layerName}&format=image/tiff`;

  try {
    const response = await fetch(wcsUrl);
    if (!response.ok) throw new Error("Fehler beim Abrufen der WCS-Coverage");

    const arrayBuffer = await response.arrayBuffer();
    const georaster = await parseGeoraster(arrayBuffer);

    function smoothGradient(value, min, max) {
      if (value == null || isNaN(value)) return null;
      const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
      const hue = 120 - 120 * t;
      const saturation = 100;
      const lightness = 50 - t * 15;
      return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    }

    rasterLayer = new GeoRasterLayer({
      georaster,
      opacity: 0.3,
      resolution: 64,
      pixelValuesToColorFn: (values) => {
        const val = values[0];
        return smoothGradient(val, georaster.mins[0], georaster.maxs[0]);
      },
    });

    rasterLayer.addTo(map);
    map.fitBounds(rasterLayer.getBounds());
    createLegendGradient(georaster.mins[0], georaster.maxs[0]);
  } catch (err) {
    console.error(err);
    alert("Fehler beim Laden des WCS-Layers: " + err.message);
  }
}

// ————————————————————————————————————————————
// Initialisierung
// ————————————————————————————————————————————
(async function init() {
  try {
    const points = await fetchPoints();
    points.forEach((p) => addPoint([p.lat, p.lon], p.db_value, p.note, p.id));
  } catch (err) {
    console.error("Fehler beim Laden der Punkte:", err);
  }
})();



const logoutBtn = document.getElementById("logoutBtn");
if (logoutBtn) {
  logoutBtn.addEventListener("click", () => {
    sessionStorage.removeItem("loggedIn");
    sessionStorage.removeItem("username");
    window.location.href = "login.html";
  });
}

const userInfo = document.getElementById("userInfo");
if (userInfo) {
  const username = sessionStorage.getItem("username") || "Gast";
  userInfo.textContent = `Hello ${username}`;
}

const infoTextContainer = document.querySelector(".info-text-container");

if (userInfo && infoTextContainer) {
  const username = sessionStorage.getItem("username") || "Gast";
  userInfo.textContent = `Hello ${username}`;

  // Show infotext only if user is guest "Gast"
  if (username === "Gast") {
    infoTextContainer.style.display = "block";
  } else {
    infoTextContainer.style.display = "none";
  }
}
