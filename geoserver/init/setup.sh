#!/bin/bash
set -euo pipefail

GEOSERVER_URL="http://localhost:8080/geoserver"
AUTH="${GEOSERVER_ADMIN_USER:-admin}:${GEOSERVER_ADMIN_PASSWORD:-geoserver}"
WORKSPACE="${GEOSERVER_WS:-distgis}"
STYLE_NAME="${GEOSERVER_STYLE:-noise_heat}"
STYLE_PATH="/opt/geoserver/data_dir/styles/${STYLE_NAME}.sld"

echo "[INFO] Starting GeoServer initialization..."

# Warten, bis GeoServer REST erreichbar ist
until curl -sf -u "$AUTH" "${GEOSERVER_URL}/rest/about/version.xml" >/dev/null; do
  echo "[WAIT] GeoServer not yet available, retrying in 5s..."
  sleep 5
done
echo "[INFO] GeoServer REST is reachable."

# Workspace prüfen oder anlegen
if ! curl -sf -u "$AUTH" "${GEOSERVER_URL}/rest/workspaces/${WORKSPACE}.xml" >/dev/null; then
  echo "[CREATE] Creating workspace '${WORKSPACE}'..."
  curl -s -u "$AUTH" -X POST -H "Content-Type: application/xml" \
    -d "<workspace><name>${WORKSPACE}</name></workspace>" \
    "${GEOSERVER_URL}/rest/workspaces" >/dev/null
else
  echo "[OK] Workspace '${WORKSPACE}' already exists."
fi

# Style prüfen oder anlegen
if ! curl -sf -u "$AUTH" "${GEOSERVER_URL}/rest/styles/${STYLE_NAME}.xml" >/dev/null; then
  echo "[CREATE] Uploading style '${STYLE_NAME}'..."
  if [ ! -f "$STYLE_PATH" ]; then
    echo "[ERROR] SLD file not found: ${STYLE_PATH}"
    exit 1
  fi
  curl -s -u "$AUTH" -X POST -H "Content-Type: application/vnd.ogc.sld+xml" \
    -d @"${STYLE_PATH}" \
    "${GEOSERVER_URL}/rest/styles?name=${STYLE_NAME}" >/dev/null
else
  echo "[OK] Style '${STYLE_NAME}' already exists."
fi

echo "[DONE] GeoServer setup completed successfully."
