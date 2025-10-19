FROM nginx:alpine

# Nginx-Konfiguration ersetzen
COPY ./nginx/nginx.conf /etc/nginx/conf.d/default.conf

# Statische Leaflet-Frontend-Dateien kopieren
COPY ./static /usr/share/nginx/html

# Expose Webport
EXPOSE 80
