FROM kartoza/geoserver:2.24.2

ENV GEOSERVER_ADMIN_USER=admin \
    GEOSERVER_ADMIN_PASSWORD=geoserver \
    CORS_ENABLED=true \
    GEOSERVER_WS=distgis \
    GEOSERVER_STYLE=noise_heat

COPY data_dir /opt/geoserver/data_dir
RUN chmod -R 777 /opt/geoserver/data_dir || true

COPY init/ /docker-entrypoint-init.d/
RUN chmod +x /docker-entrypoint-init.d/*.sh || true

EXPOSE 8080
