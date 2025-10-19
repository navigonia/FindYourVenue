# Basis: PostgreSQL 16 mit PostGIS 3.4
FROM postgis/postgis:16-3.4

# Optionale Zusatzpakete (z. B. uuid-ossp, hstore)
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-contrib \
 && rm -rf /var/lib/apt/lists/*

# Kopiere Initialisierungsskripte (wird nur beim ersten Start ausgeführt)
COPY ./init /docker-entrypoint-initdb.d/
