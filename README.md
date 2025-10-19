# FindYourVenue
This is a distributed GIS project to record and analy of sound propagation from sound systems – based on user-generated measurement points.

# Instructions for Launching the Docker Product with docker-compose

This project comprises multiple containers: Frontend (NGINX), API (FastAPI backend), PostGIS database, and GeoServer.

## Prerequisites

- Docker and Docker Compose must be installed on the system.
- Alternatively, Docker Desktop may be used.

## Starting the Project

1. Clone the repository:

```

git clone https://github.com/navigonia/FindYourVenue.git
cd YourRepository

```

2. Build and start all containers in detached mode:

```

docker-compose up -d --build

```

3. Verify the services:

- Frontend: http://localhost:8080
- GeoServer: http://localhost:8081
- API: http://localhost:8000

4. (Optional) View logs:

```

docker-compose logs -f

```

## Stopping the Containers

To stop the containers, execute the following command in the project directory:

```

docker-compose down

```

## Notes

- Database and raster data are stored in Docker volumes to ensure data persistence across container restarts.
- Environment variables in the `docker-compose.yml` file can be modified if necessary.

For inquiries or issues, please contact the maintainer.





