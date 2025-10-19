import time
import uuid
from fastapi import FastAPI, HTTPException, Depends, APIRouter, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware  
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from sqlalchemy import text
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import models
from .database import engine, SessionLocal
from .process import JOB_REGISTRY, run_sound_raster_job

app = FastAPI(title="dist-gis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for i in range(10):
    try:
        models.Base.metadata.create_all(bind=engine)
        print("Datenbankverbindung erfolgreich.")
        break
    except OperationalError:
        print(f"Warte auf Datenbank... Versuch {i + 1}/10")
        time.sleep(3)
else:
    raise RuntimeError("Konnte keine Verbindung zur Datenbank herstellen.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PointIn(BaseModel):
    lat: float
    lon: float
    db_value: int
    note: str = ""
    username: str

class PointOut(BaseModel):
    id: str
    lat: float
    lon: float
    db_value: int
    note: str

class LoginData(BaseModel):
    username: str
    password: str

router = APIRouter(prefix="/api")

@router.post("/login")
def login(data: LoginData, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == data.username).first()
    if not user:
        new_user = models.User(username=data.username, password=data.password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {
            "status": "registered",
            "user_id": new_user.id,
            "username": new_user.username
        }
    if user.password != data.password:
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    return {
        "status": "ok",
        "user_id": user.id,
        "username": user.username
    }

@router.get("/points/{username}")
def get_points(username: str, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
    points = db.query(models.NoisePoint).filter(models.NoisePoint.user_id == user.id).all()
    features = []
    for p in points:
        geom = to_shape(p.geom)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [geom.x, geom.y]},
            "properties": {
                "id": str(p.id),
                "db_value": p.db_value,
                "note": p.note,
                "created_at": p.created_at.isoformat()
            },
        })
    return {"type": "FeatureCollection", "features": features}

@router.post("/points", response_model=PointOut)
def add_point(data: PointIn, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == data.username).first()
    if not user:
        user = models.User(username=data.username, password="auto")
        db.add(user)
        db.commit()
        db.refresh(user)
    point_geom = from_shape(Point(data.lon, data.lat), srid=4326)
    new_point = models.NoisePoint(
        geom=point_geom,
        db_value=data.db_value,
        note=data.note,
        user_id=user.id
    )
    db.add(new_point)
    db.commit()
    db.refresh(new_point)
    return PointOut(
        id=str(new_point.id),
        lat=data.lat,
        lon=data.lon,
        db_value=data.db_value,
        note=data.note,
    )

@router.delete("/points/{point_id}")
def delete_point(point_id: str, db: Session = Depends(get_db)):
    point = db.query(models.NoisePoint).filter(models.NoisePoint.id == point_id).first()
    if not point:
        raise HTTPException(status_code=404, detail="Point not found")
    db.delete(point)
    db.commit()
    return {"status": "deleted"}

@router.post("/process/sound-raster")
def start_sound_raster(background_tasks: BackgroundTasks, params: dict | None = None, db: Session = Depends(get_db)):
    if params is None:
        params = {}
    username = params.get("username")
    if not username:
        raise HTTPException(status_code=400, detail="Username erforderlich")
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
    from geoalchemy2.shape import to_shape
    points = db.query(models.NoisePoint).filter(models.NoisePoint.user_id == user.id).all()
    if not points:
        raise HTTPException(status_code=400, detail="Keine Punkte für diesen Benutzer vorhanden")
    point_data = []
    for p in points:
        geom = to_shape(p.geom)
        point_data.append({
            "lon": geom.x,
            "lat": geom.y,
            "db": p.db_value
        })
    job_id = str(uuid.uuid4())
    JOB_REGISTRY[job_id] = {"status": "queued", "message": f"Starte Berechnung für Benutzer {username}"}
    background_tasks.add_task(run_sound_raster_job, job_id, params, point_data)
    return {"job_id": job_id}

@router.get("/process/status/{job_id}")
def get_status(job_id: str):
    return JOB_REGISTRY.get(job_id, {"status": "unknown", "message": "Job nicht gefunden"})

@router.get("/health")
def healthcheck(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return JSONResponse(status_code=200, content={"status": "ok"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "details": str(e)})

app.include_router(router)
