import os
import re
import math
import glob
import datetime as dt
import time
import logging
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio import features
from rasterio.io import MemoryFile
from pyproj import Transformer
from sqlalchemy.orm import Session
from geoalchemy2.shape import to_shape
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import box as shp_box

from .database import SessionLocal
from .models import NoisePoint

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JOB_REGISTRY = {}

GEOSERVER_URL   = os.getenv("GEOSERVER_URL",   "http://geoserver:8080/geoserver")
GEOSERVER_USER  = os.getenv("GEOSERVER_USER",  "admin")
GEOSERVER_PASS  = os.getenv("GEOSERVER_PASS",  "geoserver")
GEOSERVER_WS    = os.getenv("GEOSERVER_WS",    "distgis")
GEOSERVER_STYLE = os.getenv("GEOSERVER_STYLE", "noise_heat")
OUTPUT_DIR      = os.getenv("OUTPUT_RASTER_DIR", "/data/rasters")
LOD1_DIR        = os.getenv("LOD1_DIR", "/data/LoD1")

NODATA = -9999.0
_TILE_RE = re.compile(r".*LoD1_(\d{3})_(\d{4})\.(?:gml|xml)$", re.IGNORECASE)

def _pick_utm_epsg(lon: float, lat: float) -> int:
    if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
        raise ValueError(f"Ungültige Koordinaten: lon={lon}, lat={lat}")
    zone = int((lon + 180) // 6) + 1
    return int(f"327{zone:02d}") if lat < 0 else int(f"326{zone:02d}")

def _fetch_points(session: Session):
    pts = []
    for p in session.query(NoisePoint):
        try:
            geom = to_shape(p.geom)
            pts.append({"lon": geom.x, "lat": geom.y, "db": p.db_value})
        except Exception as e:
            logger.warning(f"Fehler beim Lesen eines Punkts: {e}")
    logger.info(f"{len(pts)} Punkte aus der Datenbank geladen.")
    return pts

def _validate_params(params: dict):
    def clamp(v, lo, hi, name):
        if v < lo or v > hi:
            logger.warning(f"Parameter {name}={v} außerhalb [{lo},{hi}] – wird begrenzt.")
        return max(lo, min(hi, v))
    p = dict(params or {})
    p.setdefault("cell_size_m", 10)
    p.setdefault("buffer_m", 300)
    p.setdefault("alpha_db_per_m", 0.003)
    p.setdefault("max_distance_m", 1000)
    p.setdefault("smooth_sigma", 3.0)
    p.setdefault("clip_min_db", 35)
    p.setdefault("clip_max_db", 95)
    p.setdefault("shadow_attenuation_db", 35)
    p.setdefault("ray_samples", 48)
    p["cell_size_m"] = clamp(float(p["cell_size_m"]), 2, 100, "cell_size_m")
    p["buffer_m"] = clamp(float(p["buffer_m"]), 200, 5000, "buffer_m")
    p["alpha_db_per_m"] = clamp(float(p["alpha_db_per_m"]), 0, 0.05, "alpha_db_per_m")
    p["max_distance_m"] = clamp(float(p["max_distance_m"]), 300, 3000, "max_distance_m")
    p["shadow_attenuation_db"] = clamp(float(p["shadow_attenuation_db"]), 5, 60, "shadow_attenuation_db")
    p["ray_samples"] = int(clamp(int(p["ray_samples"]), 8, 80, "ray_samples"))
    return p

def _bbox_km_indices(xmin, ymin, xmax, ymax):
    return (int(xmin // 1000), int(xmax // 1000), int(ymin // 1000), int(ymax // 1000))

def _candidate_lod1_files(lod_dir, xkm_min, xkm_max, ykm_min, ykm_max):
    files = []
    for path in glob.glob(os.path.join(lod_dir, "LoD1_*_*.xml")) + glob.glob(os.path.join(lod_dir, "LoD1_*_*.gml")):
        m = _TILE_RE.match(os.path.basename(path))
        if not m:
            continue
        xkm, ykm = int(m.group(1)), int(m.group(2))
        if xkm_min <= xkm <= xkm_max and ykm_min <= ykm <= ykm_max:
            files.append(path)
    return files

def _load_lod1_geometries_epsg(epsg, xmin, ymin, xmax, ymax, padding_km=1):
    if not os.path.isdir(LOD1_DIR):
        logger.info("Kein LoD1-Verzeichnis gefunden – keine Gebäude berücksichtigt.")
        return None
    assumed_epsg = 25833
    if epsg != assumed_epsg:
        tf = Transformer.from_crs(epsg, assumed_epsg, always_xy=True)
        xmin, ymin = tf.transform(xmin, ymin)
        xmax, ymax = tf.transform(xmax, ymax)
    pad = padding_km * 1000
    xkm_min, xkm_max, ykm_min, ykm_max = _bbox_km_indices(xmin - pad, ymin - pad, xmax + pad, ymax + pad)
    cand = _candidate_lod1_files(LOD1_DIR, xkm_min, xkm_max, ykm_min, ykm_max)
    if not cand:
        logger.info("Keine passenden LoD1-Tiles gefunden.")
        return None
    gdf_list = []
    for fp in cand:
        try:
            g = gpd.read_file(fp)
            if g.crs is None:
                g = g.set_crs(assumed_epsg)
            gdf_list.append(g)
        except Exception as e:
            logger.warning(f"LoD1-Datei konnte nicht gelesen werden: {e}")
    if not gdf_list:
        return None
    lod = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True)).to_crs(epsg=epsg)
    bbox_gdf = gpd.GeoDataFrame(geometry=[shp_box(xmin, ymin, xmax, ymax)], crs=f"EPSG:{epsg}")
    try:
        lod = gpd.clip(lod, bbox_gdf)
    except Exception:
        lod = lod.sjoin(bbox_gdf, predicate="intersects", how="inner").drop(
            columns=[c for c in lod.columns if c.startswith("index_")], errors="ignore"
        )
    if lod.empty:
        logger.info("LoD1-Filter ergab keine Geometrien im Auswertefenster.")
        return None
    logger.info(f"LoD1-Geometrien geladen: {len(lod)}")
    return lod

def _debug_shadow_mask(px, py, building_mask, transform, ray_samples, out_path):
    try:
        H, W = building_mask.shape
        try:
            inv = transform.inverse
        except AttributeError:
            inv = ~transform
        src_idx = []
        for x, y in zip(px, py):
            c, r = inv * (x, y)
            r_i, c_i = int(round(r)), int(round(c))
            if 0 <= r_i < H and 0 <= c_i < W:
                src_idx.append((r_i, c_i))
        if not src_idx:
            return
        rs = max(3, int(ray_samples))
        shadow = np.zeros_like(building_mask, dtype=np.uint8)
        for i in range(H):
            for j in range(W):
                if building_mask[i, j] == 1:
                    continue
                blocked_any = False
                for (sr, sc) in src_idx:
                    for s in range(1, rs - 1):
                        t = s / (rs - 1.0)
                        rr = int(round(sr + t * (i - sr)))
                        cc = int(round(sc + t * (j - sc)))
                        if 0 <= rr < H and 0 <= cc < W and building_mask[rr, cc] == 1:
                            blocked_any = True
                            break
                    if blocked_any:
                        break
                if blocked_any:
                    shadow[i, j] = 1
        with rasterio.open(
            out_path, "w", driver="GTiff",
            height=H, width=W, count=1, dtype="uint8",
            crs=None, transform=transform, nodata=0,
            compress="DEFLATE", tiled=True
        ) as dst:
            dst.write(shadow, 1)
    except Exception as e:
        logger.warning(f"DEBUG shadow mask fehlgeschlagen: {e}")

if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def _fast_raster(px, py, pdb, xs, ys, alpha, rmax,
                     building_mask, inv_a, inv_b, inv_c, inv_d, inv_e, inv_f,
                     clip_min, clip_max, atten_lin, ray_samples):
        H, W = building_mask.shape
        out = np.full((H, W), -9999.0, np.float32)
        N = len(px)
        src_r = np.empty(N, np.int32)
        src_c = np.empty(N, np.int32)
        for k in range(N):
            c = int(round(inv_a * px[k] + inv_b * py[k] + inv_c))
            r = int(round(inv_d * px[k] + inv_e * py[k] + inv_f))
            src_r[k] = r
            src_c[k] = c
        for i in prange(H):
            y = ys[i]
            for j in range(W):
                if building_mask[i, j] == 1:
                    continue
                E = 0.0
                valid = False
                for k in range(N):
                    dx = xs[j] - px[k]
                    dy = y - py[k]
                    r = math.hypot(dx, dy)
                    if r < 1.0 or r > rmax:
                        continue
                    Li = pdb[k] - 20.0 * math.log10(r) - alpha * r
                    Ik = 10.0 ** (Li / 10.0)
                    rs = ray_samples
                    if rs < 3:
                        rs = 3
                    rr0, cc0 = src_r[k], src_c[k]
                    blocked = False
                    for s in range(1, rs - 1):
                        t = s / (rs - 1.0)
                        rr = int(round(rr0 + t * (i - rr0)))
                        cc = int(round(cc0 + t * (j - cc0)))
                        if rr < 0 or rr >= H or cc < 0 or cc >= W:
                            continue
                        if building_mask[rr, cc] == 1:
                            blocked = True
                            break
                    if blocked:
                        Ik *= atten_lin
                    E += Ik
                    valid = True
                if valid:
                    v = 10.0 * math.log10(E)
                    if v < clip_min: v = clip_min
                    if v > clip_max: v = clip_max
                    out[i, j] = v
        return out

def _compute_raster(points, **kwargs):
    if not points:
        raise ValueError("Keine Punkte vorhanden.")
    p = _validate_params(kwargs)
    lon_c = np.mean([pt["lon"] for pt in points])
    lat_c = np.mean([pt["lat"] for pt in points])
    epsg = _pick_utm_epsg(lon_c, lat_c)
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    pts_xy = [{"x": tf.transform(pt["lon"], pt["lat"])[0],
               "y": tf.transform(pt["lon"], pt["lat"])[1],
               "db": pt["db"]} for pt in points]
    xs = [p_["x"] for p_ in pts_xy]
    ys = [p_["y"] for p_ in pts_xy]
    xmin, xmax = min(xs) - p["buffer_m"], max(xs) + p["buffer_m"]
    ymin, ymax = min(ys) - p["buffer_m"], max(ys) + p["buffer_m"]
    W = int(math.ceil((xmax - xmin) / p["cell_size_m"]))
    H = int(math.ceil((ymax - ymin) / p["cell_size_m"]))
    logger.info(
        f"Bounding Box: xmin={xmin:.1f}, ymin={ymin:.1f}, xmax={xmax:.1f}, ymax={ymax:.1f}"
    )
    logger.info(
        f"Ausdehnung: {(xmax - xmin):.1f} m x {(ymax - ymin):.1f} m (≈ {(xmax - xmin) / 1000:.2f} km x {(ymax - ymin) / 1000:.2f} km)"
    )
    max_cells = 15_000_000
    if W * H > max_cells:
        scale = math.sqrt((W * H) / max_cells)
        old = p["cell_size_m"]
        p["cell_size_m"] *= scale
        W = int(math.ceil((xmax - xmin) / p["cell_size_m"]))
        H = int(math.ceil((ymax - ymin) / p["cell_size_m"]))
        logger.info(f"Raster zu groß – Zellgröße erhöht: {old:.2f} → {p['cell_size_m']:.2f} m (neu {W}x{H})")
    xs_grid = xmin + (np.arange(W) + 0.5) * p["cell_size_m"]
    ys_grid = ymax - (np.arange(H) + 0.5) * p["cell_size_m"]
    building_mask = np.zeros((H, W), dtype="uint8")
    transform = from_origin(xmin, ymax, p["cell_size_m"], p["cell_size_m"])
    try:
        lod = _load_lod1_geometries_epsg(epsg, xmin, ymin, xmax, ymax, 1)
        if lod is not None and not lod.empty:
            geoms = (geom for geom in lod.geometry if geom and not geom.is_empty)
            building_mask = features.rasterize(
                ((geom, 1) for geom in geoms),
                out_shape=(H, W),
                transform=transform,
                fill=0,
                dtype="uint8"
            )
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            mask_path = os.path.join(OUTPUT_DIR, "last_building_mask.tif")
            with rasterio.open(
                mask_path, "w", driver="GTiff",
                height=H, width=W, count=1, dtype="uint8",
                crs=f"EPSG:{epsg}", transform=transform, nodata=0,
                compress="DEFLATE", tiled=True
            ) as dst:
                dst.write(building_mask, 1)
            logger.info(f"Gebäudezellen: {int(building_mask.sum())} (Maske: {mask_path})")
        else:
            logger.info("Kein LoD1 in Auswertefenster – keine Gebäudemaske.")
    except Exception as e:
        logger.warning(f"Gebäudeladen/Rasterisieren fehlgeschlagen: {e}")
    px, py, pdb = np.array(xs), np.array(ys), np.array([p_["db"] for p_ in pts_xy])
    if os.getenv("DEBUG_SHADOW", "0") == "1":
        _debug_shadow_mask(px, py, building_mask, transform, p["ray_samples"],
                           os.path.join(OUTPUT_DIR, "last_shadow_mask.tif"))
        logger.info("Debug: last_shadow_mask.tif geschrieben (falls aktiviert).")
    logger.info(f"Schallschatten: {p['shadow_attenuation_db']} dB, Rays={p['ray_samples']}, rmax={p['max_distance_m']} m")
    start = time.time()
    if NUMBA_AVAILABLE:
        try:
            inv = transform.inverse
        except AttributeError:
            inv = ~transform
        arr = _fast_raster(
            px, py, pdb, xs_grid, ys_grid,
            p["alpha_db_per_m"], p["max_distance_m"],
            building_mask,
            inv.a, inv.b, inv.c, inv.d, inv.e, inv.f,
            float(p["clip_min_db"]), float(p["clip_max_db"]),
            10 ** (-float(p["shadow_attenuation_db"]) / 10.0),
            int(p["ray_samples"])
        )
    else:
        raise RuntimeError("Numba erforderlich für realistische Berechnung.")
    logger.info(f"Berechnung abgeschlossen in {time.time()-start:.1f}s.")
    if SCIPY_AVAILABLE and p["smooth_sigma"] > 0:
        valid = arr != NODATA
        w = valid.astype("float32")
        arr_f = arr.copy()
        arr_f[~valid] = 0.0
        num = gaussian_filter(arr_f, sigma=p["smooth_sigma"])
        den = gaussian_filter(w, sigma=p["smooth_sigma"])
        arr = np.where(den > 1e-6, num / den, NODATA)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"noise_{stamp}.tif"
    path = os.path.join(OUTPUT_DIR, name)
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=1,
        dtype=arr.dtype, crs=f"EPSG:{epsg}", transform=transform,
        nodata=NODATA, compress="DEFLATE", tiled=True, BIGTIFF="YES"
    ) as dst:
        dst.write(arr, 1)
    logger.info(f"Raster gespeichert: {path}")
    return path, name, epsg

def _wait_for_geoserver(timeout=60):
    url = f"{GEOSERVER_URL}/rest/about/version.xml"
    for _ in range(max(1, timeout // 2)):
        try:
            if requests.get(url, auth=(GEOSERVER_USER, GEOSERVER_PASS), timeout=3).status_code == 200:
                logger.info("GeoServer erreichbar.")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("GeoServer nicht erreichbar")

def _ensure_workspace():
    r = requests.get(f"{GEOSERVER_URL}/rest/workspaces/{GEOSERVER_WS}.json", auth=(GEOSERVER_USER, GEOSERVER_PASS))
    if r.status_code != 200:
        requests.post(f"{GEOSERVER_URL}/rest/workspaces",
                      json={"workspace": {"name": GEOSERVER_WS}},
                      headers={"Content-Type": "application/json"},
                      auth=(GEOSERVER_USER, GEOSERVER_PASS))
        logger.info(f"Workspace '{GEOSERVER_WS}' erstellt.")

def _publish_geotiff(path, layer_name, style="noise_heat", epsg=None):
    _wait_for_geoserver()
    _ensure_workspace()
    store_url = f"{GEOSERVER_URL}/rest/workspaces/{GEOSERVER_WS}/coveragestores/{layer_name}/file.geotiff"
    with open(path, "rb") as f:
        data = f.read()
    r = requests.put(store_url, params={"configure": "first", "coverageName": layer_name},
                     data=data, headers={"Content-Type": "image/tiff"},
                     auth=(GEOSERVER_USER, GEOSERVER_PASS), timeout=60)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"Upload fehlgeschlagen: {r.status_code}")
    style_url = f"{GEOSERVER_URL}/rest/layers/{GEOSERVER_WS}:{layer_name}"
    r2 = requests.put(style_url,
                      json={"layer": {"defaultStyle": {"name": style}}},
                      headers={"Content-Type": "application/json"},
                      auth=(GEOSERVER_USER, GEOSERVER_PASS), timeout=10)
    if not (200 <= r2.status_code < 300):
        raise RuntimeError(f"Style-Zuweisung fehlgeschlagen: {r2.status_code}")
    logger.info(f"Layer {layer_name} publiziert.")
    return layer_name

def run_sound_raster_job(job_id: str, params: dict, points=None):
    JOB_REGISTRY[job_id] = {"status": "running", "message": "Berechnung läuft..."}
    try:
        username = params.get("username", "unbekannt")
        if not points or len(points) == 0:
            raise ValueError(f"Keine Punkte für Benutzer '{username}' übergeben – Abbruch.")
        logger.info(f"[{job_id}] Starte Berechnung für Benutzer '{username}' mit {len(points)} Punkten")
        path, name, epsg = _compute_raster(points, **(params or {}))
        layer_name = f"{username}_{os.path.splitext(name)[0]}"
        _publish_geotiff(path, layer_name, style=GEOSERVER_STYLE, epsg=epsg)
        JOB_REGISTRY[job_id] = {
            "status": "done",
            "message": f"Raster '{layer_name}' für Benutzer '{username}' publiziert.",
            "layer_name": layer_name,
        }
        logger.info(f"[{job_id}] Berechnung abgeschlossen für Benutzer '{username}'")
    except Exception as e:
        logger.exception(f"[{job_id}] Fehler: {e}")
        JOB_REGISTRY[job_id] = {"status": "error", "message": str(e)}
