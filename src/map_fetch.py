"""
Klikk på et kart -> finn nærmeste hentested -> hent Street View-bilde -> finn
søppelkassen -> vis bildet med og uten overlay, og marker estimert GPS på kartet.

En liten lokal webserver (Pythons innebygde http.server, ingen nye avhengigheter)
serverer et Leaflet-kart i nettleseren. Når du klikker et sted:

    1. Nærmeste hentested i CSV-en finnes (adresse, beholdertype, fraksjon).
    2. Hvis bildet allerede finnes (to_annotate, annotated_backup eller annotated)
       gjenbrukes det — ingen API-kall. Ellers finner Street View Metadata API
       (gratis) nærmeste panorama, kamera siktes mot hentestedet, og det betalte
       bildet hentes til data/to_annotate med samme filnavn som
       src.fetch_streetview_from_csv (streetview_<beholderid>.jpg).
    3. Seg-modellen kjøres (src.estimate_bin_positions.process_image): kassene
       segmenteres og back-projiseres til GPS via bakke-geometri.
    4. Nettleseren viser råbildet og overlegget side om side, og slipper en
       markør på kartet for hver estimert kasseposisjon.

Kameradata gjenbrukes fra data/streetview_log.csv når den finnes; ellers
rekonstrueres den med et gratis metadata-kall. Så lukk/åpne kartet fritt — et
bilde du allerede har koster ingen nye API-kall. Hver nye henting (og bilder uten
logg-rad) skrives også til data/streetview_log.csv, så de kan back-projiseres av
src.estimate_bin_positions i batch senere.

Krever GOOGLE_MAPS_API_KEY (settes i PowerShell: $env:GOOGLE_MAPS_API_KEY='din-nøkkel').
Kartfliser og estimerte posisjoner krever internett (det gjør Google-API-et uansett).

Kjør fra prosjektroten:
    py -3.14 -m src.map_fetch
    py -3.14 -m src.map_fetch --csv data/hentesteder_chunks/hentesteder_001.csv
    py -3.14 -m src.map_fetch --port 8000 --no-open
"""

import argparse
import csv
import json
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urlparse

import cv2

from src.estimate_bin_positions import (
    DEFAULT_CONF,
    DEFAULT_FOV,
    DEFAULT_MAX_GROUND_DIST,
    DEFAULT_SEG_WEIGHTS,
    CameraMeta,
    load_metadata,
    process_image,
)
from src.fetch_streetview import _api_key
from src.fetch_streetview_from_csv import (
    CAMERA_HEIGHT_M,
    LOG_FILE,
    POOL_DIR,
    SPLITS,
    TO_ANNOTATE_DIR,
    Bin,
    BinIndex,
    Location,
    _csv_delimiter,
    _log_row,
    _parse_coord,
    auto_pitch,
    bearing,
    dedupe_by_location,
    fetch_streetview_by_pano,
    haversine_m,
    load_log,
    location_to_filename,
    streetview_metadata,
    write_log,
)

CSV_FILE      = Path("data/hentesteder.csv")
CHUNKS_DIR    = Path("data/hentesteder_chunks")
ANNOTATED_DIR = Path("data/annotated")
CACHE_DIR     = Path("outputs/map_fetch")
OVERLAY_DIR   = CACHE_DIR / "overlay"

DEFAULT_SIZE   = "640x480"
DEFAULT_RADIUS = 50
OSLO_CENTER    = (59.9139, 10.7522)
DEFAULT_ZOOM   = 13

CLASS_BIN     = 0
CLASS_GROUND  = 1
COLOR_GROUND  = (0, 200, 0)     # BGR grønn — bakke
COLOR_BIN     = (0, 165, 255)   # BGR oransje — kasse
OVERLAY_ALPHA = 0.45


def _csv_rows(path: Path) -> Iterator[dict]:
    delimiter = _csv_delimiter(path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f, delimiter=delimiter)


def load_index(csv_path: Path) -> tuple[BinIndex, dict[tuple[float, float], dict]]:
    """Leser hentesteder fra en CSV-fil eller alle *.csv i en mappe.

    Gjenbruker prosjektets Bin/dedupe_by_location/BinIndex slik at filnavnet blir
    identisk med det src.fetch_streetview_from_csv lager (streetview_<beholderid>.jpg).
    Returnerer (indeks, info), der info gir adresse/type/fraksjon per koordinat (6 des.).
    """
    paths = sorted(csv_path.glob("*.csv")) if csv_path.is_dir() else [csv_path]
    bins: list[Bin] = []
    info: dict[tuple[float, float], dict] = {}
    for path in paths:
        for row in _csv_rows(path):
            if "Active" in row and (row.get("Active") or "").strip() != "SANN":
                continue
            lat = _parse_coord(row.get("Breddegrad") or row.get("Latitude"))
            lng = _parse_coord(row.get("Lengdegrad") or row.get("Longitude"))
            if lat is None or lng is None:
                continue
            bins.append(Bin(
                product_number=(row.get("Beholderid") or row.get("ProductNumber") or "").strip(),
                lat=lat,
                lng=lng,
                bin_type=(row.get("Beholdertype") or row.get("BinType") or "").strip(),
                waste=(row.get("Fraksjon") or row.get("Info1") or "").strip(),
            ))
            key = (round(lat, 6), round(lng, 6))
            if key not in info:
                info[key] = {
                    "address": (row.get("adresse") or row.get("Adresse") or "").strip(),
                    "bin_type": (row.get("Beholdertype") or row.get("BinType") or "").strip(),
                    "fraksjon": (row.get("Fraksjon") or row.get("Info1") or "").strip(),
                }
    return BinIndex(dedupe_by_location(bins)), info


def _find_image(filename: str) -> Path | None:
    """Leter etter et allerede hentet/annotert bilde, så vi slipper nytt API-kall.

    Sjekker to_annotate, master-poolen (annotated_backup) og det aktive datasettet
    (annotated). Returnerer stien hvis funnet, ellers None.
    """
    candidate = TO_ANNOTATE_DIR / filename
    if candidate.exists():
        return candidate
    for base_dir in (POOL_DIR, ANNOTATED_DIR):
        for split in SPLITS:
            candidate = base_dir / "images" / split / filename
            if candidate.exists():
                return candidate
    return None


def _append_log_row(filename: str, loc: Location, camera: CameraMeta,
                    pano_id: str | None, capture_date: str, address: str,
                    detected: bool, conf: float | None) -> None:
    """Skriver/oppdaterer raden for dette bildet i streetview_log.csv.

    Da kan src.estimate_bin_positions (batch) bruke kameraposisjonen senere. Hele
    loggen leses og skrives på nytt (samme format og dedup som batch-henteren), så
    eksisterende rader bevares.
    """
    log = load_log()
    log[filename] = _log_row(
        filename, loc, pano_id or "", camera.pano_lat, camera.pano_lng,
        camera.distance_m or 0.0, camera.heading, camera.pitch,
        capture_date, "OK", address,
        attempts=1, detected=detected, conf=conf, pano_rank=1, pair="",
    )
    write_log(log)


@dataclass
class ServerContext:
    index: BinIndex
    info: dict[tuple[float, float], dict]
    log_meta: dict[str, CameraMeta]
    seg_model: object
    device: str
    fov: int
    conf: float
    size: str
    radius: int
    lock: threading.Lock
    results: dict[str, dict]


def _write_ground_overlay(seg_model: object, image_path: Path,
                          overlay_path: Path, conf: float) -> tuple[int, int]:
    """Tegner et overlegg med bakke-masken (og evt. kassemasker) og lagrer det.

    Brukes når ingen kasse kunne GPS-posisjoneres, så brukeren ser hva modellen
    faktisk fant i stedet for et tomt panel. Returnerer (antall kasser, antall
    bakkeflater) modellen detekterte.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return 0, 0
    result = seg_model.predict(str(image_path), conf=conf,
                               retina_masks=True, verbose=False)[0]
    overlay = image.copy()
    n_bins = n_ground = 0
    if result.masks is not None and result.boxes is not None:
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for m, c in zip(masks, classes):
            binary = m > 0.5
            if c == CLASS_GROUND:
                overlay[binary] = COLOR_GROUND
                n_ground += 1
            elif c == CLASS_BIN:
                overlay[binary] = COLOR_BIN
                n_bins += 1
    blended = cv2.addWeighted(overlay, OVERLAY_ALPHA, image, 1.0 - OVERLAY_ALPHA, 0.0)
    cv2.imwrite(str(overlay_path), blended)
    return n_bins, n_ground


def _result_without_gps(ctx: ServerContext, base: dict, filename: str,
                        image_path: Path, note: str) -> dict:
    """Resultat når bildet finnes, men kameradata mangler: vis maske-overlegg uten GPS."""
    overlay_path = OVERLAY_DIR / filename
    n_bins, _ = _write_ground_overlay(ctx.seg_model, image_path, overlay_path, ctx.conf)
    extra = f" Modellen så {n_bins} kasse(r) i bildet." if n_bins else ""
    return {
        **base,
        "raw_url": f"/cache/raw/{filename}",
        "overlay_url": f"/cache/overlay/{filename}" if overlay_path.exists() else None,
        "bins": [],
        "message": note + extra,
    }


def find_and_detect(ctx: ServerContext, lat: float, lng: float) -> dict:
    """Snapper til nærmeste hentested, gjenbruker eller henter bildet, kjører deteksjon + GPS.

    Bildet hentes (betalt API) bare hvis det ikke allerede finnes i to_annotate,
    master-poolen eller det aktive datasettet. Nye bilder lagres i data/to_annotate
    med samme filnavn som src.fetch_streetview_from_csv, så de går rett inn i
    annoteringsflyten. Kameradata gjenbrukes fra streetview_log.csv når den finnes,
    ellers rekonstrueres den med et gratis metadata-kall.
    """
    loc, snap_dist = ctx.index.nearest(lat, lng)
    if loc is None:
        return {"ok": False, "error": "Ingen hentesteder i CSV-en."}

    filename = location_to_filename(loc)
    info = ctx.info.get((round(loc.lat, 6), round(loc.lng, 6)), {})
    base = {
        "ok": True,
        "address": info.get("address") or "(ukjent adresse)",
        "bin_type": info.get("bin_type", ""),
        "fraksjon": info.get("fraksjon", ""),
        "product_id": loc.product_numbers[0] if loc.product_numbers else "",
        "snap_lat": loc.lat,
        "snap_lng": loc.lng,
        "snap_dist_m": round(snap_dist, 1),
        "filename": filename,
    }

    with ctx.lock:
        if filename in ctx.results:
            return {**base, **ctx.results[filename]}

        existing = _find_image(filename)
        camera = ctx.log_meta.get(filename) if existing is not None else None
        api_key = None
        pano_id = None
        capture_date = ""

        if camera is None:
            try:
                api_key = _api_key()
            except EnvironmentError as exc:
                if existing is not None:
                    return _result_without_gps(ctx, base, filename, existing,
                        "Mangler API-nøkkel for kameradata — viser bilde uten GPS.")
                return {**base, "ok": False, "error": str(exc)}
            try:
                meta = streetview_metadata(loc.lat, loc.lng, api_key, ctx.radius)
            except Exception as exc:
                if existing is not None:
                    return _result_without_gps(ctx, base, filename, existing,
                        f"Kunne ikke hente kameradata ({exc}) — viser bilde uten GPS.")
                return {**base, "ok": False, "error": f"Metadata-feil: {exc}"}
            if meta.get("status") != "OK":
                if existing is not None:
                    return _result_without_gps(ctx, base, filename, existing,
                        f"Ingen kameradata ({meta.get('status')}) — viser bilde uten GPS.")
                return {**base, "ok": False,
                        "error": f"Ingen Street View her ({meta.get('status')})."}
            pano_lat = meta["location"]["lat"]
            pano_lng = meta["location"]["lng"]
            pano_id = meta["pano_id"]
            capture_date = meta.get("date", "")
            pano_dist = haversine_m(pano_lat, pano_lng, loc.lat, loc.lng)
            camera = CameraMeta(
                pano_lat=pano_lat,
                pano_lng=pano_lng,
                heading=bearing(pano_lat, pano_lng, loc.lat, loc.lng),
                pitch=auto_pitch(pano_dist),
                bin_lat=loc.lat,
                bin_lng=loc.lng,
                distance_m=pano_dist,
            )

        if existing is not None:
            image_path = existing
            fetched_now = False
        else:
            image_path = TO_ANNOTATE_DIR / filename
            try:
                fetch_streetview_by_pano(pano_id, camera.heading, camera.pitch,
                                         image_path, api_key, ctx.size, ctx.fov)
            except Exception as exc:
                return {**base, "ok": False, "error": f"Bilde-feil: {exc}"}
            fetched_now = True

        if cv2.imread(str(image_path)) is None:
            if fetched_now:
                try:
                    image_path.unlink()
                except OSError:
                    pass
                return {**base, "ok": False,
                        "error": "Råbildet kunne ikke leses (mulig avbrutt henting) — prøv igjen."}
            return {**base, "ok": False,
                    "error": f"Eksisterende bilde kunne ikke leses: {image_path.name}"}

        try:
            positions = process_image(
                image_path=image_path,
                seg_model=ctx.seg_model,
                depth_ctx=None,
                device=ctx.device,
                meta=camera,
                fov_h_deg=float(ctx.fov),
                conf=ctx.conf,
                method="ground",
                camera_height_m=CAMERA_HEIGHT_M,
                max_ground_dist_m=DEFAULT_MAX_GROUND_DIST,
                vis_dir=OVERLAY_DIR,
                ground_radius_m=1.0,
            )
        except Exception as exc:
            return {**base, "ok": False, "error": f"Deteksjon feilet: {exc}"}

        overlay_path = OVERLAY_DIR / filename
        bins = [{
            "est_lat": p.est_lat,
            "est_lng": p.est_lng,
            "conf": round(p.conf, 3),
            "dist_m": round(p.horizontal_dist_m, 1),
            "bearing_deg": round(p.bearing_deg, 1),
            "error_m": None if p.error_m is None else round(p.error_m, 1),
        } for p in positions]

        if bins:
            detection = f"{len(bins)} kasse(r) funnet og posisjonert."
            detected_any, best_conf = True, max(b["conf"] for b in bins)
        else:
            n_bins_seen, _ = _write_ground_overlay(
                ctx.seg_model, image_path, overlay_path, ctx.conf)
            detected_any, best_conf = n_bins_seen > 0, None
            if n_bins_seen:
                detection = (f"Fant {n_bins_seen} kasse(r), men kunne ikke posisjonere "
                             "dem (trolig for langt unna for bakke-geometri). "
                             "Overlegget viser maskene modellen fant.")
            else:
                detection = ("Fant ingen søppelkasse i bildet — overlegget viser "
                             "bakkemasken modellen fant.")

        if camera is not None and (fetched_now or filename not in ctx.log_meta):
            try:
                _append_log_row(filename, loc, camera, pano_id, capture_date,
                                info.get("address", ""), detected_any, best_conf)
                ctx.log_meta[filename] = camera
            except Exception as exc:
                print(f"  ADVARSEL: kunne ikke skrive logg for {filename}: {exc}")

        source = "nytt bilde hentet til to_annotate" if fetched_now else "gjenbrukte bilde du allerede har"
        result = {
            "pano_lat": camera.pano_lat,
            "pano_lng": camera.pano_lng,
            "pano_dist_m": None if camera.distance_m is None else round(camera.distance_m, 1),
            "heading": round(camera.heading, 1),
            "pitch": round(camera.pitch, 1),
            "raw_url": f"/cache/raw/{filename}",
            "overlay_url": f"/cache/overlay/{filename}" if overlay_path.exists() else None,
            "bins": bins,
            "message": f"{detection}  ({source})",
            "fetched_now": fetched_now,
        }
        ctx.results[filename] = result
        return {**base, **result}


PAGE = """<!doctype html>
<html lang="no">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hent via kart</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font-family: system-ui, sans-serif; }
  #wrap { display: flex; height: 100%; }
  #map { flex: 1 1 55%; min-width: 0; }
  #side { flex: 1 1 45%; overflow-y: auto; padding: 14px; background: #f5f5f7;
          border-left: 1px solid #ddd; }
  h1 { font-size: 16px; margin: 0 0 8px; }
  #status { font-size: 14px; padding: 8px 10px; border-radius: 6px; background: #e8eef7;
            margin-bottom: 10px; min-height: 20px; }
  #status.err { background: #fdeaea; color: #a12; }
  #meta { font-size: 13px; line-height: 1.5; margin-bottom: 12px; }
  #meta b { display: inline-block; min-width: 110px; color: #555; font-weight: 600; }
  .imgs { display: flex; gap: 10px; flex-wrap: wrap; }
  .imgbox { flex: 1 1 280px; min-width: 240px; }
  .imgbox span { font-size: 12px; font-weight: 600; color: #555; }
  .imgbox img { width: 100%; border: 1px solid #ccc; border-radius: 6px; background: #fff;
                display: block; margin-top: 4px; }
  .hint { font-size: 12px; color: #888; margin-top: 12px; }
</style>
</head>
<body>
<div id="wrap">
  <div id="map"></div>
  <div id="side">
    <h1>Klikk på kartet for å finne nærmeste hentested</h1>
    <div id="status">Klar. Klikk et sted i Oslo.</div>
    <div id="meta"></div>
    <div class="imgs">
      <div class="imgbox"><span>Uten overlay</span><img id="raw" alt=""></div>
      <div class="imgbox"><span>Med overlay (masker + GPS)</span><img id="ovl" alt=""></div>
    </div>
    <div class="hint">Bl&aring; = registrert hentested &middot; gul = Street View-kamera
      &middot; r&oslash;d = estimert kasseposisjon.</div>
  </div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const CFG = "__CFG__";
const map = L.map('map').setView(CFG.center, CFG.zoom);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19, attribution: '&copy; OpenStreetMap'
}).addTo(map);

const layer = L.layerGroup().addTo(map);
const statusEl = document.getElementById('status');
const metaEl = document.getElementById('meta');
const rawImg = document.getElementById('raw');
const ovlImg = document.getElementById('ovl');
let busy = false;

function setStatus(text, isErr) {
  statusEl.textContent = text;
  statusEl.className = isErr ? 'err' : '';
}

function fmt(v) { return (v === null || v === undefined) ? '-' : v; }

async function find(lat, lng) {
  if (busy) return;
  busy = true;
  layer.clearLayers();
  L.circleMarker([lat, lng], {radius: 5, color: '#888', weight: 1}).addTo(layer);
  metaEl.innerHTML = '';
  rawImg.removeAttribute('src');
  ovlImg.removeAttribute('src');
  setStatus('Henter bilde og kjorer deteksjon ...', false);
  try {
    const r = await fetch('/api/find?lat=' + lat + '&lng=' + lng);
    const d = await r.json();
    if (!d.ok) {
      setStatus(d.error || 'Ukjent feil', true);
      if (d.address) {
        metaEl.innerHTML = '<div><b>Adresse</b>' + fmt(d.address) + '</div>'
          + '<div><b>Avstand til klikk</b>' + fmt(d.snap_dist_m) + ' m</div>';
      }
      busy = false;
      return;
    }

    L.marker([d.snap_lat, d.snap_lng]).addTo(layer)
      .bindPopup('<b>' + fmt(d.address) + '</b><br>' + fmt(d.bin_type)
        + (d.fraksjon ? ' &middot; ' + d.fraksjon : ''));
    if (d.pano_lat) {
      L.circleMarker([d.pano_lat, d.pano_lng], {radius: 6, color: '#b8860b',
        fillColor: '#ffd400', fillOpacity: 0.95, weight: 2}).addTo(layer)
        .bindPopup('Street View-kamera<br>' + fmt(d.pano_dist_m) + ' m fra hentested');
    }
    (d.bins || []).forEach((b, i) => {
      L.circleMarker([b.est_lat, b.est_lng], {radius: 7, color: '#c0392b',
        fillColor: '#e74c3c', fillOpacity: 0.9, weight: 2}).addTo(layer)
        .bindPopup('Estimert kasse ' + (i + 1) + '<br>conf ' + fmt(b.conf)
          + '<br>' + fmt(b.dist_m) + ' m fra kamera'
          + (b.error_m === null ? '' : '<br>avvik fra hentested ' + b.error_m + ' m'));
      if (d.pano_lat) {
        L.polyline([[d.pano_lat, d.pano_lng], [b.est_lat, b.est_lng]],
          {color: '#e74c3c', weight: 1, dashArray: '4 4'}).addTo(layer);
      }
    });

    let html = '<div><b>Adresse</b>' + fmt(d.address) + '</div>'
      + '<div><b>Beholder</b>' + fmt(d.bin_type) + (d.fraksjon ? ' &middot; ' + d.fraksjon : '') + '</div>'
      + '<div><b>Avstand til klikk</b>' + fmt(d.snap_dist_m) + ' m</div>'
      + '<div><b>Kamera</b>' + fmt(d.pano_dist_m) + ' m unna, heading ' + fmt(d.heading) + '&deg;</div>';
    metaEl.innerHTML = html;

    rawImg.src = d.raw_url + '?t=' + Date.now();
    if (d.overlay_url) { ovlImg.src = d.overlay_url + '?t=' + Date.now(); }
    setStatus(d.message, false);
  } catch (e) {
    setStatus('Forespørsel feilet: ' + e, true);
  }
  busy = false;
}

map.on('click', e => find(e.latlng.lat.toFixed(7), e.latlng.lng.toFixed(7)));
</script>
</body>
</html>
"""


def render_page(center: tuple[float, float], zoom: int) -> bytes:
    cfg = json.dumps({"center": list(center), "zoom": zoom})
    return PAGE.replace('"__CFG__"', cfg).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        print(f"  [{self.address_string()}] {fmt % args}")

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict) -> None:
        self._send(200, "application/json; charset=utf-8",
                   json.dumps(payload).encode("utf-8"))

    def _serve_cache(self, rel: str) -> None:
        parts = rel.split("/")
        if len(parts) != 2 or parts[0] not in ("raw", "overlay"):
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        name = Path(parts[1]).name
        if not name.endswith(".jpg"):
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        path = _find_image(name) if parts[0] == "raw" else OVERLAY_DIR / name
        if path is None or not path.exists():
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        self._send(200, "image/jpeg", path.read_bytes())

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        ctx: ServerContext = self.server.ctx  # type: ignore[attr-defined]

        if parsed.path == "/":
            self._send(200, "text/html; charset=utf-8",
                       render_page(OSLO_CENTER, DEFAULT_ZOOM))
            return

        if parsed.path == "/health":
            self._send_json({"ok": True, "steder": len(ctx.index.locations)})
            return

        if parsed.path == "/api/find":
            qs = parse_qs(parsed.query)
            try:
                lat = float(qs["lat"][0])
                lng = float(qs["lng"][0])
            except (KeyError, ValueError):
                self._send_json({"ok": False, "error": "Mangler gyldig lat/lng."})
                return
            self._send_json(find_and_detect(ctx, lat, lng))
            return

        if parsed.path.startswith("/cache/"):
            self._serve_cache(parsed.path[len("/cache/"):])
            return

        self._send(404, "text/plain; charset=utf-8", b"not found")


def _resolve_csv(csv_arg: Path) -> Path | None:
    if csv_arg.exists():
        return csv_arg
    if csv_arg == CSV_FILE and CHUNKS_DIR.is_dir() and any(CHUNKS_DIR.glob("*.csv")):
        print(f"{CSV_FILE} finnes ikke — bruker chunk-mappa {CHUNKS_DIR} i stedet.")
        return CHUNKS_DIR
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Klikk på kart -> nærmeste hentested -> Street View -> finn kasse + GPS."
    )
    parser.add_argument("--csv", type=Path, default=CSV_FILE,
                        help=f"Hentesteder-CSV eller mappe med chunks (default: {CSV_FILE}, "
                             f"faller tilbake til {CHUNKS_DIR})")
    parser.add_argument("--seg-weights", type=Path, default=DEFAULT_SEG_WEIGHTS,
                        help=f"YOLO-seg vekter (default: {DEFAULT_SEG_WEIGHTS})")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Adresse å lytte på (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port (default: 8000)")
    parser.add_argument("--fov", type=int, default=int(DEFAULT_FOV),
                        help=f"FOV i grader, brukes både ved henting og back-projeksjon "
                             f"(default: {int(DEFAULT_FOV)})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"Konfidensterskel for seg-deteksjon (default: {DEFAULT_CONF})")
    parser.add_argument("--size", type=str, default=DEFAULT_SIZE,
                        help=f"Bildestørrelse WxH, maks 640x640 (default: {DEFAULT_SIZE})")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS,
                        help=f"Søkeradius i meter for nærmeste panorama (default: {DEFAULT_RADIUS})")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda eller cpu (default: auto)")
    parser.add_argument("--no-open", action="store_true",
                        help="Ikke åpne nettleseren automatisk")
    args = parser.parse_args()

    csv_path = _resolve_csv(args.csv)
    if csv_path is None:
        print(f"Fant ikke CSV: {args.csv} (og ingen chunks i {CHUNKS_DIR}).")
        return
    if not args.seg_weights.exists():
        print(f"Fant ikke seg-vekter: {args.seg_weights}")
        return

    print(f"Laster hentesteder fra {csv_path} ...")
    index, info = load_index(csv_path)
    print(f"Lastet {len(index.locations)} unike hentesteder.")
    if not index.locations:
        print("Ingen hentesteder med gyldige koordinater — avbryter.")
        return

    log_meta: dict[str, CameraMeta] = {}
    try:
        log_meta = load_metadata(LOG_FILE)
        print(f"Manifest: {len(log_meta)} bilder med kameradata (gjenbrukes uten API-kall).")
    except FileNotFoundError:
        print(f"Fant ikke {LOG_FILE} — kameradata rekonstrueres ved behov (gratis metadata-kall).")

    import torch
    from ultralytics import YOLO

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Enhet: {device}. Laster seg-modell {args.seg_weights} ...")
    seg_model = YOLO(str(args.seg_weights))

    try:
        _api_key()
    except EnvironmentError as exc:
        print(f"ADVARSEL: {exc}")
        print("Kartet starter, men nye hentinger feiler til nøkkelen er satt "
              "(bilder du allerede har vises fortsatt).")

    TO_ANNOTATE_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    ctx = ServerContext(
        index=index,
        info=info,
        log_meta=log_meta,
        seg_model=seg_model,
        device=device,
        fov=args.fov,
        conf=args.conf,
        size=args.size,
        radius=args.radius,
        lock=threading.Lock(),
        results={},
    )

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.ctx = ctx  # type: ignore[attr-defined]

    url = f"http://{args.host}:{args.port}/"
    print(f"\nServer kjører på {url}")
    print("Klikk på kartet i nettleseren. Ctrl+C for å stoppe.\n")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopper server.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
