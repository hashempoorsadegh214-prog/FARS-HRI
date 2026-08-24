import json
import math
import os
from datetime import datetime, timedelta, timezone

import requests


# ============================================================
# تنظیمات
# ============================================================

BOUNDARY_FILE = "fars.geojson"
OUTPUT_FILE = "data/fwi_fars.json"

API_URL = "https://api.effis.emergency.copernicus.eu/rest/2/burntareas/charts/wms"

MODEL = "ecmwf"

# شبکه سبک‌تر
GRID_STEP = 0.10

# محدوده تقریبی استان فارس
WEST = 50.0
EAST = 54.5
SOUTH = 27.0
NORTH = 31.5

TIMEOUT = 30


# ============================================================
# توابع هندسی
# ============================================================

def point_in_ring(lon, lat, ring):
    """
    بررسی می‌کند آیا نقطه داخل یک Polygon Ring قرار دارد یا نه.
    """

    inside = False

    j = len(ring) - 1

    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]

        intersects = (
            ((yi > lat) != (yj > lat))
            and
            (
                lon
                < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-15) + xi
            )
        )

        if intersects:
            inside = not inside

        j = i

    return inside


def point_in_polygon(lon, lat, polygon):
    """
    Polygon می‌تواند دارای سوراخ داخلی باشد.
    """

    if not polygon:
        return False

    outer = polygon[0]

    if not point_in_ring(lon, lat, outer):
        return False

    # سوراخ‌های داخلی
    for hole in polygon[1:]:
        if point_in_ring(lon, lat, hole):
            return False

    return True


def point_in_geometry(lon, lat, geometry):
    """
    پشتیبانی از Polygon و MultiPolygon.
    """

    if not geometry:
        return False

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")

    if geometry_type == "Polygon":
        return point_in_polygon(lon, lat, coordinates)

    if geometry_type == "MultiPolygon":
        for polygon in coordinates:
            if point_in_polygon(lon, lat, polygon):
                return True

    return False


# ============================================================
# خواندن مرز فارس
# ============================================================

def load_fars_boundary():
    print("Reading Fars boundary...")

    with open(BOUNDARY_FILE, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    if geojson.get("type") == "FeatureCollection":
        geometries = []

        for feature in geojson.get("features", []):
            geometry = feature.get("geometry")

            if geometry:
                geometries.append(geometry)

        return geometries

    if geojson.get("type") == "Feature":
        return [geojson["geometry"]]

    return [geojson]


# ============================================================
# ساخت شبکه
# ============================================================

def create_grid(geometries):
    print(f"Creating grid with step {GRID_STEP}° ...")

    points = []

    lat = SOUTH

    while lat <= NORTH + 1e-9:

        lon = WEST

        while lon <= EAST + 1e-9:

            for geometry in geometries:

                if point_in_geometry(lon, lat, geometry):
                    points.append({
                        "lon": round(lon, 6),
                        "lat": round(lat, 6)
                    })
                    break

            lon += GRID_STEP

        lat += GRID_STEP

    print(f"Grid points inside Fars: {len(points)}")

    return points


# ============================================================
# دریافت FWI از API
# ============================================================

def get_fwi(lon, lat, target_date):
    """

    API:
    /rest/2/burntareas/charts/wms

    مثال:
    ?model=ecmwf
    &day_gte=2026-08-25
    &day_lte=2026-08-25
    &point=(52.58 29.59)

    """

    params = {
        "model": MODEL,
        "day_gte": target_date,
        "day_lte": target_date,
        "point": f"({lon} {lat})"
    }

    try:

        response = requests.get(
            API_URL,
            params=params,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        x_data = data.get("x_data", [])
        y_data = data.get("y_data", {})

        fwi_values = y_data.get("fwi", [])

        if not x_data or not fwi_values:
            return None

        # پیدا کردن تاریخ موردنظر
        for i, date_value in enumerate(x_data):

            if date_value.startswith(target_date):

                if i < len(fwi_values):

                    value = fwi_values[i]

                    if value is not None:
                        return float(value)

        # اگر API فقط یک مقدار برگرداند
        if len(fwi_values) == 1:
            return float(fwi_values[0])

    except Exception as e:

        print(
            f"API error at "
            f"{lon}, {lat}: {e}"
        )

    return None


# ============================================================
# برنامه اصلی
# ============================================================

def main():

    print("=" * 60)
    print("FARS FWI - ECMWF Forecast")
    print("=" * 60)

    # --------------------------------------------------------
    # تاریخ فردا
    # --------------------------------------------------------

    tomorrow = (
        datetime.now(timezone.utc).date()
        + timedelta(days=1)
    )

    target_date = tomorrow.isoformat()

    print(f"Target date: {target_date}")

    # --------------------------------------------------------
    # مرز
    # --------------------------------------------------------

    geometries = load_fars_boundary()

    # --------------------------------------------------------
    # شبکه
    # --------------------------------------------------------

    points = create_grid(geometries)

    if not points:
        print("No grid points found.")
        return

    # --------------------------------------------------------
    # دریافت FWI
    # --------------------------------------------------------

    results = []

    total = len(points)

    print()
    print(f"Requesting FWI for {total} points...")
    print()

    for index, point in enumerate(points, start=1):

        lon = point["lon"]
        lat = point["lat"]

        print(
            f"[{index}/{total}] "
            f"{lon}, {lat}"
        )

        fwi = get_fwi(
            lon,
            lat,
            target_date
        )

        if fwi is None:
            print("   -> no data")
            continue

        print(f"   -> FWI = {fwi:.2f}")

        results.append({
            "lon": lon,
            "lat": lat,
            "fwi": round(fwi, 3),
            "date": target_date,
            "model": MODEL,
            "source": "Copernicus EFFIS / ECMWF"
        })

    # --------------------------------------------------------
    # ساخت پوشه خروجی
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    # --------------------------------------------------------
    # خروجی
    # --------------------------------------------------------

    output = {
        "updated_at_utc": datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S"),

        "target_date": target_date,

        "model": MODEL,

        "grid_step": GRID_STEP,

        "count": len(results),

        "source": (
            "Copernicus EFFIS "
            "ECMWF Fire Weather Index"
        ),

        "api": API_URL,

        "data": results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Valid FWI points: {len(results)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
