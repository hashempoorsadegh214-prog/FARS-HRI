```python
import json
import os
import requests
from datetime import datetime, timezone, timedelta

# ============================================================
# FARS-HRI
# دریافت پیش‌بینی FWI از Copernicus EFFIS / Meteo-France
# ============================================================

API_URL = (
    "https://api.effis.emergency.copernicus.eu/"
    "rest/2/burntareas/charts/wms"
)

OUTPUT_FILE = "data/fwi/fwi_fars.json"

# مدل Meteo-France
MODEL = "mf"

# نقاط نماینده داخل استان فارس
POINTS = [
    {"name": "Shiraz", "lon": 52.5837, "lat": 29.5918},
    {"name": "Fars_Center", "lon": 53.1525, "lat": 28.7438},
    {"name": "Fars_East", "lon": 53.5004, "lat": 28.3619},
]

# امروز تا 3 روز آینده
TODAY = datetime.now(timezone.utc).date()

START_DATE = TODAY.isoformat()
END_DATE = (TODAY + timedelta(days=3)).isoformat()

TIMEOUT = 30


# ============================================================
# دریافت داده از EFFIS
# ============================================================

def get_point_data(point):

    params = {
        "model": MODEL,
        "day_gte": START_DATE,
        "day_lte": END_DATE,
        "point": f"({point['lon']} {point['lat']})",
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# تبدیل پاسخ API
# ============================================================

def parse_point(point, data):

    dates = data.get("x_data", [])

    y_data = data.get("y_data", {})

    fwi = y_data.get("fwi", [])
    ffmc = y_data.get("ffmc", [])
    dmc = y_data.get("dmc", [])
    dc = y_data.get("dc", [])
    isi = y_data.get("isi", [])
    bui = y_data.get("bui", [])

    records = []

    for i, date_value in enumerate(dates):

        record = {
            "date": date_value[:10]
        }

        if i < len(fwi):
            record["fwi"] = round(float(fwi[i]), 2)

        if i < len(ffmc):
            record["ffmc"] = round(float(ffmc[i]), 2)

        if i < len(dmc):
            record["dmc"] = round(float(dmc[i]), 2)

        if i < len(dc):
            record["dc"] = round(float(dc[i]), 2)

        if i < len(isi):
            record["isi"] = round(float(isi[i]), 2)

        if i < len(bui):
            record["bui"] = round(float(bui[i]), 2)

        records.append(record)

    return {
        "name": point["name"],
        "lon": point["lon"],
        "lat": point["lat"],
        "records": records
    }


# ============================================================
# برنامه اصلی
# ============================================================

def main():

    print("=" * 60)
    print("FARS-HRI | EFFIS Meteo-France FWI")
    print("=" * 60)

    print(f"Model      : {MODEL}")
    print(f"Start date : {START_DATE}")
    print(f"End date   : {END_DATE}")
    print()

    results = []

    for point in POINTS:

        print(
            f"Getting Meteo-France data for "
            f"{point['name']} "
            f"({point['lon']}, {point['lat']})"
        )

        try:

            data = get_point_data(point)

            parsed = parse_point(
                point,
                data
            )

            count = len(parsed["records"])

            print(f"  OK - {count} days")

            results.append(parsed)

        except Exception as e:

            print(f"  ERROR - {e}")

    # --------------------------------------------------------
    # خروجی
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    output = {
        "updated_at_utc": datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S"),

        "model": "Meteo-France",

        "source": "Copernicus EFFIS",

        "start_date": START_DATE,

        "end_date": END_DATE,

        "points": results
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
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Points: {len(results)}"
    )


# ============================================================
# اجرا
# ============================================================

if __name__ == "__main__":
    main()
```
