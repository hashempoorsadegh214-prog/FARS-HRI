
import json
import os
import requests
from datetime import datetime, timedelta

# ==========================================
# FARS FWI - Copernicus EFFIS
# ==========================================

API_URL = "https://api.effis.emergency.copernicus.eu/rest/2/burntareas/charts/wms"

# نقاط نمونه داخل استان فارس
POINTS = [
    {"name": "Shiraz", "lon": 52.5837, "lat": 29.5918},
    {"name": "Fars_Center", "lon": 53.1525, "lat": 28.7438},
    {"name": "Fars_East", "lon": 53.5004, "lat": 28.3619},
]

OUTPUT_FILE = "data/fwi_fars.json"

# بازه داده
START_DATE = "2026-08-23"
END_DATE = "2026-09-03"


def get_fwi(point):
    """
    دریافت داده FWI برای یک نقطه از API رسمی EFFIS
    """

    point_value = f"({point['lon']} {point['lat']})"

    params = {
        "model": "ecmwf",
        "day_gte": START_DATE,
        "day_lte": END_DATE,
        "point": point_value,
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def main():

    os.makedirs("data", exist_ok=True)

    results = []

    for point in POINTS:

        print(
            f"Getting FWI: "
            f"{point['name']} "
            f"({point['lon']}, {point['lat']})"
        )

        try:

            data = get_fwi(point)

            dates = data.get("x_data", [])
            fwi_values = data.get("y_data", {}).get("fwi", [])

            records = []

            for date, fwi in zip(dates, fwi_values):

                records.append({
                    "date": date[:10],
                    "fwi": round(float(fwi), 2)
                })

            results.append({
                "name": point["name"],
                "lon": point["lon"],
                "lat": point["lat"],
                "records": records
            })

            print(
                f"  OK - {len(records)} FWI values"
            )

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

    output = {
        "source": "Copernicus EFFIS",
        "model": "ECMWF",
        "updated_at_utc": datetime.utcnow().isoformat(),
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
    print("==========================================")
    print("FWI file created:")
    print(OUTPUT_FILE)
    print("==========================================")


if __name__ == "__main__":
    main()

