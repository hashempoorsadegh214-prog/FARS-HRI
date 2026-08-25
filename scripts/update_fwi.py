#!/usr/bin/env python3
"""
FARS-HRI | Daily Canadian Fire Weather Index (FWI) updater

What this script does:
1. Reads the Fars Province boundary from fars.geojson.
2. Selects a representative point inside the province.
3. Downloads daily weather from Open-Meteo Archive API.
4. Calculates Canadian FWI components:
   FFMC, DMC, DC, ISI, BUI and FWI.
5. Saves results to data/fwi/fwi_fars.json.

Notes:
- Weather variables are sampled at 12:00 local time (Asia/Tehran).
- Daily precipitation is used as the previous 24-hour rainfall proxy.
- On the first run, calculations begin from 1 January of the current year.
- On later runs, only missing dates are calculated.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_FILE = PROJECT_ROOT / "fars.geojson"
OUTPUT_DIR = PROJECT_ROOT / "data" / "fwi"
OUTPUT_FILE = OUTPUT_DIR / "fwi_fars.json"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

TIMEZONE_NAME = "Asia/Tehran"
HTTP_TIMEOUT_SECONDS = 60

# Standard Canadian FWI initial values.
INITIAL_FFMC = 85.0
INITIAL_DMC = 6.0
INITIAL_DC = 15.0

# The script processes completed days only.
# This avoids using incomplete weather observations for the current day.
END_DATE = date.today() - timedelta(days=1)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def clamp(value: float, minimum: float, maximum: float) -> float:
    """Limit a number to a specified range."""
    return max(minimum, min(value, maximum))


def round_value(value: float, digits: int = 3) -> float:
    """Round values consistently for JSON output."""
    return round(float(value), digits)


def read_json_file(file_path: Path) -> dict[str, Any]:
    """Read an existing JSON file safely."""
    if not file_path.exists():
        return {}

    try:
        with file_path.open("r", encoding="utf-8") as file:
            content = json.load(file)

        if isinstance(content, dict):
            return content

    except (json.JSONDecodeError, OSError):
        pass

    return {}


def save_json_file(file_path: Path, data: dict[str, Any]) -> None:
    """Save JSON output with readable formatting."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


# ---------------------------------------------------------------------
# Geographic location
# ---------------------------------------------------------------------

def get_fars_representative_point() -> tuple[float, float]:
    """
    Read Fars boundary and return a point guaranteed to be inside it.

    Returns:
        (latitude, longitude)
    """
    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Boundary file was not found: {BOUNDARY_FILE}"
        )

    boundary = gpd.read_file(BOUNDARY_FILE)

    if boundary.empty:
        raise ValueError("fars.geojson contains no geographic features.")

    # Ensure coordinates are longitude/latitude (EPSG:4326).
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    else:
        boundary = boundary.to_crs("EPSG:4326")

    merged_geometry = boundary.geometry.union_all()

    if merged_geometry.is_empty:
        raise ValueError("Fars boundary geometry is empty.")

    # representative_point() is safer than centroid because it is inside
    # the province even for complex polygon shapes.
    point = merged_geometry.representative_point()

    return float(point.y), float(point.x)


# ---------------------------------------------------------------------
# Open-Meteo weather retrieval
# ---------------------------------------------------------------------

def fetch_weather(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    """
    Download weather values needed for Canadian FWI.

    Required variables:
    - temperature_2m
    - relative_humidity_2m
    - wind_speed_10m
    - precipitation

    Temperature, humidity and wind are taken at 12:00 local time.
    Total precipitation is calculated for each local calendar day.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "precipitation"
        ),
        "timezone": TIMEZONE_NAME,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params=params,
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    payload = response.json()

    if "hourly" not in payload:
        reason = payload.get("reason", "Unknown Open-Meteo response.")
        raise RuntimeError(f"Open-Meteo did not return hourly weather: {reason}")

    hourly = payload["hourly"]

    times = hourly.get("time", [])
    temperatures = hourly.get("temperature_2m", [])
    humidities = hourly.get("relative_humidity_2m", [])
    winds = hourly.get("wind_speed_10m", [])
    precipitation = hourly.get("precipitation", [])

    expected_length = len(times)

    if not all(
        len(values) == expected_length
        for values in (temperatures, humidities, winds, precipitation)
    ):
        raise RuntimeError("Open-Meteo returned weather arrays with inconsistent lengths.")

    daily_weather: dict[str, dict[str, Any]] = {}

    for time_text, temperature, humidity, wind, rain in zip(
        times,
        temperatures,
        humidities,
        winds,
        precipitation,
    ):
        if None in (temperature, humidity, wind, rain):
            continue

        local_datetime = datetime.fromisoformat(time_text)
        day_key = local_datetime.date().isoformat()

        if day_key not in daily_weather:
            daily_weather[day_key] = {
                "date": day_key,
                "temperature_c": None,
                "relative_humidity_pct": None,
                "wind_speed_kmh": None,
                "rainfall_mm": 0.0,
            }

        # Open-Meteo precipitation is hourly precipitation in millimetres.
        daily_weather[day_key]["rainfall_mm"] += max(0.0, float(rain))

        # Canadian FWI convention uses noon weather observations.
        if local_datetime.hour == 12:
            daily_weather[day_key]["temperature_c"] = float(temperature)
            daily_weather[day_key]["relative_humidity_pct"] = float(humidity)
            daily_weather[day_key]["wind_speed_kmh"] = float(wind)

    valid_days: list[dict[str, Any]] = []

    for day_key in sorted(daily_weather):
        item = daily_weather[day_key]

        if None in (
            item["temperature_c"],
            item["relative_humidity_pct"],
            item["wind_speed_kmh"],
        ):
            continue

        item["temperature_c"] = round_value(item["temperature_c"], 2)
        item["relative_humidity_pct"] = round_value(
            clamp(item["relative_humidity_pct"], 0.0, 100.0),
            2,
        )
        item["wind_speed_kmh"] = round_value(
            max(0.0, item["wind_speed_kmh"]),
            2,
        )
        item["rainfall_mm"] = round_value(item["rainfall_mm"], 2)

        valid_days.append(item)

    return valid_days


# ---------------------------------------------------------------------
# Canadian Fire Weather Index equations
# ---------------------------------------------------------------------

def calculate_ffmc(
    previous_ffmc: float,
    temperature_c: float,
    relative_humidity_pct: float,
    wind_speed_kmh: float,
    rainfall_mm: float,
) -> float:
    """Calculate Fine Fuel Moisture Code (FFMC)."""
    ffmc = clamp(previous_ffmc, 0.0, 101.0)
    temperature = temperature_c
    humidity = clamp(relative_humidity_pct, 0.0, 100.0)
    wind = max(0.0, wind_speed_kmh)
    rain = max(0.0, rainfall_mm)

    moisture = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)

    if rain > 0.5:
        effective_rain = rain - 0.5

        if moisture > 150.0:
            moisture += (
                42.5
                * effective_rain
                * math.exp(-100.0 / (251.0 - moisture))
                * (1.0 - math.exp(-6.93 / effective_rain))
            ) + (
                0.0015
                * (moisture - 150.0) ** 2
                * math.sqrt(effective_rain)
            )
        else:
            moisture += (
                42.5
                * effective_rain
                * math.exp(-100.0 / (251.0 - moisture))
                * (1.0 - math.exp(-6.93 / effective_rain))
            )

        moisture = min(moisture, 250.0)

    equilibrium_drying = (
        0.942 * humidity**0.679
        + 11.0 * math.exp((humidity - 100.0) / 10.0)
        + 0.18 * (21.1 - temperature) * (1.0 - math.exp(-0.115 * humidity))
    )

    equilibrium_wetting = (
        0.618 * humidity**0.753
        + 10.0 * math.exp((humidity - 100.0) / 10.0)
        + 0.18 * (21.1 - temperature) * (1.0 - math.exp(-0.115 * humidity))
    )

    if moisture < equilibrium_drying:
        drying_rate = (
            0.424 * (1.0 - (humidity / 100.0) ** 1.7)
            + 0.0694 * math.sqrt(wind) * (1.0 - (humidity / 100.0) ** 8)
        )
        drying_rate *= 0.581 * math.exp(0.0365 * temperature)

        moisture = equilibrium_wetting - (
            (equilibrium_wetting - moisture) * 10.0 ** (-drying_rate)
        )

    elif moisture > equilibrium_drying:
        wetting_rate = (
            0.424 * (1.0 - ((100.0 - humidity) / 100.0) ** 1.7)
            + 0.0694 * math.sqrt(wind) * (1.0 - ((100.0 - humidity) / 100.0) ** 8)
        )
        wetting_rate *= 0.581 * math.exp(0.0365 * temperature)

        moisture = equilibrium_drying + (
            (moisture - equilibrium_drying) * 10.0 ** (-wetting_rate)
        )

    moisture = clamp(moisture, 0.0, 250.0)

    result = 59.5 * (250.0 - moisture) / (147.2 + moisture)

    return clamp(result, 0.0, 101.0)


def calculate_dmc(
    previous_dmc: float,
    temperature_c: float,
    relative_humidity_pct: float,
    rainfall_mm: float,
    month: int,
) -> float:
    """Calculate Duff Moisture Code (DMC)."""
    dmc = max(0.0, previous_dmc)
    temperature = max(-1.1, temperature_c)
    humidity = clamp(relative_humidity_pct, 0.0, 100.0)
    rain = max(0.0, rainfall_mm)

    # Day-length adjustment factors for Northern Hemisphere.
    day_length_factors = [
        6.5, 7.5, 9.0, 12.8, 13.9, 13.9,
        12.4, 10.9, 9.4, 8.0, 7.0, 6.0,
    ]
    day_length_factor = day_length_factors[month - 1]

    if rain > 1.5:
        effective_rain = 0.92 * rain - 1.27
        initial_moisture = 20.0 + math.exp(5.6348 - dmc / 43.43)

        if dmc <= 33.0:
            b_factor = 100.0 / (0.5 + 0.3 * dmc)
        elif dmc <= 65.0:
            b_factor = 14.0 - 1.3 * math.log(dmc)
        else:
            b_factor = 6.2 * math.log(dmc) - 17.2

        final_moisture = initial_moisture + (
            1000.0 * effective_rain / (48.77 + b_factor * effective_rain)
        )

        if final_moisture > 20.0:
            dmc = 43.43 * (5.6348 - math.log(final_moisture - 20.0))
        else:
            dmc = 0.0

        dmc = max(0.0, dmc)

    drying = 1.894 * (temperature + 1.1) * (100.0 - humidity) * day_length_factor * 0.000001

    return max(0.0, dmc + drying)


def calculate_dc(
    previous_dc: float,
    temperature_c: float,
    rainfall_mm: float,
    month: int,
) -> float:
    """Calculate Drought Code (DC)."""
    dc = max(0.0, previous_dc)
    temperature = max(-2.8, temperature_c)
    rain = max(0.0, rainfall_mm)

    # Seasonal adjustment factors for Northern Hemisphere.
    seasonal_factors = [
        -1.6, -1.6, -1.6, 0.9, 3.8, 5.8,
        6.4, 5.0, 2.4, 0.4, -1.6, -1.6,
    ]
    seasonal_factor = seasonal_factors[month - 1]

    if rain > 2.8:
        effective_rain = 0.83 * rain - 1.27
        initial_moisture = 800.0 * math.exp(-dc / 400.0)
        final_moisture = initial_moisture + 3.937 * effective_rain

        dc = 400.0 * math.log(800.0 / final_moisture)
        dc = max(0.0, dc)

    drying = 0.36 * (temperature + 2.8) + seasonal_factor
    drying = max(0.0, drying)

    return dc + 0.5 * drying


def calculate_isi(ffmc: float, wind_speed_kmh: float) -> float:
    """Calculate Initial Spread Index (ISI)."""
    moisture = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)

    fuel_moisture_function = (
        91.9
        * math.exp(-0.1386 * moisture)
        * (1.0 + moisture**5.31 / 49_300_000.0)
    )

    wind_function = math.exp(0.05039 * max(0.0, wind_speed_kmh))

    return 0.208 * wind_function * fuel_moisture_function


def calculate_bui(dmc: float, dc: float) -> float:
    """Calculate Build Up Index (BUI)."""
    if dmc <= 0.4 * dc:
        denominator = dmc + 0.4 * dc

        if denominator == 0.0:
            return 0.0

        return (0.8 * dmc * dc) / denominator

    denominator = dmc + 0.4 * dc

    if denominator == 0.0:
        return 0.0

    return dmc - (
        (1.0 - (0.8 * dc / denominator))
        * (0.92 + (0.0114 * dmc) ** 1.7)
    )


def calculate_fwi(isi: float, bui: float) -> float:
    """Calculate final Fire Weather Index (FWI)."""
    if bui <= 80.0:
        duff_moisture_function = 0.626 * bui**0.809 + 2.0
    else:
        duff_moisture_function = (
            1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui))
        )

    intermediate = 0.1 * isi * duff_moisture_function

    if intermediate <= 1.0:
        return max(0.0, intermediate)

    return math.exp(
        2.72 * (0.434 * math.log(intermediate)) ** 0.647
    )


# ---------------------------------------------------------------------
# Main update process
# ---------------------------------------------------------------------

def determine_start_date(existing_data: dict[str, Any]) -> date:
    """
    Decide the first date that still needs FWI calculation.

    If no valid previous calculation exists, start on 1 January of
    the current year with standard Canadian FWI initial values.
    """
    last_calculated_date = existing_data.get("last_calculated_date")

    if isinstance(last_calculated_date, str):
        try:
            return date.fromisoformat(last_calculated_date) + timedelta(days=1)
        except ValueError:
            pass

    return date(END_DATE.year, 1, 1)


def get_previous_codes(existing_data: dict[str, Any]) -> tuple[float, float, float]:
    """Get previously calculated FFMC, DMC and DC values."""
    latest_codes = existing_data.get("latest_codes", {})

    if not isinstance(latest_codes, dict):
        latest_codes = {}

    ffmc = float(latest_codes.get("ffmc", INITIAL_FFMC))
    dmc = float(latest_codes.get("dmc", INITIAL_DMC))
    dc = float(latest_codes.get("dc", INITIAL_DC))

    return (
        clamp(ffmc, 0.0, 101.0),
        max(0.0, dmc),
        max(0.0, dc),
    )


def main() -> None:
    """Run the daily FWI update."""
    print("Starting FARS-HRI FWI update...")

    if END_DATE < date(END_DATE.year, 1, 1):
        print("No completed date is available for calculation.")
        return

    existing_data = read_json_file(OUTPUT_FILE)
    start_date = determine_start_date(existing_data)

    if start_date > END_DATE:
        print(
            "FWI data is already up to date. "
            f"Last calculated date: {existing_data.get('last_calculated_date')}"
        )
        return

    latitude, longitude = get_fars_representative_point()

    print(f"Representative point: latitude={latitude:.5f}, longitude={longitude:.5f}")
    print(f"Downloading weather from {start_date.isoformat()} to {END_DATE.isoformat()}...")

    weather_days = fetch_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=END_DATE,
    )

    if not weather_days:
        raise RuntimeError("No usable weather records were returned by Open-Meteo.")

    ffmc, dmc, dc = get_previous_codes(existing_data)

    existing_history = existing_data.get("daily_history", [])

    if not isinstance(existing_history, list):
        existing_history = []

    new_history: list[dict[str, Any]] = []

    for weather in weather_days:
        weather_date = date.fromisoformat(weather["date"])

        ffmc = calculate_ffmc(
            previous_ffmc=ffmc,
            temperature_c=weather["temperature_c"],
            relative_humidity_pct=weather["relative_humidity_pct"],
            wind_speed_kmh=weather["wind_speed_kmh"],
            rainfall_mm=weather["rainfall_mm"],
        )

        dmc = calculate_dmc(
            previous_dmc=dmc,
            temperature_c=weather["temperature_c"],
            relative_humidity_pct=weather["relative_humidity_pct"],
            rainfall_mm=weather["rainfall_mm"],
            month=weather_date.month,
        )

        dc = calculate_dc(
            previous_dc=dc,
            temperature_c=weather["temperature_c"],
            rainfall_mm=weather["rainfall_mm"],
            month=weather_date.month,
        )

        isi = calculate_isi(
            ffmc=ffmc,
            wind_speed_kmh=weather["wind_speed_kmh"],
        )

        bui = calculate_bui(dmc=dmc, dc=dc)
        fwi = calculate_fwi(isi=isi, bui=bui)

        new_history.append(
            {
                "date": weather["date"],
                "weather": {
                    "temperature_c": weather["temperature_c"],
                    "relative_humidity_pct": weather["relative_humidity_pct"],
                    "wind_speed_kmh": weather["wind_speed_kmh"],
                    "rainfall_mm": weather["rainfall_mm"],
                },
                "ffmc": round_value(ffmc),
                "dmc": round_value(dmc),
                "dc": round_value(dc),
                "isi": round_value(isi),
                "bui": round_value(bui),
                "fwi": round_value(fwi),
            }
        )

    # Keep a maximum of 400 days in the JSON file.
    # This is enough for monitoring and keeps the repository lightweight.
    combined_history = (existing_history + new_history)[-400:]

    latest = combined_history[-1]

    output_data = {
        "project": "FARS-HRI",
        "index_system": "Canadian Fire Weather Index System",
        "weather_source": "Open-Meteo Archive API",
        "timezone": TIMEZONE_NAME,
        "location": {
            "name": "Fars Province representative point",
            "latitude": round_value(latitude, 6),
            "longitude": round_value(longitude, 6),
        },
        "last_calculated_date": latest["date"],
        "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "latest_codes": {
            "ffmc": latest["ffmc"],
            "dmc": latest["dmc"],
            "dc": latest["dc"],
            "isi": latest["isi"],
            "bui": latest["bui"],
            "fwi": latest["fwi"],
        },
        "daily_history": combined_history,
    }

    save_json_file(OUTPUT_FILE, output_data)

    print(f"FWI update completed successfully: {OUTPUT_FILE}")
    print(
        "Latest FWI | "
        f"date={latest['date']} | "
        f"FFMC={latest['ffmc']} | "
        f"DMC={latest['dmc']} | "
        f"DC={latest['dc']} | "
        f"FWI={latest['fwi']}"
    )


if __name__ == "__main__":
    main()
