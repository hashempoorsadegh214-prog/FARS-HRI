#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
آرشیو روزانهٔ خروجی پیش‌بینی FWI و خلاصهٔ رستر HRI برای سامانه FARS-HRI

ورودی‌ها:
    data/fwi/fwi_fars_grid.json
    data/output/fars_hri.tif

خروجی‌ها:
    data/archive/YYYY-MM-DD.json
    data/archive_index.json

نکته:
- فایل روزانه شامل نقاط FWI موجود و خلاصهٔ آماری رستر HRI است.
- برای جلوگیری از سنگین‌شدن مخزن، خودِ رستر HRI برای هر روز کپی نمی‌شود.
  رستر جاری همچنان در data/output/fars_hri.tif باقی می‌ماند.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FWI_GRID_PATH = PROJECT_ROOT / "data" / "fwi" / "fwi_fars_grid.json"
HRI_RASTER_PATH = PROJECT_ROOT / "data" / "output" / "fars_hri.tif"

ARCHIVE_DIR = PROJECT_ROOT / "data" / "archive"
ARCHIVE_INDEX_PATH = PROJECT_ROOT / "data" / "archive_index.json"

MAX_ARCHIVE_DAYS = 90


def read_json(path: Path) -> Any:
    """خواندن فایل JSON با کدگذاری UTF-8."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    """ثبت فایل JSON با خوانایی مناسب و پشتیبانی از فارسی."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def to_float(value: Any) -> float | None:
    """تبدیل امن مقدار به عدد اعشاری."""
    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return None


def fwi_risk_level(fwi_value: float) -> str:
    """طبقه‌بندی استاندارد شدت خطر FWI."""
    if fwi_value < 11.2:
        return "کم"

    if fwi_value < 21.3:
        return "متوسط"

    if fwi_value < 38.0:
        return "زیاد"

    if fwi_value < 50.0:
        return "خیلی زیاد"

    if fwi_value < 70.0:
        return "شدید"

    return "بسیار شدید"


def hri_risk_level(hri_value: float | None) -> str:
    """
    طبقه‌بندی سادهٔ HRI نرمال‌شده در بازهٔ 0 تا 100.
    این مقدار برای نمایش خلاصهٔ مدیریتی است.
    """
    if hri_value is None:
        return "نامشخص"

    if hri_value < 20:
        return "کم"

    if hri_value < 40:
        return "متوسط"

    if hri_value < 60:
        return "زیاد"

    if hri_value < 80:
        return "خیلی زیاد"

    return "بسیار شدید"


def summarize_values(values: list[float]) -> dict[str, float | None]:
    """محاسبهٔ کمینه، میانگین و بیشینهٔ یک فهرست عددی."""
    if not values:
        return {
            "min": None,
            "mean": None,
            "max": None,
        }

    array = np.asarray(values, dtype=np.float64)

    return {
        "min": round(float(np.min(array)), 2),
        "mean": round(float(np.mean(array)), 2),
        "max": round(float(np.max(array)), 2),
    }


def calculate_hri_statistics(hri_path: Path) -> dict[str, float | int | None]:
    """محاسبهٔ آمار پیکسل‌های معتبر رستر HRI."""
    with rasterio.open(hri_path) as dataset:
        hri_data = dataset.read(1, masked=True)

        values = np.asarray(hri_data.compressed(), dtype=np.float64)
        values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "valid_pixels": 0,
            "min_hri": None,
            "mean_hri": None,
            "max_hri": None,
        }

    return {
        "valid_pixels": int(values.size),
        "min_hri": round(float(np.min(values)), 2),
        "mean_hri": round(float(np.mean(values)), 2),
        "max_hri": round(float(np.max(values)), 2),
    }


def clean_points(raw_points: Any) -> list[dict[str, float]]:
    """
    اعتبارسنجی و پاک‌سازی نقاط FWI.
    فقط نقاطی با lat، lon و fwi عددی در خروجی آرشیو ثبت می‌شوند.
    """
    if not isinstance(raw_points, list):
        return []

    cleaned_points: list[dict[str, float]] = []

    for item in raw_points:
        if not isinstance(item, dict):
            continue

        latitude = to_float(item.get("lat"))
        longitude = to_float(item.get("lon"))
        fwi_value = to_float(item.get("fwi"))

        if latitude is None or longitude is None or fwi_value is None:
            continue

        point: dict[str, float] = {
            "lat": round(latitude, 6),
            "lon": round(longitude, 6),
            "fwi": round(fwi_value, 2),
        }

        hri_value = to_float(item.get("hri"))

        if hri_value is not None:
            point["hri"] = round(hri_value, 2)

        cleaned_points.append(point)

    return cleaned_points


def load_existing_index() -> list[dict[str, Any]]:
    """خواندن ایمن فهرست آرشیو قبلی."""
    if not ARCHIVE_INDEX_PATH.exists():
        return []

    try:
        data = read_json(ARCHIVE_INDEX_PATH)

        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]

    except (OSError, json.JSONDecodeError) as error:
        print(f"Warning: Could not read archive index: {error}")

    return []


def remove_old_archives(index_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """فقط آخرین تعداد تعیین‌شده از آرشیوها را نگه می‌دارد."""
    sorted_items = sorted(
        index_items,
        key=lambda item: str(item.get("date_gregorian", "")),
        reverse=True,
    )

    kept_items = sorted_items[:MAX_ARCHIVE_DAYS]
    removed_items = sorted_items[MAX_ARCHIVE_DAYS:]

    for item in removed_items:
        relative_path = item.get("file")

        if not isinstance(relative_path, str):
            continue

        archive_file = PROJECT_ROOT / relative_path

        if archive_file.exists():
            archive_file.unlink()
            print(f"Removed old archive: {archive_file.relative_to(PROJECT_ROOT)}")

    return kept_items


def main() -> None:
    """اجرای فرآیند آرشیو روزانه."""
    if not FWI_GRID_PATH.is_file():
        raise FileNotFoundError(
            f"Required FWI grid JSON was not found: {FWI_GRID_PATH}"
        )

    if not HRI_RASTER_PATH.is_file():
        raise FileNotFoundError(
            f"Required HRI raster was not found: {HRI_RASTER_PATH}"
        )

    fwi_data = read_json(FWI_GRID_PATH)

    if not isinstance(fwi_data, dict):
        raise ValueError("FWI grid JSON must contain a JSON object.")

    forecast_gregorian = str(fwi_data.get("forecast_gregorian", "")).strip()

    if not forecast_gregorian:
        raise ValueError(
            "The key 'forecast_gregorian' is missing in fwi_fars_grid.json."
        )

    forecast_shamsi = str(fwi_data.get("forecast_shamsi", "")).strip()
    generated_at_iran = str(fwi_data.get("generated_at_iran", "")).strip()
    source = str(fwi_data.get("source", "")).strip()

    points = clean_points(fwi_data.get("points", []))
    fwi_values = [point["fwi"] for point in points]

    fwi_statistics = summarize_values(fwi_values)
    hri_statistics = calculate_hri_statistics(HRI_RASTER_PATH)

    archive_payload = {
        "forecast_gregorian": forecast_gregorian,
        "forecast_shamsi": forecast_shamsi,
        "generated_at_iran": generated_at_iran,
        "archived_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "source": source,
        "summary": {
            "points_count": len(points),
            "min_fwi": fwi_statistics["min"],
            "mean_fwi": fwi_statistics["mean"],
            "max_fwi": fwi_statistics["max"],
            "fwi_risk_level": (
                fwi_risk_level(fwi_statistics["max"])
                if fwi_statistics["max"] is not None
                else "نامشخص"
            ),
            "valid_hri_pixels": hri_statistics["valid_pixels"],
            "min_hri": hri_statistics["min_hri"],
            "mean_hri": hri_statistics["mean_hri"],
            "max_hri": hri_statistics["max_hri"],
            "hri_risk_level": hri_risk_level(hri_statistics["max_hri"]),
        },
        "points": points,
    }

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    archive_file_name = f"{forecast_gregorian}.json"
    archive_path = ARCHIVE_DIR / archive_file_name
    archive_relative_path = f"data/archive/{archive_file_name}"

    write_json(archive_path, archive_payload)

    index_items = load_existing_index()

    index_entry = {
        "date_gregorian": forecast_gregorian,
        "date_shamsi": forecast_shamsi,
        "file": archive_relative_path,
        "generated_at_iran": generated_at_iran,
        "points_count": len(points),
        "mean_fwi": fwi_statistics["mean"],
        "max_fwi": fwi_statistics["max"],
        "mean_hri": hri_statistics["mean_hri"],
        "max_hri": hri_statistics["max_hri"],
        "risk_level": hri_risk_level(hri_statistics["max_hri"]),
    }

    index_without_current_date = [
        item
        for item in index_items
        if str(item.get("date_gregorian", "")) != forecast_gregorian
    ]

    updated_index = [index_entry, *index_without_current_date]
    updated_index = remove_old_archives(updated_index)

    write_json(ARCHIVE_INDEX_PATH, updated_index)

    print("Daily FWI/HRI archive created successfully. ✅")
    print(f"Archive file: {archive_path.relative_to(PROJECT_ROOT)}")
    print(f"Archive index: {ARCHIVE_INDEX_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Forecast date: {forecast_gregorian}")
    print(f"Archived FWI points: {len(points)}")
    print(f"Valid HRI pixels: {hri_statistics['valid_pixels']}")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(f"Archive creation failed: {error}", file=sys.stderr)
        sys.exit(1)
