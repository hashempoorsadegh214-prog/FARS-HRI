#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
دریافت و اعتبارسنجی دادهٔ آمادهٔ FWI استان فارس.

منبع:
    مخزن عمومی FARS-FWI
    https://raw.githubusercontent.com/hashempoorsadegh214-prog/FARS-FWI/main/data/fwi_fars.json

خروجی محلی در پروژه FARS-HRI:
    data/fwi/fwi_fars_grid.json

این اسکریپت عمداً از WMS، EFFIS endpoint و دریافت مستقیم متئو استفاده
نمی‌کند؛ منبع FWI آماده و قابل‌استفاده را از پروژهٔ FARS-FWI می‌گیرد.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests


# ---------------------------------------------------------------------
# تنظیمات پروژه
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FWI_URL = (
    "https://raw.githubusercontent.com/"
    "hashempoorsadegh214-prog/FARS-FWI/main/data/fwi_fars.json"
)

OUTPUT_FWI_PATH = PROJECT_ROOT / "data" / "fwi" / "fwi_fars_grid.json"

REQUEST_TIMEOUT_SECONDS = 60

# حداقل تعداد نقطه برای جلوگیری از پذیرش فایل ناقص یا دادهٔ آزمایشی.
MIN_EXPECTED_POINT_COUNT = 1000

# محدودهٔ منطقی دادهٔ استان فارس با اندکی حاشیه.
MIN_EXPECTED_LONGITUDE = 49.0
MAX_EXPECTED_LONGITUDE = 56.0
MIN_EXPECTED_LATITUDE = 26.0
MAX_EXPECTED_LATITUDE = 33.0


# ---------------------------------------------------------------------
# توابع کمکی
# ---------------------------------------------------------------------

def get_points(data: Any) -> list[dict[str, Any]]:
    """
    آرایهٔ نقاط FWI را از ساختار استاندارد منبع استخراج می‌کند.

    ساختار مورد انتظار:
    {
      "forecast_gregorian": "...",
      "forecast_shamsi": "...",
      "points": [
        {"lat": ..., "lon": ..., "fwi": ...}
      ]
    }
    """
    if not isinstance(data, dict):
        raise ValueError(
            "ساختار دادهٔ دریافتی باید یک شیء JSON باشد، اما این‌گونه نیست."
        )

    points = data.get("points")

    if not isinstance(points, list):
        raise ValueError(
            "کلید 'points' در فایل FWI وجود ندارد یا آرایه نیست."
        )

    if not points:
        raise ValueError("آرایهٔ 'points' خالی است.")

    return points


def validate_fwi_data(data: Any) -> tuple[int, float, float, float, float, float, float]:
    """
    دادهٔ FWI را اعتبارسنجی می‌کند و آمار لازم را برمی‌گرداند.

    خروجی:
        تعداد نقاط،
        کمینه/بیشینه طول،
        کمینه/بیشینه عرض،
        کمینه/بیشینه FWI
    """
    points = get_points(data)

    if len(points) < MIN_EXPECTED_POINT_COUNT:
        raise ValueError(
            "تعداد نقاط FWI کمتر از مقدار قابل‌قبول است.\n"
            f"تعداد دریافت‌شده: {len(points):,}\n"
            f"حداقل مورد انتظار: {MIN_EXPECTED_POINT_COUNT:,}"
        )

    longitudes: list[float] = []
    latitudes: list[float] = []
    fwi_values: list[float] = []

    for record_number, point in enumerate(points, start=1):
        if not isinstance(point, dict):
            raise ValueError(
                f"نقطهٔ شمارهٔ {record_number} یک شیء JSON معتبر نیست."
            )

        try:
            longitude = float(point["lon"])
            latitude = float(point["lat"])
            fwi_value = float(point["fwi"])
        except KeyError as exc:
            raise ValueError(
                f"کلید لازم در نقطهٔ شمارهٔ {record_number} وجود ندارد: {exc}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"مقادیر lon، lat یا fwi در نقطهٔ شمارهٔ "
                f"{record_number} عددی و معتبر نیستند."
            ) from exc

        if not np.isfinite(longitude):
            raise ValueError(
                f"طول جغرافیایی نامعتبر در نقطهٔ {record_number}: {longitude}"
            )

        if not np.isfinite(latitude):
            raise ValueError(
                f"عرض جغرافیایی نامعتبر در نقطهٔ {record_number}: {latitude}"
            )

        if not np.isfinite(fwi_value):
            raise ValueError(
                f"مقدار FWI نامعتبر در نقطهٔ {record_number}: {fwi_value}"
            )

        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(
                f"طول جغرافیایی خارج از دامنه در نقطهٔ {record_number}: {longitude}"
            )

        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(
                f"عرض جغرافیایی خارج از دامنه در نقطهٔ {record_number}: {latitude}"
            )

        if fwi_value < 0.0:
            raise ValueError(
                f"FWI منفی در نقطهٔ {record_number}: {fwi_value}"
            )

        longitudes.append(longitude)
        latitudes.append(latitude)
        fwi_values.append(fwi_value)

    longitude_array = np.asarray(longitudes, dtype=np.float64)
    latitude_array = np.asarray(latitudes, dtype=np.float64)
    fwi_array = np.asarray(fwi_values, dtype=np.float64)

    coordinates = np.column_stack((longitude_array, latitude_array))
    unique_coordinates = np.unique(coordinates, axis=0)

    if len(unique_coordinates) != len(coordinates):
        duplicate_count = len(coordinates) - len(unique_coordinates)

        raise ValueError(
            "در دادهٔ FWI مختصات تکراری وجود دارد.\n"
            f"تعداد نقاط تکراری: {duplicate_count:,}"
        )

    min_lon = float(longitude_array.min())
    max_lon = float(longitude_array.max())
    min_lat = float(latitude_array.min())
    max_lat = float(latitude_array.max())
    min_fwi = float(fwi_array.min())
    max_fwi = float(fwi_array.max())

    if (
        min_lon < MIN_EXPECTED_LONGITUDE
        or max_lon > MAX_EXPECTED_LONGITUDE
        or min_lat < MIN_EXPECTED_LATITUDE
        or max_lat > MAX_EXPECTED_LATITUDE
    ):
        raise ValueError(
            "محدودهٔ مختصات فایل FWI با محدودهٔ منطقی استان فارس سازگار نیست.\n"
            f"طول جغرافیایی: {min_lon:.4f} تا {max_lon:.4f}\n"
            f"عرض جغرافیایی: {min_lat:.4f} تا {max_lat:.4f}"
        )

    return (
        len(points),
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        min_fwi,
        max_fwi,
    )


def download_fwi_data() -> Any:
    """فایل JSON را از مخزن FARS-FWI دریافت می‌کند."""
    headers = {
        "Accept": "application/json",
        "User-Agent": "FARS-HRI-FWI-Updater/1.0",
    }

    try:
        response = requests.get(
            SOURCE_FWI_URL,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "دانلود فایل FWI از مخزن FARS-FWI ناموفق بود.\n"
            f"آدرس: {SOURCE_FWI_URL}\n"
            f"جزئیات: {exc}"
        ) from exc

    if not response.content:
        raise RuntimeError("پاسخ دریافت‌شده برای فایل FWI خالی است.")

    try:
        return response.json()
    except ValueError as exc:
        preview = response.text[:300].replace("\n", " ")

        raise ValueError(
            "پاسخ منبع، JSON معتبر نیست.\n"
            f"ابتدای پاسخ: {preview}"
        ) from exc


def save_fwi_data(data: Any) -> None:
    """
    JSON را ابتدا در فایل موقت می‌نویسد و سپس به‌صورت اتمی جایگزین می‌کند
    تا فایل ناقص در مخزن باقی نماند.
    """
    OUTPUT_FWI_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_output_path = OUTPUT_FWI_PATH.with_suffix(".json.tmp")

    with temporary_output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")

    temporary_output_path.replace(OUTPUT_FWI_PATH)


def main() -> int:
    """نقطهٔ شروع برنامه."""
    print("=" * 72)
    print("دریافت دادهٔ آمادهٔ FWI فارس")
    print("=" * 72)
    print(f"منبع: {SOURCE_FWI_URL}")
    print(f"مقصد: {OUTPUT_FWI_PATH}")
    print()

    try:
        print("در حال دانلود JSON FWI ...")
        fwi_data = download_fwi_data()

        print("در حال اعتبارسنجی داده ...")
        (
            point_count,
            min_lon,
            max_lon,
            min_lat,
            max_lat,
            min_fwi,
            max_fwi,
        ) = validate_fwi_data(fwi_data)

        source_name = str(fwi_data.get("source", "نامشخص"))
        forecast_date = str(fwi_data.get("forecast_gregorian", "نامشخص"))
        forecast_shamsi = str(fwi_data.get("forecast_shamsi", "نامشخص"))

        fwi_data["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat()
        fwi_data["download_source"] = SOURCE_FWI_URL

        print("در حال ذخیره‌سازی فایل معتبر ...")
        save_fwi_data(fwi_data)

        print()
        print("=" * 72)
        print("دادهٔ FWI با موفقیت دریافت و ذخیره شد. ✅")
        print("=" * 72)
        print(f"تاریخ پیش‌بینی میلادی: {forecast_date}")
        print(f"تاریخ پیش‌بینی شمسی: {forecast_shamsi}")
        print(f"منبع اعلام‌شده: {source_name}")
        print(f"تعداد نقاط: {point_count:,}")
        print(f"محدوده طول جغرافیایی: {min_lon:.4f} تا {max_lon:.4f}")
        print(f"محدوده عرض جغرافیایی: {min_lat:.4f} تا {max_lat:.4f}")
        print(f"محدوده FWI: {min_fwi:.3f} تا {max_fwi:.3f}")
        print(f"فایل ذخیره‌شده: {OUTPUT_FWI_PATH}")

        return 0

    except Exception as exc:
        print()
        print("=" * 72, file=sys.stderr)
        print("خطا در دریافت یا اعتبارسنجی دادهٔ FWI", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print(str(exc), file=sys.stderr)

        return 1


if __name__ == "__main__":
    sys.exit(main())
