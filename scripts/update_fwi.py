#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اعتبارسنجی دادهٔ شبکه‌ای FWI استان فارس.

سیاست پروژه:
    دادهٔ FWI از فایل شبکه‌ای محلی و معتبر زیر خوانده می‌شود:
        data/fwi/fwi_fars_grid.json

    این اسکریپت عمداً هیچ درخواست WMS، EFFIS یا ECMWF ارسال نمی‌کند.
    دانلود مستقیم WMS در مراحل قبلی ناپایدار بود و با خطای HTTP 404
    مواجه می‌شد.

وظیفه:
    1. بررسی وجود فایل JSON گریدی FWI
    2. شناسایی رکوردهای دارای lat، lon و fwi
    3. اعتبارسنجی مختصات، مقادیر FWI و نقاط تکراری
    4. ثبت خلاصهٔ داده در خروجی اجرای GitHub Actions

خروجی:
    در صورت معتبر بودن داده، برنامه با exit code صفر تمام می‌شود.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------
# مسیرهای پروژه
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FWI_JSON_PATH = PROJECT_ROOT / "data" / "fwi" / "fwi_fars_grid.json"

# دامنهٔ مورد انتظار برای دادهٔ شبکه‌ای فارس
MIN_EXPECTED_POINT_COUNT = 1000
MIN_EXPECTED_LON = 49.0
MAX_EXPECTED_LON = 56.0
MIN_EXPECTED_LAT = 26.0
MAX_EXPECTED_LAT = 33.0


# ---------------------------------------------------------------------
# خواندن داده
# ---------------------------------------------------------------------

def find_point_records(data: Any) -> list[dict[str, Any]]:
    """
    آرایه‌ای از رکوردهای دارای lat، lon و fwi را در JSON پیدا می‌کند.

    فایل ممکن است دارای یک لایهٔ متادیتا یا کلیدهای تو در تو باشد؛
    بنابراین نام کلید بیرونی ثابت فرض نمی‌شود.
    """
    if isinstance(data, list):
        if data and all(
            isinstance(item, dict)
            and {"lat", "lon", "fwi"}.issubset(item.keys())
            for item in data
        ):
            return data

        for item in data:
            result = find_point_records(item)
            if result:
                return result

    if isinstance(data, dict):
        for value in data.values():
            result = find_point_records(value)
            if result:
                return result

    return []


def load_and_validate_fwi_data(
    json_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    JSON را می‌خواند، رکوردهای FWI را استخراج می‌کند و اعتبارسنجی انجام می‌دهد.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            "فایل گرید FWI پیدا نشد.\n\n"
            f"مسیر مورد انتظار:\n{json_path}\n\n"
            "فایل دادهٔ معتبر FWI را با نام زیر در مخزن قرار دهید:\n"
            "data/fwi/fwi_fars_grid.json"
        )

    if json_path.stat().st_size == 0:
        raise ValueError(
            f"فایل گرید FWI خالی است:\n{json_path}"
        )

    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"ساختار JSON فایل FWI معتبر نیست:\n{json_path}\n"
            f"جزئیات خطا: {exc}"
        ) from exc

    records = find_point_records(data)

    if not records:
        raise ValueError(
            "هیچ آرایه‌ای شامل رکوردهای FWI با کلیدهای "
            "'lat'، 'lon' و 'fwi' در فایل JSON پیدا نشد."
        )

    longitudes: list[float] = []
    latitudes: list[float] = []
    fwi_values: list[float] = []

    for index, record in enumerate(records, start=1):
        try:
            longitude = float(record["lon"])
            latitude = float(record["lat"])
            fwi_value = float(record["fwi"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"رکورد شمارهٔ {index} معتبر نیست:\n{record}"
            ) from exc

        if not np.isfinite(longitude):
            raise ValueError(f"طول جغرافیایی نامعتبر در رکورد {index}: {longitude}")

        if not np.isfinite(latitude):
            raise ValueError(f"عرض جغرافیایی نامعتبر در رکورد {index}: {latitude}")

        if not np.isfinite(fwi_value):
            raise ValueError(f"مقدار FWI نامعتبر در رکورد {index}: {fwi_value}")

        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"طول جغرافیایی خارج از دامنه در رکورد {index}: {longitude}")

        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(f"عرض جغرافیایی خارج از دامنه در رکورد {index}: {latitude}")

        if fwi_value < 0.0:
            raise ValueError(f"مقدار منفی FWI در رکورد {index}: {fwi_value}")

        longitudes.append(longitude)
        latitudes.append(latitude)
        fwi_values.append(fwi_value)

    lon_array = np.asarray(longitudes, dtype=np.float64)
    lat_array = np.asarray(latitudes, dtype=np.float64)
    fwi_array = np.asarray(fwi_values, dtype=np.float32)

    if len(fwi_array) < MIN_EXPECTED_POINT_COUNT:
        raise ValueError(
            "تعداد نقاط FWI کمتر از حد قابل‌قبول است.\n"
            f"تعداد موجود: {len(fwi_array):,}\n"
            f"حداقل مورد انتظار: {MIN_EXPECTED_POINT_COUNT:,}\n\n"
            "به‌جای دادهٔ نقطه‌ای یا ناقص، فایل گریدی کامل فارس را استفاده کنید."
        )

    coordinates = np.column_stack((lon_array, lat_array))
    unique_coordinates = np.unique(coordinates, axis=0)

    if len(unique_coordinates) != len(coordinates):
        duplicate_count = len(coordinates) - len(unique_coordinates)
        raise ValueError(
            "مختصات تکراری در دادهٔ FWI وجود دارد.\n"
            f"تعداد نقاط تکراری: {duplicate_count:,}"
        )

    if (
        lon_array.min() < MIN_EXPECTED_LON
        or lon_array.max() > MAX_EXPECTED_LON
        or lat_array.min() < MIN_EXPECTED_LAT
        or lat_array.max() > MAX_EXPECTED_LAT
    ):
        raise ValueError(
            "محدودهٔ مختصات داده با محدودهٔ منطقی استان فارس سازگار نیست.\n"
            f"Lon: {lon_array.min():.4f} تا {lon_array.max():.4f}\n"
            f"Lat: {lat_array.min():.4f} تا {lat_array.max():.4f}"
        )

    return lon_array, lat_array, fwi_array


# ---------------------------------------------------------------------
# اجرای اصلی
# ---------------------------------------------------------------------

def main() -> int:
    """اجرای اعتبارسنجی فایل گریدی FWI."""
    print("=" * 70)
    print("اعتبارسنجی دادهٔ شبکه‌ای FWI فارس")
    print("=" * 70)
    print("حالت داده: Local gridded JSON")
    print("درخواست WMS/اینترنت: غیرفعال")
    print()

    try:
        longitudes, latitudes, fwi_values = load_and_validate_fwi_data(
            FWI_JSON_PATH
        )

        unique_lons = np.unique(longitudes)
        unique_lats = np.unique(latitudes)

        lon_steps = np.diff(unique_lons)
        lat_steps = np.diff(unique_lats)

        lon_step = float(np.median(lon_steps)) if len(lon_steps) else 0.0
        lat_step = float(np.median(lat_steps)) if len(lat_steps) else 0.0

        print(f"فایل معتبر: {FWI_JSON_PATH}")
        print(f"تعداد نقاط: {len(fwi_values):,}")
        print(
            f"محدودهٔ طول جغرافیایی: "
            f"{longitudes.min():.4f} تا {longitudes.max():.4f}"
        )
        print(
            f"محدودهٔ عرض جغرافیایی: "
            f"{latitudes.min():.4f} تا {latitudes.max():.4f}"
        )
        print(f"تعداد طول‌های یکتا: {len(unique_lons):,}")
        print(f"تعداد عرض‌های یکتا: {len(unique_lats):,}")
        print(f"گام تقریبی طولی شبکه: {lon_step:.4f} درجه")
        print(f"گام تقریبی عرضی شبکه: {lat_step:.4f} درجه")
        print(
            f"FWI حداقل: {fwi_values.min():.3f} | "
            f"حداکثر: {fwi_values.max():.3f} | "
            f"میانگین: {fwi_values.mean():.3f}"
        )
        print()
        print("دادهٔ FWI معتبر است. ✅")
        print(
            "مرحلهٔ بعدی: اجرای scripts/build_fwi_raster.py "
            "برای تولید data/fwi/fwi_fars.tif"
        )

        return 0

    except Exception as exc:
        print()
        print("=" * 70, file=sys.stderr)
        print("خطا در اعتبارسنجی دادهٔ FWI", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
