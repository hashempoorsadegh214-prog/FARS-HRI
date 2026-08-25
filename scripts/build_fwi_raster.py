#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ساخت رستر FWI هم‌تراز با لایه سوخت استان فارس.

ورودی:
    data/fwi/fwi_fars_grid.json
    data/fuel/fars_fuel.tif

خروجی:
    data/fwi/fwi_fars.tif

روش:
    برای هر پیکسل معتبر در رستر سوخت، نزدیک‌ترین نقطه شبکه‌ای FWI
    انتخاب می‌شود. خروجی از نظر CRS، ابعاد، Transform و محدوده دقیقاً
    با fars_fuel.tif یکسان خواهد بود.

نیازمندی‌ها:
    numpy
    rasterio
    scipy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------
# مسیرهای پروژه
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FWI_JSON_PATH = PROJECT_ROOT / "data" / "fwi" / "fwi_fars_grid.json"
FUEL_RASTER_PATH = PROJECT_ROOT / "data" / "fuel" / "fars_fuel.tif"
OUTPUT_RASTER_PATH = PROJECT_ROOT / "data" / "fwi" / "fwi_fars.tif"

OUTPUT_NODATA = np.float32(-9999.0)
OUTPUT_DTYPE = "float32"

# اندازه دسته‌ها برای کنترل حافظه در زمان پردازش پیکسل‌های معتبر
CHUNK_SIZE = 100_000


# ---------------------------------------------------------------------
# توابع کمکی برای خواندن و اعتبارسنجی JSON
# ---------------------------------------------------------------------

def find_point_records(data: Any) -> list[dict[str, Any]]:
    """
    به‌صورت بازگشتی، آرایه‌ای از رکوردهای دارای کلیدهای lat، lon و fwi
    را در ساختار JSON پیدا می‌کند.

    این روش باعث می‌شود اسکریپت به نام دقیق کلید لایهٔ بیرونی JSON
    وابسته نباشد.
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

    elif isinstance(data, dict):
        for value in data.values():
            result = find_point_records(value)
            if result:
                return result

    return []


def load_fwi_points(json_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    فایل JSON را می‌خواند و آرایه‌های longitude، latitude و fwi را برمی‌گرداند.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"فایل داده FWI پیدا نشد:\n{json_path}\n\n"
            "فایل را با نام دقیق زیر در مخزن قرار دهید:\n"
            "data/fwi/fwi_fars_grid.json"
        )

    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"فایل JSON معتبر نیست: {json_path}\n"
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
            lon = float(record["lon"])
            lat = float(record["lat"])
            fwi = float(record["fwi"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"رکورد شماره {index} دارای lon، lat یا fwi معتبر نیست:\n{record}"
            ) from exc

        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"طول جغرافیایی نامعتبر در رکورد {index}: {lon}")

        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"عرض جغرافیایی نامعتبر در رکورد {index}: {lat}")

        if not np.isfinite(fwi):
            raise ValueError(f"مقدار FWI نامعتبر در رکورد {index}: {fwi}")

        longitudes.append(lon)
        latitudes.append(lat)
        fwi_values.append(fwi)

    lon_array = np.asarray(longitudes, dtype=np.float64)
    lat_array = np.asarray(latitudes, dtype=np.float64)
    fwi_array = np.asarray(fwi_values, dtype=np.float32)

    coordinates = np.column_stack((lon_array, lat_array))
    unique_coordinates = np.unique(coordinates, axis=0)

    if len(unique_coordinates) != len(coordinates):
        raise ValueError(
            "مختصات تکراری در داده‌های FWI وجود دارد. "
            "برای هر نقطه فقط یک مقدار FWI باید موجود باشد."
        )

    if len(fwi_array) < 10:
        raise ValueError(
            f"تعداد نقاط FWI بسیار کم است ({len(fwi_array)} نقطه). "
            "دادهٔ شبکه‌ای کامل فارس باید استفاده شود."
        )

    return lon_array, lat_array, fwi_array


# ---------------------------------------------------------------------
# تولید رستر FWI
# ---------------------------------------------------------------------

def build_fwi_raster() -> None:
    """
    رستر FWI را با شبکه، CRS و محدودهٔ رستر سوخت تولید می‌کند.
    """
    print("=" * 70)
    print("شروع ساخت رستر FWI هم‌تراز با رستر سوخت فارس")
    print("=" * 70)

    if not FUEL_RASTER_PATH.exists():
        raise FileNotFoundError(
            f"رستر سوخت پیدا نشد:\n{FUEL_RASTER_PATH}\n\n"
            "مسیر مورد انتظار:\n"
            "data/fuel/fars_fuel.tif"
        )

    print(f"\nخواندن داده‌های FWI از:\n{FWI_JSON_PATH}")
    longitudes, latitudes, fwi_values = load_fwi_points(FWI_JSON_PATH)

    print(f"تعداد نقاط FWI: {len(fwi_values):,}")
    print(
        "محدوده نقاط FWI: "
        f"Lon={longitudes.min():.4f} تا {longitudes.max():.4f} | "
        f"Lat={latitudes.min():.4f} تا {latitudes.max():.4f}"
    )
    print(
        "دامنه FWI: "
        f"{fwi_values.min():.3f} تا {fwi_values.max():.3f} | "
        f"میانگین: {fwi_values.mean():.3f}"
    )

    print(f"\nخواندن رستر مرجع سوخت از:\n{FUEL_RASTER_PATH}")

    with rasterio.open(FUEL_RASTER_PATH) as fuel_source:
        if fuel_source.crs is None:
            raise ValueError("رستر سوخت فاقد CRS است.")

        if fuel_source.crs.to_epsg() != 4326:
            raise ValueError(
                "رستر سوخت باید در EPSG:4326 باشد، اما CRS فعلی آن برابر است با:\n"
                f"{fuel_source.crs}"
            )

        fuel_data = fuel_source.read(1, masked=True)

        reference_profile = fuel_source.profile.copy()
        reference_transform = fuel_source.transform
        reference_crs = fuel_source.crs
        reference_width = fuel_source.width
        reference_height = fuel_source.height
        reference_bounds = fuel_source.bounds

    valid_fuel_mask = ~fuel_data.mask

    if not np.any(valid_fuel_mask):
        raise ValueError(
            "هیچ پیکسل معتبر در رستر سوخت پیدا نشد؛ "
            "نمی‌توان رستر FWI را با ماسک آن تولید کرد."
        )

    valid_row_indices, valid_col_indices = np.where(valid_fuel_mask)
    valid_pixel_count = len(valid_row_indices)

    print("\nمشخصات رستر مرجع:")
    print(f"CRS: {reference_crs}")
    print(f"ابعاد: {reference_width} × {reference_height}")
    print(
        "محدوده: "
        f"غرب={reference_bounds.left:.6f}, "
        f"جنوب={reference_bounds.bottom:.6f}, "
        f"شرق={reference_bounds.right:.6f}, "
        f"شمال={reference_bounds.top:.6f}"
    )
    print(f"تعداد پیکسل‌های معتبر فارس: {valid_pixel_count:,}")

    # درخت جست‌وجوی نزدیک‌ترین همسایه بر اساس مختصات lon و lat
    # مختصات داده‌ها درجه‌ای هستند؛ بنابراین CRS صحیح EPSG:4326 است.
    source_coordinates = np.column_stack((longitudes, latitudes))
    fwi_tree = cKDTree(source_coordinates)

    output_array = np.full(
        (reference_height, reference_width),
        OUTPUT_NODATA,
        dtype=np.float32,
    )

    print("\nتخصیص نزدیک‌ترین مقدار FWI به پیکسل‌های معتبر فارس...")

    for start_index in range(0, valid_pixel_count, CHUNK_SIZE):
        end_index = min(start_index + CHUNK_SIZE, valid_pixel_count)

        rows_chunk = valid_row_indices[start_index:end_index]
        cols_chunk = valid_col_indices[start_index:end_index]

        x_coordinates, y_coordinates = xy(
            reference_transform,
            rows_chunk,
            cols_chunk,
            offset="center",
        )

        target_coordinates = np.column_stack(
            (
                np.asarray(x_coordinates, dtype=np.float64),
                np.asarray(y_coordinates, dtype=np.float64),
            )
        )

        _, nearest_indices = fwi_tree.query(target_coordinates, k=1)

        output_array[rows_chunk, cols_chunk] = fwi_values[nearest_indices]

        processed_percent = (end_index / valid_pixel_count) * 100.0
        print(
            f"\rپیشرفت: {end_index:,} از {valid_pixel_count:,} "
            f"پیکسل ({processed_percent:.1f}%)",
            end="",
            flush=True,
        )

    print()

    valid_output_values = output_array[output_array != OUTPUT_NODATA]

    if valid_output_values.size == 0:
        raise RuntimeError("هیچ مقدار معتبری در رستر خروجی FWI تولید نشد.")

    if not np.all(np.isfinite(valid_output_values)):
        raise RuntimeError("رستر خروجی شامل مقدار NaN یا Infinity است.")

    OUTPUT_RASTER_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_profile = reference_profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype=OUTPUT_DTYPE,
        count=1,
        nodata=float(OUTPUT_NODATA),
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    print(f"\nنوشتن خروجی در:\n{OUTPUT_RASTER_PATH}")

    with rasterio.open(OUTPUT_RASTER_PATH, "w", **output_profile) as output_file:
        output_file.write(output_array, 1)

    # اعتبارسنجی نهایی خروجی
    with rasterio.open(OUTPUT_RASTER_PATH) as result:
        if result.crs != reference_crs:
            raise RuntimeError("CRS فایل خروجی با رستر سوخت یکسان نیست.")

        if result.width != reference_width or result.height != reference_height:
            raise RuntimeError("ابعاد فایل خروجی با رستر سوخت یکسان نیست.")

        if result.transform != reference_transform:
            raise RuntimeError("Transform فایل خروجی با رستر سوخت یکسان نیست.")

        result_data = result.read(1, masked=True)
        result_valid_count = int((~result_data.mask).sum())

    if result_valid_count != valid_pixel_count:
        raise RuntimeError(
            "تعداد پیکسل‌های معتبر خروجی با پیکسل‌های معتبر رستر سوخت برابر نیست.\n"
            f"خروجی: {result_valid_count:,}\n"
            f"سوخت: {valid_pixel_count:,}"
        )

    print("\n" + "=" * 70)
    print("رستر FWI با موفقیت ساخته و اعتبارسنجی شد. ✅")
    print("=" * 70)
    print(f"فایل خروجی: {OUTPUT_RASTER_PATH}")
    print(f"پیکسل‌های معتبر: {result_valid_count:,}")
    print(
        "دامنه FWI خروجی: "
        f"{valid_output_values.min():.3f} تا {valid_output_values.max():.3f}"
    )


def main() -> int:
    """نقطهٔ شروع اجرای برنامه."""
    try:
        build_fwi_raster()
        return 0

    except Exception as exc:
        print("\n" + "=" * 70, file=sys.stderr)
        print("خطا در ساخت رستر FWI", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
