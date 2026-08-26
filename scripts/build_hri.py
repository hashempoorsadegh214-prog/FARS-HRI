#!/usr/bin/env python3
"""
ساخت نقشه نهایی شاخص خطر حریق فارس (HRI).

فرمول:
HRI = 100 * (0.45 * FWI_n + 0.35 * Fuel_n + 0.20 * Topo_n)

ورودی‌ها:
- data/fwi/fwi_fars.tif
- data/fuel/fars_fuel.tif
- data/fuel/Global_fuelbeds_parameters_v1.2.xlsx
- data/dem_fars.tif

خروجی:
- data/output/fars_hri.tif
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


ROOT = Path(__file__).resolve().parents[1]

FWI_FILE = ROOT / "data" / "fwi" / "fwi_fars.tif"
FUEL_FILE = ROOT / "data" / "fuel" / "fars_fuel.tif"
FUEL_PARAMETERS_FILE = ROOT / "data" / "fuel" / "Global_fuelbeds_parameters_v1.2.xlsx"
DEM_FILE = ROOT / "data" / "dem_fars.tif"
OUTPUT_FILE = ROOT / "data" / "output" / "fars_hri.tif"

OUTPUT_NODATA = -9999.0

FWI_WEIGHT = 0.45
FUEL_WEIGHT = 0.35
TOPO_WEIGHT = 0.20


def ensure_exists(path: Path, title: str) -> None:
    """بررسی وجود فایل ورودی."""
    if not path.is_file():
        raise FileNotFoundError(f"فایل {title} پیدا نشد: {path}")


def normalize(values: np.ndarray, mask: np.ndarray, title: str) -> np.ndarray:
    """نرمال‌سازی مقادیر معتبر به بازه صفر تا یک."""
    output = np.full(values.shape, np.nan, dtype=np.float32)

    valid_values = values[mask]
    if valid_values.size == 0:
        raise ValueError(f"برای {title} هیچ پیکسل معتبری وجود ندارد.")

    minimum = float(np.nanmin(valid_values))
    maximum = float(np.nanmax(valid_values))

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError(f"مقادیر {title} معتبر نیستند.")

    if np.isclose(minimum, maximum):
        output[mask] = 0.0
    else:
        output[mask] = (values[mask] - minimum) / (maximum - minimum)

    print(
        f"{title}: حداقل={minimum:.4f} | "
        f"حداکثر={maximum:.4f} | "
        f"پیکسل معتبر={int(mask.sum()):,}"
    )

    return output


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    """خواندن اولین باند رستر و تبدیل NoData به NaN."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        data[np.isclose(data, nodata)] = np.nan

    return data, profile


def align_to_fuel_grid(source_file: Path, fuel_profile: dict) -> np.ndarray:
    """هم‌تراز کردن رستر ورودی با شبکه fars_fuel.tif."""
    destination = np.full(
        (fuel_profile["height"], fuel_profile["width"]),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(source_file) as src:
        source = src.read(1).astype(np.float32)

        if src.nodata is not None:
            source[np.isclose(source, src.nodata)] = np.nan

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=fuel_profile["transform"],
            dst_crs=fuel_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return destination


def get_fuel_scores(excel_file: Path) -> dict[int, float]:
    """
    خواندن جدول Fuelbeds_metric و ساخت امتیاز سوخت.

    FuelRaw =
        بار علف
        + بار چوبی ۱ ساعته
        + بار چوبی ۱۰ ساعته
        + ۰٫۲۵ × عمق لاشبرگ

    تمام مقادیر منفی، شامل -1 و -3، به معنی نبود/نامعتبر بودن
    پارامتر هستند و به صفر تبدیل می‌شوند.
    """
    columns = [
        "JOIN_VALUE",
        "G_Load (Mg/ha)",
        "W_1hLoad (Mg/ha)",
        "W_10h Load (Mg/ha)",
        "L_depth (cm)",
    ]

    table = pd.read_excel(
        excel_file,
        sheet_name="Fuelbeds_metric",
        engine="openpyxl",
    )

    missing_columns = [column for column in columns if column not in table.columns]
    if missing_columns:
        raise ValueError(
            "این ستون‌ها در شیت Fuelbeds_metric پیدا نشدند: "
            + ", ".join(missing_columns)
        )

    table = table[columns].copy()
    table["JOIN_VALUE"] = pd.to_numeric(table["JOIN_VALUE"], errors="coerce")
    table = table.dropna(subset=["JOIN_VALUE"])
    table["JOIN_VALUE"] = table["JOIN_VALUE"].astype(np.int64)

    for column in columns[1:]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
        table[column] = table[column].fillna(0.0)
        table.loc[table[column] < 0, column] = 0.0

    table["fuel_raw"] = (
        table["G_Load (Mg/ha)"]
        + table["W_1hLoad (Mg/ha)"]
        + table["W_10h Load (Mg/ha)"]
        + 0.25 * table["L_depth (cm)"]
    )

    if table["JOIN_VALUE"].duplicated().any():
        raise ValueError("ستون JOIN_VALUE در فایل پارامتر سوخت دارای مقدار تکراری است.")

    return dict(zip(table["JOIN_VALUE"], table["fuel_raw"]))


def build_fuel_component(
    fuel_codes: np.ndarray,
    fuel_mask: np.ndarray,
    fuel_scores: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """تبدیل کدهای سوخت رستر به امتیاز FuelRaw."""
    fuel_raw = np.full(fuel_codes.shape, np.nan, dtype=np.float32)

    codes_in_raster = np.unique(fuel_codes[fuel_mask]).astype(np.int64)
    missing_codes = []

    for code in codes_in_raster:
        score = fuel_scores.get(int(code))

        if score is None:
            missing_codes.append(int(code))
            continue

        fuel_raw[fuel_codes == code] = np.float32(score)

    if missing_codes:
        print(
            "هشدار: این کدهای رستر سوخت در ستون JOIN_VALUE اکسل پیدا نشدند:"
        )
        print(sorted(missing_codes))

    mapped_mask = np.isfinite(fuel_raw)

    if not mapped_mask.any():
        raise ValueError("هیچ کد سوختی از رستر با جدول اکسل تطبیق پیدا نکرد.")

    print(f"پیکسل‌های دارای امتیاز سوخت: {int(mapped_mask.sum()):,}")

    return fuel_raw, mapped_mask


def calculate_slope_degrees(dem: np.ndarray, transform) -> np.ndarray:
    """
    محاسبه شیب بر حسب درجه از DEM در EPSG:4326.

    تبدیل اندازه پیکسل جغرافیایی به متر، بر اساس عرض جغرافیایی هر ردیف
    انجام می‌شود.
    """
    slope = np.full(dem.shape, np.nan, dtype=np.float32)
    valid_mask = np.isfinite(dem)

    if not valid_mask.any():
        raise ValueError("پس از هم‌ترازی، DEM هیچ مقدار معتبری ندارد.")

    dem_filled = dem.copy()
    dem_filled[~valid_mask] = float(np.nanmedian(dem[valid_mask]))

    pixel_lon_degree = abs(transform.a)
    pixel_lat_degree = abs(transform.e)

    rows = np.arange(dem.shape[0], dtype=np.float64)
    latitudes = transform.f + (rows + 0.5) * transform.e

    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = 111_320.0 * np.cos(np.deg2rad(latitudes))
    meters_per_degree_lon = np.maximum(meters_per_degree_lon, 1.0)

    pixel_x_meter = meters_per_degree_lon * pixel_lon_degree
    pixel_y_meter = meters_per_degree_lat * pixel_lat_degree

    gradient_y_pixel, gradient_x_pixel = np.gradient(dem_filled)

    gradient_x = gradient_x_pixel / pixel_x_meter[np.newaxis, :]
    gradient_y = gradient_y_pixel / pixel_y_meter

    slope_radian = np.arctan(np.sqrt(gradient_x**2 + gradient_y**2))
    slope[valid_mask] = np.degrees(slope_radian[valid_mask])

    return slope


def main() -> None:
    """اجرای کامل ساخت HRI."""
    print("=" * 70)
    print("شروع ساخت نقشه نهایی HRI فارس")
    print("=" * 70)

    ensure_exists(FWI_FILE, "FWI")
    ensure_exists(FUEL_FILE, "رستر سوخت")
    ensure_exists(FUEL_PARAMETERS_FILE, "اکسل پارامتر سوخت")
    ensure_exists(DEM_FILE, "DEM")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] خواندن رستر سوخت به‌عنوان شبکه مرجع...")
    fuel_data, fuel_profile = read_raster(FUEL_FILE)
    fuel_mask = np.isfinite(fuel_data)
    fuel_codes = np.where(fuel_mask, fuel_data, 0).astype(np.int64)

    print(
        f"شبکه مرجع: {fuel_profile['width']} × {fuel_profile['height']} | "
        f"CRS: {fuel_profile['crs']}"
    )
    print(f"پیکسل‌های معتبر رستر سوخت: {int(fuel_mask.sum()):,}")

    print("\n[2/5] ساخت مؤلفه Fuel...")
    fuel_scores = get_fuel_scores(FUEL_PARAMETERS_FILE)
    fuel_raw, fuel_mask_mapped = build_fuel_component(
        fuel_codes,
        fuel_mask,
        fuel_scores,
    )
    fuel_normalized = normalize(fuel_raw, fuel_mask_mapped, "Fuel")

    print("\n[3/5] خواندن و نرمال‌سازی FWI...")
    fwi_data, fwi_profile = read_raster(FWI_FILE)

    fwi_is_aligned = (
        fwi_profile["crs"] == fuel_profile["crs"]
        and fwi_profile["width"] == fuel_profile["width"]
        and fwi_profile["height"] == fuel_profile["height"]
        and fwi_profile["transform"] == fuel_profile["transform"]
    )

    if not fwi_is_aligned:
        print("FWI با شبکه سوخت هم‌تراز نیست؛ بازنمونه‌برداری انجام می‌شود...")
        fwi_data = align_to_fuel_grid(FWI_FILE, fuel_profile)

    fwi_mask = np.isfinite(fwi_data) & (fwi_data >= 0)
    fwi_normalized = normalize(fwi_data, fwi_mask, "FWI")

    print("\n[4/5] ساخت مؤلفه Topography از شیب DEM...")
    dem_aligned = align_to_fuel_grid(DEM_FILE, fuel_profile)
    slope = calculate_slope_degrees(dem_aligned, fuel_profile["transform"])

    topo_mask = np.isfinite(slope)
    topo_normalized = normalize(slope, topo_mask, "Slope / Topography")

    print("\n[5/5] محاسبه و ذخیره HRI...")
    valid_mask = fuel_mask_mapped & fwi_mask & topo_mask

    if not valid_mask.any():
        raise ValueError("پیکسل معتبر مشترکی بین FWI، Fuel و Topography وجود ندارد.")

    hri = np.full(fuel_codes.shape, OUTPUT_NODATA, dtype=np.float32)

    hri[valid_mask] = 100.0 * (
        FWI_WEIGHT * fwi_normalized[valid_mask]
        + FUEL_WEIGHT * fuel_normalized[valid_mask]
        + TOPO_WEIGHT * topo_normalized[valid_mask]
    )

    output_profile = fuel_profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(OUTPUT_FILE, "w", **output_profile) as dst:
        dst.write(hri, 1)
        dst.set_band_description(1, "Fars Hazard Risk Index (HRI 0-100)")
        dst.update_tags(
            TITLE="Fars Hazard Risk Index",
            FORMULA="HRI = 100 * (0.45*FWI + 0.35*Fuel + 0.20*Topo)",
            FWI_WEIGHT=str(FWI_WEIGHT),
            FUEL_WEIGHT=str(FUEL_WEIGHT),
            TOPO_WEIGHT=str(TOPO_WEIGHT),
            FUEL_METHOD="G_Load + W_1hLoad + W_10hLoad + 0.25*L_depth",
            TOPO_METHOD="Slope degrees derived from dem_fars.tif",
        )

    valid_hri = hri[valid_mask]

    print("\n" + "=" * 70)
    print("نقشه نهایی HRI با موفقیت ساخته شد. ✅")
    print("=" * 70)
    print(f"فایل خروجی: {OUTPUT_FILE}")
    print(f"پیکسل‌های معتبر: {int(valid_mask.sum()):,}")
    print(f"دامنه HRI: {float(valid_hri.min()):.2f} تا {float(valid_hri.max()):.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nخطا: {error}", file=sys.stderr)
        sys.exit(1)
