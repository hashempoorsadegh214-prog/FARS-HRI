```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FARS-HRI
محاسبه شاخص خطر حریق (FLI)

فرمول:

FLI = 100 * (
    0.45 * F_FWI +
    0.35 * F_Fuel +
    0.20 * F_Topo
)

F_FWI  = FWI / 50  محدود شده بین 0 و 1
F_Topo = Slope / 45 محدود شده بین 0 و 1

Fuel:
    Woody Cover
    W_1h Load
    W_10h Load
    W_100h Load
    W_1000h Load
    Litter Cover
    L_depth

NoData:
    -3
    -1
"""

from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.enums import Resampling


# ============================================================
# مسیرها
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"

FUEL_DIR = DATA / "fuel"
TOPO_DIR = DATA / "topography"
FWI_DIR = DATA / "fwi"

OUTPUT_DIR = DATA / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# فایل‌ها
# ============================================================

DEM_FILE = TOPO_DIR / "fars_dem.tif"

FWI_FILE = FWI_DIR / "fwi_fars.tif"

FUEL_FILES = {
    "woody": FUEL_DIR / "woody_cover.tif",
    "w1h": FUEL_DIR / "w_1h_load.tif",
    "w10h": FUEL_DIR / "w_10h_load.tif",
    "w100h": FUEL_DIR / "w_100h_load.tif",
    "w1000h": FUEL_DIR / "w_1000h_load.tif",
    "litter": FUEL_DIR / "litter_cover.tif",
    "ldepth": FUEL_DIR / "l_depth.tif",
}

FLI_FILE = OUTPUT_DIR / "fli_fars.tif"

TOPO_INDEX_FILE = OUTPUT_DIR / "topo_index.tif"

FUEL_INDEX_FILE = OUTPUT_DIR / "fuel_index.tif"

HRI_JSON_FILE = DATA / "hri.json"


# ============================================================
# تنظیمات
# ============================================================

FWI_MAX = 50.0
SLOPE_MAX = 45.0

FWI_WEIGHT = 0.45
FUEL_WEIGHT = 0.35
TOPO_WEIGHT = 0.20

NODATA = -9999.0


# ============================================================
# توابع کمکی
# ============================================================

def check_file(path):
    """بررسی وجود فایل"""
    if not path.exists():
        raise FileNotFoundError(
            f"\nفایل پیدا نشد:\n{path}\n"
        )


def read_raster(path):
    """خواندن Raster"""
    check_file(path)

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    return data, profile


def clean_nodata(data):
    """
    تبدیل NoDataهای پروژه:
    -3
    -1
    NaN
    به NaN
    """

    data = data.astype(np.float32)

    invalid = (
        np.isnan(data) |
        (data == -3) |
        (data == -1)
    )

    data[invalid] = np.nan

    return data


def save_raster(path, data, profile):
    """ذخیره Raster"""

    out_profile = profile.copy()

    out_profile.update(
        dtype="float32",
        count=1,
        nodata=NODATA,
        compress="deflate"
    )

    output = np.where(
        np.isnan(data),
        NODATA,
        data
    ).astype(np.float32)

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(output, 1)


def normalize_0_1(data):
    """
    تبدیل داده به مقیاس 0 تا 1
    بر اساس min/max واقعی داده‌های معتبر.
    """

    result = np.full(
        data.shape,
        np.nan,
        dtype=np.float32
    )

    valid = np.isfinite(data)

    if not np.any(valid):
        return result

    minimum = np.nanmin(data)
    maximum = np.nanmax(data)

    if maximum == minimum:
        result[valid] = 0.0
    else:
        result[valid] = (
            (data[valid] - minimum) /
            (maximum - minimum)
        )

    result = np.clip(result, 0, 1)

    return result


# ============================================================
# هم‌تراز کردن Rasterها
# ============================================================

def read_match_raster(path, reference_profile):
    """
    Raster را با Grid مربوط به FWI هم‌تراز می‌کند.
    """

    check_file(path)

    height = reference_profile["height"]
    width = reference_profile["width"]
    transform = reference_profile["transform"]
    crs = reference_profile["crs"]

    with rasterio.open(path) as src:

        data = src.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.bilinear
        ).astype(np.float32)

    return data


# ============================================================
# محاسبه شیب از DEM
# ============================================================

def calculate_slope(dem, transform):
    """
    محاسبه شیب تقریبی از DEM.

    خروجی بر حسب درجه است.
    """

    dem = dem.astype(np.float32)

    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)

    if pixel_width == 0 or pixel_height == 0:
        raise ValueError("Resolution مربوط به DEM نامعتبر است.")

    # گرادیان ارتفاع
    dz_dy, dz_dx = np.gradient(
        dem,
        pixel_height,
        pixel_width
    )

    slope_radians = np.arctan(
        np.sqrt(
            dz_dx ** 2 +
            dz_dy ** 2
        )
    )

    slope_degrees = np.degrees(
        slope_radians
    )

    slope_degrees[~np.isfinite(dem)] = np.nan

    return slope_degrees.astype(np.float32)


# ============================================================
# شروع محاسبات
# ============================================================

def main():

    print("=" * 60)
    print("FARS-HRI : BUILD HRI")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. FWI
    # --------------------------------------------------------

    print("\n[1/5] خواندن FWI...")

    fwi, fwi_profile = read_raster(FWI_FILE)

    fwi = clean_nodata(fwi)

    print(
        f"FWI min: {np.nanmin(fwi):.2f}"
    )

    print(
        f"FWI max: {np.nanmax(fwi):.2f}"
    )

    # نرمال سازی FWI
    f_fwi = np.clip(
        fwi / FWI_MAX,
        0,
        1
    )

    # --------------------------------------------------------
    # 2. Fuel
    # --------------------------------------------------------

    print("\n[2/5] محاسبه Fuel Index...")

    fuel_arrays = []

    for name, path in FUEL_FILES.items():

        print(f"  - {name}")

        data = read_match_raster(
            path,
            fwi_profile
        )

        data = clean_nodata(data)

        # ----------------------------------------------------
        # نرمال سازی هر متغیر
        # ----------------------------------------------------

        if name == "woody":
            normalized = np.clip(
                data / 100.0,
                0,
                1
            )

        elif name == "litter":
            normalized = np.clip(
                data / 100.0,
                0,
                1
            )

        elif name == "ldepth":

            # نرمال سازی بر اساس حداکثر واقعی
            normalized = normalize_0_1(
                data
            )

        else:

            # بار سوخت
            normalized = normalize_0_1(
                data
            )

        fuel_arrays.append(normalized)

    fuel_stack = np.stack(
        fuel_arrays,
        axis=0
    )

    # میانگین فقط از مقادیر معتبر
    valid_count = np.sum(
        np.isfinite(fuel_stack),
        axis=0
    )

    fuel_sum = np.nansum(
        fuel_stack,
        axis=0
    )

    f_fuel = np.full(
        fuel_sum.shape,
        np.nan,
        dtype=np.float32
    )

    valid = valid_count > 0

    f_fuel[valid] = (
        fuel_sum[valid] /
        valid_count[valid]
    )

    f_fuel = np.clip(
        f_fuel,
        0,
        1
    )

    save_raster(
        FUEL_INDEX_FILE,
        f_fuel,
        fwi_profile
    )

    print(
        f"Fuel Index mean: "
        f"{np.nanmean(f_fuel):.3f}"
    )

    # --------------------------------------------------------
    # 3. Topography
    # --------------------------------------------------------

    print("\n[3/5] محاسبه Topography...")

    dem = read_match_raster(
        DEM_FILE,
        fwi_profile
    )

    dem = clean_nodata(dem)

    slope = calculate_slope(
        dem,
        fwi_profile["transform"]
    )

    # تبدیل شیب به 0 تا 1
    f_topo = np.clip(
        slope / SLOPE_MAX,
        0,
        1
    )

    save_raster(
        TOPO_INDEX_FILE,
        f_topo,
        fwi_profile
    )

    print(
        f"Slope min: "
        f"{np.nanmin(slope):.2f}"
    )

    print(
        f"Slope max: "
        f"{np.nanmax(slope):.2f}"
    )

    print(
        f"Topo Index mean: "
        f"{np.nanmean(f_topo):.3f}"
    )

    # --------------------------------------------------------
    # 4. محاسبه FLI
    # --------------------------------------------------------

    print("\n[4/5] محاسبه FLI...")

    all_valid = (
        np.isfinite(f_fwi) &
        np.isfinite(f_fuel) &
        np.isfinite(f_topo)
    )

    fli = np.full(
        f_fwi.shape,
        np.nan,
        dtype=np.float32
    )

    fli[all_valid] = 100.0 * (
        FWI_WEIGHT * f_fwi[all_valid] +
        FUEL_WEIGHT * f_fuel[all_valid] +
        TOPO_WEIGHT * f_topo[all_valid]
    )

    fli = np.clip(
        fli,
        0,
        100
    )

    save_raster(
        FLI_FILE,
        fli,
        fwi_profile
    )

    # --------------------------------------------------------
    # 5. آمار
    # --------------------------------------------------------

    print("\n[5/5] تولید آمار...")

    valid_fli = fli[np.isfinite(fli)]

    if valid_fli.size == 0:
        raise RuntimeError(
            "هیچ مقدار معتبر FLI تولید نشد."
        )

    minimum = float(np.min(valid_fli))
    mean = float(np.mean(valid_fli))
    maximum = float(np.max(valid_fli))

    very_low = int(
        np.sum(
            (valid_fli >= 0) &
            (valid_fli < 20)
        )
    )

    low = int(
        np.sum(
            (valid_fli >= 20) &
            (valid_fli < 40)
        )
    )

    moderate = int(
        np.sum(
            (valid_fli >= 40) &
            (valid_fli < 60)
        )
    )

    high = int(
        np.sum(
            (valid_fli >= 60) &
            (valid_fli < 80)
        )
    )

    very_high = int(
        np.sum(
            valid_fli >= 80
        )
    )

    # --------------------------------------------------------
    # تاریخ تولید
    # --------------------------------------------------------

    generated_at = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    result = {
        "project": "FARS-HRI",
        "index": "FLI",
        "generated_at": generated_at,

        "formula": {
            "FWI_weight": FWI_WEIGHT,
            "Fuel_weight": FUEL_WEIGHT,
            "Topography_weight": TOPO_WEIGHT
        },

        "normalization": {
            "FWI_max": FWI_MAX,
            "Slope_max": SLOPE_MAX
        },

        "statistics": {
            "min": round(minimum, 3),
            "mean": round(mean, 3),
            "max": round(maximum, 3)
        },

        "classes": {
            "very_low": very_low,
            "low": low,
            "moderate": moderate,
            "high": high,
            "very_high": very_high
        },

        "outputs": {
            "fli": "data/outputs/fli_fars.tif",
            "fuel_index": "data/outputs/fuel_index.tif",
            "topography_index": "data/outputs/topo_index.tif"
        }
    }

    with open(
        HRI_JSON_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # پایان
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("محاسبه FLI با موفقیت انجام شد")
    print("=" * 60)

    print(
        f"\nFLI Min  : {minimum:.2f}"
    )

    print(
        f"FLI Mean : {mean:.2f}"
    )

    print(
        f"FLI Max  : {maximum:.2f}"
    )

    print("\nکلاس‌ها:")

    print(
        f"بسیار کم   : {very_low}"
    )

    print(
        f"کم         : {low}"
    )

    print(
        f"متوسط      : {moderate}"
    )

    print(
        f"زیاد       : {high}"
    )

    print(
        f"بسیار زیاد : {very_high}"
    )

    print("\nخروجی‌ها:")

    print(
        FLI_FILE
    )

    print(
        FUEL_INDEX_FILE
    )

    print(
        TOPO_INDEX_FILE
    )

    print(
        HRI_JSON_FILE
    )

    print("\nتمام شد.")


if __name__ == "__main__":
    main()
```
