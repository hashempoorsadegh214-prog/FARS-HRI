
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FARS-HRI
سامانه پیش‌بینی و پهنه‌بندی خطر حریق در عرصه‌های طبیعی استان فارس

محاسبه شاخص خطر حریق:

HRI = 100 * (
    0.45 * F_FWI +
    0.35 * F_Fuel +
    0.20 * F_Topo
)

------------------------------------------------------------
اجزای شاخص
------------------------------------------------------------

F_FWI:
    FWI / 50
    محدود شده بین 0 و 1

F_Fuel:
    میانگین نرمال‌شده:
        Woody Cover
        W_1h Load
        W_10h Load
        W_100h Load
        W_1000h Load
        Litter Cover
        L_depth

F_Topo:
    Slope / 45
    محدود شده بین 0 و 1

------------------------------------------------------------
NoData
------------------------------------------------------------

مقادیر زیر به عنوان NoData در نظر گرفته می‌شوند:

    -3
    -1
    NaN
    NoData تعریف‌شده در Raster

------------------------------------------------------------
خروجی‌ها
------------------------------------------------------------

data/outputs/hri_fars.tif
data/outputs/fli_fars.tif
data/outputs/fuel_index.tif
data/outputs/topo_index.tif
data/outputs/hri_classes.tif

data/hri.json
"""

from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


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


# ============================================================
# خروجی‌ها
# ============================================================

HRI_FILE = OUTPUT_DIR / "hri_fars.tif"

# برای سازگاری با نسخه قبلی
FLI_FILE = OUTPUT_DIR / "fli_fars.tif"

TOPO_INDEX_FILE = OUTPUT_DIR / "topo_index.tif"

FUEL_INDEX_FILE = OUTPUT_DIR / "fuel_index.tif"

HRI_CLASSES_FILE = OUTPUT_DIR / "hri_classes.tif"

HRI_JSON_FILE = DATA / "hri.json"


# ============================================================
# تنظیمات شاخص
# ============================================================

FWI_MAX = 50.0

SLOPE_MAX = 45.0


FWI_WEIGHT = 0.45

FUEL_WEIGHT = 0.35

TOPO_WEIGHT = 0.20


# ============================================================
# تنظیمات Fuel
# ============================================================

# حداقل تعداد متغیرهای معتبر Fuel برای محاسبه Fuel Index
#
# از 7 متغیر موجود، حداقل 4 متغیر باید معتبر باشند.
#
MIN_VALID_FUEL_VARIABLES = 4


# ============================================================
# NoData خروجی
# ============================================================

NODATA = -9999.0


# ============================================================
# کلاس‌های HRI
# ============================================================

CLASS_VERY_LOW = 1
CLASS_LOW = 2
CLASS_MODERATE = 3
CLASS_HIGH = 4
CLASS_VERY_HIGH = 5


# ============================================================
# توابع کمکی
# ============================================================

def check_file(path):
    """
    بررسی وجود فایل.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"\nفایل پیدا نشد:\n{path}\n"
        )


# ============================================================

def read_raster(path):
    """
    خواندن Raster.
    """

    check_file(path)

    with rasterio.open(path) as src:

        data = src.read(1).astype(np.float32)

        profile = src.profile.copy()

        nodata = src.nodata

    return data, profile, nodata


# ============================================================

def clean_nodata(data, source_nodata=None):
    """
    تبدیل NoDataهای پروژه به NaN.

    NoDataهای پروژه:
        -3
        -1

    همچنین:
        NaN
        source_nodata
    """

    data = data.astype(np.float32)

    invalid = ~np.isfinite(data)

    invalid |= data == -3

    invalid |= data == -1

    if source_nodata is not None:

        if np.isfinite(source_nodata):

            invalid |= data == source_nodata

    data[invalid] = np.nan

    return data


# ============================================================

def save_raster(path, data, profile):
    """
    ذخیره Raster Float32.
    """

    out_profile = profile.copy()

    out_profile.update(
        dtype="float32",
        count=1,
        nodata=NODATA,
        compress="deflate",
        predictor=3
    )

    output = np.where(
        np.isfinite(data),
        data,
        NODATA
    ).astype(np.float32)

    with rasterio.open(
        path,
        "w",
        **out_profile
    ) as dst:

        dst.write(
            output,
            1
        )


# ============================================================

def save_class_raster(path, classes, profile):
    """
    ذخیره Raster طبقه‌بندی خطر.

    کلاس‌ها:

        0 = NoData
        1 = بسیار کم
        2 = کم
        3 = متوسط
        4 = زیاد
        5 = بسیار زیاد
    """

    out_profile = profile.copy()

    out_profile.update(
        dtype="uint8",
        count=1,
        nodata=0,
        compress="deflate"
    )

    with rasterio.open(
        path,
        "w",
        **out_profile
    ) as dst:

        dst.write(
            classes.astype(np.uint8),
            1
        )


# ============================================================

def normalize_0_1(data):
    """
    نرمال‌سازی Min-Max بین 0 و 1.

    فقط برای متغیرهای Fuel که حد ثابت علمی
    برای آن‌ها تعیین نشده است استفاده می‌شود.
    """

    result = np.full(
        data.shape,
        np.nan,
        dtype=np.float32
    )

    valid = np.isfinite(data)

    if not np.any(valid):
        return result

    minimum = float(
        np.nanmin(data)
    )

    maximum = float(
        np.nanmax(data)
    )

    if maximum == minimum:

        result[valid] = 0.0

    else:

        result[valid] = (
            (data[valid] - minimum)
            /
            (maximum - minimum)
        )

    result = np.clip(
        result,
        0,
        1
    )

    return result.astype(np.float32)


# ============================================================
# هم‌تراز کردن Raster
# ============================================================

def read_match_raster(
    path,
    reference_profile,
    resampling=Resampling.bilinear
):
    """
    Raster را واقعاً به Grid مرجع تبدیل می‌کند.

    برخلاف نسخه قبلی، فقط width/height تغییر نمی‌کند؛
    transform و CRS نیز در نظر گرفته می‌شوند.
    """

    check_file(path)

    height = reference_profile["height"]

    width = reference_profile["width"]

    destination = np.full(
        (height, width),
        np.nan,
        dtype=np.float32
    )

    dst_transform = reference_profile["transform"]

    dst_crs = reference_profile["crs"]

    with rasterio.open(path) as src:

        source = src.read(1).astype(np.float32)

        source_nodata = src.nodata

        source = clean_nodata(
            source,
            source_nodata
        )

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling
        )

    return destination


# ============================================================
# محاسبه شیب
# ============================================================

def calculate_slope(dem, transform, crs):
    """
    محاسبه شیب بر حسب درجه.

    اگر CRS جغرافیایی باشد (مثلاً EPSG:4326)،
    اندازه پیکسل از درجه به متر تبدیل می‌شود.

    این موضوع مهم است چون استفاده مستقیم از درجه
    به عنوان متر باعث خطای جدی در شیب می‌شود.
    """

    dem = dem.astype(np.float32)

    valid = np.isfinite(dem)

    if not np.any(valid):

        raise RuntimeError(
            "DEM هیچ مقدار معتبری ندارد."
        )

    # --------------------------------------------------------
    # اندازه پیکسل
    # --------------------------------------------------------

    pixel_width = abs(
        transform.a
    )

    pixel_height = abs(
        transform.e
    )

    if pixel_width <= 0 or pixel_height <= 0:

        raise ValueError(
            "Resolution مربوط به Raster نامعتبر است."
        )

    # --------------------------------------------------------
    # اگر CRS جغرافیایی است
    # --------------------------------------------------------

    if crs is not None and crs.is_geographic:

        rows = np.where(
            valid
        )[0]

        if rows.size == 0:

            raise RuntimeError(
                "هیچ پیکسل معتبری برای محاسبه شیب وجود ندارد."
            )

        center_row = int(
            np.median(rows)
        )

        latitude = (
            transform.f
            +
            (
                center_row + 0.5
            )
            *
            transform.e
        )

        lat_rad = np.radians(
            latitude
        )

        meters_per_degree_lat = (
            111132.92
            - 559.82 * np.cos(2 * lat_rad)
            + 1.175 * np.cos(4 * lat_rad)
            - 0.0023 * np.cos(6 * lat_rad)
        )

        meters_per_degree_lon = (
            111412.84 * np.cos(lat_rad)
            - 93.5 * np.cos(3 * lat_rad)
            + 0.118 * np.cos(5 * lat_rad)
        )

        dx = (
            pixel_width
            *
            meters_per_degree_lon
        )

        dy = (
            pixel_height
            *
            meters_per_degree_lat
        )

    else:

        dx = pixel_width

        dy = pixel_height

    if dx <= 0 or dy <= 0:

        raise ValueError(
            "اندازه پیکسل برای محاسبه شیب نامعتبر است."
        )

    # --------------------------------------------------------
    # محاسبه Gradient
    # --------------------------------------------------------

    work_dem = dem.copy()

    # پر کردن موقت NoData برای جلوگیری از انتشار NaN
    # سپس در انتها NoDataها دوباره حذف می‌شوند.

    valid_values = work_dem[valid]

    fill_value = float(
        np.nanmedian(valid_values)
    )

    work_dem[~valid] = fill_value

    dz_dy, dz_dx = np.gradient(
        work_dem,
        dy,
        dx
    )

    slope_radians = np.arctan(
        np.sqrt(
            dz_dx ** 2
            +
            dz_dy ** 2
        )
    )

    slope_degrees = np.degrees(
        slope_radians
    ).astype(np.float32)

    slope_degrees[~valid] = np.nan

    return slope_degrees


# ============================================================
# طبقه‌بندی HRI
# ============================================================

def classify_hri(hri):
    """
    طبقه‌بندی HRI:

        0 = NoData
        1 = بسیار کم   0-20
        2 = کم         20-40
        3 = متوسط      40-60
        4 = زیاد       60-80
        5 = بسیار زیاد 80-100
    """

    classes = np.zeros(
        hri.shape,
        dtype=np.uint8
    )

    valid = np.isfinite(hri)

    classes[
        valid & (hri < 20)
    ] = CLASS_VERY_LOW

    classes[
        valid &
        (hri >= 20) &
        (hri < 40)
    ] = CLASS_LOW

    classes[
        valid &
        (hri >= 40) &
        (hri < 60)
    ] = CLASS_MODERATE

    classes[
        valid &
        (hri >= 60) &
        (hri < 80)
    ] = CLASS_HIGH

    classes[
        valid &
        (hri >= 80)
    ] = CLASS_VERY_HIGH

    return classes


# ============================================================
# آمار
# ============================================================

def calculate_statistics(hri):
    """
    محاسبه آمار HRI.
    """

    valid = hri[
        np.isfinite(hri)
    ]

    if valid.size == 0:

        raise RuntimeError(
            "هیچ مقدار معتبر HRI تولید نشد."
        )

    classes = classify_hri(
        hri
    )

    statistics = {

        "min": round(
            float(np.min(valid)),
            3
        ),

        "mean": round(
            float(np.mean(valid)),
            3
        ),

        "max": round(
            float(np.max(valid)),
            3
        ),

        "valid_pixels": int(
            valid.size
        ),

        "very_low": int(
            np.sum(
                classes == CLASS_VERY_LOW
            )
        ),

        "low": int(
            np.sum(
                classes == CLASS_LOW
            )
        ),

        "moderate": int(
            np.sum(
                classes == CLASS_MODERATE
            )
        ),

        "high": int(
            np.sum(
                classes == CLASS_HIGH
            )
        ),

        "very_high": int(
            np.sum(
                classes == CLASS_VERY_HIGH
            )
        )
    }

    return statistics


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "FARS-HRI : BUILD HRI"
    )

    print("=" * 70)

    # ========================================================
    # 1. FWI
    # ========================================================

    print(
        "\n[1/5] خواندن FWI..."
    )

    fwi, fwi_profile, fwi_nodata = read_raster(
        FWI_FILE
    )

    fwi = clean_nodata(
        fwi,
        fwi_nodata
    )

    valid_fwi = np.isfinite(
        fwi
    )

    if not np.any(valid_fwi):

        raise RuntimeError(
            "FWI هیچ مقدار معتبری ندارد."
        )

    print(
        f"FWI min: "
        f"{np.nanmin(fwi):.2f}"
    )

    print(
        f"FWI max: "
        f"{np.nanmax(fwi):.2f}"
    )

    # --------------------------------------------------------
    # نرمال‌سازی FWI
    # --------------------------------------------------------

    f_fwi = np.full(
        fwi.shape,
        np.nan,
        dtype=np.float32
    )

    f_fwi[valid_fwi] = (
        fwi[valid_fwi]
        /
        FWI_MAX
    )

    f_fwi = np.clip(
        f_fwi,
        0,
        1
    )

    # ========================================================
    # 2. Fuel
    # ========================================================

    print(
        "\n[2/5] محاسبه Fuel Index..."
    )

    fuel_arrays = []

    fuel_names = []

    for name, path in FUEL_FILES.items():

        print(
            f"  - {name}"
        )

        data = read_match_raster(
            path,
            fwi_profile,
            resampling=Resampling.bilinear
        )

        if not np.any(
            np.isfinite(data)
        ):

            print(
                f"    WARNING: "
                f"{name} فاقد داده معتبر است."
            )

            normalized = np.full(
                data.shape,
                np.nan,
                dtype=np.float32
            )

        else:

            # ------------------------------------------------
            # Woody Cover
            # ------------------------------------------------

            if name == "woody":

                normalized = np.clip(
                    data / 100.0,
                    0,
                    1
                )

            # ------------------------------------------------
            # Litter Cover
            # ------------------------------------------------

            elif name == "litter":

                normalized = np.clip(
                    data / 100.0,
                    0,
                    1
                )

            # ------------------------------------------------
            # L_depth
            # ------------------------------------------------

            elif name == "ldepth":

                normalized = normalize_0_1(
                    data
                )

            # ------------------------------------------------
            # بار سوخت
            # ------------------------------------------------

            else:

                normalized = normalize_0_1(
                    data
                )

        fuel_arrays.append(
            normalized
        )

        fuel_names.append(
            name
        )

    # --------------------------------------------------------
    # Stack
    # --------------------------------------------------------

    fuel_stack = np.stack(
        fuel_arrays,
        axis=0
    )

    # --------------------------------------------------------
    # تعداد متغیرهای معتبر
    # --------------------------------------------------------

    valid_count = np.sum(
        np.isfinite(
            fuel_stack
        ),
        axis=0
    )

    # --------------------------------------------------------
    # مجموع
    # --------------------------------------------------------

    fuel_sum = np.nansum(
        fuel_stack,
        axis=0
    )

    f_fuel = np.full(
        fuel_sum.shape,
        np.nan,
        dtype=np.float32
    )

    valid_fuel = (
        valid_count
        >=
        MIN_VALID_FUEL_VARIABLES
    )

    f_fuel[
        valid_fuel
    ] = (
        fuel_sum[
            valid_fuel
        ]
        /
        valid_count[
            valid_fuel
        ]
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
        f"Fuel valid pixels: "
        f"{np.sum(valid_fuel)}"
    )

    print(
        f"Fuel Index mean: "
        f"{np.nanmean(f_fuel):.3f}"
    )

    # ========================================================
    # 3. Topography
    # ========================================================

    print(
        "\n[3/5] محاسبه Topography..."
    )

    dem = read_match_raster(
        DEM_FILE,
        fwi_profile,
        resampling=Resampling.bilinear
    )

    slope = calculate_slope(
        dem,
        fwi_profile["transform"],
        fwi_profile["crs"]
    )

    # --------------------------------------------------------
    # تبدیل شیب به 0 تا 1
    # --------------------------------------------------------

    f_topo = np.full(
        slope.shape,
        np.nan,
        dtype=np.float32
    )

    valid_slope = np.isfinite(
        slope
    )

    f_topo[
        valid_slope
    ] = (
        slope[
            valid_slope
        ]
        /
        SLOPE_MAX
    )

    f_topo = np.clip(
        f_topo,
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
        f"Slope mean: "
        f"{np.nanmean(slope):.2f}"
    )

    print(
        f"Topo Index mean: "
        f"{np.nanmean(f_topo):.3f}"
    )

    # ========================================================
    # 4. محاسبه HRI
    # ========================================================

    print(
        "\n[4/5] محاسبه HRI..."
    )

    all_valid = (
        np.isfinite(f_fwi)
        &
        np.isfinite(f_fuel)
        &
        np.isfinite(f_topo)
    )

    hri = np.full(
        f_fwi.shape,
        np.nan,
        dtype=np.float32
    )

    hri[
        all_valid
    ] = 100.0 * (

        FWI_WEIGHT
        *
        f_fwi[
            all_valid
        ]

        +

        FUEL_WEIGHT
        *
        f_fuel[
            all_valid
        ]

        +

        TOPO_WEIGHT
        *
        f_topo[
            all_valid
        ]
    )

    hri = np.clip(
        hri,
        0,
        100
    )

    # --------------------------------------------------------
    # ذخیره HRI
    # --------------------------------------------------------

    save_raster(
        HRI_FILE,
        hri,
        fwi_profile
    )

    # --------------------------------------------------------
    # ذخیره نسخه FLI برای سازگاری
    # --------------------------------------------------------

    save_raster(
        FLI_FILE,
        hri,
        fwi_profile
    )

    # --------------------------------------------------------
    # طبقه‌بندی
    # --------------------------------------------------------

    hri_classes = classify_hri(
        hri
    )

    save_class_raster(
        HRI_CLASSES_FILE,
        hri_classes,
        fwi_profile
    )

    # ========================================================
    # 5. آمار و JSON
    # ========================================================

    print(
        "\n[5/5] تولید آمار..."
    )

    statistics = calculate_statistics(
        hri
    )

    generated_at = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    # --------------------------------------------------------
    # درصد کلاس‌ها
    # --------------------------------------------------------

    valid_pixels = statistics[
        "valid_pixels"
    ]

    class_percentages = {}

    if valid_pixels > 0:

        class_percentages = {

            "very_low": round(
                100.0
                *
                statistics["very_low"]
                /
                valid_pixels,
                3
            ),

            "low": round(
                100.0
                *
                statistics["low"]
                /
                valid_pixels,
                3
            ),

            "moderate": round(
                100.0
                *
                statistics["moderate"]
                /
                valid_pixels,
                3
            ),

            "high": round(
                100.0
                *
                statistics["high"]
                /
                valid_pixels,
                3
            ),

            "very_high": round(
                100.0
                *
                statistics["very_high"]
                /
                valid_pixels,
                3
            )
        }

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    result = {

        "project": "FARS-HRI",

        "index": "HRI",

        "generated_at": generated_at,

        "formula": {

            "expression":
                "HRI = 100 * "
                "(0.45*F_FWI + "
                "0.35*F_Fuel + "
                "0.20*F_Topo)",

            "FWI_weight":
                FWI_WEIGHT,

            "Fuel_weight":
                FUEL_WEIGHT,

            "Topography_weight":
                TOPO_WEIGHT
        },

        "normalization": {

            "FWI_max":
                FWI_MAX,

            "Slope_max":
                SLOPE_MAX,

            "Woody_Cover":
                "value / 100",

            "Litter_Cover":
                "value / 100",

            "Fuel_loads":
                "min-max normalization",

            "L_depth":
                "min-max normalization"
        },

        "fuel": {

            "variables":
                fuel_names,

            "minimum_valid_variables":
                MIN_VALID_FUEL_VARIABLES
        },

        "nodata": {

            "input_values":
                [-3, -1],

            "output_value":
                NODATA
        },

        "statistics": {

            "min":
                statistics["min"],

            "mean":
                statistics["mean"],

            "max":
                statistics["max"],

            "valid_pixels":
                statistics["valid_pixels"]
        },

        "classes": {

            "1_very_low": {
                "range": "0-20",
                "pixels":
                    statistics["very_low"],
                "percent":
                    class_percentages["very_low"]
            },

            "2_low": {
                "range": "20-40",
                "pixels":
                    statistics["low"],
                "percent":
                    class_percentages["low"]
            },

            "3_moderate": {
                "range": "40-60",
                "pixels":
                    statistics["moderate"],
                "percent":
                    class_percentages["moderate"]
            },

            "4_high": {
                "range": "60-80",
                "pixels":
                    statistics["high"],
                "percent":
                    class_percentages["high"]
            },

            "5_very_high": {
                "range": "80-100",
                "pixels":
                    statistics["very_high"],
                "percent":
                    class_percentages["very_high"]
            }
        },

        "outputs": {

            "hri":
                "data/outputs/hri_fars.tif",

            "fli_compatibility":
                "data/outputs/fli_fars.tif",

            "fuel_index":
                "data/outputs/fuel_index.tif",

            "topography_index":
                "data/outputs/topo_index.tif",

            "hri_classes":
                "data/outputs/hri_classes.tif"
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

    # ========================================================
    # پایان
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "محاسبه HRI با موفقیت انجام شد"
    )

    print(
        "=" * 70
    )

    print(
        f"\nHRI Min  : "
        f"{statistics['min']:.2f}"
    )

    print(
        f"HRI Mean : "
        f"{statistics['mean']:.2f}"
    )

    print(
        f"HRI Max  : "
        f"{statistics['max']:.2f}"
    )

    print(
        f"\nValid pixels: "
        f"{statistics['valid_pixels']}"
    )

    print(
        "\nکلاس‌های خطر:"
    )

    print(
        f"بسیار کم   : "
        f"{statistics['very_low']}"
        f" ({class_percentages['very_low']}%)"
    )

    print(
        f"کم         : "
        f"{statistics['low']}"
        f" ({class_percentages['low']}%)"
    )

    print(
        f"متوسط      : "
        f"{statistics['moderate']}"
        f" ({class_percentages['moderate']}%)"
    )

    print(
        f"زیاد       : "
        f"{statistics['high']}"
        f" ({class_percentages['high']}%)"
    )

    print(
        f"بسیار زیاد : "
        f"{statistics['very_high']}"
        f" ({class_percentages['very_high']}%)"
    )

    print(
        "\nخروجی‌ها:"
    )

    print(
        HRI_FILE
    )

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
        HRI_CLASSES_FILE
    )

    print(
        HRI_JSON_FILE
    )

    print(
        "\nتمام شد."
    )


# ============================================================
# اجرای برنامه
# ============================================================

if __name__ == "__main__":
    main()
