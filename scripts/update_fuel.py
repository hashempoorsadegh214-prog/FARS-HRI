
import os
import json
import zipfile
import urllib.request
import shutil

import rasterio
from rasterio.mask import mask


# ============================================================
# FARS-HRI
# Global Fuelbed Dataset - Pettinari et al.
# PANGAEA DOI: 10.1594/PANGAEA.849808
# ============================================================

FUEL_URL = (
    "https://store.pangaea.de/Publications/"
    "Pettinari_2015/Global_fuelbeds_map_Tile4.zip"
)

ZIP_FILE = "temp_fuel_tile4.zip"
EXTRACT_DIR = "temp_fuel_tile4"

BOUNDARY_FILE = "fars.geojson"

OUTPUT_DIR = "data/fuel"
OUTPUT_FILE = "data/fuel/fars_fuel.tif"
STATS_FILE = "data/fuel/fars_fuel_stats.json"


# ============================================================
# دانلود Tile 4
# ============================================================

def download_fuel():

    print("=" * 60)
    print("STEP 1 - Downloading official Fuel Tile 4")
    print("=" * 60)

    print("URL:")
    print(FUEL_URL)
    print()

    urllib.request.urlretrieve(
        FUEL_URL,
        ZIP_FILE
    )

    if not os.path.exists(ZIP_FILE):

        raise RuntimeError(
            "Fuel ZIP was not downloaded."
        )

    size_mb = (
        os.path.getsize(ZIP_FILE)
        / 1024
        / 1024
    )

    print(
        f"Downloaded: {size_mb:.2f} MB"
    )


# ============================================================
# استخراج ZIP
# ============================================================

def extract_fuel():

    print()
    print("=" * 60)
    print("STEP 2 - Extracting Fuel Tile 4")
    print("=" * 60)

    if os.path.exists(EXTRACT_DIR):

        shutil.rmtree(
            EXTRACT_DIR
        )

    os.makedirs(
        EXTRACT_DIR,
        exist_ok=True
    )

    with zipfile.ZipFile(
        ZIP_FILE,
        "r"
    ) as archive:

        archive.extractall(
            EXTRACT_DIR
        )

    print(
        "Extraction completed."
    )


# ============================================================
# پیدا کردن TIFF
# ============================================================

def find_fuel_tif():

    print()
    print("=" * 60)
    print("STEP 3 - Searching for Fuel GeoTIFF")
    print("=" * 60)

    expected_name = (
        "Global_fuelbeds_map_Tile4.tif"
    )

    for root, dirs, files in os.walk(
        EXTRACT_DIR
    ):

        for filename in files:

            if filename == expected_name:

                tif_path = os.path.join(
                    root,
                    filename
                )

                print(
                    "Fuel TIFF found:"
                )

                print(
                    tif_path
                )

                return tif_path

    # اگر نام دقیق پیدا نشد
    # هر TIFF را بررسی می‌کنیم

    print(
        "Exact filename not found."
    )

    print(
        "Searching for any TIFF..."
    )

    for root, dirs, files in os.walk(
        EXTRACT_DIR
    ):

        for filename in files:

            if filename.lower().endswith(
                ".tif"
            ):

                tif_path = os.path.join(
                    root,
                    filename
                )

                print(
                    "Alternative TIFF found:"
                )

                print(
                    tif_path
                )

                return tif_path

    raise FileNotFoundError(
        "No TIFF file found inside Fuel Tile 4 ZIP."
    )


# ============================================================
# خواندن مرز فارس
# ============================================================

def load_fars_boundary():

    print()
    print("=" * 60)
    print("STEP 4 - Reading Fars boundary")
    print("=" * 60)

    if not os.path.exists(
        BOUNDARY_FILE
    ):

        raise FileNotFoundError(
            f"Boundary file not found: {BOUNDARY_FILE}"
        )

    with open(
        BOUNDARY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        geojson = json.load(f)

    if geojson.get(
        "type"
    ) == "FeatureCollection":

        geometries = []

        for feature in geojson.get(
            "features",
            []
        ):

            geometry = feature.get(
                "geometry"
            )

            if geometry:

                geometries.append(
                    geometry
                )

    elif geojson.get(
        "type"
    ) == "Feature":

        geometries = [
            geojson["geometry"]
        ]

    else:

        geometries = [
            geojson
        ]

    if not geometries:

        raise RuntimeError(
            "No geometry found in fars.geojson."
        )

    print(
        f"Boundary geometries: {len(geometries)}"
    )

    return geometries


# ============================================================
# برش Fuel برای فارس
# ============================================================

def clip_fuel(
    fuel_tif,
    geometries
):

    print()
    print("=" * 60)
    print("STEP 5 - Clipping Fuel map to Fars")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with rasterio.open(
        fuel_tif
    ) as src:

        print(
            "Source CRS:",
            src.crs
        )

        print(
            "Source size:",
            src.width,
            "x",
            src.height
        )

        print(
            "Source bounds:",
            src.bounds
        )

        print()

        clipped, transform = mask(
            src,
            geometries,
            crop=True,
            filled=True,
            nodata=src.nodata
        )

        profile = src.profile.copy()

        profile.update(
            {
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": transform,
                "compress": "deflate"
            }
        )

        with rasterio.open(
            OUTPUT_FILE,
            "w",
            **profile
        ) as dst:

            dst.write(
                clipped
            )

    if not os.path.exists(
        OUTPUT_FILE
    ):

        raise RuntimeError(
            "Fars Fuel TIFF was not created."
        )

    size_mb = (
        os.path.getsize(
            OUTPUT_FILE
        )
        / 1024
        / 1024
    )

    print()
    print(
        "Fars Fuel TIFF created:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"Output size: {size_mb:.2f} MB"
    )


# ============================================================
# آمار Fuel
# ============================================================

def create_statistics():

    print()
    print("=" * 60)
    print("STEP 6 - Creating Fuel statistics")
    print("=" * 60)

    with rasterio.open(
        OUTPUT_FILE
    ) as src:

        data = src.read(
            1
        )

        nodata = src.nodata

        if nodata is not None:

            valid = data[
                data != nodata
            ]

        else:

            valid = data

        valid = valid[
            valid >= 0
        ]

        if len(valid) == 0:

            raise RuntimeError(
                "No valid Fuel values found."
            )

        unique_values = sorted(
            set(
                int(value)
                for value in valid.flatten()
            )
        )

        stats = {

            "source": (
                "Global Fuelbed Dataset"
            ),

            "provider": (
                "Pettinari et al."
            ),

            "pangaea_doi": (
                "10.1594/PANGAEA.849808"
            ),

            "source_tile": (
                "Global_fuelbeds_map_Tile4"
            ),

            "output": OUTPUT_FILE,

            "crs": str(
                src.crs
            ),

            "width": src.width,

            "height": src.height,

            "fuelbed_classes": unique_values,

            "fuelbed_class_count": len(
                unique_values
            )
        }

    with open(
        STATS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        "Fuel classes:"
    )

    print(
        unique_values
    )

    print()
    print(
        "Statistics file:"
    )

    print(
        STATS_FILE
    )


# ============================================================
# پاکسازی
# ============================================================

def cleanup():

    print()
    print(
        "Cleaning temporary files..."
    )

    if os.path.exists(
        ZIP_FILE
    ):

        os.remove(
            ZIP_FILE
        )

    if os.path.exists(
        EXTRACT_DIR
    ):

        shutil.rmtree(
            EXTRACT_DIR
        )


# ============================================================
# برنامه اصلی
# ============================================================

def main():

    print()
    print("=" * 60)
    print("FARS-HRI | OFFICIAL FUEL DATA")
    print("=" * 60)
    print()
    print(
        "Source: Pettinari Global Fuelbed Dataset"
    )
    print(
        "PANGAEA DOI: 10.1594/PANGAEA.849808"
    )
    print(
        "Tile: 4"
    )
    print()

    try:

        download_fuel()

        extract_fuel()

        fuel_tif = find_fuel_tif()

        geometries = (
            load_fars_boundary()
        )

        clip_fuel(
            fuel_tif,
            geometries
        )

        create_statistics()

        print()
        print("=" * 60)
        print("FUEL PROCESSING SUCCESSFUL")
        print("=" * 60)

        print()
        print(
            f"Output: {OUTPUT_FILE}"
        )

    except Exception as error:

        print()
        print("=" * 60)
        print("FUEL PROCESSING FAILED")
        print("=" * 60)

        print()
        print(
            "ERROR:"
        )

        print(
            repr(error)
        )

        raise

    finally:

        cleanup()


# ============================================================
# اجرا
# ============================================================

if __name__ == "__main__":

    main()

