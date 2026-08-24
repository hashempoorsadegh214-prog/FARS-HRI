```python
import os
import json
import zipfile
import urllib.request
import shutil

import rasterio
from rasterio.mask import mask


# ============================================================
# FARS-HRI
# Download official Global Fuelbed Dataset from PANGAEA
# Clip it to Fars Province
# ============================================================

FUEL_URL = (
    "https://store.pangaea.de/Publications/Pettinari_2015/"
    "Global_fuelbeds_map_Tile4.zip"
)

ZIP_FILE = "temp_fuel_tile4.zip"
EXTRACT_DIR = "temp_fuel_tile4"

BOUNDARY_FILE = "fars.geojson"

OUTPUT_DIR = "data/fuel"
OUTPUT_FILE = "data/fuel/fars_fuel.tif"
STATS_FILE = "data/fuel/fars_fuel_stats.json"


def download_file():

    print("Downloading official Fuelbed Tile 4...")

    urllib.request.urlretrieve(
        FUEL_URL,
        ZIP_FILE
    )

    print("Download complete.")


def extract_file():

    print("Extracting Tile 4...")

    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)

    with zipfile.ZipFile(
        ZIP_FILE,
        "r"
    ) as zip_ref:

        zip_ref.extractall(
            EXTRACT_DIR
        )

    print("Extraction complete.")


def find_tif():

    for root, dirs, files in os.walk(
        EXTRACT_DIR
    ):

        for file in files:

            if file.lower().endswith(".tif"):

                return os.path.join(
                    root,
                    file
                )

    raise FileNotFoundError(
        "Fuel GeoTIFF not found inside ZIP."
    )


def load_boundary():

    print("Reading Fars boundary...")

    with open(
        BOUNDARY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        boundary = json.load(f)

    if boundary["type"] == "FeatureCollection":

        geometries = [
            feature["geometry"]
            for feature in boundary["features"]
            if feature.get("geometry")
        ]

    elif boundary["type"] == "Feature":

        geometries = [
            boundary["geometry"]
        ]

    else:

        geometries = [
            boundary
        ]

    return geometries


def clip_fuel():

    fuel_tif = find_tif()

    geometries = load_boundary()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("Clipping Fuel map to Fars...")

    with rasterio.open(
        fuel_tif
    ) as src:

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

    print("Fars Fuel map created.")


def create_stats():

    with rasterio.open(
        OUTPUT_FILE
    ) as src:

        data = src.read(1)

        nodata = src.nodata

        valid = data

        if nodata is not None:

            valid = data[
                data != nodata
            ]

        unique_values = sorted(
            [
                int(value)
                for value in set(
                    valid.flatten()
                )
            ]
        )

        stats = {
            "source": (
                "Global Fuelbed Dataset"
            ),

            "citation": (
                "Pettinari, M. L. (2015): "
                "Global Fuelbed Dataset. "
                "PANGAEA. "
                "https://doi.org/10.1594/PANGAEA.849808"
            ),

            "source_file": (
                "Global_fuelbeds_map_Tile4"
            ),

            "output_file": OUTPUT_FILE,

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

    print("Fuel statistics created.")


def cleanup():

    print("Cleaning temporary files...")

    if os.path.exists(ZIP_FILE):

        os.remove(
            ZIP_FILE
        )

    if os.path.exists(EXTRACT_DIR):

        shutil.rmtree(
            EXTRACT_DIR
        )


def main():

    print("=" * 60)
    print("FARS-HRI | FUEL DATA")
    print("=" * 60)

    try:

        download_file()

        extract_file()

        clip_fuel()

        create_stats()

        print()
        print("=" * 60)
        print("DONE")
        print("=" * 60)

        print(
            f"Output: {OUTPUT_FILE}"
        )

        print(
            f"Stats: {STATS_FILE}"
        )

    finally:

        cleanup()


if __name__ == "__main__":
    main()
```
