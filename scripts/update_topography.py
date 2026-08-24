import json
import os
import requests
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import from_bounds
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pyproj import Transformer
from io import BytesIO
import numpy as np


# ============================================================
# FARS-HRI
# Topography
# DEM + Slope + Aspect
# ============================================================

BOUNDARY_FILE = "fars.geojson"

OUTPUT_DIR = "data/topography"

DEM_FILE = os.path.join(OUTPUT_DIR, "fars_dem.tif")
SLOPE_FILE = os.path.join(OUTPUT_DIR, "fars_slope.tif")
ASPECT_FILE = os.path.join(OUTPUT_DIR, "fars_aspect.tif")


# Copernicus DEM GLO-30
# AWS Open Data
COPERNICUS_DEM_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
)


# ============================================================
# خواندن مرز فارس
# ============================================================

def load_boundary():

    print("Reading Fars boundary...")

    with open(
        BOUNDARY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    geometries = []

    if data["type"] == "FeatureCollection":

        for feature in data["features"]:

            geometry = feature.get("geometry")

            if geometry:
                geometries.append(
                    shape(geometry)
                )

    elif data["type"] == "Feature":

        geometries.append(
            shape(data["geometry"])
        )

    else:

        geometries.append(
            shape(data)
        )

    boundary = unary_union(
        geometries
    )

    print("Fars boundary loaded.")

    return boundary


# ============================================================
# پیدا کردن Tile های Copernicus DEM
# ============================================================

def get_tiles(boundary):

    minx, miny, maxx, maxy = boundary.bounds

    min_lon = int(np.floor(minx))
    max_lon = int(np.floor(maxx))

    min_lat = int(np.floor(miny))
    max_lat = int(np.floor(maxy))

    tiles = []

    for lat in range(
        min_lat,
        max_lat + 1
    ):

        for lon in range(
            min_lon,
            max_lon + 1
        ):

            if lat >= 0:

                lat_name = f"N{lat:02d}"

            else:

                lat_name = f"S{abs(lat):02d}"

            if lon >= 0:

                lon_name = f"E{lon:03d}"

            else:

                lon_name = f"W{abs(lon):03d}"

            tile_name = (
                f"Copernicus_DSM_COG_10_"
                f"{lat_name}_00_"
                f"{lon_name}_00_DEM"
            )

            url = (
                COPERNICUS_DEM_URL
                + tile_name
                + "/"
                + tile_name
                + ".tif"
            )

            tiles.append(
                {
                    "name": tile_name,
                    "url": url
                }
            )

    return tiles


# ============================================================
# دانلود DEM
# ============================================================

def download_tiles(
    tiles,
    boundary
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    downloaded = []

    for tile in tiles:

        print()
        print(
            "Downloading:",
            tile["name"]
        )

        local_file = os.path.join(
            OUTPUT_DIR,
            tile["name"] + ".tif"
        )

        try:

            response = requests.get(
                tile["url"],
                timeout=120
            )

            if response.status_code != 200:

                print(
                    "Tile unavailable:",
                    response.status_code
                )

                continue

            with open(
                local_file,
                "wb"
            ) as f:

                f.write(
                    response.content
                )

            downloaded.append(
                local_file
            )

            print("OK")

        except Exception as e:

            print(
                "Download error:",
                e
            )

    return downloaded


# ============================================================
# ساخت DEM فارس
# ============================================================

def build_dem(
    files,
    boundary
):

    print()
    print(
        "Building Fars DEM..."
    )

    datasets = []

    try:

        for file in files:

            datasets.append(
                rasterio.open(file)
            )

        mosaic, transform = merge(
            datasets
        )

        source = datasets[0]

        profile = source.profile.copy()

        profile.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": 1,
                "dtype": "float32",
                "compress": "deflate",
                "nodata": -9999
            }
        )

        # ----------------------------------------------------
        # Clip
        # ----------------------------------------------------

        temp_file = os.path.join(
            OUTPUT_DIR,
            "temp_dem.tif"
        )

        with rasterio.open(
            temp_file,
            "w",
            **profile
        ) as dst:

            dst.write(
                mosaic[0].astype(
                    "float32"
                ),
                1
            )

        with rasterio.open(
            temp_file
        ) as src:

            clipped, clipped_transform = mask(
                src,
                [mapping(boundary)],
                crop=True,
                nodata=-9999
            )

            profile = src.profile.copy()

            profile.update(
                {
                    "height": clipped.shape[1],
                    "width": clipped.shape[2],
                    "transform": clipped_transform,
                    "count": 1,
                    "dtype": "float32",
                    "nodata": -9999,
                    "compress": "deflate"
                }
            )

        with rasterio.open(
            DEM_FILE,
            "w",
            **profile
        ) as dst:

            dst.write(
                clipped[0].astype(
                    "float32"
                ),
                1
            )

    finally:

        for dataset in datasets:

            dataset.close()

    if os.path.exists(
        temp_file
    ):

        os.remove(
            temp_file
        )

    print(
        "DEM created:",
        DEM_FILE
    )


# ============================================================
# محاسبه Slope و Aspect
# ============================================================

def calculate_slope_aspect():

    print()
    print(
        "Calculating slope and aspect..."
    )

    with rasterio.open(
        DEM_FILE
    ) as src:

        dem = src.read(
            1
        ).astype(
            "float32"
        )

        profile = src.profile.copy()

        transform = src.transform

        xres = abs(
            transform.a
        )

        yres = abs(
            transform.e
        )

        nodata = src.nodata

    valid = (
        dem != nodata
    )

    # --------------------------------------------------------
    # ارتفاع
    # --------------------------------------------------------

    dzdy, dzdx = np.gradient(
        dem,
        yres,
        xres
    )

    # --------------------------------------------------------
    # Slope
    # --------------------------------------------------------

    slope = np.degrees(
        np.arctan(
            np.sqrt(
                dzdx ** 2
                +
                dzdy ** 2
            )
        )
    )

    # --------------------------------------------------------
    # Aspect
    # --------------------------------------------------------

    aspect = np.degrees(
        np.arctan2(
            -dzdx,
            dzdy
        )
    )

    aspect = (
        360
        -
        (aspect + 360) % 360
    )

    aspect[
        aspect == 360
    ] = 0

    slope[
        ~valid
    ] = -9999

    aspect[
        ~valid
    ] = -9999

    profile.update(
        {
            "dtype": "float32",
            "count": 1,
            "nodata": -9999,
            "compress": "deflate"
        }
    )

    # --------------------------------------------------------
    # Slope
    # --------------------------------------------------------

    with rasterio.open(
        SLOPE_FILE,
        "w",
        **profile
    ) as dst:

        dst.write(
            slope.astype(
                "float32"
            ),
            1
        )

    # --------------------------------------------------------
    # Aspect
    # --------------------------------------------------------

    with rasterio.open(
        ASPECT_FILE,
        "w",
        **profile
    ) as dst:

        dst.write(
            aspect.astype(
                "float32"
            ),
            1
        )

    print(
        "Slope created:",
        SLOPE_FILE
    )

    print(
        "Aspect created:",
        ASPECT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "FARS-HRI | TOPOGRAPHY"
    )
    print(
        "DEM + Slope + Aspect"
    )
    print("=" * 60)

    boundary = load_boundary()

    tiles = get_tiles(
        boundary
    )

    print()
    print(
        "Required DEM tiles:",
        len(tiles)
    )

    files = download_tiles(
        tiles,
        boundary
    )

    if not files:

        raise RuntimeError(
            "No DEM tiles downloaded."
        )

    build_dem(
        files,
        boundary
    )

    calculate_slope_aspect()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":

    main()
