import json
import os
import math

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import shape, mapping
from shapely.ops import unary_union


# ============================================================
# FARS-HRI
# Lightweight Topography
# DEM + Slope + Aspect
# Copernicus GLO-90
# ============================================================

BOUNDARY_FILE = "fars.geojson"

OUTPUT_DIR = "data/topography"

DEM_FILE = os.path.join(
    OUTPUT_DIR,
    "fars_dem.tif"
)

SLOPE_FILE = os.path.join(
    OUTPUT_DIR,
    "fars_slope.tif"
)

ASPECT_FILE = os.path.join(
    OUTPUT_DIR,
    "fars_aspect.tif"
)


# ============================================================
# Copernicus GLO-90
# ============================================================

BASE_URL = (
    "https://copernicus-dem-90m.s3.amazonaws.com/"
)

RESOLUTION = "30"


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

    return unary_union(
        geometries
    )


# ============================================================
# نام Tile
# ============================================================

def tile_name(lat, lon):

    if lat >= 0:
        north = f"N{lat:02d}"
    else:
        north = f"S{abs(lat):02d}"

    if lon >= 0:
        east = f"E{lon:03d}"
    else:
        east = f"W{abs(lon):03d}"

    return (
        f"Copernicus_DSM_COG_{RESOLUTION}_"
        f"{north}_00_"
        f"{east}_00_DEM"
    )


# ============================================================
# ساخت URL
# ============================================================

def tile_url(name):

    return (
        BASE_URL
        + name
        + "/"
        + name
        + ".tif"
    )


# ============================================================
# پیدا کردن Tile های موردنیاز
# ============================================================

def get_tile_urls(boundary):

    minx, miny, maxx, maxy = boundary.bounds

    min_lon = math.floor(minx)
    max_lon = math.floor(maxx)

    min_lat = math.floor(miny)
    max_lat = math.floor(maxy)

    urls = []

    for lat in range(
        min_lat,
        max_lat + 1
    ):

        for lon in range(
            min_lon,
            max_lon + 1
        ):

            name = tile_name(
                lat,
                lon
            )

            urls.append(
                tile_url(name)
            )

    return urls


# ============================================================
# دریافت DEM به‌صورت Remote COG
# ============================================================

def build_dem(boundary, urls):

    print()
    print(
        "Opening DEM tiles remotely..."
    )

    datasets = []

    try:

        for url in urls:

            print(
                "Opening:",
                url.split("/")[-2]
            )

            try:

                src = rasterio.open(
                    "/vsicurl/" + url
                )

                datasets.append(src)

            except Exception as e:

                print(
                    "Tile unavailable:",
                    e
                )

        if not datasets:

            raise RuntimeError(
                "No DEM tile could be opened."
            )

        print()
        print(
            "Merging required DEM tiles..."
        )

        mosaic, transform = merge(
            datasets
        )

        profile = datasets[0].profile.copy()

        profile.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": 1,
                "dtype": "float32",
                "nodata": -9999,
                "compress": "deflate"
            }
        )

        temp_file = os.path.join(
            OUTPUT_DIR,
            "temp_dem.tif"
        )

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

        with rasterio.open(
            temp_file,
            "w",
            **profile
        ) as dst:

            dst.write(
                mosaic[0].astype(
                    np.float32
                ),
                1
            )

        print(
            "Clipping to Fars boundary..."
        )

        with rasterio.open(
            temp_file
        ) as src:

            clipped, transform = mask(
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
                    "transform": transform,
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
                    np.float32
                ),
                1
            )

        os.remove(
            temp_file
        )

    finally:

        for src in datasets:

            src.close()

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
        "Calculating Slope and Aspect..."
    )

    with rasterio.open(
        DEM_FILE
    ) as src:

        dem = src.read(
            1
        ).astype(
            np.float32
        )

        profile = src.profile.copy()

        transform = src.transform

        nodata = src.nodata

        xres = abs(
            transform.a
        )

        yres = abs(
            transform.e
        )

    valid = (
        dem != nodata
    )

    dem_work = dem.copy()

    dem_work[
        ~valid
    ] = np.nan

    dz_dy, dz_dx = np.gradient(
        dem_work,
        yres,
        xres
    )

    # --------------------------------------------------------
    # Slope
    # --------------------------------------------------------

    slope = np.degrees(
        np.arctan(
            np.sqrt(
                dz_dx ** 2
                +
                dz_dy ** 2
            )
        )
    )

    # --------------------------------------------------------
    # Aspect
    # --------------------------------------------------------

    aspect = (
        180
        -
        np.degrees(
            np.arctan2(
                dz_dx,
                dz_dy
            )
        )
    ) % 360

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

    with rasterio.open(
        SLOPE_FILE,
        "w",
        **profile
    ) as dst:

        dst.write(
            slope.astype(
                np.float32
            ),
            1
        )

    with rasterio.open(
        ASPECT_FILE,
        "w",
        **profile
    ) as dst:

        dst.write(
            aspect.astype(
                np.float32
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
        "FARS-HRI | LIGHT TOPOGRAPHY"
    )
    print(
        "Copernicus GLO-90"
    )
    print("=" * 60)

    boundary = load_boundary()

    urls = get_tile_urls(
        boundary
    )

    print()
    print(
        "Required tiles:",
        len(urls)
    )

    build_dem(
        boundary,
        urls
    )

    calculate_slope_aspect()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
