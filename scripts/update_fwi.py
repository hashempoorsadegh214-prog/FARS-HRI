#!/usr/bin/env python3
"""
FARS-HRI | Download gridded Fire Weather Index (FWI) for Fars Province

Data source:
    Copernicus GWIS / EFFIS WMS
    Meteorological model: Meteo-France
    Layer: mf010.fwi
    Approximate spatial resolution: 10 km

Outputs:
    data/fwi/fwi_fars.tif
    data/fwi/fwi_fars_metadata.json

Optional environment variable:
    FWI_DATE=YYYY-MM-DD

Example:
    FWI_DATE=2026-08-27 python scripts/update_fwi.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask


# ---------------------------------------------------------------------
# Project paths and service configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FARS_BOUNDARY_FILE = PROJECT_ROOT / "fars.geojson"

OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "fwi"
OUTPUT_RASTER_FILE = OUTPUT_DIRECTORY / "fwi_fars.tif"
OUTPUT_METADATA_FILE = OUTPUT_DIRECTORY / "fwi_fars_metadata.json"

# Official Copernicus EFFIS WMS endpoint.
WMS_URL = "https://maps.effis.emergency.copernicus.eu/effis"

WMS_VERSION = "1.3.0"
WMS_LAYER = "mf010.fwi"
WMS_CRS = "EPSG:4326"
WMS_IMAGE_FORMAT = "image/tiff"

HTTP_TIMEOUT_SECONDS = 120

# The layer itself has about 10 km resolution. Requesting very large
# dimensions would only enlarge the image, not add real meteorological detail.
TARGET_WIDTH_PIXELS = 180
TARGET_HEIGHT_PIXELS = 150

# Small buffer around the provincial boundary to avoid edge cropping.
BOUNDING_BOX_BUFFER_DEGREES = 0.08


# ---------------------------------------------------------------------
# General utility functions
# ---------------------------------------------------------------------

def save_json(file_path: Path, data: dict[str, Any]) -> None:
    """Write a UTF-8 JSON file with readable formatting."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def get_requested_date() -> str:
    """
    Read FWI_DATE from the environment, if supplied.

    If not supplied, use today's date in UTC. The EFFIS service determines
    whether that date is available according to its current forecast period.
    """
    requested_date = os.getenv("FWI_DATE", "").strip()

    if not requested_date:
        return date.today().isoformat()

    try:
        return date.fromisoformat(requested_date).isoformat()
    except ValueError as error:
        raise ValueError(
            "FWI_DATE must use YYYY-MM-DD format. "
            f"Received: {requested_date!r}"
        ) from error


def read_fars_boundary() -> gpd.GeoDataFrame:
    """Read the Fars boundary and ensure it is in EPSG:4326."""
    if not FARS_BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            "Fars boundary file was not found:\n"
            f"{FARS_BOUNDARY_FILE}"
        )

    boundary = gpd.read_file(FARS_BOUNDARY_FILE)

    if boundary.empty:
        raise ValueError("fars.geojson does not contain any features.")

    if boundary.crs is None:
        boundary = boundary.set_crs(WMS_CRS)
    else:
        boundary = boundary.to_crs(WMS_CRS)

    return boundary


def get_request_bbox(boundary: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """
    Return WMS BBOX as min longitude, min latitude, max longitude, max latitude.

    For WMS 1.3.0 with EPSG:4326, axis order can be latitude/longitude.
    To avoid ambiguity, the actual WMS request below uses CRS:84.
    """
    min_x, min_y, max_x, max_y = boundary.total_bounds

    return (
        float(min_x - BOUNDING_BOX_BUFFER_DEGREES),
        float(min_y - BOUNDING_BOX_BUFFER_DEGREES),
        float(max_x + BOUNDING_BOX_BUFFER_DEGREES),
        float(max_y + BOUNDING_BOX_BUFFER_DEGREES),
    )


# ---------------------------------------------------------------------
# WMS download
# ---------------------------------------------------------------------

def download_fwi_geotiff(
    bbox: tuple[float, float, float, float],
    requested_date: str,
) -> bytes:
    """
    Request the complete FWI map for Fars from EFFIS WMS as GeoTIFF.

    CRS:84 is used intentionally because it has fixed longitude/latitude
    axis order: longitude, latitude.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    parameters = {
        "SERVICE": "WMS",
        "VERSION": WMS_VERSION,
        "REQUEST": "GetMap",
        "LAYERS": WMS_LAYER,
        "STYLES": "",
        "CRS": "CRS:84",
        "BBOX": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "WIDTH": TARGET_WIDTH_PIXELS,
        "HEIGHT": TARGET_HEIGHT_PIXELS,
        "FORMAT": WMS_IMAGE_FORMAT,
        "TRANSPARENT": "FALSE",
        "TIME": requested_date,
    }

    print("Requesting gridded FWI from Copernicus EFFIS WMS...")
    print(f"Layer: {WMS_LAYER}")
    print(f"Date: {requested_date}")
    print(f"BBOX: {parameters['BBOX']}")

    response = requests.get(
        WMS_URL,
        params=parameters,
        timeout=HTTP_TIMEOUT_SECONDS,
    )

    content_type = response.headers.get("Content-Type", "").lower()

    if response.status_code != 200:
        raise RuntimeError(
            "EFFIS WMS request failed.\n"
            f"HTTP status: {response.status_code}\n"
            f"Response: {response.text[:1000]}"
        )

    # WMS errors are often returned as XML/text with HTTP 200.
    if "xml" in content_type or "text" in content_type:
        raise RuntimeError(
            "EFFIS WMS returned a service message instead of a TIFF file.\n"
            f"Response: {response.text[:1500]}"
        )

    if len(response.content) < 500:
        raise RuntimeError(
            "EFFIS WMS returned an unexpectedly small response; "
            "a valid map TIFF was not received."
        )

    return response.content


# ---------------------------------------------------------------------
# Clip and save output
# ---------------------------------------------------------------------

def clip_and_save_raster(
    raster_bytes: bytes,
    boundary: gpd.GeoDataFrame,
) -> dict[str, Any]:
    """Clip downloaded raster to Fars Province and save it as GeoTIFF."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with MemoryFile(raster_bytes) as memory_file:
        with memory_file.open() as source:
            source_crs = source.crs

            if source_crs is None:
                raise RuntimeError(
                    "The EFFIS WMS TIFF has no CRS information. "
                    "The output cannot be safely clipped."
                )

            boundary_for_raster = boundary.to_crs(source_crs)

            clipped_data, clipped_transform = mask(
                source,
                boundary_for_raster.geometry,
                crop=True,
                filled=True,
                nodata=source.nodata,
            )

            profile = source.profile.copy()
            profile.update(
                driver="GTiff",
                height=clipped_data.shape[1],
                width=clipped_data.shape[2],
                transform=clipped_transform,
                compress="deflate",
                tiled=False,
            )

            with rasterio.open(OUTPUT_RASTER_FILE, "w", **profile) as destination:
                destination.write(clipped_data)

            bounds = rasterio.transform.array_bounds(
                clipped_data.shape[1],
                clipped_data.shape[2],
                clipped_transform,
            )

            return {
                "source_crs": str(source_crs),
                "band_count": int(clipped_data.shape[0]),
                "width_pixels": int(clipped_data.shape[2]),
                "height_pixels": int(clipped_data.shape[1]),
                "nodata_value": source.nodata,
                "bounds_in_source_crs": {
                    "left": float(bounds[0]),
                    "bottom": float(bounds[1]),
                    "right": float(bounds[2]),
                    "top": float(bounds[3]),
                },
            }


# ---------------------------------------------------------------------
# Main process
# ---------------------------------------------------------------------

def main() -> None:
    """Download, clip and document the current FWI grid for Fars."""
    try:
        requested_date = get_requested_date()
        fars_boundary = read_fars_boundary()
        request_bbox = get_request_bbox(fars_boundary)

        raster_bytes = download_fwi_geotiff(
            bbox=request_bbox,
            requested_date=requested_date,
        )

        raster_information = clip_and_save_raster(
            raster_bytes=raster_bytes,
            boundary=fars_boundary,
        )

        metadata = {
            "project": "FARS-HRI",
            "updated_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "requested_forecast_date": requested_date,
            "source": "Copernicus GWIS / EFFIS",
            "meteorological_model": "Meteo-France",
            "service": "WMS",
            "wms_url": WMS_URL,
            "wms_version": WMS_VERSION,
            "layer": WMS_LAYER,
            "variable": "Fire Weather Index (FWI)",
            "approximate_native_resolution": "~10 km",
            "output_file": str(
                OUTPUT_RASTER_FILE.relative_to(PROJECT_ROOT)
            ),
            "boundary_file": str(
                FARS_BOUNDARY_FILE.relative_to(PROJECT_ROOT)
            ),
            "wms_request_bbox_crs84": {
                "min_longitude": request_bbox[0],
                "min_latitude": request_bbox[1],
                "max_longitude": request_bbox[2],
                "max_latitude": request_bbox[3],
            },
            "raster_information": raster_information,
            "important_note": (
                "This TIFF is the official map rendering returned by the "
                "EFFIS WMS. Before using it as a numeric FWI input in the "
                "HRI equation, the map legend/value encoding must be "
                "validated against the EFFIS service."
            ),
        }

        save_json(OUTPUT_METADATA_FILE, metadata)

        print("\nSuccess.")
        print(f"FWI raster saved: {OUTPUT_RASTER_FILE}")
        print(f"Metadata saved:   {OUTPUT_METADATA_FILE}")

    except Exception as error:
        print("\nFWI update failed.", file=sys.stderr)
        print(str(error), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
