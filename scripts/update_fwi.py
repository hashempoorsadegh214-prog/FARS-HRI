#!/usr/bin/env python3
"""
FARS-HRI | Download and validate gridded FWI for Fars Province

Source:
    Copernicus GWIS / EFFIS WMS
    Meteorological model: Meteo-France
    Layer: mf010.fwi

Outputs created only after validation:
    data/fwi/fwi_fars.tif
    data/fwi/fwi_fars_metadata.json

Important:
    The script tries available forecast dates automatically.
    It refuses to overwrite the valid output with an all-zero or invalid TIFF.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FARS_BOUNDARY_FILE = PROJECT_ROOT / "fars.geojson"

OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "fwi"
OUTPUT_RASTER_FILE = OUTPUT_DIRECTORY / "fwi_fars.tif"
OUTPUT_METADATA_FILE = OUTPUT_DIRECTORY / "fwi_fars_metadata.json"

# ---------------------------------------------------------------------
# Official EFFIS WMS configuration
# ---------------------------------------------------------------------

WMS_URL = "https://maps.effis.emergency.copernicus.eu/effis"
WMS_VERSION = "1.3.0"
WMS_LAYER = "mf010.fwi"

# CRS:84 is longitude, latitude and avoids WMS 1.3 axis-order ambiguity.
WMS_CRS = "CRS:84"
WMS_IMAGE_FORMAT = "image/tiff"

HTTP_TIMEOUT_SECONDS = 120

# The native FWI grid is approximately 10 km.
# These dimensions are suitable for the Fars bounding box.
TARGET_WIDTH_PIXELS = 180
TARGET_HEIGHT_PIXELS = 150

# Add a small margin before clipping exactly to the Fars boundary.
BOUNDING_BOX_BUFFER_DEGREES = 0.08

# A correct FWI raster must contain at least one value above this threshold.
# FWI = 0 everywhere indicates an empty/unusable WMS response.
MINIMUM_VALID_FWI_VALUE = 0.01


# ---------------------------------------------------------------------
# General functions
# ---------------------------------------------------------------------

def save_json(file_path: Path, data: dict[str, Any]) -> None:
    """Write formatted JSON using UTF-8."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def get_requested_date() -> date | None:
    """
    Read an optional exact date from FWI_DATE.

    Example:
        FWI_DATE=2026-08-27 python scripts/update_fwi.py

    If FWI_DATE is not set, automatic date selection is used.
    """
    value = os.getenv("FWI_DATE", "").strip()

    if not value:
        return None

    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            "FWI_DATE must have this format: YYYY-MM-DD. "
            f"Received: {value!r}"
        ) from error


def get_candidate_dates() -> list[date]:
    """
    Build dates to try.

    EFFIS FWI provides a forecast horizon of up to about 3 days.
    We try yesterday, today and the next three dates. This makes the
    workflow resilient when the server updates at a different UTC hour.
    """
    manually_requested_date = get_requested_date()

    if manually_requested_date is not None:
        return [manually_requested_date]

    today = date.today()

    return [
        today - timedelta(days=1),
        today,
        today + timedelta(days=1),
        today + timedelta(days=2),
        today + timedelta(days=3),
    ]


def read_fars_boundary() -> gpd.GeoDataFrame:
    """Load the Fars polygon in longitude/latitude coordinates."""
    if not FARS_BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Boundary file not found: {FARS_BOUNDARY_FILE}"
        )

    boundary = gpd.read_file(FARS_BOUNDARY_FILE)

    if boundary.empty:
        raise ValueError("fars.geojson has no geographic features.")

    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    else:
        boundary = boundary.to_crs("EPSG:4326")

    return boundary


def get_request_bbox(
    boundary: gpd.GeoDataFrame,
) -> tuple[float, float, float, float]:
    """Return buffered BBOX as min_lon, min_lat, max_lon, max_lat."""
    min_lon, min_lat, max_lon, max_lat = boundary.total_bounds

    return (
        float(min_lon - BOUNDING_BOX_BUFFER_DEGREES),
        float(min_lat - BOUNDING_BOX_BUFFER_DEGREES),
        float(max_lon + BOUNDING_BOX_BUFFER_DEGREES),
        float(max_lat + BOUNDING_BOX_BUFFER_DEGREES),
    )


# ---------------------------------------------------------------------
# WMS request and validation
# ---------------------------------------------------------------------

def request_fwi_tiff(
    bbox: tuple[float, float, float, float],
    forecast_date: date,
) -> bytes:
    """Request one FWI GeoTIFF from the EFFIS WMS."""
    min_lon, min_lat, max_lon, max_lat = bbox

    parameters = {
        "SERVICE": "WMS",
        "VERSION": WMS_VERSION,
        "REQUEST": "GetMap",
        "LAYERS": WMS_LAYER,
        "STYLES": "",
        "CRS": WMS_CRS,
        "BBOX": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "WIDTH": TARGET_WIDTH_PIXELS,
        "HEIGHT": TARGET_HEIGHT_PIXELS,
        "FORMAT": WMS_IMAGE_FORMAT,
        "TRANSPARENT": "FALSE",
        "TIME": forecast_date.isoformat(),
    }

    response = requests.get(
        WMS_URL,
        params=parameters,
        timeout=HTTP_TIMEOUT_SECONDS,
    )

    content_type = response.headers.get("Content-Type", "").lower()

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code}: {response.text[:800]}"
        )

    # WMS errors sometimes have HTTP 200 but are XML/text responses.
    if (
        "xml" in content_type
        or "text" in content_type
        or response.content.lstrip().startswith(b"<?xml")
    ):
        raise RuntimeError(
            "WMS returned a service message instead of GeoTIFF: "
            f"{response.text[:1000]}"
        )

    if len(response.content) < 500:
        raise RuntimeError(
            "WMS response is unexpectedly small and is not a valid map TIFF."
        )

    return response.content


def inspect_raster_bytes(raster_bytes: bytes) -> dict[str, Any]:
    """
    Inspect TIFF values before saving.

    Returns statistics and valid=False for all-zero, empty or nonnumeric data.
    """
    with MemoryFile(raster_bytes) as memory_file:
        with memory_file.open() as dataset:
            if dataset.count != 1:
                return {
                    "valid": False,
                    "reason": (
                        f"Expected one FWI band, but received {dataset.count} bands."
                    ),
                }

            if not np.issubdtype(np.dtype(dataset.dtypes[0]), np.number):
                return {
                    "valid": False,
                    "reason": f"FWI band is not numeric: {dataset.dtypes[0]}",
                }

            values = dataset.read(1, masked=True)
            valid_values = values.compressed()

            if valid_values.size == 0:
                return {
                    "valid": False,
                    "reason": "The TIFF contains no valid pixels.",
                }

            minimum = float(np.min(valid_values))
            maximum = float(np.max(valid_values))
            mean = float(np.mean(valid_values))
            unique_count = int(np.unique(valid_values).size)

            # FWI must not be uniformly zero.
            if maximum <= MINIMUM_VALID_FWI_VALUE:
                return {
                    "valid": False,
                    "reason": (
                        "The TIFF contains only zero or near-zero values. "
                        "It is not usable as a gridded FWI input."
                    ),
                    "minimum": minimum,
                    "maximum": maximum,
                    "mean": mean,
                    "unique_value_count": unique_count,
                }

            # A completely uniform raster cannot represent a useful FWI grid.
            if unique_count <= 1:
                return {
                    "valid": False,
                    "reason": (
                        "The TIFF has only one unique value and is not a "
                        "usable spatial FWI grid."
                    ),
                    "minimum": minimum,
                    "maximum": maximum,
                    "mean": mean,
                    "unique_value_count": unique_count,
                }

            return {
                "valid": True,
                "reason": "Numeric FWI raster contains non-zero spatial values.",
                "crs": str(dataset.crs),
                "width_pixels": int(dataset.width),
                "height_pixels": int(dataset.height),
                "band_count": int(dataset.count),
                "dtype": dataset.dtypes[0],
                "nodata_value": dataset.nodata,
                "minimum": minimum,
                "maximum": maximum,
                "mean": mean,
                "unique_value_count": unique_count,
            }


# ---------------------------------------------------------------------
# Clip and write output
# ---------------------------------------------------------------------

def clip_and_save(
    raster_bytes: bytes,
    boundary: gpd.GeoDataFrame,
) -> dict[str, Any]:
    """
    Clip valid FWI raster to Fars Province and save the final GeoTIFF.

    This runs only after the downloaded TIFF has passed numeric validation.
    """
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with MemoryFile(raster_bytes) as memory_file:
        with memory_file.open() as source:
            if source.crs is None:
                raise RuntimeError("Downloaded WMS TIFF has no CRS.")

            boundary_for_raster = boundary.to_crs(source.crs)

            clipped_data, clipped_transform = mask(
                source,
                boundary_for_raster.geometry,
                crop=True,
                filled=True,
                nodata=source.nodata,
            )

            clipped_values = clipped_data[0]

            if source.nodata is not None:
                clipped_values = clipped_values[
                    clipped_values != source.nodata
                ]

            if clipped_values.size == 0:
                raise RuntimeError("No valid FWI pixels remain after clipping.")

            clipped_maximum = float(np.max(clipped_values))

            if clipped_maximum <= MINIMUM_VALID_FWI_VALUE:
                raise RuntimeError(
                    "After clipping to Fars, the raster has only zero values. "
                    "The output was not saved."
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

            temporary_output = OUTPUT_RASTER_FILE.with_suffix(".tmp.tif")

            with rasterio.open(
                temporary_output,
                "w",
                **profile,
            ) as destination:
                destination.write(clipped_data)

            # Replace the old file only after a complete successful write.
            temporary_output.replace(OUTPUT_RASTER_FILE)

            return {
                "source_crs": str(source.crs),
                "width_pixels": int(clipped_data.shape[2]),
                "height_pixels": int(clipped_data.shape[1]),
                "band_count": int(clipped_data.shape[0]),
                "nodata_value": source.nodata,
                "minimum_after_clip": float(np.min(clipped_values)),
                "maximum_after_clip": float(np.max(clipped_values)),
                "mean_after_clip": float(np.mean(clipped_values)),
                "unique_value_count_after_clip": int(
                    np.unique(clipped_values).size
                ),
            }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Try forecast dates, validate FWI data, clip and save the first valid grid."""
    boundary = read_fars_boundary()
    bbox = get_request_bbox(boundary)
    candidate_dates = get_candidate_dates()

    print("Starting EFFIS gridded FWI update...")
    print(f"Layer: {WMS_LAYER}")
    print(
        "Fars request BBOX: "
        f"{bbox[0]:.6f}, {bbox[1]:.6f}, {bbox[2]:.6f}, {bbox[3]:.6f}"
    )

    attempt_log: list[dict[str, str]] = []

    for forecast_date in candidate_dates:
        print(f"\nTrying forecast date: {forecast_date.isoformat()}")

        try:
            raster_bytes = request_fwi_tiff(
                bbox=bbox,
                forecast_date=forecast_date,
            )

            inspection = inspect_raster_bytes(raster_bytes)

            if not inspection["valid"]:
                reason = str(inspection["reason"])
                print(f"Rejected: {reason}")

                attempt_log.append(
                    {
                        "date": forecast_date.isoformat(),
                        "result": "rejected",
                        "reason": reason,
                    }
                )
                continue

            clipping_information = clip_and_save(
                raster_bytes=raster_bytes,
                boundary=boundary,
            )

            metadata = {
                "project": "FARS-HRI",
                "updated_at_utc": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "source": "Copernicus GWIS / EFFIS",
                "meteorological_model": "Meteo-France",
                "service": "WMS",
                "wms_url": WMS_URL,
                "wms_version": WMS_VERSION,
                "layer": WMS_LAYER,
                "variable": "Fire Weather Index (FWI)",
                "approximate_native_resolution": "~10 km",
                "selected_forecast_date": forecast_date.isoformat(),
                "output_file": str(
                    OUTPUT_RASTER_FILE.relative_to(PROJECT_ROOT)
                ),
                "boundary_file": str(
                    FARS_BOUNDARY_FILE.relative_to(PROJECT_ROOT)
                ),
                "wms_request": {
                    "crs": WMS_CRS,
                    "bbox": {
                        "min_longitude": bbox[0],
                        "min_latitude": bbox[1],
                        "max_longitude": bbox[2],
                        "max_latitude": bbox[3],
                    },
                    "width_pixels": TARGET_WIDTH_PIXELS,
                    "height_pixels": TARGET_HEIGHT_PIXELS,
                },
                "download_validation": inspection,
                "clipped_raster_information": clipping_information,
                "attempt_log": attempt_log
                + [
                    {
                        "date": forecast_date.isoformat(),
                        "result": "accepted",
                        "reason": str(inspection["reason"]),
                    }
                ],
            }

            save_json(OUTPUT_METADATA_FILE, metadata)

            print("\nSuccess: valid FWI grid saved.")
            print(f"Date: {forecast_date.isoformat()}")
            print(f"Raster: {OUTPUT_RASTER_FILE}")
            print(f"Metadata: {OUTPUT_METADATA_FILE}")
            return

        except Exception as error:
            reason = str(error)

            print(f"Failed: {reason}")

            attempt_log.append(
                {
                    "date": forecast_date.isoformat(),
                    "result": "error",
                    "reason": reason,
                }
            )

    error_summary = "\n".join(
        f"- {item['date']}: {item['result']} — {item['reason']}"
        for item in attempt_log
    )

    raise RuntimeError(
        "No valid non-zero FWI grid was returned by EFFIS for the tested "
        "forecast dates. The previous output was not replaced.\n"
        f"{error_summary}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nFWI update failed.", file=sys.stderr)
        print(str(error), file=sys.stderr)
        sys.exit(1)
