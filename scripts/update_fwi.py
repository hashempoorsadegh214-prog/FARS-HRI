#!/usr/bin/env python3
"""
FARS-HRI | Download validated gridded Fire Weather Index for Fars Province

Source:
    Copernicus EFFIS / GWIS
    Meteorological model: Meteo-France
    Layer: mf010.fwi
    Service: EFFIS forecast WMS

Outputs, only when a valid non-zero raster is received:
    data/fwi/fwi_fars.tif
    data/fwi/fwi_fars_metadata.json

Optional:
    Set FWI_DATE to request a specific forecast date.

Example:
    FWI_DATE=2026-08-27 python scripts/update_fwi.py
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
# Project files
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FARS_BOUNDARY_FILE = PROJECT_ROOT / "fars.geojson"

OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "fwi"
OUTPUT_RASTER_FILE = OUTPUT_DIRECTORY / "fwi_fars.tif"
OUTPUT_METADATA_FILE = OUTPUT_DIRECTORY / "fwi_fars_metadata.json"


# ---------------------------------------------------------------------
# Copernicus EFFIS / GWIS forecast WMS settings
# ---------------------------------------------------------------------

# Important: "effist" is the forecast endpoint.
WMS_URL = "https://maps.effis.emergency.copernicus.eu/effist"

WMS_VERSION = "1.3.0"
WMS_LAYER = "mf010.fwi"

# CRS:84 keeps BBOX axis order fixed as longitude, latitude.
WMS_CRS = "CRS:84"
WMS_FORMAT = "image/tiff"

HTTP_TIMEOUT_SECONDS = 120

# Native layer resolution is approximately 10 km.
# These dimensions are appropriate for the Fars extent.
REQUEST_WIDTH = 180
REQUEST_HEIGHT = 150

# Small geographic margin before clipping exactly to the province boundary.
BBOX_BUFFER_DEGREES = 0.08

# Reject empty and all-zero TIFF responses.
MINIMUM_VALID_FWI = 0.01


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def save_json(file_path: Path, content: dict[str, Any]) -> None:
    """Save formatted UTF-8 JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
        file.write("\n")


def read_fars_boundary() -> gpd.GeoDataFrame:
    """Read Fars Province boundary and ensure EPSG:4326 coordinates."""
    if not FARS_BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Boundary file not found: {FARS_BOUNDARY_FILE}"
        )

    boundary = gpd.read_file(FARS_BOUNDARY_FILE)

    if boundary.empty:
        raise ValueError("fars.geojson contains no geographic features.")

    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    else:
        boundary = boundary.to_crs("EPSG:4326")

    return boundary


def get_fars_bbox(
    boundary: gpd.GeoDataFrame,
) -> tuple[float, float, float, float]:
    """Return buffered BBOX as min_lon, min_lat, max_lon, max_lat."""
    min_lon, min_lat, max_lon, max_lat = boundary.total_bounds

    return (
        float(min_lon - BBOX_BUFFER_DEGREES),
        float(min_lat - BBOX_BUFFER_DEGREES),
        float(max_lon + BBOX_BUFFER_DEGREES),
        float(max_lat + BBOX_BUFFER_DEGREES),
    )


def get_candidate_dates() -> list[date]:
    """
    Return forecast dates to try.

    If FWI_DATE is defined, only that exact date is used.
    Otherwise, try yesterday through three days ahead. This covers the
    normal EFFIS forecast window and publication-time differences.
    """
    requested_date = os.getenv("FWI_DATE", "").strip()

    if requested_date:
        try:
            return [date.fromisoformat(requested_date)]
        except ValueError as error:
            raise ValueError(
                "FWI_DATE must use YYYY-MM-DD format. "
                f"Received: {requested_date!r}"
            ) from error

    today = date.today()

    return [
        today - timedelta(days=1),
        today,
        today + timedelta(days=1),
        today + timedelta(days=2),
        today + timedelta(days=3),
    ]


# ---------------------------------------------------------------------
# Download and validation
# ---------------------------------------------------------------------

def request_fwi_tiff(
    bbox: tuple[float, float, float, float],
    forecast_date: date,
) -> bytes:
    """Download one FWI GeoTIFF request from the EFFIS forecast WMS."""
    min_lon, min_lat, max_lon, max_lat = bbox

    parameters = {
        "SERVICE": "WMS",
        "VERSION": WMS_VERSION,
        "REQUEST": "GetMap",
        "LAYERS": WMS_LAYER,
        "STYLES": "",
        "CRS": WMS_CRS,
        "BBOX": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "WIDTH": REQUEST_WIDTH,
        "HEIGHT": REQUEST_HEIGHT,
        "FORMAT": WMS_FORMAT,
        "TRANSPARENT": "FALSE",
        "TIME": forecast_date.isoformat(),
    }

    response = requests.get(
        WMS_URL,
        params=parameters,
        timeout=HTTP_TIMEOUT_SECONDS,
    )

    content_type = response.headers.get("Content-Type", "").lower()
    response_start = response.content.lstrip()[:100].lower()

    if response.status_code != 200:
        raise RuntimeError(
            f"WMS request returned HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    # WMS errors can return HTTP 200 together with XML or plain text.
    if (
        "xml" in content_type
        or "text" in content_type
        or response_start.startswith(b"<?xml")
        or response_start.startswith(b"<serviceexception")
    ):
        raise RuntimeError(
            "WMS returned a service message instead of a GeoTIFF: "
            f"{response.text[:1500]}"
        )

    if len(response.content) < 500:
        raise RuntimeError(
            "WMS response is too small to be a valid GeoTIFF."
        )

    return response.content


def inspect_downloaded_tiff(raster_bytes: bytes) -> dict[str, Any]:
    """
    Validate the downloaded TIFF before it can overwrite the current output.

    A valid grid must be:
    - exactly one numeric band;
    - non-empty;
    - not entirely zero;
    - spatially variable (more than one unique value).
    """
    with MemoryFile(raster_bytes) as memory_file:
        with memory_file.open() as dataset:
            if dataset.count != 1:
                return {
                    "valid": False,
                    "reason": (
                        "Expected one numeric FWI band, "
                        f"but received {dataset.count} bands."
                    ),
                }

            data_type = np.dtype(dataset.dtypes[0])

            if not np.issubdtype(data_type, np.number):
                return {
                    "valid": False,
                    "reason": (
                        f"FWI raster band is not numeric: {dataset.dtypes[0]}"
                    ),
                }

            values = dataset.read(1, masked=True)
            valid_values = values.compressed()

            if valid_values.size == 0:
                return {
                    "valid": False,
                    "reason": "Downloaded TIFF contains no valid pixels.",
                }

            minimum = float(np.min(valid_values))
            maximum = float(np.max(valid_values))
            mean = float(np.mean(valid_values))
            unique_count = int(np.unique(valid_values).size)

            if maximum <= MINIMUM_VALID_FWI:
                return {
                    "valid": False,
                    "reason": (
                        "Downloaded TIFF contains only zero or near-zero "
                        "values, so it is not a usable FWI grid."
                    ),
                    "minimum": minimum,
                    "maximum": maximum,
                    "mean": mean,
                    "unique_value_count": unique_count,
                }

            if unique_count <= 1:
                return {
                    "valid": False,
                    "reason": (
                        "Downloaded TIFF has only one unique value and is "
                        "not a usable spatial FWI grid."
                    ),
                    "minimum": minimum,
                    "maximum": maximum,
                    "mean": mean,
                    "unique_value_count": unique_count,
                }

            return {
                "valid": True,
                "reason": "Non-zero, spatially variable numeric FWI grid received.",
                "crs": str(dataset.crs),
                "band_count": int(dataset.count),
                "dtype": dataset.dtypes[0],
                "width_pixels": int(dataset.width),
                "height_pixels": int(dataset.height),
                "nodata_value": dataset.nodata,
                "minimum": minimum,
                "maximum": maximum,
                "mean": mean,
                "unique_value_count": unique_count,
            }


# ---------------------------------------------------------------------
# Clip and write final output
# ---------------------------------------------------------------------

def clip_and_save_fwi(
    raster_bytes: bytes,
    boundary: gpd.GeoDataFrame,
) -> dict[str, Any]:
    """
    Clip the validated WMS GeoTIFF to Fars and write the final raster.

    The old output is replaced only after successful validation and writing.
    """
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with MemoryFile(raster_bytes) as memory_file:
        with memory_file.open() as source:
            if source.crs is None:
                raise RuntimeError(
                    "Downloaded GeoTIFF has no CRS and cannot be clipped safely."
                )

            boundary_in_raster_crs = boundary.to_crs(source.crs)

            clipped_data, clipped_transform = mask(
                source,
                boundary_in_raster_crs.geometry,
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
                raise RuntimeError(
                    "No valid FWI pixels remain after clipping to Fars."
                )

            clipped_minimum = float(np.min(clipped_values))
            clipped_maximum = float(np.max(clipped_values))
            clipped_mean = float(np.mean(clipped_values))
            clipped_unique_count = int(np.unique(clipped_values).size)

            if clipped_maximum <= MINIMUM_VALID_FWI:
                raise RuntimeError(
                    "All FWI values are zero after clipping to Fars. "
                    "The old output was preserved."
                )

            if clipped_unique_count <= 1:
                raise RuntimeError(
                    "FWI raster is uniform after clipping to Fars. "
                    "The old output was preserved."
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

            temporary_file = OUTPUT_RASTER_FILE.with_suffix(".temporary.tif")

            with rasterio.open(temporary_file, "w", **profile) as output:
                output.write(clipped_data)

            temporary_file.replace(OUTPUT_RASTER_FILE)

            return {
                "source_crs": str(source.crs),
                "band_count": int(clipped_data.shape[0]),
                "width_pixels": int(clipped_data.shape[2]),
                "height_pixels": int(clipped_data.shape[1]),
                "nodata_value": source.nodata,
                "minimum_after_clip": clipped_minimum,
                "maximum_after_clip": clipped_maximum,
                "mean_after_clip": clipped_mean,
                "unique_value_count_after_clip": clipped_unique_count,
            }


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main() -> None:
    """Download a valid FWI grid and save it for the FARS-HRI pipeline."""
    fars_boundary = read_fars_boundary()
    bbox = get_fars_bbox(fars_boundary)
    candidate_dates = get_candidate_dates()

    print("Starting FARS-HRI gridded FWI update...")
    print(f"Source endpoint: {WMS_URL}")
    print(f"Layer: {WMS_LAYER}")
    print(
        "Fars request BBOX: "
        f"{bbox[0]:.6f}, {bbox[1]:.6f}, "
        f"{bbox[2]:.6f}, {bbox[3]:.6f}"
    )

    attempts: list[dict[str, str]] = []

    for forecast_date in candidate_dates:
        print(f"\nTrying FWI date: {forecast_date.isoformat()}")

        try:
            raster_bytes = request_fwi_tiff(
                bbox=bbox,
                forecast_date=forecast_date,
            )

            validation = inspect_downloaded_tiff(raster_bytes)

            if not validation["valid"]:
                reason = str(validation["reason"])
                print(f"Rejected: {reason}")

                attempts.append(
                    {
                        "date": forecast_date.isoformat(),
                        "result": "rejected",
                        "reason": reason,
                    }
                )
                continue

            clipped_information = clip_and_save_fwi(
                raster_bytes=raster_bytes,
                boundary=fars_boundary,
            )

            metadata = {
                "project": "FARS-HRI",
                "updated_at_utc": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "meteorological_source": "Copernicus EFFIS / GWIS",
                "meteorological_model": "Meteo-France",
                "variable": "Fire Weather Index (FWI)",
                "service": "EFFIS forecast WMS",
                "endpoint": WMS_URL,
                "wms_version": WMS_VERSION,
                "layer": WMS_LAYER,
                "selected_forecast_date": forecast_date.isoformat(),
                "approximate_native_resolution": "~10 km",
                "boundary_file": str(
                    FARS_BOUNDARY_FILE.relative_to(PROJECT_ROOT)
                ),
                "output_file": str(
                    OUTPUT_RASTER_FILE.relative_to(PROJECT_ROOT)
                ),
                "wms_request": {
                    "crs": WMS_CRS,
                    "bbox": {
                        "min_longitude": bbox[0],
                        "min_latitude": bbox[1],
                        "max_longitude": bbox[2],
                        "max_latitude": bbox[3],
                    },
                    "width_pixels": REQUEST_WIDTH,
                    "height_pixels": REQUEST_HEIGHT,
                },
                "download_validation": validation,
                "clipped_raster_information": clipped_information,
                "attempts": attempts
                + [
                    {
                        "date": forecast_date.isoformat(),
                        "result": "accepted",
                        "reason": str(validation["reason"]),
                    }
                ],
            }

            save_json(OUTPUT_METADATA_FILE, metadata)

            print("\nSuccess: valid FWI grid saved.")
            print(f"Selected date: {forecast_date.isoformat()}")
            print(f"Raster: {OUTPUT_RASTER_FILE}")
            print(f"Metadata: {OUTPUT_METADATA_FILE}")
            return

        except Exception as error:
            reason = str(error)
            print(f"Failed: {reason}")

            attempts.append(
                {
                    "date": forecast_date.isoformat(),
                    "result": "error",
                    "reason": reason,
                }
            )

    attempt_report = "\n".join(
        f"- {item['date']}: {item['result']} | {item['reason']}"
        for item in attempts
    )

    raise RuntimeError(
        "No valid FWI grid was received from the EFFIS forecast WMS. "
        "Existing output files were not replaced.\n"
        f"{attempt_report}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nFWI update failed.", file=sys.stderr)
        print(str(error), file=sys.stderr)
        sys.exit(1)
