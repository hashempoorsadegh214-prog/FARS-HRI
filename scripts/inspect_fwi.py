#!/usr/bin/env python3
"""
FARS-HRI | Inspect downloaded EFFIS FWI GeoTIFF

This script checks whether data/fwi/fwi_fars.tif contains:
- a single numeric FWI band, or
- a rendered RGB/RGBA image from the WMS.

It writes a report to:
    data/fwi/fwi_fars_inspection.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "fwi" / "fwi_fars.tif"
OUTPUT_FILE = PROJECT_ROOT / "data" / "fwi" / "fwi_fars_inspection.json"


def to_json_value(value: Any) -> Any:
    """Convert NumPy values into normal JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()

    return value


def main() -> None:
    """Inspect TIFF properties and pixel-value distribution."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"FWI raster was not found: {INPUT_FILE}"
        )

    with rasterio.open(INPUT_FILE) as dataset:
        report: dict[str, Any] = {
            "project": "FARS-HRI",
            "inspected_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
            "driver": dataset.driver,
            "crs": str(dataset.crs),
            "width_pixels": dataset.width,
            "height_pixels": dataset.height,
            "band_count": dataset.count,
            "dtypes": list(dataset.dtypes),
            "nodata_value": dataset.nodata,
            "color_interpretations": [
                color.name for color in dataset.colorinterp
            ],
            "bands": [],
        }

        for band_number in range(1, dataset.count + 1):
            values = dataset.read(band_number, masked=True)

            valid_values = values.compressed()

            band_report: dict[str, Any] = {
                "band": band_number,
                "valid_pixel_count": int(valid_values.size),
            }

            if valid_values.size > 0:
                unique_values = np.unique(valid_values)

                band_report.update(
                    {
                        "minimum": to_json_value(valid_values.min()),
                        "maximum": to_json_value(valid_values.max()),
                        "mean": float(valid_values.mean()),
                        "unique_value_count": int(unique_values.size),
                        "first_unique_values": [
                            to_json_value(value)
                            for value in unique_values[:30]
                        ],
                    }
                )

            report["bands"].append(band_report)

        # Practical decision for the next pipeline step.
        if dataset.count == 1 and dataset.dtypes[0].startswith(
            ("float", "int", "uint")
        ):
            report["assessment"] = (
                "Single numeric band detected. Values still need to be "
                "checked against the official EFFIS legend before using "
                "them as direct FWI values."
            )
        elif dataset.count in (3, 4):
            report["assessment"] = (
                "Multiple bands detected. This is likely a rendered "
                "RGB/RGBA WMS map image, not a direct numeric FWI raster."
            )
        else:
            report["assessment"] = (
                "Unexpected band structure. Manual review is required."
            )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Inspection report saved: {OUTPUT_FILE}")
    print(report["assessment"])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Inspection failed: {error}", file=sys.stderr)
        sys.exit(1)
