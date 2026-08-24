import os
import json

import numpy as np
import rasterio


# ============================================================
# FARS-HRI
# Fuel Processing
# Source:
# Pettinari et al. - Global Fuelbed Dataset
# PANGAEA DOI: 10.1594/PANGAEA.849808
# ============================================================

INPUT_FILE = "data/fuel/fars_fuel.tif"

OUTPUT_DIR = "data/fuel"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "fars_fuel_processed.tif"
)

STATS_FILE = os.path.join(
    OUTPUT_DIR,
    "fars_fuel_stats.json"
)


# ============================================================
# پردازش Fuel
# ============================================================

def main():

    print("=" * 60)
    print("FARS-HRI | FUEL PROCESSING")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Fuel file not found: {INPUT_FILE}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print()
    print("Input:")
    print(INPUT_FILE)

    # --------------------------------------------------------
    # خواندن Raster
    # --------------------------------------------------------

    with rasterio.open(
        INPUT_FILE
    ) as src:

        data = src.read(
            1
        ).astype(
            np.float32
        )

        profile = src.profile.copy()

        nodata = src.nodata

        transform = src.transform

        crs = src.crs

    print()
    print("CRS:", crs)

    print(
        "Raster size:",
        data.shape[1],
        "x",
        data.shape[0]
    )

    # --------------------------------------------------------
    # تعیین NoData
    # --------------------------------------------------------

    if nodata is None:

        nodata = -9999

    valid = (
        np.isfinite(data)
        &
        (data != nodata)
    )

    valid_values = data[
        valid
    ]

    if len(valid_values) == 0:

        raise RuntimeError(
            "No valid Fuel values found."
        )

    # --------------------------------------------------------
    # آمار
    # --------------------------------------------------------

    minimum = float(
        np.min(valid_values)
    )

    maximum = float(
        np.max(valid_values)
    )

    mean = float(
        np.mean(valid_values)
    )

    print()
    print("Fuel statistics:")
    print(
        "Min :",
        minimum
    )
    print(
        "Max :",
        maximum
    )
    print(
        "Mean:",
        mean
    )

    # --------------------------------------------------------
    # خروجی
    # --------------------------------------------------------

    profile.update(
        {
            "dtype": "float32",
            "count": 1,
            "nodata": -9999,
            "compress": "deflate"
        }
    )

    output_data = data.copy()

    output_data[
        ~valid
    ] = -9999

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        **profile
    ) as dst:

        dst.write(
            output_data,
            1
        )

    # --------------------------------------------------------
    # ذخیره Metadata
    # --------------------------------------------------------

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

        "input": INPUT_FILE,

        "output": OUTPUT_FILE,

        "minimum": minimum,

        "maximum": maximum,

        "mean": mean,

        "valid_pixels": int(
            len(valid_values)
        ),

        "crs": str(crs),

        "pixel_width": float(
            transform.a
        ),

        "pixel_height": float(
            abs(transform.e)
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

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print()
    print(
        "Processed Fuel:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Statistics:"
    )

    print(
        STATS_FILE
    )


if __name__ == "__main__":
    main()
