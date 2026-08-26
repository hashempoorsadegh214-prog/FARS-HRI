#!/usr/bin/env python3
"""
Build the Fars Hazard Risk Index (HRI) raster.

Formula:
    HRI = 100 * (0.45 * FWI_n + 0.35 * Fuel_n + 0.20 * Topo_n)

Inputs:
    data/fwi/fwi_fars.tif
    data/fuel/fars_fuel.tif
    data/fuel/Global_fuelbeds_parameters_v1.2.xlsx
    data/dem_fars.tif

Output:
    data/output/fars_hri.tif

Fuel score:
    FineFuelLoad = G_Load + W_1hLoad + W_10hLoad
    FuelRaw      = FineFuelLoad + 0.25 * L_depth

Negative values in the fuelbed parameter table (-1, -3) mean
"not present" or "not applicable" and are converted to zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FWI_PATH = PROJECT_ROOT / "data" / "fwi" / "fwi_fars.tif"
FUEL_RASTER_PATH = PROJECT_ROOT / "data" / "fuel" / "fars_fuel.tif"
FUEL_PARAMETERS_PATH = (
    PROJECT_ROOT / "data" / "fuel" / "Global_fuelbeds_parameters_v1.2.xlsx"
)
DEM_PATH = PROJECT_ROOT / "data" / "dem_fars.tif"
OUTPUT_PATH = PROJECT_ROOT / "data" / "output" / "fars_hri.tif"

FUEL_SHEET_NAME = "Fuelbeds_metric"

# HRI weights
FWI_WEIGHT = 0.45
FUEL_WEIGHT = 0.35
TOPO_WEIGHT = 0.20

# HRI output NoData value
OUTPUT_NODATA = -9999.0


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def require_file(path: Path, label: str) -> None:
    """Stop with a clear message if a required input file does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired {label} file was not found:\n"
            f"  {path}\n\n"
            "Check that the file exists in the repository and that its "
            "path matches the project structure."
        )


def normalize_valid_values(
    array: np.ndarray,
    valid_mask: np.ndarray,
    label: str,
) -> np.ndarray:
    """
    Normalize valid values to [0, 1] using min-max normalization.

    Invalid pixels remain NaN.
    """
    result = np.full(array.shape, np.nan, dtype=np.float32)

    values = array[valid_mask]
    if values.size == 0:
        raise ValueError(f"No valid pixels were available for {label} normalization.")

    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError(f"{label} contains no finite values.")

    if np.isclose(maximum, minimum):
        result[valid_mask] = 0.0
        print(f"{label}: constant value {minimum:.4f}; normalized to 0.")
        return result

    result[valid_mask] = (array[valid_mask] - minimum) / (maximum - minimum)

    print(
        f"{label}: min={minimum:.4f}, max={maximum:.4f}, "
        f"valid_pixels={int(valid_mask.sum()):,}"
    )

    return result


def read_raster_as_float(path: Path) -> tuple[np.ndarray, dict]:
    """Read first raster band as float32 and convert nodata to NaN."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        data[np.isclose(data, nodata)] = np.nan

    return data, profile


def reproject_to_fuel_grid(
    source_path: Path,
    fuel_profile: dict,
    resampling: Resampling,
) -> np.ndarray:
    """
    Reproject/resample a source raster onto the fuel raster grid.

    The returned raster has the exact dimensions, CRS and transform
    of fars_fuel.tif.
    """
    destination = np.full(
        (fuel_profile["height"], fuel_profile["width"]),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(source_path) as src:
        source = src.read(1).astype(np.float32)

        if src.nodata is not None:
            source[np.isclose(source, src.nodata)] = np.nan

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=fuel_profile["transform"],
            dst_crs=fuel_profile["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )

    return destination


# ---------------------------------------------------------------------
# Fuel component
# ---------------------------------------------------------------------

def build_fuel_lookup(excel_path: Path) -> dict[int, float]:
    """
    Read fuelbed parameters and calculate one FuelRaw score per JOIN_VALUE.

    FuelRaw formula:
        G_Load + W_1hLoad + W_10hLoad + 0.25 * L_depth

    Fine fuels receive the strongest importance because they promote
    ignition and rapid surface-fire spread. Litter depth is included
    with a lower weight.
    """
    required_columns = [
        "JOIN_VALUE",
        "G_Load (Mg/ha)",
        "W_1hLoad (Mg/ha)",
        "W_10h Load (Mg/ha)",
        "L_depth (cm)",
    ]

    fuel_table = pd.read_excel(
        excel_path,
        sheet_name=FUEL_SHEET_NAME,
        engine="openpyxl",
    )

    missing_columns = [
        column for column in required_columns
        if column not in fuel_table.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing from the Fuelbeds_metric sheet:\n"
            + "\n".join(f"  - {column}" for column in missing_columns)
        )

    fuel_table = fuel_table[required_columns].copy()

    numeric_columns = required_columns[1:]
    for column in numeric_columns:
        fuel_table[column] = pd.to_numeric(
            fuel_table[column],
            errors="coerce",
        )

        # -1 and -3 (and any other negative value) mean no usable fuel value.
        fuel_table.loc[fuel_table[column] < 0, column] = 0.0
        fuel_table[column] = fuel_table[column].fillna(0.0)

    fuel_table["JOIN_VALUE"] = pd.to_numeric(
        fuel_table["JOIN_VALUE"],
        errors="coerce",
    )

    fuel_table = fuel_table.dropna(subset=["JOIN_VALUE"]).copy()
    fuel_table["JOIN_VALUE"] = fuel_table["JOIN_VALUE"].astype(np.int64)

    fuel_table["FuelRaw"] = (
        fuel_table["G_Load (Mg/ha)"]
        + fuel_table["W_1hLoad (Mg/ha)"]
        + fuel_table["W_10h Load (Mg/ha)"]
        + 0.25 * fuel_table["L_depth (cm)"]
    )

    if fuel_table["JOIN_VALUE"].duplicated().any():
        duplicates = fuel_table.loc[
            fuel_table["JOIN_VALUE"].duplicated(),
            "JOIN_VALUE",
        ].tolist()

        raise ValueError(
            "Duplicate JOIN_VALUE records were found in Fuelbeds_metric:\n"
            f"  {duplicates}"
        )

    lookup = dict(
        zip(
            fuel_table["JOIN_VALUE"].astype(int),
            fuel_table["FuelRaw"].astype(float),
        )
    )

    print(f"Fuel parameter records loaded: {len(lookup):,}")
    return lookup


def build_fuel_raster(
    fuel_codes: np.ndarray,
    fuel_valid_mask: np.ndarray,
    fuel_lookup: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert integer fuelbed codes in fars_fuel.tif to FuelRaw values.

    Returns:
        fuel_raw: FuelRaw raster, NaN where unavailable.
        mapped_mask: true where a valid fuel pixel was successfully mapped.
    """
    fuel_raw = np.full(fuel_codes.shape, np.nan, dtype=np.float32)

    valid_codes = np.unique(fuel_codes[fuel_valid_mask]).astype(np.int64)
    missing_codes = []

    for fuel_code in valid_codes:
        score = fuel_lookup.get(int(fuel_code))

        if score is None:
            missing_codes.append(int(fuel_code))
            continue

        fuel_raw[fuel_codes == fuel_code] = np.float32(score)

    mapped_mask = np.isfinite(fuel_raw)

    if missing_codes:
        print(
            "Warning: fuel codes found in fars_fuel.tif but missing from "
            "Fuelbeds_metric JOIN_VALUE:"
        )
        print(f"  {sorted(missing_codes)}")

    print(
        f"Fuel pixels mapped successfully: "
        f"{int(mapped_mask.sum()):,} / {int(fuel_valid_mask.sum()):,}"
    )

    if not mapped_mask.any():
        raise ValueError(
            "No fuel codes from fars_fuel.tif were matched to JOIN_VALUE "
            "in the fuelbed parameter table."
        )

    return fuel_raw, mapped_mask


# ---------------------------------------------------------------------
# Topographic component
# ---------------------------------------------------------------------

def calculate_slope_degrees(
    dem: np.ndarray,
    transform,
) -> np.ndarray:
    """
    Calculate terrain slope in degrees from a DEM.

    The DEM is geographic (EPSG:4326), so pixel size in degrees is converted
    approximately to metres using the latitude at each row.
    """
    slope_degrees = np.full(dem.shape, np.nan, dtype=np.float32)

    valid_dem = np.isfinite(dem)
    if valid_dem.sum() == 0:
        raise ValueError("DEM has no valid values after reprojection.")

    # Fill gaps temporarily to avoid invalid gradient calculation.
    filled_dem = dem.copy()
    fill_value = float(np.nanmedian(filled_dem[valid_dem]))
    filled_dem[~valid_dem] = fill_value

    pixel_size_lon_deg = abs(transform.a)
    pixel_size_lat_deg = abs(transform.e)

    rows = np.arange(dem.shape[0], dtype=np.float64)
    latitudes = transform.f + (rows + 0.5) * transform.e

    metres_per_degree_lat = 111_320.0
    metres_per_degree_lon = 111_320.0 * np.cos(np.deg2rad(latitudes))

    metres_per_degree_lon = np.maximum(metres_per_degree_lon, 1.0)

    pixel_size_x_m = metres_per_degree_lon * pixel_size_lon_deg
    pixel_size_y_m = metres_per_degree_lat * pixel_size_lat_deg

    # First calculate gradients in pixel units.
    gradient_y_pixels, gradient_x_pixels = np.gradient(filled_dem)

    # Convert elevation change per pixel to elevation change per metre.
    gradient_x = gradient_x_pixels / pixel_size_x_m[np.newaxis, :]
    gradient_y = gradient_y_pixels / pixel_size_y_m

    slope_radians = np.arctan(
        np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    )

    slope_degrees[valid_dem] = np.degrees(slope_radians[valid_dem])

    return slope_degrees


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    """Build and save the final Fars HRI raster."""
    print("=" * 70)
    print("FARS-HRI: Building final Hazard Risk Index raster")
    print("=" * 70)

    require_file(FWI_PATH, "FWI raster")
    require_file(FUEL_RASTER_PATH, "fuel raster")
    require_file(FUEL_PARAMETERS_PATH, "fuel parameter Excel")
    require_file(DEM_PATH, "DEM raster")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Read fuel raster: this is the master/reference grid.
    # -------------------------------------------------------------
    print("\n[1/6] Reading fuel raster (reference grid)...")

    fuel_codes_float, fuel_profile = read_raster_as_float(FUEL_RASTER_PATH)
    fuel_valid_mask = np.isfinite(fuel_codes_float)
    fuel_codes = np.where(
        fuel_valid_mask,
        fuel_codes_float,
        0,
    ).astype(np.int64)

    print(
        f"Reference grid: {fuel_profile['width']} x {fuel_profile['height']}"
    )
    print(f"Reference CRS: {fuel_profile['crs']}")
    print(f"Valid fuel pixels: {int(fuel_valid_mask.sum()):,}")

    # -------------------------------------------------------------
    # 2. Build and normalize Fuel component.
    # -------------------------------------------------------------
    print("\n[2/6] Building Fuel component from fuelbed parameters...")

    fuel_lookup = build_fuel_lookup(FUEL_PARAMETERS_PATH)
    fuel_raw, fuel_mapped_mask = build_fuel_raster(
        fuel_codes=fuel_codes,
        fuel_valid_mask=fuel_valid_mask,
        fuel_lookup=fuel_lookup,
    )

    fuel_normalized = normalize_valid_values(
        array=fuel_raw,
        valid_mask=fuel_mapped_mask,
        label="Fuel",
    )

    # -------------------------------------------------------------
    # 3. Read/reproject and normalize FWI component.
    # -------------------------------------------------------------
    print("\n[3/6] Reading FWI raster...")

    with rasterio.open(FWI_PATH) as fwi_src:
        same_grid = (
            fwi_src.crs == fuel_profile["crs"]
            and fwi_src.width == fuel_profile["width"]
            and fwi_src.height == fuel_profile["height"]
            and fwi_src.transform == fuel_profile["transform"]
        )

    if same_grid:
        fwi, _ = read_raster_as_float(FWI_PATH)
        print("FWI raster is already aligned with the fuel grid.")
    else:
        print("FWI raster is not aligned; resampling to the fuel grid...")
        fwi = reproject_to_fuel_grid(
            source_path=FWI_PATH,
            fuel_profile=fuel_profile,
            resampling=Resampling.bilinear,
        )

    fwi_valid_mask = np.isfinite(fwi) & (fwi >= 0)
    fwi_normalized = normalize_valid_values(
        array=fwi,
        valid_mask=fwi_valid_mask,
        label="FWI",
    )

    # -------------------------------------------------------------
    # 4. Reproject DEM, calculate slope and normalize Topo component.
    # -------------------------------------------------------------
    print("\n[4/6] Building Topography component from DEM slope...")

    dem_on_fuel_grid = reproject_to_fuel_grid(
        source_path=DEM_PATH,
        fuel_profile=fuel_profile,
        resampling=Resampling.bilinear,
    )

    slope_degrees = calculate_slope_degrees(
        dem=dem_on_fuel_grid,
        transform=fuel_profile["transform"],
    )

    topo_valid_mask = np.isfinite(slope_degrees)
    topo_normalized = normalize_valid_values(
        array=slope_degrees,
        valid_mask=topo_valid_mask,
        label="Slope / Topography",
    )

    # -------------------------------------------------------------
    # 5. Combine components to create HRI.
    # -------------------------------------------------------------
    print("\n[5/6] Calculating HRI...")

    final_valid_mask = (
        fuel_mapped_mask
        & fwi_valid_mask
        & topo_valid_mask
    )

    if not final_valid_mask.any():
        raise ValueError(
            "No overlapping valid pixels were found among FWI, Fuel and DEM."
        )

    hri = np.full(fuel_codes.shape, OUTPUT_NODATA, dtype=np.float32)

    hri[final_valid_mask] = (
        100.0
        * (
            FWI_WEIGHT * fwi_normalized[final_valid_mask]
            + FUEL_WEIGHT * fuel_normalized[final_valid_mask]
            + TOPO_WEIGHT * topo_normalized[final_valid_mask]
        )
    )

    hri_valid_values = hri[final_valid_mask]

    print(f"Final valid HRI pixels: {int(final_valid_mask.sum()):,}")
    print(
        f"HRI range: {float(np.nanmin(hri_valid_values)):.2f} "
        f"to {float(np.nanmax(hri_valid_values)):.2f}"
    )

    # -------------------------------------------------------------
    # 6. Save final GeoTIFF.
    # -------------------------------------------------------------
    print("\n[6/6] Saving final HRI raster...")

    output_profile = fuel_profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(OUTPUT_PATH, "w", **output_profile) as dst:
        dst.write(hri, 1)
        dst.set_band_description(1, "Fars Hazard Risk Index (HRI: 0-100)")

        dst.update_tags(
            TITLE="Fars Hazard Risk Index",
            FORMULA="HRI = 100 * (0.45*FWI + 0.35*Fuel + 0.20*Topo)",
            FWI_WEIGHT=str(FWI_WEIGHT),
            FUEL_WEIGHT=str(FUEL_WEIGHT),
            TOPO_WEIGHT=str(TOPO_WEIGHT),
            FUEL_METHOD=(
                "FuelRaw = G_Load + W_1hLoad + W_10hLoad + "
                "0.25 * L_depth; min-max normalized"
            ),
            TOPO_METHOD=(
                "DEM reprojected to fuel grid; slope in degrees; "
                "min-max normalized"
            ),
            FWI_METHOD="FWI raster min-max normalized",
            SOURCE_FWI=str(FWI_PATH.relative_to(PROJECT_ROOT)),
            SOURCE_FUEL=str(FUEL_RASTER_PATH.relative_to(PROJECT_ROOT)),
            SOURCE_DEM=str(DEM_PATH.relative_to(PROJECT_ROOT)),
        )

    print("\n" + "=" * 70)
    print("HRI raster created successfully.")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
