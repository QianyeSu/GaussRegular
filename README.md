
<div align="center">
  <a href="https://github.com/QianyeSu/GaussRegular">
    <img src="https://raw.githubusercontent.com/QianyeSu/GaussRegular/main/assets/logo.svg" alt="GaussRegular Logo" width="800">
  </a>
</div>

A lightweight, high-performance Python library for converting ECMWF reduced Gaussian grids (N-grids and O-grids) to regular Gaussian grids. Designed for efficient in-memory processing with zero intermediate file I/O.

## Overview

**GaussRegular** specializes in converting ECMWF reduced Gaussian grids to full (regular) Gaussian grids:

- **N-grids**: Classical reduced Gaussian grids (e.g., N320, N640)
- **O-grids**: Octahedral reduced Gaussian grids (e.g., O320, O640, O1280)

Both grid types carry a `pl` array (points per latitude row) and use identical row-wise interpolation logic, matching CDO's `setgridtype,regular` and `setgridtype,regularnn` behavior.

## Features

- **Dual grid support**: Handles both classical N-grids (e.g., N320, N640) and octahedral O-grids (e.g., O320, O640, O1280)
- **Two interpolation modes**: 
  - **Bilinear (default)** — matching CDO `setgridtype,regular`
  - **Nearest-neighbor** — matching CDO `setgridtype,regularnn`
- **C-accelerated**: Core interpolation in compiled C, ~10-100x faster than pure NumPy
- **In-memory processing**: Read GRIB2 data → convert → analyze → output, no intermediate files
- **Double-precision internals**: Matching CDO's numerical accuracy
- **xarray/cfgrib integration**: Direct support for GRIB2 data
- **Batch processing**: Efficient multi-dimensional array handling (time, level, etc.)

## Installation

```bash
pip install gaussregular
```

### With xarray support:
```bash
pip install "gaussregular[xarray]"
```

## Quick Start

### Case 1: Direct Numpy Array Conversion

If you have a NumPy array with known grid metadata:

```python
import numpy as np
import gaussregular as gr

# Your reduced Gaussian data (1D array, one value per grid point)
values = np.random.rand(1040640)  # Example data

# Gaussian row length array (from GRIB metadata, e.g., GRIB_pl)
pl = np.array([20, 27, 36, 40, 45, ..., 360, 360], dtype=np.int32)  # 161 latitude rows in this example

# Create regularizer
regularizer = gr.GaussRegularizer(method="linear")

# Convert to regular Gaussian (2D array)
regular_data, lon = regularizer.regularize_values(
    values=values,
    pl=pl,
    missval=np.nan
)

print(f"Output shape: {regular_data.shape}")  # e.g. (161, 320) in this example
print(f"Longitude points: {lon.size}")
```

### Case 2: xarray DataArray (from cfgrib)

If you're reading GRIB2 files with `cfgrib`:

```python
import xarray as xr
import gaussregular as gr

# Open GRIB2 file with cfgrib backend
ds = xr.open_dataset("era5_sample.grib", engine="cfgrib")
da = ds["temperature"]  # xarray.DataArray

# Create regularizer
regularizer = gr.GaussRegularizer(method="linear", cache=True)

# Convert directly
regular_da = regularizer.regularize_xarray(da)

print(f"Regular grid shape: {regular_da.shape}")
print(f"Coordinates preserved: {list(regular_da.coords.keys())}")
```

The library automatically extracts `GRIB_pl`, grid type, and other metadata from the DataArray attributes.

### Case 3: Multi-dimensional Data (Time series, Levels)

Batch processing with leading dimensions:

```python
import xarray as xr
import gaussregular as gr

# Multi-dimensional xarray: (time, level, values)
ds = xr.open_dataset("era5_multi_level.grib", engine="cfgrib")
da = ds["temperature"]  # Shape: (24, 137, 1040640) in this example

regularizer = gr.GaussRegularizer(method="linear", cache=True)
regular_da = regularizer.regularize_xarray(da)

print(f"Output shape: {regular_da.shape}")  # e.g. (24, 137, 161, 320) in this example
```

The regularizer automatically handles reshaping and maintains coordinate dimensions.

### Case 4: xarray Dataset (Multiple Variables)

You can pass a full `xr.Dataset` directly. All convertible reduced-Gaussian variables are regularized.

```python
import xarray as xr
import gaussregular as gr

ds = xr.open_dataset("era5_multi_vars.grib", engine="cfgrib")

regularizer = gr.GaussRegularizer(method="linear", cache=True)
regular_ds = regularizer.regularize_dataset(ds)

# Or use auto dispatch:
# regular_ds = regularizer.regularize(ds)

print(regular_ds)
```

### Case 5: Regional Sub-area Data

For data covering only a portion of the globe (e.g., a regional cut from a model run), specify the longitude bounds and grid number:

```python
import numpy as np
import gaussregular as gr

# Regional reduced Gaussian data
values = np.random.rand(342080)  # Subset of N320 (e.g., Europe region)
pl = np.array([...])  # Row lengths for the region

regularizer = gr.GaussRegularizer(method="linear")

# Specify regional bounds and grid number for proper output sizing
regular_data, lon = regularizer.regularize_values(
    values=values,
    pl=pl,
    missval=np.nan,
    grid_number=320,  # N320 grid
    xfirst=0.0,       # Start longitude (degrees east)
    xlast=40.0,       # End longitude (degrees east)
)

print(f"Regional grid shape: {regular_data.shape}")
print(f"Longitude range: {lon[0]:.2f}–{lon[-1]:.2f}°E")
```

**Parameters for regional data:**
- **`grid_number`** (int): Gaussian truncation number (e.g., 320 for N320). Required to distinguish global vs. regional grids.
- **`xfirst`** (float, default=0.0): Longitude of the first data point (degrees east, range 0–360).
- **`xlast`** (float, default=359.9999): Longitude of the last data point (degrees east).

The library automatically infers the output longitude count based on the regional bounds.

### Advanced: Fast Mode (Skip Missing-Value Detection)

If your input is guaranteed to have **no missing values**, enable `fast=True` to skip per-row missing-value detection for ~1.8x speedup:

```python
import gaussregular as gr

engine = gr.GaussRegularizer(method="linear", cache=True)

# Standard mode with missing-value checking
regular_ds_safe = engine.regularize_dataset(ds, fast=False)  # Default

# Fast mode: assumes no missing values (dangerous if false!)
regular_ds_fast = engine.regularize_dataset(ds, fast=True)  # ~1.8x faster
```

**Warning:** Use `fast=True` **only** when you are certain the input contains no missing values (NaN, sentinel values, etc.). Incorrect use may produce garbage output.

## API Reference

### `GaussRegularizer`

Main converter class.

#### Parameters:
- **`method`** (str, default="linear"): Interpolation method, either `"linear"` or `"nearest"`
- **`grid_number`** (int, optional): Default Gaussian truncation number N (e.g., 320 for N320/O320). Used when `GRIB_N` is missing for xarray inputs and as a fallback when `regularize_values` is called without a per-call `grid_number`. When neither metadata nor this attribute provide N, a heuristic based on the equatorial row length is used
- **`cache`** (bool, default=False): Enable xarray metadata plan caching for repeated calls on same-structure data
- **`max_plan_cache`** (int, default=32): Maximum number of cached conversion plans

#### Methods:

**`regularize(data, nlon=None, grid_type_hint=None, method=None)`**
- Auto-detects input type (`xr.Dataset`, `xr.DataArray`, or NumPy) and calls appropriate method
- Returns regularized data in same type as input

**`regularize_values(values, pl, missval, nlon=None, method=None, fast=False)`**
- **`values`**: 1D NumPy array (`len(values) == sum(pl)`)
- **`pl`**: 1D NumPy array of row lengths (from GRIB_pl)
- **`missval`**: Missing-value sentinel (float). Points equal to this value (or NaN, if `missval` is NaN) are treated as missing and handled by the interpolation kernel
- **`nlon`** (optional): Number of columns in output regular grid. Auto-inferred if not provided
- **`method`** (optional): Per-call method override: `"linear"` or `"nearest"`
- **`fast`** (optional): Skip missing-value detection when input has no missing values
- Returns: `(out, lon)` where `out` is NumPy array and `lon` is longitude vector

**`regularize_xarray(dataarray, nlon=None, grid_type_hint=None, method=None, fast=False)`**
- **`dataarray`**: xarray.DataArray with GRIB metadata in `.attrs`
- Requires: `GRIB_pl` attribute. If `GRIB_missingValue` is present it is used as `missval`; otherwise NaN values are treated as missing
- **`method`** (optional): Per-call method override: `"linear"` or `"nearest"`
- **`fast`** (optional): Skip missing-value detection when input has no missing values
- Returns: xarray.DataArray with regularized data and preserved coordinates

**`regularize_dataset(dataset, nlon=None, grid_type_hint=None, method=None, fast=False)`**
- **`dataset`**: xarray.Dataset with one or more reduced Gaussian variables
- Converts each convertible variable and keeps non-convertible variables unchanged
- **`method`** (optional): Per-call method override: `"linear"` or `"nearest"`
- **`fast`** (optional): Skip missing-value detection when input has no missing values
- Returns: xarray.Dataset

**`clear_cache()`**
- Clears internal xarray plan cache

### Module-level Functions

**`regularize_values(values, pl, missval, nlon=None, method="linear", fast=False)`**
- Standalone function using default regularizer configuration

**`regularize_xarray(dataarray, grid_type_hint=None, method="linear", fast=False)`**
- Standalone function using default regularizer configuration

**`regularize_dataset(dataset, grid_type_hint=None, method="linear", fast=False)`**
- Standalone function using default regularizer configuration

## Grid Specifics

### N-grids (Classical Gaussian)
- **Regular spacing in longitude**
- Example: N320 → 4×N = 1280 columns at the equator in the regular grid
- Total reduced-grid points: 4×N×(2N+1)

### O-grids (Octahedral)
- **Refinement at equator** for better accuracy
- Example: O320, O640, O1280
- Formula: nlon = 4×N + 16 for O-grid

The library automatically detects grid type from GRIB metadata. You can also override with `grid_type_hint` parameter.

## Supported Grid Types

GaussRegular recognizes many GRIB metadata variants:
- `reduced_gg`
- `reduced-gg`
- `reduced gaussian`
- `octahedral_gg`
- `octahedral_reduced_gg`
- And several aliases

See `GRID_TYPE_FULL_NAMES` in the API for the full list.

## Performance Tuning

### Interpolation Method
- **`method="linear"` (default)**: Bilinear interpolation. Slightly slower but more accurate.
- **`method="nearest"`**: Nearest-neighbor. Faster, less smooth.

### Missing-Value Detection
- **`fast=False` (default)**: Safe mode. Detects missing values in each row and handles them gracefully (~11.8 sec for N320 ERA5 example).
- **`fast=True`**: Speed mode. Skips per-row missing-value detection and assumes no missing values exist. **Only use when input has no missing values.** Typical speedup: ~1.8x (~6.3 sec for same example).

### Caching
- Enable `cache=True` when processing multiple DataArrays with identical structure to reuse parsed metadata.

Typical throughput:
- N320 data (bilinear): ~100M points/second (safe mode), ~180M points/sec (fast mode)
- N640 data (bilinear): ~80M points/second (safe mode), ~150M points/sec (fast mode)

## Limitations

1. **Missing values must be consistently marked** — via NaN or a numeric `missval`/`GRIB_missingValue`; when `fast=True` you must guarantee there are no missing values
2. **Requires GRIB metadata** — `pl` array must be provided or in DataArray attrs
3. **Supports only Gaussian grids** — other map projections not supported
4. **Precision**: Outputs double-precision computation; results cast back to input dtype
5. **No projection support** — only Gaussian grid conversions


## Testing & Validation

GaussRegular has been validated against CDO `setgridtype,regular` and `setgridtype,regularnn` with double-precision agreement to machine epsilon.

Test with real ERA5 data:
```bash
python examples/run_era5_demo.py
```

## Requirements

- **Python**: 3.9+
- **NumPy**: ≥1.23
- **Optional**: xarray ≥2023.1, cfgrib ≥0.9.10



## Citation

If you use GaussRegular in research, please cite:
```
@software{gaussregular,
  author = {Qianye Su},
  title = {GaussRegular: ECMWF Reduced Gaussian to Regular Grid Converter},
  year = {2026},
  url = {https://github.com/QianyeSu/GaussRegular}
}
```

## Troubleshooting

### "Missing GRIB_pl in DataArray attrs"
- Ensure you opened the file with `engine="cfgrib"`
- Check that cfgrib correctly parsed the GRIB2 file

### "Input does not look like reduced Gaussian grid"
- Verify your file contains a reduced Gaussian grid (not full Gaussian, lat-lon, or other projection)
- Pass `grid_type_hint` if metadata is nonstandard

### "Results look wrong near missing values"
- Ensure you are not using `fast=True` when missing values are present
- Verify that missing values are either NaN (and `missval` is left as default) or match the `GRIB_missingValue`/`missval` you pass
- If necessary, preprocess your data（例如插值或填充）以减少大块缺测区域对结果的影响

### Performance is slower than expected
- Use `method="linear"` unless you specifically want nearest-neighbour behavior
- Use caching: `regularizer.cache=True` for repeated calls
- Profile with larger batch operations instead of single conversions

## Contributing

Contributions welcome! Please:
1. Read the source code and existing patterns
2. Write tests for new features
3. Ensure CI passes before submitting PR

