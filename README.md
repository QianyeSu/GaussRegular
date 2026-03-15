# GaussRegular

A lightweight, high-performance Python library for converting ECMWF reduced Gaussian grids (N-grids and O-grids) to regular Gaussian grids. Designed for efficient in-memory processing with zero intermediate file I/O.

## 🔴 CRITICAL: Input Data Must Have NO Missing Values

**This library requires that all input grid points contain valid data.** Missing values (NaN, masked values, etc.) are **strictly not allowed**. If your GRIB2 data contains missing values:

1. **Pre-process** to interpolate or fill missing data, OR
2. **Subset** data to regions with complete coverage, OR
3. **Use CDO** if you need missing-value-aware regridding

**Why?** Missing-value-aware interpolation requires case-by-case boundary handling that significantly complicates the algorithm. By enforcing complete data, GaussRegular achieves maximum performance while maintaining numerical accuracy.

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
values = np.random.rand(1040640)  # Example: N320 reduced grid

# Gaussian row length array (from GRIB metadata, e.g., GRIB_pl)
pl = np.array([20, 27, 36, 40, 45, ..., 360, 360], dtype=np.int32)  # 161 entries for N320

# Create regularizer
regularizer = gr.GaussRegularizer(nearest=False)

# Convert to regular Gaussian (2D array)
regular_data = regularizer.regularize_values(
    values=values,
    pl=pl,
    missval=np.nan
)

print(f"Output shape: {regular_data.shape}")  # (161, 320) for N320
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
regularizer = gr.GaussRegularizer(nearest=False, cache=True)

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

# Multi-dimensional xarray: (time, level, latitude_reduced, longitude_reduced)
ds = xr.open_dataset("era5_multi_level.grib", engine="cfgrib")
da = ds["temperature"]  # Shape: (24, 137, 1040640)

regularizer = gr.GaussRegularizer(nearest=False, cache=True)
regular_da = regularizer.regularize_xarray(da)

print(f"Output shape: {regular_da.shape}")  # (24, 137, 161, 320) for N320
```

The regularizer automatically handles reshaping and maintains coordinate dimensions.

## API Reference

### `GaussRegularizer`

Main converter class.

#### Parameters:
- **`nearest`** (bool, default=False): Use nearest-neighbour instead of bilinear interpolation
- **`default_np_value`** (int, optional): Default Gaussian truncation number N (e.g., 320). Used only if grid metadata doesn't specify it
- **`cache`** (bool, default=False): Enable xarray metadata plan caching for repeated calls on same-structure data
- **`max_plan_cache`** (int, default=32): Maximum number of cached conversion plans

#### Methods:

**`regularize(data, nlon=None, grid_type_hint=None)`**
- Auto-detects input type (xarray or NumPy) and calls appropriate method
- Returns regularized data in same type as input

**`regularize_values(values, pl, missval, nlon=None, nearest=None)`**
- **`values`**: 1D or N-D NumPy array (flattened last dimension is grid points)
- **`pl`**: 1D NumPy array of row lengths (from GRIB_pl)
- **`missval`**: Missing value sentinel (float). **Input must NOT contain this value**
- **`nlon`** (optional): Number of columns in output regular grid. Auto-inferred if not provided
- **`nearest`** (optional): Override instance setting for this call only
- Returns: NumPy array with same dtype as input

**`regularize_xarray(dataarray, nlon=None, grid_type_hint=None, nearest=None)`**
- **`dataarray`**: xarray.DataArray with GRIB metadata in `.attrs`
- Requires: `GRIB_pl` and `GRIB_missingValue` attributes
- Returns: xarray.DataArray with regularized data and preserved coordinates

**`clear_cache()`**
- Clears internal xarray plan cache

### Module-level Functions

**`regularize_values(values, pl, missval, nlon=None, nearest=False)`**
- Standalone function using default regularizer configuration

**`regularize_xarray(dataarray, grid_type_hint=None, nearest=False)`**
- Standalone function using default regularizer configuration

## Grid Specifics

### N-grids (Classical Gaussian)
- **Regular spacing in longitude**
- Example: N320 → 320 columns in regular grid
- Total points: 4×N×(2N+1) for N-grid

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

## Performance Notes

- **C extension backend**: Core interpolation is compiled from proven CDO algorithms for maximum speed
- **Bilinear vs. Nearest**: Bilinear is slightly slower but more accurate; nearest-neighbour is faster
- **Batching**: Processing multi-dimensional data (time, level) is more efficient than repeated 1D conversions
- **Caching**: Enable `cache=True` for repeated calls on identically-structured data

Typical throughput:
- N320 data: ~100M points/second (bilinear)
- N640 data: ~80M points/second (bilinear)

## Limitations

1. **Input must be missing-value-free** — no NaN or masked values allowed
2. **Requires GRIB metadata** — `pl` array must be provided or in DataArray attrs
3. **Supports only Gaussian grids** — other map projections not supported
4. **Precision**: Outputs double-precision computation; results cast back to input dtype
5. **No projection support** — only Gaussian grid conversions

## Examples

See the `examples/` directory (not in wheel) for:
- `run_era5_demo.py` — Real ERA5 data conversion
- `test_real_numpy_xarray.py` — Detailed input validation examples
- `benchmark_speed.py` — Performance benchmarking

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

## License

[Specify your license, e.g., MIT, GPL, etc.]

## Author

Qianye Su

## Citation

If you use GaussRegular in research, please cite:
```
@software{gaussregular,
  author = {Qianye Su},
  title = {GaussRegular: ECMWF Reduced Gaussian to Regular Grid Converter},
  year = {2024},
  url = {https://github.com/yourusername/GaussRegular}
}
```

## Troubleshooting

### "Missing GRIB_pl in DataArray attrs"
- Ensure you opened the file with `engine="cfgrib"`
- Check that cfgrib correctly parsed the GRIB2 file

### "Input does not look like reduced Gaussian grid"
- Verify your file contains a reduced Gaussian grid (not full Gaussian, lat-lon, or other projection)
- Pass `grid_type_hint` if metadata is nonstandard

### "Input contains missing values"
- Preprocess your data: `da = da.interpolate_na(dim='...')`  or mask/remove missing values
- Check data before calling GaussRegular

### Performance is slower than expected
- Ensure you're not calling with `nearest=True` unless necessary
- Use caching: `regularizer.cache=True` for repeated calls
- Profile with larger batch operations instead of single conversions

## Contributing

Contributions welcome! Please:
1. Read the source code and existing patterns
2. Write tests for new features
3. Ensure CI passes before submitting PR

## Related Projects

- **CDO** (Climate Data Operators) — Original algorithms
- **cfgrib** — GRIB2 file reading
- **xarray** — Array metadata and labels
