from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
from . import _core

if TYPE_CHECKING:  # pragma: no cover - used only for static typing
    import xarray as xr


# Supported reduced-Gaussian grid type names and their meaning.
# If metadata does not expose a known name, callers may pass grid_type_hint.
GRID_TYPE_FULL_NAMES = {
    "reduced_gg": "Reduced Gaussian Grid (generic, most GRIB/cfgrib outputs)",
    "reduced-gg": "Reduced Gaussian Grid (dash variant)",
    "reduced gaussian": "Reduced Gaussian Grid (spaced variant)",
    "reduced_gaussian": "Reduced Gaussian Grid (underscore variant)",
    "gaussian_reduced": "Reduced Gaussian Grid (reversed-token variant)",
    "reduced_gg_ml": "Reduced Gaussian Grid on model levels",
    "reduced_gg_pl": "Reduced Gaussian Grid on pressure levels",
    "reduced_gg_sfc": "Reduced Gaussian Grid on surface levels",
    "octahedral_gg": "Octahedral Reduced Gaussian Grid",
    "reduced_gg_octahedral": "Octahedral Reduced Gaussian Grid (alt name)",
    "o_gaussian": "Octahedral Gaussian Grid (short alias)",
    "octahedral_reduced_gg": "Octahedral Reduced Gaussian Grid (long alias)",
    "octahedral_gg_ml": "Octahedral Reduced Gaussian Grid on model levels",
    "octahedral_gg_pl": "Octahedral Reduced Gaussian Grid on pressure levels",
    "octahedral_gg_sfc": "Octahedral Reduced Gaussian Grid on surface levels",
}

REDUCED_GAUSSIAN_ALIASES = set(GRID_TYPE_FULL_NAMES)
VALID_METHODS = {"linear", "nearest"}
PrecisionMode = Literal["auto", "single", "double"]
PRECISION_ALIASES = {
    "auto": "auto",
    "single": "single",
    "float32": "single",
    "double": "double",
    "float64": "double",
}


def _supported_grid_types_message() -> str:
    lines = [
        f"- {name}: {desc}" for name, desc in sorted(GRID_TYPE_FULL_NAMES.items())
    ]
    return "Supported reduced Gaussian grid type names:\n" + "\n".join(lines)


def _to_int32_pl(pl: Iterable[int]) -> np.ndarray:
    arr = np.asarray(pl, dtype=np.int32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("pl must be a 1D non-empty array")
    if np.any(arr <= 0):
        raise ValueError("pl must contain positive integers")
    return arr


def _normalize_method(method: Optional[str], default: str = "linear") -> str:
    chosen = default if method is None else str(method).strip().lower()
    if chosen not in VALID_METHODS:
        supported = ", ".join(sorted(VALID_METHODS))
        raise ValueError(f"method must be one of: {supported}")
    return chosen


def _normalize_precision(
    precision: Optional[str],
    default: str = "auto",
) -> str:
    chosen = default if precision is None else str(precision).strip().lower()
    normalized = PRECISION_ALIASES.get(chosen)
    if normalized is None:
        supported = ", ".join(sorted(PRECISION_ALIASES))
        raise ValueError(f"precision must be one of: {supported}")
    return normalized


def _precision_dtypes(dtype: np.dtype, precision: str) -> tuple[np.dtype, np.dtype]:
    """Return (compute dtype, public output dtype)."""
    dtype = np.dtype(dtype)
    if precision == "single":
        return np.dtype(np.float32), np.dtype(np.float32)
    if precision == "double":
        return np.dtype(np.float64), np.dtype(np.float64)

    if dtype == np.dtype(np.float32):
        return np.dtype(np.float32), np.dtype(np.float32)
    if dtype == np.dtype(np.float64):
        return np.dtype(np.float64), np.dtype(np.float64)
    return np.dtype(np.float64), dtype


def _is_supported_reduced_grid(attrs: Mapping[str, Any]) -> bool:
    grid_type = str(attrs.get("GRIB_gridType", attrs.get(
        "gridType", ""))).strip().lower()
    if grid_type in REDUCED_GAUSSIAN_ALIASES:
        return True
    if "GRIB_pl" in attrs:
        return True
    desc = str(attrs.get("GRIB_gridDefinitionDescription", "")).lower()
    if "reduced" in desc and "gaussian" in desc:
        return True
    return False


def _extract_lat_1d(values: Any, target_size: int) -> Optional[np.ndarray]:
    """Normalise latitude coordinate to a 1D array of length target_size.

    Accepts either a 1D latitude vector or a 2D/ND mesh that can be
    reduced to unique latitude values.
    """
    latv = np.asarray(values)
    if latv.ndim == 1 and latv.size == target_size:
        return latv.astype(np.float64)
    if latv.ndim >= 1 and latv.size >= target_size:
        arr = np.unique(latv.reshape(-1)).astype(np.float64)
        if arr.size == target_size:
            return np.sort(arr)[::-1]
    return None


def _require_xarray():
    """Import xarray on demand for xarray-based APIs.

    This keeps ``gaussregular`` importable without xarray installed while still
    providing a clear error message when xarray-specific functions are used.
    """
    try:
        import xarray as xr  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - import error path
        raise ImportError(
            "xarray is required for xarray-based APIs; "
            "install it via 'pip install \"gaussregular[xarray]\"'."
        ) from exc
    return xr


@dataclass
class GaussRegularizer:
    """Convert ECMWF reduced Gaussian grids to regular (full) Gaussian grids.

    Supports both classical **N-grids** (e.g. N320, N640) and octahedral
    **O-grids** (e.g. O320, O640, O1280).  Both grid families carry a ``pl``
    array and use the identical row-wise interpolation path.

    Precision defaults to ``"auto"``: float32 input uses the single-precision
    C path and returns float32, while float64 input uses the double-precision
    C path and returns float64. Set ``precision="single"`` or
    ``precision="double"`` to force a public output precision.

    Optimisation mode
    -----------------
    * **Precise (default):** compiled with ``/fp:precise`` (MSVC) or
        ``-fno-math-errno`` (GCC/Clang).  NaN / missing-value detection is
        IEEE-754 compliant.  This is the recommended mode.
    * **Fast-math:** set ``GAUSSREGULAR_FAST_MATH=1`` *before building* the
        extension (``pip install``), then rebuild.  Adds ``/fp:fast`` or
        ``-ffast-math``; gives ~15 % throughput gain but relaxes IEEE 754
        semantics (do not rely on exact NaN propagation).

    Parameters
    ----------
    method:
        Interpolation method. ``linear`` (default) matches CDO
        ``setgridtype,regular``; ``nearest`` matches CDO
        ``setgridtype,regularnn``.
    precision:
        Default precision policy: ``"auto"``, ``"single"``, or ``"double"``.
        ``"auto"`` preserves float32/float64 input precision.
    grid_number:
        Optional default Gaussian truncation number N (e.g. 320 for
        N320/O320).  Used when ``GRIB_N`` is missing for xarray inputs and
        when :meth:`regularize_values` is called without a per-call
        ``grid_number``.  When both metadata and this attribute are
        absent, a heuristic based on the equatorial row length is used.
    cache:
        Enable xarray metadata plan cache.  When ``True``, repeated
        calls on same-structure DataArray reuse parsed ``GRIB_pl``, inferred
        ``nlon``, coordinate templates and grid flags.
    max_plan_cache:
        Maximum number of cached plans.  Oldest entries are evicted first.

    Typical usage
    -------------
    ``engine.regularize(data)`` auto-detects input type:

    - xarray Dataset input -> ``regularize_dataset`` path
    - xarray DataArray input -> ``regularize_xarray`` path
    - numpy input -> ``regularize_values`` path (requires ``pl`` and ``missval``)

    Example::

        import gaussregular as gr
        engine = gr.GaussRegularizer(cache=False)
        out = engine.regularize(da_day1)
    """

    method: str = "linear"
    precision: PrecisionMode = "auto"
    grid_number: Optional[int] = None
    cache: bool = False
    max_plan_cache: int = 100000
    _xarray_plan_cache: dict[tuple, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.method = _normalize_method(self.method)
        self.precision = _normalize_precision(self.precision)  # type: ignore[assignment]

    def clear_cache(self) -> None:
        """Clear internal xarray plan cache."""
        self._xarray_plan_cache.clear()

    def _build_xarray_plan(
        self,
        dataarray: "xr.DataArray",
        nlon: Optional[int],
        grid_type_hint: Optional[str],
    ) -> dict[str, Any]:
        attrs = dict(dataarray.attrs)
        hint = None if grid_type_hint is None else str(
            grid_type_hint).strip().lower()
        if hint and hint not in REDUCED_GAUSSIAN_ALIASES:
            raise ValueError(
                f"Unsupported grid_type_hint: {grid_type_hint!r}.\n"
                + _supported_grid_types_message()
            )

        if not _is_supported_reduced_grid(attrs) and hint is None:
            raise ValueError(
                "Input does not look like reduced Gaussian grid. "
                "Expected reduced Gaussian aliases or GRIB_pl metadata.\n"
                + _supported_grid_types_message()
                + "\nIf your metadata is incomplete, pass grid_type_hint='<known_name>'."
            )

        if "GRIB_pl" not in attrs:
            raise ValueError("Missing GRIB_pl in DataArray attrs")

        pl = _to_int32_pl(attrs["GRIB_pl"])
        missval = float(attrs.get("GRIB_missingValue", np.nan))

        # Determine truncation N (np_value) in order of preference:
        # 1) GRIB_N from metadata when present and non-zero;
        # 2) the regulariser's grid_number attribute when set;
        # 3) 0, which triggers the equatorial-row heuristic in infer_nlon.
        np_value = int(attrs.get("GRIB_N", 0) or self.grid_number or 0)

        nlon_val = int(_core.infer_nlon(pl, np_value, 0.0,
                       359.9999)) if nlon is None else int(nlon)

        expected = int(pl.sum())
        lead_dims = tuple(dataarray.dims[:-1])

        lat_1d = None
        lat_attrs: dict = {}
        target_size = int(pl.size)

        # 1) Name-based quick check for common latitude coordinate names.
        for cand in ("latitude", "lat"):
            if cand in dataarray.coords:
                lat_1d = _extract_lat_1d(
                    dataarray.coords[cand].values, target_size)
                if lat_1d is not None:
                    lat_attrs = dict(
                        getattr(dataarray.coords[cand], "attrs", {}))
                    break

        # 2) Fallback: search coords with CF-style metadata (standard_name/units)
        # when latitude coordinate uses a non-standard name.
        if lat_1d is None:
            for coord in dataarray.coords.values():
                attrs_c = getattr(coord, "attrs", {})
                standard_name = str(attrs_c.get("standard_name", "")).lower()
                units = str(attrs_c.get("units", "")).lower()
                if standard_name == "latitude" or "degrees_north" in units:
                    lat_1d = _extract_lat_1d(coord.values, target_size)
                    if lat_1d is not None:
                        lat_attrs = dict(attrs_c)
                        break

        if lat_1d is None or lat_1d.size != int(pl.size):
            lat_1d = np.arange(int(pl.size), dtype=np.float64)

        # Collect longitude attrs from input if present.
        lon_attrs: dict = {}
        for cand in ("longitude", "lon"):
            if cand in dataarray.coords:
                lon_attrs = dict(getattr(dataarray.coords[cand], "attrs", {}))
                break
        if not lon_attrs:
            for coord in dataarray.coords.values():
                attrs_c = getattr(coord, "attrs", {})
                standard_name = str(attrs_c.get("standard_name", "")).lower()
                units = str(attrs_c.get("units", "")).lower()
                if standard_name == "longitude" or "degrees_east" in units:
                    lon_attrs = dict(attrs_c)
                    break

        lon_1d = np.linspace(0.0, 360.0, nlon_val,
                             endpoint=False, dtype=np.float64)
        is_global = bool(_core.is_global(np_value, pl, 0.0,
                         359.9999)) if np_value > 0 else True

        return {
            "attrs": attrs,
            "pl": pl,
            "missval": missval,
            "np_value": np_value,
            "nlon": nlon_val,
            "expected": expected,
            "lead_dims": lead_dims,
            "lat_1d": lat_1d,
            "lon_1d": lon_1d,
            "lat_attrs": lat_attrs,
            "lon_attrs": lon_attrs,
            "is_global": is_global,
        }

    def regularize_values(
        self,
        values: np.ndarray,
        pl: Iterable[int],
        missval: float,
        nlon: Optional[int] = None,
        grid_number: Optional[int] = None,
        xfirst: float = 0.0,
        xlast: float = 359.9999,
        method: Optional[str] = None,
        fast: bool = True,
        precision: Optional[PrecisionMode] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate a flat reduced Gaussian array to a regular longitude grid.

        Parameters
        ----------
        values:
            1-D array of reduced-grid values packed row-by-row from north to
            south.  Length must equal ``sum(pl)``.  Any numeric dtype is
            accepted (float32, float64, etc.); float32 and float64 arrays keep
            their dtype through the C path, while other numeric dtypes are
            converted through float64 and cast back to the input dtype.
        pl:
            **Points per latitude row** — the ``GRIB_pl`` attribute.  1-D
            integer array with one entry per latitude circle (north → south).

            * **N320 (classical):** 640 rows; values range from ~18 near the
              pole to **1280** at the Equator; ``sum(pl) = 542 080``.
            * **O320 (octahedral):** 640 rows; Equatorial row = **1296**
              (formula: ``20 + 4*N`` per side); ``sum(pl) = 542 080 + extra``.
            * **O1280:** 2560 rows; ``sum(pl) ≈ 6 599 680``.

            For both grid types, ``pl[nlat // 2]`` gives the Equatorial (=
            maximum) row count used as the default *nlon*.
        missval:
            Sentinel value that marks missing data.  Common choices:
            ``9.999e20`` (GRIB default), ``-9999.0``, ``float('nan')``.
            Points equal to *missval* are replaced by the nearest valid
            neighbour rather than being interpolated.
        nlon:
            Number of output longitude columns.  When *None* (default) *nlon*
            is inferred via ``pl[nlat // 2]`` (global) or
            ``grib_get_reduced_row`` (regional sub-area).
        grid_number:
            Gaussian truncation number N (e.g. 320 for N320 or O320).
            Needed only to distinguish global vs. regional grids and to infer
            *nlon* for sub-areas when metadata does not provide it.  If
            omitted, this falls back to the regulariser's ``grid_number``
            attribute when set, otherwise to a heuristic based on equatorial
            row length (``np_in = 0``).
        xfirst:
            Longitude of the first grid point in degrees.  Default ``0.0``.
        xlast:
            Longitude of the last grid point in degrees.  Default ``359.9999``.
        method:
            Per-call interpolation method override (``linear`` or ``nearest``).
        fast:
            If ``True``, skip row-wise missing-value detection for speed.
            Use only when input is guaranteed to contain no missing values.
        precision:
            ``"auto"`` preserves float32/float64 input precision.
            ``"single"`` computes and returns float32; ``"double"`` computes
            and returns float64. Defaults to the regularizer's precision.

        Notes
        -----
        ``grid_number`` is **not** a regridding target selector.
        It does not transform O640 -> F320 by itself.  This converter keeps the
        input latitude count; it regularizes longitudes on that latitude set.
        Producing a true F320 target grid requires a separate remapping step.

        Returns
        -------
        out : np.ndarray
            2-D array of shape ``(nlat, nlon)`` with the regularised field.
            With ``precision="auto"``, float32 input returns float32 and
            float64 input returns float64.
        lon : np.ndarray
            1-D longitude coordinate array of length *nlon*, degrees east.
        """
        pl_arr = _to_int32_pl(pl)
        method_name = _normalize_method(method, default=self.method)
        nearest_flag = method_name == "nearest"

        # Determine truncation N (np_in) for the numeric path.  Per-call
        # grid_number wins; then the regulariser's grid_number attribute;
        # finally 0 (equatorial-row heuristic).
        np_in = int(grid_number or self.grid_number or 0)

        nlon = int(_core.infer_nlon(pl_arr, np_in, float(xfirst),
                   float(xlast))) if nlon is None else int(nlon)
        if nlon <= 0:
            raise ValueError("nlon must be positive")

        vals_in = np.asarray(values)
        precision_name = _normalize_precision(precision, default=self.precision)
        compute_dtype, public_dtype = _precision_dtypes(
            vals_in.dtype, precision_name)
        vals = vals_in.astype(compute_dtype, copy=False)
        if vals.ndim != 1:
            raise ValueError(
                "values must be a 1D array with length == sum(pl)")

        expected = int(pl_arr.sum())
        if vals.size != expected:
            raise ValueError(
                f"values length {vals.size} does not match sum(pl) {expected}")

        out = _core.regularize_values(
            vals, pl_arr, float(missval), nearest_flag, int(nlon), bool(fast))
        if out.dtype != public_dtype:
            out = out.astype(public_dtype, copy=False)

        is_glob = bool(_core.is_global(np_in, pl_arr, float(
            xfirst), float(xlast))) if np_in > 0 else True

        if np_in > 0 and not is_glob:
            np4 = np_in * 4
            _, ilon_first, _ = _core.reduced_row(
                np4, float(xfirst), float(xlast))
            lon = ((ilon_first + np.arange(int(nlon), dtype=np.float64))
                   * 360.0) / np4
            if xfirst > xlast:
                lon = lon - 360.0
        else:
            lon = np.linspace(0.0, 360.0, int(
                nlon), endpoint=False, dtype=np.float64)

        return out, lon

    def regularize_xarray(
        self,
        dataarray: "xr.DataArray",
        nlon: Optional[int] = None,
        method: Optional[str] = None,
        grid_type_hint: Optional[str] = None,
        fast: bool = True,
        precision: Optional[PrecisionMode] = None,
    ) -> "xr.DataArray":
        """Interpolate an xarray DataArray from reduced to regular Gaussian grid.

        The DataArray must carry ``GRIB_pl`` in its ``.attrs`` (present by
        default when opened via cfgrib / eccodes).  Leading dimensions (time,
        level, ensemble member, …) are handled automatically; only the last
        dimension (the reduced-grid "values" axis) is regularised.

        Parameters
        ----------
        dataarray:
            ``xarray.DataArray`` whose **last dimension** spans the reduced-
            grid points.  Accepts any number of leading dimensions.
        nlon:
            Override the inferred output longitude count.  When *None* it is
            derived from ``pl[nlat // 2]`` (global grids) or
            ``grib_get_reduced_row`` (regional sub-areas).
        method:
            Per-call interpolation method override (``linear`` or ``nearest``).
        fast:
            If ``True``, skip row-wise missing-value detection for speed.
            Use only when the input has no missing values.
        grid_type_hint:
            Optional manual grid type name when metadata is incomplete.
            Must be one of the supported names listed by
            ``gaussregular.api.GRID_TYPE_FULL_NAMES``.
        precision:
            ``"auto"`` preserves float32/float64 input precision.
            ``"single"`` computes and returns float32; ``"double"`` computes
            and returns float64. Defaults to the regularizer's precision.

        Returns
        -------
        xarray.DataArray
            Same shape except the last dimension is replaced by
            ``(latitude, longitude)``. With ``precision="auto"``, float32
            input returns float32 and float64 input returns float64. Added attributes:
            ``gaussregular_converted``, ``gaussregular_mode``,
            ``gaussregular_is_global``.
        """
        xr = _require_xarray()

        if not hasattr(dataarray, "attrs"):
            raise TypeError("dataarray must be an xarray.DataArray")

        cache_key = None
        plan = None
        if self.cache:
            # Use shape + dims + GRIB_pl + effective N + requested nlon as key.
            raw_pl = np.asarray(dataarray.attrs.get(
                "GRIB_pl", []), dtype=np.int32)
            np_key = int(dataarray.attrs.get("GRIB_N", 0)
                         or self.grid_number or 0)
            cache_key = (
                tuple(dataarray.dims),
                tuple(dataarray.shape),
                int(nlon) if nlon is not None else -1,
                raw_pl.tobytes(),
                np_key,
                str(grid_type_hint).strip().lower()
                if grid_type_hint is not None
                else "",
            )
            plan = self._xarray_plan_cache.get(cache_key)

        if plan is None:
            plan = self._build_xarray_plan(dataarray, nlon, grid_type_hint)
            if self.cache and cache_key is not None:
                if len(self._xarray_plan_cache) >= max(1, int(self.max_plan_cache)):
                    # FIFO-style eviction by insertion order.
                    oldest = next(iter(self._xarray_plan_cache))
                    self._xarray_plan_cache.pop(oldest, None)
                self._xarray_plan_cache[cache_key] = plan

        attrs = dict(plan["attrs"])
        pl = plan["pl"]
        missval = float(plan["missval"])
        np_value = int(plan["np_value"])
        nlon_val = int(plan["nlon"])
        expected = int(plan["expected"])
        lead_dims = list(plan["lead_dims"])
        lat_1d = plan["lat_1d"]
        lon_1d = plan["lon_1d"]
        is_global = bool(plan["is_global"])
        lat_attrs = dict(plan["lat_attrs"])
        lon_attrs = dict(plan["lon_attrs"])
        method_name = _normalize_method(method, default=self.method)
        nearest_flag = method_name == "nearest"

        arr_in = np.asarray(dataarray.values)
        precision_name = _normalize_precision(precision, default=self.precision)
        compute_dtype, public_dtype = _precision_dtypes(
            arr_in.dtype, precision_name)
        arr = arr_in.astype(compute_dtype, copy=False)
        last = arr.shape[-1]
        if last != expected:
            raise ValueError(
                f"Last dimension size {last} does not match sum(GRIB_pl) {expected}"
            )

        lead_shape = arr.shape[:-1]
        batch = int(np.prod(lead_shape)) if lead_shape else 1
        flat = arr.reshape(batch, last)

        try:
            # Prefer batched C path when available to reduce Python overhead.
            out = _core.regularize_values_batch(
                flat, pl, missval, nearest_flag, nlon_val, bool(fast)
            )
        except AttributeError:
            # Fallback for older cores: loop in Python.
            out = np.empty((batch, int(pl.size), nlon_val),
                           dtype=compute_dtype)
            for i in range(batch):
                out[i] = _core.regularize_values(
                    flat[i], pl, missval, nearest_flag, nlon_val, bool(fast)
                )

        out = out.reshape(*lead_shape, int(pl.size), nlon_val)

        if out.dtype != public_dtype:
            out = out.astype(public_dtype, copy=False)

        dims = tuple(lead_dims + ["latitude", "longitude"])

        coords = {d: dataarray.coords[d]
                  for d in lead_dims if d in dataarray.coords}
        coords["latitude"] = xr.Variable("latitude", lat_1d, attrs=lat_attrs)
        coords["longitude"] = xr.Variable("longitude", lon_1d, attrs=lon_attrs)

        out_da = xr.DataArray(
            out,
            dims=dims,
            coords=coords,
            name=dataarray.name,
            attrs=attrs,
        )
        out_da.attrs["gaussregular_converted"] = "reduced_to_regular"
        out_da.attrs["gaussregular_mode"] = method_name
        out_da.attrs["gaussregular_is_global"] = int(is_global)
        return out_da

    def regularize_dataset(
        self,
        dataset: "xr.Dataset",
        nlon: Optional[int] = None,
        method: Optional[str] = None,
        grid_type_hint: Optional[str] = None,
        fast: bool = True,
        precision: Optional[PrecisionMode] = None,
    ) -> "xr.Dataset":
        """Interpolate all convertible variables in an xarray Dataset.

        Variables that do not look like reduced Gaussian fields are kept
        unchanged. For cfgrib datasets where ``GRIB_pl`` is only present in
        dataset attributes, that metadata is propagated to each variable before
        conversion.
        """
        xr = _require_xarray()

        if not isinstance(dataset, xr.Dataset):
            raise TypeError("dataset must be an xarray.Dataset")

        out_vars: dict[str, xr.DataArray] = {}
        for name, da in dataset.data_vars.items():
            merged_attrs = dict(dataset.attrs)
            merged_attrs.update(dict(da.attrs))

            if "GRIB_pl" not in merged_attrs:
                out_vars[name] = da
                continue

            if not _is_supported_reduced_grid(merged_attrs) and grid_type_hint is None:
                out_vars[name] = da
                continue

            work = da.copy(deep=False)
            work.attrs = merged_attrs
            out_vars[name] = self.regularize_xarray(
                work,
                nlon=nlon,
                method=method,
                grid_type_hint=grid_type_hint,
                fast=fast,
                precision=precision,
            )

        out_ds = xr.Dataset(out_vars, attrs=dict(dataset.attrs))
        out_ds.attrs["gaussregular_converted"] = "reduced_to_regular_dataset"
        out_ds.attrs["gaussregular_mode"] = _normalize_method(
            method, default=self.method)
        return out_ds

    def regularize(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Union["xr.Dataset", "xr.DataArray", Tuple[np.ndarray, np.ndarray]]:
        """Auto-detect input type and delegate to the appropriate method.

        Parameters
        ----------
        data:
                        * ``xarray.Dataset`` → calls :meth:`regularize_dataset`.
            * ``xarray.DataArray`` → calls :meth:`regularize_xarray`.
            * numpy array → calls :meth:`regularize_values` (``pl`` and
              ``missval`` are required keyword arguments in this case).
        **kwargs:
            Forwarded to the selected method.

        Returns
        -------
        For dataset input → ``xarray.Dataset`` with converted variables.
        For xarray DataArray input → ``xarray.DataArray`` on the regular grid.
        For numpy input → ``(out, lon)`` tuple.

        Examples
        --------
        xarray input (auto path)::

            out = engine.regularize(da_day1)

        numpy input (auto path)::

            out, lon = engine.regularize(values, pl=pl, missval=9.999e20, grid_number=320)
        """
        # xarray Dataset-like input (duck-typed to avoid hard dependency).
        if hasattr(data, "data_vars") and hasattr(data, "attrs"):
            return self.regularize_dataset(data, **kwargs)

        # Auto-detect xarray DataArray-like objects next.
        if hasattr(data, "attrs") and hasattr(data, "dims") and hasattr(data, "values"):
            return self.regularize_xarray(data, **kwargs)

        pl = kwargs.pop("pl", None)
        missval = kwargs.pop("missval", None)
        if pl is None or missval is None:
            raise ValueError(
                "For numpy input, 'pl' and 'missval' are required")
        return self.regularize_values(data, pl=pl, missval=missval, **kwargs)

    __call__ = regularize


_DEFAULT = GaussRegularizer()


def regularize_values(
    values: np.ndarray,
    pl: Iterable[int],
    missval: float,
    method: str = "linear",
    precision: PrecisionMode = "auto",
    nlon: Optional[int] = None,
    grid_number: Optional[int] = None,
    xfirst: float = 0.0,
    xlast: float = 359.9999,
    fast: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Module-level shortcut — see :meth:`GaussRegularizer.regularize_values`.

    Parameters
    ----------
    values:
        1-D reduced-grid data array.  Any dtype; output matches input dtype.
    pl:
        Points per latitude row (``GRIB_pl``).  See class docstring for full
        description and N-grid vs. O-grid examples.
    missval:
        Missing-value sentinel (e.g. ``9.999e20``, ``float('nan')``).
    method:
        ``linear`` (default) or ``nearest``.
    precision:
        ``"auto"`` preserves float32/float64 input precision.
        ``"single"`` returns float32; ``"double"`` returns float64.
    nlon:
        Output longitude count; inferred from ``pl`` when *None*.
    grid_number:
        Gaussian truncation N (e.g. 320 for N320 / O320).
    xfirst:
        First longitude in degrees (default 0.0).
    xlast:
        Last longitude in degrees (default 359.9999).
    fast:
        If ``True``, skip row-wise missing-value detection for speed.

    Returns
    -------
    out : np.ndarray
        Shape ``(nlat, nlon)``, dtype = input dtype.
    lon : np.ndarray
        1-D longitude array (degrees east).
    """
    return _DEFAULT.regularize_values(
        values=values,
        pl=pl,
        missval=missval,
        method=method,
        precision=precision,
        nlon=nlon,
        grid_number=grid_number,
        xfirst=xfirst,
        xlast=xlast,
        fast=fast,
    )


def regularize_xarray(
    dataarray: "xr.DataArray",
    method: str = "linear",
    precision: PrecisionMode = "auto",
    nlon: Optional[int] = None,
    grid_type_hint: Optional[str] = None,
    fast: bool = True,
) -> "xr.DataArray":
    """Module-level shortcut — see :meth:`GaussRegularizer.regularize_xarray`.

    Parameters
    ----------
    dataarray:
        ``xarray.DataArray`` on a reduced Gaussian grid with ``GRIB_pl`` in
        attrs (N-grids and O-grids both supported).
    method:
        ``linear`` (default) or ``nearest``.
    precision:
        ``"auto"`` preserves float32/float64 input precision.
        ``"single"`` returns float32; ``"double"`` returns float64.
    nlon:
        Override output longitude count.
    grid_type_hint:
        Optional manual grid type name when metadata is incomplete.
    fast:
        If ``True``, skip row-wise missing-value detection for speed.

    Returns
    -------
    xarray.DataArray
        Regular Gaussian grid; dtype matches input.
    """
    return _DEFAULT.regularize_xarray(
        dataarray,
        method=method,
        precision=precision,
        nlon=nlon,
        grid_type_hint=grid_type_hint,
        fast=fast,
    )


def regularize_dataset(
    dataset: "xr.Dataset",
    method: str = "linear",
    precision: PrecisionMode = "auto",
    nlon: Optional[int] = None,
    grid_type_hint: Optional[str] = None,
    fast: bool = True,
) -> "xr.Dataset":
    """Module-level shortcut — see :meth:`GaussRegularizer.regularize_dataset`."""
    return _DEFAULT.regularize_dataset(
        dataset,
        method=method,
        precision=precision,
        nlon=nlon,
        grid_type_hint=grid_type_hint,
        fast=fast,
    )
