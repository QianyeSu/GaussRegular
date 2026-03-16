from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import xarray as xr
from . import _core


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


def _is_supported_reduced_grid(attrs: dict[str, Any]) -> bool:
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


@dataclass
class GaussRegularizer:
    """Convert ECMWF reduced Gaussian grids to regular (full) Gaussian grids.

    Supports both classical **N-grids** (e.g. N320, N640) and octahedral
    **O-grids** (e.g. O320, O640, O1280).  Both grid families carry a ``pl``
    array and use the identical row-wise interpolation path.

    Computation is always performed in **double precision** (matching CDO
    ``setgridtype,regular`` / ``setgridtype,regularnn``).  The *output* array
    is cast back to the same dtype as the input so that float32 data does not
    silently bloat to float64.

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
    default_np_value:
        Default Gaussian truncation number N (e.g. 320 for N320/O320).  Used
        only when *grid_number* is not supplied per-call.
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
    default_np_value: Optional[int] = None
    cache: bool = False
    max_plan_cache: int = 32
    _xarray_plan_cache: dict[tuple, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.method = _normalize_method(self.method)

    def clear_cache(self) -> None:
        """Clear internal xarray plan cache."""
        self._xarray_plan_cache.clear()

    def _build_xarray_plan(self, dataarray, nlon: Optional[int], grid_type_hint: Optional[str]) -> dict[str, Any]:
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
        np_value = int(attrs.get("GRIB_N", self.default_np_value or 0) or 0)

        if nlon is None:
            nlon_val = int(_core.infer_nlon(pl, np_value, 0.0, 359.9999))
        else:
            nlon_val = int(nlon)

        expected = int(pl.sum())
        lead_dims = tuple(dataarray.dims[:-1])

        lat_1d = None
        for cand in ("latitude", "lat"):
            if cand in dataarray.coords:
                latv = np.asarray(dataarray.coords[cand].values)
                if latv.ndim == 1 and latv.size == int(pl.size):
                    lat_1d = latv.astype(np.float64)
                elif latv.ndim >= 1 and latv.size >= int(pl.size):
                    lat_1d = np.unique(latv.reshape(-1)).astype(np.float64)
                    if lat_1d.size == int(pl.size):
                        lat_1d = np.sort(lat_1d)[::-1]
                if lat_1d is not None and lat_1d.size == int(pl.size):
                    break

        if lat_1d is None or lat_1d.size != int(pl.size):
            lat_1d = np.arange(int(pl.size), dtype=np.float64)

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
        fast: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate a flat reduced Gaussian array to a regular longitude grid.

        Parameters
        ----------
        values:
            1-D array of reduced-grid values packed row-by-row from north to
            south.  Length must equal ``sum(pl)``.  Any numeric dtype is
            accepted (float32, float64, …); the interpolation is performed in
            float64 internally and the result is cast back to the input dtype.
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
            *nlon* for sub-areas.  Falls back to ``self.default_np_value``.
        xfirst:
            Longitude of the first grid point in degrees.  Default ``0.0``.
        xlast:
            Longitude of the last grid point in degrees.  Default ``359.9999``.
        method:
            Per-call interpolation method override (``linear`` or ``nearest``).
        fast:
            If ``True``, skip row-wise missing-value detection for speed.
            Use only when input is guaranteed to contain no missing values.

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
            **Dtype matches the input** *values* dtype.
        lon : np.ndarray
            1-D longitude coordinate array of length *nlon*, degrees east.
        """
        pl_arr = _to_int32_pl(pl)
        method_name = _normalize_method(method, default=self.method)
        nearest_flag = method_name == "nearest"
        np_in = int(self.default_np_value or 0) if grid_number is None else int(
            grid_number)

        if nlon is None:
            nlon = int(_core.infer_nlon(
                pl_arr, np_in, float(xfirst), float(xlast)))
        if nlon <= 0:
            raise ValueError("nlon must be positive")

        orig_dtype = np.asarray(values).dtype
        # Pass as-is; the C extension converts to float64 internally.
        vals = np.asarray(values)
        if vals.ndim != 1:
            raise ValueError(
                "values must be a 1D array with length == sum(pl)")

        expected = int(pl_arr.sum())
        if vals.size != expected:
            raise ValueError(
                f"values length {vals.size} does not match sum(pl) {expected}")

        # C extension always returns float64; cast back to input dtype.
        out = _core.regularize_values(
            vals, pl_arr, float(missval), nearest_flag, int(nlon), bool(fast))
        if orig_dtype != np.float64:
            out = out.astype(orig_dtype, copy=False)

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
        dataarray,
        nlon: Optional[int] = None,
        method: Optional[str] = None,
        grid_type_hint: Optional[str] = None,
        fast: bool = False,
    ):
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

        Returns
        -------
        xarray.DataArray
            Same shape except the last dimension is replaced by
            ``(latitude, longitude)``.  **Output dtype matches the input
            dtype** (e.g. float32 in → float32 out).  Added attributes:
            ``gaussregular_converted``, ``gaussregular_mode``,
            ``gaussregular_is_global``.
        """
        if not hasattr(dataarray, "attrs"):
            raise TypeError("dataarray must be an xarray.DataArray")

        cache_key = None
        plan = None
        if self.cache:
            # Use shape + dims + GRIB_pl + requested nlon as cache key.
            raw_pl = np.asarray(dataarray.attrs.get(
                "GRIB_pl", []), dtype=np.int32)
            cache_key = (
                tuple(dataarray.dims),
                tuple(dataarray.shape),
                int(nlon) if nlon is not None else -1,
                raw_pl.tobytes(),
                int(dataarray.attrs.get("GRIB_N", self.default_np_value or 0) or 0),
                str(grid_type_hint).strip().lower(
                ) if grid_type_hint is not None else "",
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
        method_name = _normalize_method(method, default=self.method)
        nearest_flag = method_name == "nearest"

        orig_dtype = dataarray.values.dtype
        # Do not force dtype here; the C extension converts internally.
        arr = np.asarray(dataarray.values)
        last = arr.shape[-1]
        if last != expected:
            raise ValueError(
                f"Last dimension size {last} does not match sum(GRIB_pl) {expected}"
            )

        lead_shape = arr.shape[:-1]
        batch = int(np.prod(lead_shape)) if lead_shape else 1
        flat = arr.reshape(batch, last)

        out = np.empty((batch, int(pl.size), nlon_val), dtype=np.float64)
        for i in range(batch):
            out[i] = _core.regularize_values(
                flat[i], pl, missval, nearest_flag, nlon_val, bool(fast))

        out = out.reshape(*lead_shape, int(pl.size), nlon_val)

        # Cast output back to the original dtype (avoids float32 → float64 bloat).
        if orig_dtype != np.float64:
            out = out.astype(orig_dtype, copy=False)

        dims = tuple(lead_dims + ["latitude", "longitude"])

        coords = {d: dataarray.coords[d]
                  for d in lead_dims if d in dataarray.coords}
        coords["latitude"] = lat_1d
        coords["longitude"] = lon_1d

        out_da = xr.DataArray(
            out,
            dims=dims,
            coords=coords,
            name=dataarray.name,
            attrs=attrs,
        )
        out_da.attrs["gaussregular_converted"] = "reduced_to_regular"
        out_da.attrs["gaussregular_mode"] = method_name
        out_da.attrs["gaussregular_is_global"] = is_global
        out_da.attrs["gaussregular_plan_cache"] = "on" if self.cache else "off"
        return out_da

    def regularize_dataset(
        self,
        dataset,
        nlon: Optional[int] = None,
        method: Optional[str] = None,
        grid_type_hint: Optional[str] = None,
        fast: bool = False,
    ):
        """Interpolate all convertible variables in an xarray Dataset.

        Variables that do not look like reduced Gaussian fields are kept
        unchanged. For cfgrib datasets where ``GRIB_pl`` is only present in
        dataset attributes, that metadata is propagated to each variable before
        conversion.
        """
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
            )

        out_ds = xr.Dataset(out_vars, attrs=dict(dataset.attrs))
        out_ds.attrs["gaussregular_converted"] = "reduced_to_regular_dataset"
        out_ds.attrs["gaussregular_mode"] = _normalize_method(
            method, default=self.method)
        out_ds.attrs["gaussregular_plan_cache"] = "on" if self.cache else "off"
        return out_ds

    def regularize(self, data: Any, **kwargs):
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
        if isinstance(data, xr.Dataset):
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
    nlon: Optional[int] = None,
    grid_number: Optional[int] = None,
    xfirst: float = 0.0,
    xlast: float = 359.9999,
    fast: bool = False,
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
        nlon=nlon,
        grid_number=grid_number,
        xfirst=xfirst,
        xlast=xlast,
        fast=fast,
    )


def regularize_xarray(
    dataarray,
    method: str = "linear",
    nlon: Optional[int] = None,
    grid_type_hint: Optional[str] = None,
    fast: bool = False,
):
    """Module-level shortcut — see :meth:`GaussRegularizer.regularize_xarray`.

    Parameters
    ----------
    dataarray:
        ``xarray.DataArray`` on a reduced Gaussian grid with ``GRIB_pl`` in
        attrs (N-grids and O-grids both supported).
    method:
        ``linear`` (default) or ``nearest``.
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
        nlon=nlon,
        grid_type_hint=grid_type_hint,
        fast=fast,
    )


def regularize_dataset(
    dataset,
    method: str = "linear",
    nlon: Optional[int] = None,
    grid_type_hint: Optional[str] = None,
    fast: bool = False,
):
    """Module-level shortcut — see :meth:`GaussRegularizer.regularize_dataset`."""
    return _DEFAULT.regularize_dataset(
        dataset,
        method=method,
        nlon=nlon,
        grid_type_hint=grid_type_hint,
        fast=fast,
    )
