#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

/*
 * gaussregular._core
 * -------------------
 *
 * C implementation of the inner loops for converting ECMWF reduced
 * Gaussian grids (N-grids / O-grids) to regular Gaussian grids.
 *
 * Author : Qianye Su
 * Email  : suqianye2000@gmail.com
 * Created: 2026-03-16
 *
 * Conventions used here:
 *   - values are always passed in as 1D float64 arrays in "row-packed"
 *     order (all points of latitude 0, then latitude 1, ...);
 *   - pl[j] holds the number of points in latitude row j;
 *   - output is a dense (nlat, nlon) float64 array in row-major order;
 *   - missing values are detected via a numeric sentinel 'missval',
 *     including the special case where missval is NaN.
 */

/* Check if a value v matches the missing-value sentinel missval.
   Handles both NaN and non-NaN missing values correctly. */
static int is_missing(double v, double missval)
{
    if (isnan(missval))
        return isnan(v);
    return v == missval;
}

/*
 * grib_get_reduced_row
 * ---------------------
 * Given a full reduced-Gaussian row with 'pl' grid points over 0..360,
 * compute how many points fall inside the longitudinal window
 * [lon_first, lon_last], and the integer start/end indices (ilon_first,
 * ilon_last) in the reduced row.  This mirrors the behaviour of the
 * ECMWF GRIB API utility of the same name.
 */
static void grib_get_reduced_row(long pl, double lon_first, double lon_last,
                                 long *npoints, long *ilon_first, long *ilon_last)
{
    double range = lon_last - lon_first;
    if (range < 0.0)
    {
        range += 360.0;
        lon_first -= 360.0;
    }

    *npoints = (long)((range * pl) / 360.0) + 1;
    *ilon_first = (long)((lon_first * pl) / 360.0);
    *ilon_last = (long)((lon_last * pl) / 360.0);

    long irange = *ilon_last - *ilon_first + 1;

    if (irange != *npoints)
    {
        if (irange > *npoints)
        {
            double dlon_first = ((*ilon_first) * 360.0) / pl;
            if (dlon_first < lon_first)
            {
                (*ilon_first)++;
                irange--;
            }

            double dlon_last = ((*ilon_last) * 360.0) / pl;
            if (dlon_last > lon_last)
            {
                (*ilon_last)--;
                irange--;
            }
        }
        else
        {
            int ok = 0;
            double dlon_first = ((*ilon_first - 1) * 360.0) / pl;
            if (dlon_first > lon_first)
            {
                (*ilon_first)--;
                irange++;
                ok = 1;
            }

            double dlon_last = ((*ilon_last + 1) * 360.0) / pl;
            if (dlon_last < lon_last)
            {
                (*ilon_last)++;
                irange++;
                ok = 1;
            }

            if (!ok)
                (*npoints)--;
        }
    }
    else
    {
        double dlon_first = ((*ilon_first) * 360.0) / pl;
        if (dlon_first < lon_first)
        {
            (*ilon_first)++;
            (*ilon_last)++;
        }
    }

    if (*ilon_first < 0)
        *ilon_first += pl;
}

/*
 * reduced_grid_is_global
 * ----------------------
 * Decide whether a reduced Gaussian grid should be treated as global.
 *
 * np    : Gaussian truncation number (N); controls the theoretical
 *         meridional resolution (90 / np degrees).
 * nxmax : maximum number of points per latitude row.
 * xfirst/xlast : longitudinal data window.
 *
 * Returns non-zero when the data window is wide enough to be considered
 * a global field; otherwise returns 0 (regional / sub-area grid).
 */
static int reduced_grid_is_global(int np, int nxmax, double xfirst, double xlast)
{
    double dx_global = (np > 0) ? (90.0 / np) : 999.0;
    double dx_data = 360.0 - (xlast - xfirst);
    if ((dx_data > dx_global) && (dx_data * nxmax > 360.0))
        dx_data = 360.0 / nxmax;
    return !(dx_data > dx_global);
}

/* Bilinear interpolation: map src_n source points to dst_n destination points.
   Scale factor: each output point is computed via linear interpolation between
   two adjacent source points. If has_missing is 1, missing values are handled
   by fallback logic: if both neighbors are missing, output is missing;
   if one is missing, use the valid neighbor; if both valid, apply linear blend. */
static int regularize_row_linear(const double *src, int src_n, double *dst, int dst_n, int has_missing, double missval)
{
    if (src_n <= 0 || dst_n <= 0)
        return -1;
    /* Fast path: if source and destination size match, copy as-is */
    if (src_n == dst_n)
    {
        for (int i = 0; i < dst_n; ++i)
            dst[i] = src[i];
        return 0;
    }

    const double scale = (double)src_n / (double)dst_n;
    for (int i = 0; i < dst_n; ++i)
    {
        /* Map output index i to source position x */
        double x = i * scale;
        int i0 = (int)x;
        double t = x - i0; /* Fractional part for blending weight */
        int i1 = i0 + 1;
        if (i1 >= src_n)
            i1 -= src_n; /* Wrap around for cyclic longitude */

        double v0 = src[i0];
        double v1 = src[i1];
        if (has_missing)
        {
            int m0 = is_missing(v0, missval);
            int m1 = is_missing(v1, missval);
            if (m0 && m1)
                dst[i] = missval; /* Both missing: output missing */
            else if (m0)
                dst[i] = v1; /* v0 missing: use v1 */
            else if (m1)
                dst[i] = v0; /* v1 missing: use v0 */
            else
                dst[i] = v0 * (1.0 - t) + v1 * t; /* Both valid: linear blend */
        }
        else
        {
            /* No missing values: unconditional linear interpolation */
            dst[i] = v0 * (1.0 - t) + v1 * t;
        }
    }
    return 0;
}

/* Nearest-neighbour interpolation: find the closest source point for each
   destination index.  When missing values are present, the nearest point
   may itself be missing; in that case we expand a search radius and look
   alternately forward/backward until a valid neighbour is found.  If no
   valid neighbour exists in the row, the output is marked as missing. */
static int regularize_row_nearest(const double *src, int src_n, double *dst, int dst_n, int has_missing, double missval)
{
    if (src_n <= 0 || dst_n <= 0)
        return -1;
    /* Fast path: if source and destination size match, copy as-is */
    if (src_n == dst_n)
    {
        for (int i = 0; i < dst_n; ++i)
            dst[i] = src[i];
        return 0;
    }

    const double scale = (double)src_n / (double)dst_n;
    for (int i = 0; i < dst_n; ++i)
    {
        /* Find nearest source index idx for output position i */
        double x = i * scale;
        int idx = (int)(x + 0.5); /* Round to nearest integer */
        if (idx >= src_n)
            idx -= src_n; /* Wrap around */
        double v = src[idx];

        /* If no missing values or nearest point is valid, use it directly */
        if (!has_missing || !is_missing(v, missval))
        {
            dst[i] = v;
            continue;
        }

        /* Nearest point is missing; search for valid neighbor by expanding radius */
        int found = 0;
        for (int d = 1; d < src_n; ++d)
        {
            int p = idx + d; /* Forward neighbor at distance d */
            int m = idx - d; /* Backward neighbor at distance d */
            if (p >= src_n)
                p -= src_n; /* Wrap around */
            if (m < 0)
                m += src_n; /* Wrap around */
            /* Try forward neighbor first */
            if (!is_missing(src[p], missval))
            {
                dst[i] = src[p];
                found = 1;
                break;
            }
            /* Try backward neighbor */
            if (!is_missing(src[m], missval))
            {
                dst[i] = src[m];
                found = 1;
                break;
            }
        }
        /* If no valid neighbor found, mark output as missing */
        if (!found)
            dst[i] = missval;
    }
    return 0;
}

/* Python wrapper for grib_get_reduced_row.
   Python signature: reduced_row(pl, xfirst, xlast) -> (npoints, ilon_first, ilon_last). */
static PyObject *py_reduced_row(PyObject *self, PyObject *args)
{
    long pl = 0;
    double xfirst = 0.0;
    double xlast = 359.9999;
    if (!PyArg_ParseTuple(args, "ldd", &pl, &xfirst, &xlast))
        return NULL;
    if (pl <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "pl must be positive");
        return NULL;
    }

    long npoints = 0, ilon_first = 0, ilon_last = 0;
    grib_get_reduced_row(pl, xfirst, xlast, &npoints, &ilon_first, &ilon_last);
    return Py_BuildValue("lll", npoints, ilon_first, ilon_last);
}

/* Python wrapper around reduced_grid_is_global.
   Python signature: is_global(np, pl, xfirst, xlast) -> bool. */
static PyObject *py_is_global(PyObject *self, PyObject *args)
{
    int np = 0;
    PyObject *pl_obj = NULL;
    double xfirst = 0.0;
    double xlast = 359.9999;
    if (!PyArg_ParseTuple(args, "iOdd", &np, &pl_obj, &xfirst, &xlast))
        return NULL;

    PyArrayObject *pl = (PyArrayObject *)PyArray_FROM_OTF(pl_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!pl)
        return NULL;
    if (PyArray_NDIM(pl) != 1 || PyArray_DIM(pl, 0) == 0)
    {
        Py_DECREF(pl);
        PyErr_SetString(PyExc_ValueError, "pl must be a 1D non-empty int32 array");
        return NULL;
    }

    npy_intp nlat = PyArray_DIM(pl, 0);
    const int *pl_data = (const int *)PyArray_DATA(pl);
    int nxmax = 0;
    for (npy_intp j = 0; j < nlat; ++j)
        if (pl_data[j] > nxmax)
            nxmax = pl_data[j];

    int ok = reduced_grid_is_global(np, nxmax, xfirst, xlast);
    Py_DECREF(pl);
    if (ok)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/* Infer the regular-grid longitude count for a reduced Gaussian grid.
   Python signature: infer_nlon(pl, np, xfirst, xlast) -> nlon. */
static PyObject *py_infer_nlon(PyObject *self, PyObject *args)
{
    PyObject *pl_obj = NULL;
    int np = 0;
    double xfirst = 0.0;
    double xlast = 359.9999;
    if (!PyArg_ParseTuple(args, "Oidd", &pl_obj, &np, &xfirst, &xlast))
        return NULL;

    PyArrayObject *pl = (PyArrayObject *)PyArray_FROM_OTF(pl_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!pl)
        return NULL;
    if (PyArray_NDIM(pl) != 1 || PyArray_DIM(pl, 0) == 0)
    {
        Py_DECREF(pl);
        PyErr_SetString(PyExc_ValueError, "pl must be a 1D non-empty int32 array");
        return NULL;
    }

    npy_intp nlat = PyArray_DIM(pl, 0);
    const int *pl_data = (const int *)PyArray_DATA(pl);

    int nlon = 0;
    if (np > 0)
    {
        int nxmax = 0;
        for (npy_intp j = 0; j < nlat; ++j)
            if (pl_data[j] > nxmax)
                nxmax = pl_data[j];

        if (reduced_grid_is_global(np, nxmax, xfirst, xlast))
        {
            nlon = pl_data[nlat / 2];
        }
        else
        {
            long row_count = 0, ilon_first = 0, ilon_last = 0;
            long np4 = (long)np * 4L;
            grib_get_reduced_row(np4, xfirst, xlast, &row_count, &ilon_first, &ilon_last);
            nlon = (int)row_count;
        }
    }
    else
    {
        nlon = pl_data[nlat / 2];
    }

    Py_DECREF(pl);
    return PyLong_FromLong((long)nlon);
}

/* Helper: regularise a single field (one reduced Gaussian grid) into
   a regular Gaussian grid.

   in_data  : pointer to 1D values, length == sum(pl_data[j])
   pl_data  : row lengths (points per latitude)
   nlat     : number of latitude rows
   nlon     : number of longitudes in the target regular grid
   missval  : missing-value sentinel
   nearest  : 0 -> linear, non-zero -> nearest-neighbour
   fast     : 0 -> scan each row for missing values; non-zero assumes
              there are no missing values and skips the scan
   out_data : pointer to preallocated (nlat * nlon) output buffer. */
static int regularize_field(
    const double *in_data,
    const int *pl_data,
    npy_intp nlat,
    int nlon,
    double missval,
    int nearest,
    int fast,
    double *out_data)
{
    npy_intp ptr = 0;
    for (npy_intp j = 0; j < nlat; ++j)
    {
        const int src_n = pl_data[j];
        const double *src = in_data + ptr;
        double *dst = out_data + j * (npy_intp)nlon;

        /* Detect if this row contains any missing values.
           Skip detection if fast=1 (assumes input has no missing values). */
        int has_missing = 0;
        if (!fast)
        {
            for (int k = 0; k < src_n; ++k)
            {
                if (is_missing(src[k], missval))
                {
                    has_missing = 1;
                    break;
                }
            }
        }

        int rc = nearest ? regularize_row_nearest(src, src_n, dst, nlon, has_missing, missval)
                         : regularize_row_linear(src, src_n, dst, nlon, has_missing, missval);
        if (rc != 0)
        {
            return rc;
        }

        ptr += src_n;
    }

    return 0;
}

/* Main Python entry point for grid regularization.
   Reads reduced-Gaussian values, converts them to regular-Gaussian via
   per-row interpolation (linear or nearest-neighbor).

   Args:
     values_obj: 1D float64 array of reduced-grid values (row-packed)
     pl_obj: 1D int32 array of row lengths (points per latitude)
     missval: Sentinel for missing data
     nearest: 1 for nearest-neighbor, 0 for linear interpolation
     nlon: Output longitude count
     fast: (optional) If 1, skip per-row missing-value detection for speed.
           Use only when input has no missing values.

   Returns: 2D float64 array of shape (nlat, nlon)
 */
static PyObject *py_regularize_values(PyObject *self, PyObject *args)
{
    PyObject *values_obj = NULL;
    PyObject *pl_obj = NULL;
    double missval = 0.0;
    int nearest = 0;
    int nlon = 0;
    int fast = 0; /* Renamed from assume_no_missing */

    if (!PyArg_ParseTuple(args, "OOdpi|p", &values_obj, &pl_obj, &missval, &nearest, &nlon, &fast))
    {
        return NULL;
    }

    PyArrayObject *values = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pl = (PyArrayObject *)PyArray_FROM_OTF(pl_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!values || !pl)
    {
        Py_XDECREF(values);
        Py_XDECREF(pl);
        return NULL;
    }

    if (PyArray_NDIM(values) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "values must be 1D float64 array");
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }
    if (PyArray_NDIM(pl) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "pl must be 1D int32 array");
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }
    if (nlon <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "nlon must be positive");
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }

    npy_intp nlat = PyArray_DIM(pl, 0);
    const int *pl_data = (const int *)PyArray_DATA(pl);

    npy_intp sum_pl = 0;
    for (npy_intp j = 0; j < nlat; ++j)
    {
        if (pl_data[j] <= 0)
        {
            PyErr_SetString(PyExc_ValueError, "pl contains non-positive values");
            Py_DECREF(values);
            Py_DECREF(pl);
            return NULL;
        }
        sum_pl += pl_data[j];
    }

    if (PyArray_DIM(values, 0) != sum_pl)
    {
        PyErr_SetString(PyExc_ValueError, "values length must equal sum(pl)");
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }

    npy_intp out_dims[2] = {nlat, (npy_intp)nlon};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_FLOAT64);
    if (!out)
    {
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }

    const double *in_data = (const double *)PyArray_DATA(values);
    double *out_data = (double *)PyArray_DATA(out);

    int rc = regularize_field(in_data, pl_data, nlat, nlon, missval, nearest, fast, out_data);
    if (rc != 0)
    {
        Py_DECREF(values);
        Py_DECREF(pl);
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, "row interpolation failed");
        return NULL;
    }

    Py_DECREF(values);
    Py_DECREF(pl);
    return (PyObject *)out;
}

/* Batch interface: process a stack of fields in one call.
   values2d has shape (batch, sum(pl)), out has shape (batch, nlat, nlon). */
static PyObject *py_regularize_values_batch(PyObject *self, PyObject *args)
{
    PyObject *values_obj = NULL;
    PyObject *pl_obj = NULL;
    double missval = 0.0;
    int nearest = 0;
    int nlon = 0;
    int fast = 0;

    if (!PyArg_ParseTuple(args, "OOdpi|p", &values_obj, &pl_obj, &missval, &nearest, &nlon, &fast))
    {
        return NULL;
    }

    PyArrayObject *values = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pl = (PyArrayObject *)PyArray_FROM_OTF(pl_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!values || !pl)
    {
        Py_XDECREF(values);
        Py_XDECREF(pl);
        return NULL;
    }

    if (PyArray_NDIM(values) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "values must be 2D float64 array");
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }
    if (PyArray_NDIM(pl) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "pl must be 1D int32 array");
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }
    if (nlon <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "nlon must be positive");
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }

    npy_intp batch = PyArray_DIM(values, 0);
    npy_intp nvals = PyArray_DIM(values, 1);
    npy_intp nlat = PyArray_DIM(pl, 0);
    const int *pl_data = (const int *)PyArray_DATA(pl);

    npy_intp sum_pl = 0;
    for (npy_intp j = 0; j < nlat; ++j)
    {
        if (pl_data[j] <= 0)
        {
            PyErr_SetString(PyExc_ValueError, "pl contains non-positive values");
            Py_DECREF(values);
            Py_DECREF(pl);
            return NULL;
        }
        sum_pl += pl_data[j];
    }

    if (nvals != sum_pl)
    {
        PyErr_SetString(PyExc_ValueError, "second dimension of values must equal sum(pl)");
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }

    npy_intp out_dims[3] = {batch, nlat, (npy_intp)nlon};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(3, out_dims, NPY_FLOAT64);
    if (!out)
    {
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }

    const double *in_data = (const double *)PyArray_DATA(values);
    double *out_data = (double *)PyArray_DATA(out);

    for (npy_intp b = 0; b < batch; ++b)
    {
        const double *in_b = in_data + b * nvals;
        double *out_b = out_data + b * (nlat * (npy_intp)nlon);

        int rc = regularize_field(in_b, pl_data, nlat, nlon, missval, nearest, fast, out_b);
        if (rc != 0)
        {
            Py_DECREF(values);
            Py_DECREF(pl);
            Py_DECREF(out);
            PyErr_SetString(PyExc_RuntimeError, "row interpolation failed");
            return NULL;
        }
    }

    Py_DECREF(values);
    Py_DECREF(pl);
    return (PyObject *)out;
}

static PyMethodDef methods[] = {
    {"regularize_values_batch", py_regularize_values_batch, METH_VARARGS,
     "regularize_values_batch(values2d, pl, missval, nearest, nlon, fast=False) -> 3D float64 array"},
    {"regularize_values", py_regularize_values, METH_VARARGS,
     "regularize_values(values, pl, missval, nearest, nlon, fast=False) -> 2D float64 array"},
    {"reduced_row", py_reduced_row, METH_VARARGS,
     "reduced_row(pl, xfirst, xlast) -> (npoints, ilon_first, ilon_last)"},
    {"is_global", py_is_global, METH_VARARGS,
     "is_global(np, pl, xfirst, xlast) -> bool"},
    {"infer_nlon", py_infer_nlon, METH_VARARGS,
     "infer_nlon(pl, np, xfirst, xlast) -> output nlon"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "gaussregular C core",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__core(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
