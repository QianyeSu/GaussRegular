#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

static int is_missing(double v, double missval) {
    if (isnan(missval)) return isnan(v);
    return v == missval;
}

static void grib_get_reduced_row(long pl, double lon_first, double lon_last,
                                 long *npoints, long *ilon_first, long *ilon_last) {
    double range = lon_last - lon_first;
    if (range < 0.0) {
        range += 360.0;
        lon_first -= 360.0;
    }

    *npoints = (long)((range * pl) / 360.0) + 1;
    *ilon_first = (long)((lon_first * pl) / 360.0);
    *ilon_last = (long)((lon_last * pl) / 360.0);

    long irange = *ilon_last - *ilon_first + 1;

    if (irange != *npoints) {
        if (irange > *npoints) {
            double dlon_first = ((*ilon_first) * 360.0) / pl;
            if (dlon_first < lon_first) {
                (*ilon_first)++;
                irange--;
            }

            double dlon_last = ((*ilon_last) * 360.0) / pl;
            if (dlon_last > lon_last) {
                (*ilon_last)--;
                irange--;
            }
        } else {
            int ok = 0;
            double dlon_first = ((*ilon_first - 1) * 360.0) / pl;
            if (dlon_first > lon_first) {
                (*ilon_first)--;
                irange++;
                ok = 1;
            }

            double dlon_last = ((*ilon_last + 1) * 360.0) / pl;
            if (dlon_last < lon_last) {
                (*ilon_last)++;
                irange++;
                ok = 1;
            }

            if (!ok) (*npoints)--;
        }
    } else {
        double dlon_first = ((*ilon_first) * 360.0) / pl;
        if (dlon_first < lon_first) {
            (*ilon_first)++;
            (*ilon_last)++;
        }
    }

    if (*ilon_first < 0) *ilon_first += pl;
}

static int reduced_grid_is_global(int np, int nxmax, double xfirst, double xlast) {
    double dx_global = (np > 0) ? (90.0 / np) : 999.0;
    double dx_data = 360.0 - (xlast - xfirst);
    if ((dx_data > dx_global) && (dx_data * nxmax > 360.0)) dx_data = 360.0 / nxmax;
    return !(dx_data > dx_global);
}

static int regularize_row_linear(const double *src, int src_n, double *dst, int dst_n, int has_missing, double missval) {
    if (src_n <= 0 || dst_n <= 0) return -1;
    if (src_n == dst_n) {
        for (int i = 0; i < dst_n; ++i) dst[i] = src[i];
        return 0;
    }

    const double scale = (double)src_n / (double)dst_n;
    for (int i = 0; i < dst_n; ++i) {
        double x = i * scale;
        int i0 = (int)x;
        double t = x - i0;
        int i1 = i0 + 1;
        if (i1 >= src_n) i1 -= src_n;

        double v0 = src[i0];
        double v1 = src[i1];
        if (has_missing) {
            int m0 = is_missing(v0, missval);
            int m1 = is_missing(v1, missval);
            if (m0 && m1) dst[i] = missval;
            else if (m0) dst[i] = v1;
            else if (m1) dst[i] = v0;
            else dst[i] = v0 * (1.0 - t) + v1 * t;
        } else {
            dst[i] = v0 * (1.0 - t) + v1 * t;
        }
    }
    return 0;
}

static int regularize_row_nearest(const double *src, int src_n, double *dst, int dst_n, int has_missing, double missval) {
    if (src_n <= 0 || dst_n <= 0) return -1;
    if (src_n == dst_n) {
        for (int i = 0; i < dst_n; ++i) dst[i] = src[i];
        return 0;
    }

    const double scale = (double)src_n / (double)dst_n;
    for (int i = 0; i < dst_n; ++i) {
        double x = i * scale;
        int idx = (int)(x + 0.5);
        if (idx >= src_n) idx -= src_n;
        double v = src[idx];

        if (!has_missing || !is_missing(v, missval)) {
            dst[i] = v;
            continue;
        }

        int found = 0;
        for (int d = 1; d < src_n; ++d) {
            int p = idx + d;
            int m = idx - d;
            if (p >= src_n) p -= src_n;
            if (m < 0) m += src_n;
            if (!is_missing(src[p], missval)) {
                dst[i] = src[p];
                found = 1;
                break;
            }
            if (!is_missing(src[m], missval)) {
                dst[i] = src[m];
                found = 1;
                break;
            }
        }
        if (!found) dst[i] = missval;
    }
    return 0;
}

static PyObject *py_reduced_row(PyObject *self, PyObject *args) {
    long pl = 0;
    double xfirst = 0.0;
    double xlast = 359.9999;
    if (!PyArg_ParseTuple(args, "ldd", &pl, &xfirst, &xlast)) return NULL;
    if (pl <= 0) {
        PyErr_SetString(PyExc_ValueError, "pl must be positive");
        return NULL;
    }

    long npoints = 0, ilon_first = 0, ilon_last = 0;
    grib_get_reduced_row(pl, xfirst, xlast, &npoints, &ilon_first, &ilon_last);
    return Py_BuildValue("lll", npoints, ilon_first, ilon_last);
}

static PyObject *py_is_global(PyObject *self, PyObject *args) {
    int np = 0;
    PyObject *pl_obj = NULL;
    double xfirst = 0.0;
    double xlast = 359.9999;
    if (!PyArg_ParseTuple(args, "iOdd", &np, &pl_obj, &xfirst, &xlast)) return NULL;

    PyArrayObject *pl = (PyArrayObject *)PyArray_FROM_OTF(pl_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!pl) return NULL;
    if (PyArray_NDIM(pl) != 1 || PyArray_DIM(pl, 0) == 0) {
        Py_DECREF(pl);
        PyErr_SetString(PyExc_ValueError, "pl must be a 1D non-empty int32 array");
        return NULL;
    }

    npy_intp nlat = PyArray_DIM(pl, 0);
    const int *pl_data = (const int *)PyArray_DATA(pl);
    int nxmax = 0;
    for (npy_intp j = 0; j < nlat; ++j) if (pl_data[j] > nxmax) nxmax = pl_data[j];

    int ok = reduced_grid_is_global(np, nxmax, xfirst, xlast);
    Py_DECREF(pl);
    if (ok) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *py_infer_nlon(PyObject *self, PyObject *args) {
    PyObject *pl_obj = NULL;
    int np = 0;
    double xfirst = 0.0;
    double xlast = 359.9999;
    if (!PyArg_ParseTuple(args, "Oidd", &pl_obj, &np, &xfirst, &xlast)) return NULL;

    PyArrayObject *pl = (PyArrayObject *)PyArray_FROM_OTF(pl_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!pl) return NULL;
    if (PyArray_NDIM(pl) != 1 || PyArray_DIM(pl, 0) == 0) {
        Py_DECREF(pl);
        PyErr_SetString(PyExc_ValueError, "pl must be a 1D non-empty int32 array");
        return NULL;
    }

    npy_intp nlat = PyArray_DIM(pl, 0);
    const int *pl_data = (const int *)PyArray_DATA(pl);

    int nlon = 0;
    if (np > 0) {
        int nxmax = 0;
        for (npy_intp j = 0; j < nlat; ++j) if (pl_data[j] > nxmax) nxmax = pl_data[j];

        if (reduced_grid_is_global(np, nxmax, xfirst, xlast)) {
            nlon = pl_data[nlat / 2];
        } else {
            long row_count = 0, ilon_first = 0, ilon_last = 0;
            long np4 = (long)np * 4L;
            grib_get_reduced_row(np4, xfirst, xlast, &row_count, &ilon_first, &ilon_last);
            nlon = (int)row_count;
        }
    } else {
        nlon = pl_data[nlat / 2];
    }

    Py_DECREF(pl);
    return PyLong_FromLong((long)nlon);
}

static PyObject *py_regularize_values(PyObject *self, PyObject *args) {
    PyObject *values_obj = NULL;
    PyObject *pl_obj = NULL;
    double missval = 0.0;
    int nearest = 0;
    int nlon = 0;

    if (!PyArg_ParseTuple(args, "OOdpi", &values_obj, &pl_obj, &missval, &nearest, &nlon)) {
        return NULL;
    }

    PyArrayObject *values = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pl = (PyArrayObject *)PyArray_FROM_OTF(pl_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!values || !pl) {
        Py_XDECREF(values);
        Py_XDECREF(pl);
        return NULL;
    }

    if (PyArray_NDIM(values) != 1) {
        PyErr_SetString(PyExc_ValueError, "values must be 1D float64 array");
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }
    if (PyArray_NDIM(pl) != 1) {
        PyErr_SetString(PyExc_ValueError, "pl must be 1D int32 array");
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }
    if (nlon <= 0) {
        PyErr_SetString(PyExc_ValueError, "nlon must be positive");
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }

    npy_intp nlat = PyArray_DIM(pl, 0);
    const int *pl_data = (const int *)PyArray_DATA(pl);

    npy_intp sum_pl = 0;
    for (npy_intp j = 0; j < nlat; ++j) {
        if (pl_data[j] <= 0) {
            PyErr_SetString(PyExc_ValueError, "pl contains non-positive values");
            Py_DECREF(values);
            Py_DECREF(pl);
            return NULL;
        }
        sum_pl += pl_data[j];
    }

    if (PyArray_DIM(values, 0) != sum_pl) {
        PyErr_SetString(PyExc_ValueError, "values length must equal sum(pl)");
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }

    npy_intp out_dims[2] = {nlat, (npy_intp)nlon};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_FLOAT64);
    if (!out) {
        Py_DECREF(values);
        Py_DECREF(pl);
        return NULL;
    }

    const double *in_data = (const double *)PyArray_DATA(values);
    double *out_data = (double *)PyArray_DATA(out);

    npy_intp ptr = 0;
    for (npy_intp j = 0; j < nlat; ++j) {
        const int src_n = pl_data[j];
        const double *src = in_data + ptr;
        double *dst = out_data + j * nlon;

        int has_missing = 0;
        for (int k = 0; k < src_n; ++k) {
            if (is_missing(src[k], missval)) {
                has_missing = 1;
                break;
            }
        }

        int rc = nearest ? regularize_row_nearest(src, src_n, dst, nlon, has_missing, missval)
                         : regularize_row_linear(src, src_n, dst, nlon, has_missing, missval);
        if (rc != 0) {
            Py_DECREF(values);
            Py_DECREF(pl);
            Py_DECREF(out);
            PyErr_SetString(PyExc_RuntimeError, "row interpolation failed");
            return NULL;
        }
        ptr += src_n;
    }

    Py_DECREF(values);
    Py_DECREF(pl);
    return (PyObject *)out;
}

static PyMethodDef methods[] = {
    {"regularize_values", py_regularize_values, METH_VARARGS,
     "regularize_values(values, pl, missval, nearest, nlon) -> 2D float64 array"},
    {"reduced_row", py_reduced_row, METH_VARARGS,
     "reduced_row(pl, xfirst, xlast) -> (npoints, ilon_first, ilon_last)"},
    {"is_global", py_is_global, METH_VARARGS,
     "is_global(np, pl, xfirst, xlast) -> bool"},
    {"infer_nlon", py_infer_nlon, METH_VARARGS,
     "infer_nlon(pl, np, xfirst, xlast) -> output nlon"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "gaussregular C core",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__core(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
