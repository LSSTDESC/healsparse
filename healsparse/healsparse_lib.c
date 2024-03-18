#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

uint8_t _count_bits_uint8(uint8_t value) {
    static int has_lut = 0;
    static uint64_t lut[256];

    if (!has_lut) {
        // Make a lookup table.
        for (int i=0; i<256; i++) {
            lut[i] = i - ((i >> 1) & 0x55);
            lut[i] = (lut[i] & 0x33) + ((lut[i] >> 2) & 0x33);
            lut[i] = (lut[i] + (lut[i] >> 4)) & 0x0F;
            lut[i] *= 0x01;
        }
        has_lut = 1;
    }

    return lut[value];
}

PyDoc_STRVAR(sum_bits_uint8_doc,
             "sum_bits_uint8(arr, axis=None)\n"
             "--\n\n"
             "Count and sum the bits in an unsigned 8-bit integer array.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "arr : `np.ndarray`\n"
             "    Array to count bits.  At most 2D.\n"
             "axis : `int`, optional\n"
             "    Axis to sum over.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "bit_sum : `int` or `np.ndarray`\n"
             "    Sum total of bits, or reduced by an axis.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    If array is not of integer type, or dimensionality too large.\n");

static PyObject *sum_bits_uint8(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *array_obj = NULL;
    PyObject *array_arr = NULL;
    PyObject *sum_arr = NULL;

    NpyIter *iter = NULL;
    int64_t sum = 0;

    int axis = NPY_MAXDIMS;
    static char *kwlist[] = {"arr", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O&", kwlist, &array_obj, PyArray_AxisConverter, &axis))
        goto fail;

    // Check that we have an array of type NPY_UINT8.
    if (!PyArray_CheckExact(array_obj) || (PyArray_TYPE((PyArrayObject *)array_obj) != NPY_UINT8)) {
        PyErr_SetString(PyExc_ValueError,
                        "arr must be a numpy array with type np.uint8");
        goto fail;
    }

    // This will flatten the array if no axis is specified.
    array_arr = PyArray_CheckAxis((PyArrayObject *)array_obj, &axis, 0);
    if (array_arr == NULL) goto fail;

    // Do an intitial check if the array is empty.
    if (PyArray_SIZE((PyArrayObject *)array_arr) == 0) {
        sum = 0;
        goto cleanup;
    }

    int ndim = PyArray_NDIM((PyArrayObject *)array_arr);

    // This only works with flattened or 2D data.
    if (ndim > 2) {
        PyErr_SetString(PyExc_ValueError,
                        "arr must be at most 2 dimensions if reducing by an axis.");
        goto fail;
    }

    // One path if it's flattened.
    if (ndim == 1) {
        NpyIter_IterNextFunc *iternext;
        char** dataptr;
        npy_intp *strideptr, *innersizeptr;

        iter = NpyIter_New((PyArrayObject *)array_arr,
                           NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP,
                           NPY_KEEPORDER, NPY_NO_CASTING, NULL);
        if (iter == NULL) goto fail;
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) goto fail;

        dataptr = NpyIter_GetDataPtrArray(iter);
        strideptr = NpyIter_GetInnerStrideArray(iter);
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

        do {
            uint8_t *data = (uint8_t *) *dataptr;
            npy_intp stride = *strideptr;
            npy_intp count = *innersizeptr;

            while (count--) {
                sum += (int64_t) _count_bits_uint8(*data);
                data += stride;
            }

        } while(iternext(iter));

    } else {
        // We have a different code path when collapsing an axis.
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        NpyIter_GetMultiIndexFunc *get_multi_index;
        npy_intp multi_index[NPY_MAXDIMS];

        npy_intp *dims = PyArray_DIMS((PyArrayObject *)array_arr);
        npy_intp new_dims[NPY_MAXDIMS];
        int j = 0;
        for (int i = 0; i < ndim; i++) {
            if (i != axis) {
                new_dims[j++] = dims[i];
            }
        }
        sum_arr = PyArray_ZEROS(ndim - 1, new_dims, NPY_INT64, 0);

        iter = NpyIter_New((PyArrayObject *)array_arr,
                           NPY_ITER_READONLY | NPY_ITER_MULTI_INDEX,
                           NPY_KEEPORDER, NPY_NO_CASTING, NULL);
        if (iter == NULL) goto fail;

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) goto fail;
        get_multi_index = NpyIter_GetGetMultiIndex(iter, NULL);
        if (get_multi_index == NULL) goto fail;

        dataptr = NpyIter_GetDataPtrArray(iter);

        int64_t *sum_data = (int64_t *)PyArray_DATA((PyArrayObject *)sum_arr);
        do {
            uint8_t *data = (uint8_t *) *dataptr;

            get_multi_index(iter, multi_index);
            if (axis == 0) {
                sum_data[multi_index[1]] += (int64_t) _count_bits_uint8(*data);
            } else {
                sum_data[multi_index[0]] += (int64_t) _count_bits_uint8(*data);
            }
        } while(iternext(iter));

        goto cleanup;
    }

 cleanup:
    Py_DECREF(array_arr);
    if (iter != NULL) {
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            iter = NULL;
            goto fail;
        }
    }
    if (sum_arr == NULL) {
        // Turn the sum into a numpy integer.
        sum_arr = PyArray_EMPTY(0, NULL, NPY_INT64, 0);
        int64_t *data = (int64_t *) PyArray_DATA((PyArrayObject *)sum_arr);
        data[0] = sum;
    }

    return PyArray_Return((PyArrayObject *)sum_arr);

 fail:
    Py_XDECREF(array_arr);
    Py_XDECREF(sum_arr);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    return NULL;
}

static PyMethodDef healsparse_lib_methods[] = {
    {"sum_bits_uint8", (PyCFunction)(void (*)(void))sum_bits_uint8,
     METH_VARARGS | METH_KEYWORDS, sum_bits_uint8_doc},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef healsparse_lib_module = {
    PyModuleDef_HEAD_INIT,
    "_healsparse_lib",
    NULL,
    -1,
    healsparse_lib_methods
};

PyMODINIT_FUNC PyInit__healsparse_lib(void) {
    import_array();
    return PyModule_Create(&healsparse_lib_module);
}
