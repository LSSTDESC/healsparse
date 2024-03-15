#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

static PyMethodDef healsparse_lib_methods[] = {
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef healsparse_lib_module = {PyModuleDef_HEAD_INIT, "_healsparse_lib", NULL, -1,
                                                   healsparse_lib_methods};

PyMODINIT_FUNC PyInit__healsparse_lib(void) {
    import_array();
    return PyModule_Create(&healsparse_lib_module);
}
