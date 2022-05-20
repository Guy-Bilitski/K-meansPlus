#define PY_SSIZE_T_CLEAN
#include <Python.h>

static double geo_c(double z, int n){
    return 1.;
}

static PyObject* kmeans_capi(PyObject *self, PyObject *args){ //The called function, extracts the input and calls kmeans C func.
    double z;
    int n;
    if(!PyArg_ParseTuple(args, "di", &z, &n)) {
        return NULL;
    }
    return Py_BuildValue("d", geo_c(z,n));
}

static PyMethodDef capiMethods[] = {
    {"geo",
    (PyCFunction) kmeans_capi,
    METH_VARARGS,
    PyDoc_STR("My KMeans Documentation")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "kmeansC",
    NULL,
    -1,
    capiMethods
}

PyMODINIT_FUNC Py