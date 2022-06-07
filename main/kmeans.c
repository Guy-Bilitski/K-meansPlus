#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <math.h>
#include <ctype.h>

void init_new_centroids(Py_ssize_t dim, Py_ssize_t k, PyObject *new_centroids);
PyObject * kmeans(PyObject *data_points, PyObject *centroids, int maxiter, double epsilon);
double max_distance_between_centroids(PyObject *old_centroids, PyObject *new_centroids);
void kmeans_iteration(PyObject *data_points , PyObject *centroids, PyObject *new_centroids);
Py_ssize_t find_closest_centroid(PyObject *vector, PyObject *centroids);
void initarray(PyObject *matrix);
int checkForZeros(int k, int dim, double **centroids);
void print_pymatrix(PyObject *matrix);
void print_centroids(PyObject *matrix);
void fix_final_centroids_matrix(PyObject *matrix);
void add_sum_entry_to_centroids(PyObject *centroids);
void checkForError();


PyObject * kmeans(PyObject *data_points, PyObject *centroids, int maxiter, double epsilon)
{
    PyObject *new_centroids, *temp;
    Py_ssize_t dim;
    Py_ssize_t k;
    int iter;
    double maxd;

    //Setting variables
    dim = PyList_Size(PyList_GetItem(data_points, 0));
    k = PyList_Size(centroids);
    maxiter = maxiter == -1 ? INT_MAX: maxiter;
    new_centroids = PyList_New(k);
    init_new_centroids(dim, k, new_centroids);
    add_sum_entry_to_centroids(centroids);

    for (iter=0; iter < maxiter; iter++) {
        kmeans_iteration(data_points, centroids, new_centroids);
        maxd = max_distance_between_centroids(centroids, new_centroids);

        temp = centroids;
        centroids = new_centroids;
        new_centroids = temp;
        
        if (maxd < epsilon) {
            break;
        }
        initarray(new_centroids);
    }
    fix_final_centroids_matrix(centroids);
    checkForError();

    return centroids;
}

void checkForError(){
    if (PyErr_Occurred() != NULL){
        printf("An Error Has Occurred\n");
        exit(1);
    }
}

void add_sum_entry_to_centroids(PyObject *centroids){
    Py_ssize_t i, k;
    PyObject *cent;
    k = PyList_Size(centroids);
    for (i=0; i<k; i++){
        cent = PyList_GetItem(centroids, i);
        if (PyList_Append(cent, PyFloat_FromDouble(1.))){
            printf("An Error Has Occurred\n");
            exit(1);
        }
    }
}

void init_new_centroids(Py_ssize_t dim, Py_ssize_t k, PyObject *new_centroids){
    Py_ssize_t i, j;
    PyObject * temp;

    if (new_centroids == NULL){
        printf("An Error Has Occurred\n");
        exit(1);
    }

    for (i=0; i < k; i++) {
        temp = PyList_New(dim+1);
        PyList_SetItem(new_centroids, i, temp);
        for (j=0; j <= dim; j++){
            PyList_SetItem(temp, j, PyFloat_FromDouble(0.));
        }
    }
}

double max_distance_between_centroids(PyObject *old_centroids, PyObject *new_centroids) {
    Py_ssize_t dim, k;
    Py_ssize_t i,j;
    PyObject *old_c, *new_c;

    double max_value = DBL_MIN;
    double current_value;
    double old_denomin, new_denomin;

    dim = PyList_Size(PyList_GetItem(new_centroids, 0))-1;
    k = PyList_Size(new_centroids);

    for (i=0; i < k; i++) {
        current_value = 0.;
        old_c = PyList_GetItem(old_centroids, i);
        new_c = PyList_GetItem(new_centroids, i);
        for (j=0; j < dim; j++) {
            old_denomin = PyFloat_AsDouble(PyList_GetItem(old_c, dim));
            new_denomin = PyFloat_AsDouble(PyList_GetItem(new_c, dim));
            if (old_denomin == 0 || new_denomin == 0){
                printf("An Error Has Occurred\n");
                exit(1);
            }
            current_value += pow(
                (PyFloat_AsDouble(PyList_GetItem(old_c, j)) / old_denomin) - 
                (PyFloat_AsDouble(PyList_GetItem(new_c, j)) / new_denomin), 2
                );
        }
        if (current_value > max_value) {
            max_value = current_value;
        }
    }

    return sqrt(max_value);
}


void kmeans_iteration(PyObject *data_points , PyObject *centroids, PyObject *new_centroids) {
    Py_ssize_t dim, n;
    Py_ssize_t i,j;
    Py_ssize_t closet_centroid_index;
    PyObject *closest_centroid, *current_vector;
    double entry_value;
    
    dim = PyList_Size(PyList_GetItem(data_points, 0));
    n = PyList_Size(data_points);


    for (i=0; i<n; i++) {
        current_vector = PyList_GetItem(data_points, i);
        closet_centroid_index = find_closest_centroid(current_vector, centroids);
        closest_centroid = PyList_GetItem(new_centroids, closet_centroid_index);

        for (j = 0; j < dim; j++) {
            entry_value = PyFloat_AsDouble(PyList_GetItem(closest_centroid, j)) + PyFloat_AsDouble(PyList_GetItem(current_vector, j));
            if (PyList_SetItem(closest_centroid, j, PyFloat_FromDouble(entry_value))){
                printf("An Error Has Occurred\n");
                exit(1);
            }
        }
        entry_value = PyFloat_AsDouble(PyList_GetItem(closest_centroid, dim)) + 1.;
        PyList_SetItem(closest_centroid, dim, PyFloat_FromDouble(entry_value));
    }
}


Py_ssize_t find_closest_centroid(PyObject *vector, PyObject *centroids) {
    Py_ssize_t dim, k;
    Py_ssize_t i,j;
    PyObject * centroid_i;
    dim = PyList_Size(PyList_GetItem(centroids, 0))-1;
    k = PyList_Size(centroids);

    double closest_value = DBL_MAX;
    double current_value = 0;
    double denomin;
    Py_ssize_t closest_index = -1;

    for (i = 0; i < k; i++) {
        current_value = 0;
        centroid_i = PyList_GetItem(centroids, i);
        for (j=0; j < dim; j++) {
            denomin = PyFloat_AsDouble(PyList_GetItem(centroid_i, dim));
            if (denomin == 0){
                printf("An Error Has Occurred\n");
                exit(1);
            }
            current_value += pow(
                PyFloat_AsDouble(PyList_GetItem(vector, j)) - 
                PyFloat_AsDouble(PyList_GetItem(centroid_i, j)) / denomin, 2);
        }
        if (current_value < closest_value) {
            closest_value = current_value;
            closest_index = i;
        }
    }

    return closest_index;
}


void initarray(PyObject *junk_centroids){
    Py_ssize_t i,j;
    Py_ssize_t dim, k;
    PyObject * centroid_i;
    dim = PyList_Size(PyList_GetItem(junk_centroids, 0))-1;
    k = PyList_Size(junk_centroids);

    for (i=0; i<k; i++){
        centroid_i = PyList_GetItem(junk_centroids, i);
        for (j=0; j<dim+1; j++){
            PyList_SetItem(centroid_i, j, PyFloat_FromDouble(0.));
        }
    }
}



void print_pymatrix(PyObject *matrix){
    Py_ssize_t i, j;
    Py_ssize_t dim;
    Py_ssize_t k;
    PyObject *pyfloat;

    k = PyList_Size(matrix);
    dim = PyList_Size(PyList_GetItem(matrix, 0));
    for (i=0; i < k; i++) {
        for (j=0; j < dim; j++){
            pyfloat = PyList_GetItem(PyList_GetItem(matrix, i), j);
            printf("%f,",PyFloat_AsDouble(pyfloat));
        }
        printf("\n");
    }
}

void print_centroids(PyObject *matrix){
    Py_ssize_t i, j;
    Py_ssize_t dim;
    Py_ssize_t k;
    PyObject *pyfloat;
    double num;

    k = PyList_Size(matrix);
    dim = PyList_Size(PyList_GetItem(matrix, 0));
    for (i=0; i < k; i++) {
        num = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(matrix, i), dim-1));
        for (j=0; j < dim-1; j++){
            pyfloat = PyList_GetItem(PyList_GetItem(matrix, i), j);
            printf("%f,",PyFloat_AsDouble(pyfloat)/num);
        }
        printf("\n");
    }
}

void fix_final_centroids_matrix(PyObject *matrix){
    Py_ssize_t i, j;
    Py_ssize_t dim;
    Py_ssize_t k;
    PyObject *pyfloat, *sublist;
    double num, entry;

    k = PyList_Size(matrix);
    dim = PyList_Size(PyList_GetItem(matrix, 0));

    for (i=0; i < k; i++) { //dividing each entry by sum
        num = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(matrix, i), dim-1));
        for (j=0; j < dim-1; j++){
            pyfloat = PyList_GetItem(PyList_GetItem(matrix, i), j);
            entry = PyFloat_AsDouble(pyfloat)/num;
            PyList_SetItem(PyList_GetItem(matrix, i), j, PyFloat_FromDouble(entry));
        }
    }

    for (i=0; i < k; i++) { //popping the sum entry
        sublist = PyList_GetSlice(PyList_GetItem(matrix, i), 0, dim-1);
        PyList_SetItem(matrix, i, sublist);
    }
}


static PyObject* kmeans_capi(PyObject *self, PyObject *args){
    PyObject *data_points;
    PyObject *init_centroids;
    int maxiter;
    double epsilon;

    if (!(PyArg_ParseTuple(args, "OOid", &data_points, &init_centroids, &maxiter, &epsilon))){
        printf("An Error Has Occurred\n");
        exit(1);
    }

    return Py_BuildValue("O", kmeans(data_points, init_centroids, maxiter, epsilon));
}

static PyMethodDef capiMethods[] = {
    {
        "fit",
        (PyCFunction) kmeans_capi,
        METH_VARARGS,
        PyDoc_STR("Args:\nData-Points: ndarray,\nCentroids: list[list]\nmaxiter: int\nepsilon: float")
    },
    {
        NULL, NULL, 0, NULL
    }
};

static struct PyModuleDef moduledef =
{
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    "My Kmeans Module",
    -1,
    capiMethods
};


PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    PyObject *m;
    m=PyModule_Create(&moduledef);
    if (!m) {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    return m;
}