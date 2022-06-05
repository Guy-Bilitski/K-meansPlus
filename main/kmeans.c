#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <math.h>
#include <ctype.h>

void init_new_centroids(Py_ssize_t dim, Py_ssize_t k, PyObject *new_centroids);
int kmeans(PyObject *data_points, PyObject *centroids, int maxiter, double epsilon);
double max_distance_between_centroids(PyObject *old_centroids, PyObject *new_centroids);
void kmeans_iteration(PyObject *data_points , PyObject *centroids, PyObject *new_centroids);
Py_ssize_t find_closest_centroid(PyObject *vector, PyObject *centroids);
void initarray(PyObject *matrix);
int write_result(int k, int dim, char *outname, double **data);
int checkForZeros(int k, int dim, double **centroids);
void print_pymatrix(PyObject *matrix);


int kmeans(PyObject *data_points, PyObject *centroids, int maxiter, double epsilon)
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
    

    for (iter=0; iter < maxiter; iter++) {
        kmeans_iteration(data_points, centroids, new_centroids);
        maxd = max_distance_between_centroids(centroids, new_centroids);

        printf("\nC:\n");
        print_pymatrix(centroids);
        printf("\nNC:\n");
        print_pymatrix(new_centroids);

        temp = centroids;
        centroids = new_centroids;
        new_centroids = temp;

        printf("\nC after:\n");
        print_pymatrix(centroids);
        printf("\nNC after:\n");
        print_pymatrix(new_centroids);
        
        if (maxd < epsilon) {
            break;
        }
        initarray(new_centroids);
    }
    return 0;
}

void init_new_centroids(Py_ssize_t dim, Py_ssize_t k, PyObject *new_centroids){
    Py_ssize_t i, j;
    PyObject * temp;

    if (new_centroids == NULL){
        printf("An Error Has Occurred\n");
        exit(1);
    }

    for (i=0; i < k; i++) {
        temp = PyList_New(dim);
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

    dim = PyList_Size(PyList_GetItem(new_centroids, 0))-1;
    k = PyList_Size(new_centroids);

    for (i=0; i < k; i++) {
        current_value = 0.;
        old_c = PyList_GetItem(old_centroids, i);
        new_c = PyList_GetItem(new_centroids, i);
        for (j=0; j < dim; j++) {
            current_value += pow(
                PyFloat_AsDouble(PyList_GetItem(old_c, j)) / PyFloat_AsDouble(PyList_GetItem(old_c, dim)) - 
                PyFloat_AsDouble(PyList_GetItem(new_c, j)) / PyFloat_AsDouble(PyList_GetItem(new_c, dim)), 2
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

int checkForZeros(int k, int dim, double **centroids){
    int i;
    for (i = 0; i < k; i++) {
        if (centroids[i][dim] == 0){
            return 1;
        }
    }
    return 0;
}

Py_ssize_t find_closest_centroid(PyObject *vector, PyObject *centroids) {
    Py_ssize_t dim, k;
    Py_ssize_t i,j;
    PyObject * centroid_i;
    dim = PyList_Size(PyList_GetItem(centroids, 0))-1;
    k = PyList_Size(centroids);

    double closest_value = DBL_MAX;
    double current_value = 0;
    Py_ssize_t closest_index = -1;

    for (i = 0; i < k; i++) {
        current_value = 0;
        centroid_i = PyList_GetItem(centroids, i);
        for (j=0; j < dim; j++) {
            current_value += pow(
                PyFloat_AsDouble(PyList_GetItem(vector, j)) - 
                PyFloat_AsDouble(PyList_GetItem(centroid_i, j)) / PyFloat_AsDouble(PyList_GetItem(centroid_i, dim)), 2);
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



int write_result(int k, int dim, char *outname, double **data){
    FILE *ofp;
    int i,j;
    ofp = fopen(outname, "w");
    if (ofp == NULL) {
        return 1;
    }
    for (i = 0; i<k; i++){
        for (j=0; j<dim; j++){
            fprintf(ofp, "%.4f",data[i][j]/data[i][dim]);
            if (j < dim-1){
                fprintf(ofp, ",");
            } else {
                fprintf(ofp, "\n");
            }
        }
    }
    fclose(ofp);
    return 0;
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


static PyObject* kmeans_capi(PyObject *self, PyObject *args){
    PyObject *data_points;
    PyObject *init_centroids;
    int maxiter;
    double epsilon;

    if (!PyArg_ParseTuple(args, "OOid", &data_points, &init_centroids, &maxiter, &epsilon)){
        printf("An Error Has Occurred303\n");
        exit(1);
    }

    return Py_BuildValue("O", kmeans(data_points, init_centroids, maxiter, epsilon));
}

static PyMethodDef capiMethods[] = {
    {
        "getKmeans",
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
        printf("An Error Has Occurred336\n");
        exit(1);
    }
    return m;
}