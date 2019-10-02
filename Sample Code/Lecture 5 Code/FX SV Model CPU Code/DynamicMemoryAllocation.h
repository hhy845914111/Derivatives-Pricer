
//    These are required for dynamic memory allocation, Numerical Recipes in C
#define NR_END 1
#define FREE_ARG char*

double **matrix_fp64(int nrow, int ncol);
void free_matrix_fp64(double **m);
unsigned int **uint_matrix(int nrl, int nrh, int ncl, int nch);
void free_uint_matrix(unsigned int **m, int nrl, int nrh, int ncl, int nch);
int **int_matrix(int nrl, int nrh, int ncl, int nch);
void free_int_matrix(int **m, int nrl, int nrh, int ncl, int nch);