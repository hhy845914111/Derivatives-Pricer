
//    These are required for dynamic memory allocation, Numerical Recipes in C
#define NR_END 1
#define FREE_ARG char*

double **matrix_fp64(int nrow, int ncol);
void free_matrix_fp64(double **m);
int **int_matrix(int nrl, int nrh, int ncl, int nch);
void free_int_matrix(int **m, int nrl, int nrh, int ncl, int nch);
double ***d3Tensor(int row, int col, int ndep);
void free_d3Tensor(double ***t);