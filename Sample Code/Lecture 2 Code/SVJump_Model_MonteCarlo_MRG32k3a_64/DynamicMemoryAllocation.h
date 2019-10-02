
//    These are required for dynamic memory allocation, Numerical Recipes in C
#define NR_END 1
#define FREE_ARG char*

double **dmatrix(int nrl, int nrh, int ncl, int nch) ;
void free_dmatrix(double **m, int nrl, int nrh, int ncl, int nch) ;
unsigned int **uint_matrix(int nrl, int nrh, int ncl, int nch);
void free_uint_matrix(unsigned int **m, int nrl, int nrh, int ncl, int nch);
