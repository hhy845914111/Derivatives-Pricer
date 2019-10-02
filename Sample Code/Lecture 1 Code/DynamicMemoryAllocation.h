
//    These are required for dynamic memory allocation, Numerical Recipes in C
#define NR_END 1
#define FREE_ARG char*

float **matrix_fp32(int nrow, int ncol);
void free_matrix_fp32(float **m);
