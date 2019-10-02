#include "malloc.h"

#include "DynamicMemoryAllocation.h"

double **matrix_fp64(int nrow, int ncol)
// allocate a double precision matrix using malloc
{
	int i;
	double **m;
	// allocate pointers to rows 
	m = (double **)malloc((size_t)((nrow + 1) * sizeof(double*)));
	// allocate the necessary memory and set pointers to the rows 
	m[0] = (double *)malloc((size_t)((nrow + 1)*(ncol + 1) * sizeof(double)));
	for (i = 1; i <= nrow; i++) m[i] = m[i - 1] + (ncol + 1);
	// return pointer to array of pointers to rows
	return m;
}

void free_matrix_fp64(double **m)
// free a double precision matrix allocated by matrix_f64() 
{
	free(m[0]);
	free(m);
}

unsigned int **uint_matrix(int nrl, int nrh, int ncl, int nch)
/* allocate an unsigned int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	int i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
	unsigned int **m;

	/* allocate pointers to rows */
	m = (unsigned int **)malloc((size_t)((nrow + NR_END) * sizeof(unsigned int*)));
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl] = (unsigned int *)malloc((size_t)((nrow*ncol + NR_END) * sizeof(unsigned int)));
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for (i = nrl + 1; i <= nrh; i++) m[i] = m[i - 1] + ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

void free_uint_matrix(unsigned int **m, int nrl, int nrh, int ncl, int nch)
/* free a double matrix allocated by dmatrix() */
{
	free((FREE_ARG)(m[nrl] + ncl - NR_END));
	free((FREE_ARG)(m + nrl - NR_END));
}

int **int_matrix(int nrl, int nrh, int ncl, int nch)
/* allocate an unsigned int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	int i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
	int **m;

	/* allocate pointers to rows */
	m = (int **)malloc((size_t)((nrow + NR_END) * sizeof(int*)));
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl] = (int *)malloc((size_t)((nrow*ncol + NR_END) * sizeof(int)));
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for (i = nrl + 1; i <= nrh; i++) m[i] = m[i - 1] + ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

void free_int_matrix(int **m, int nrl, int nrh, int ncl, int nch)
/* free a int matrix allocated by int_matrix() */
{
	free((FREE_ARG)(m[nrl] + ncl - NR_END));
	free((FREE_ARG)(m + nrl - NR_END));
}

