#include "stdafx.h"

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

double ***d3tensor(int nrl, int nrh, int ncl, int nch, int ndl, int ndh)
/* allocate a double 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
	int i, j, nrow = nrh - nrl + 1, ncol = nch - ncl + 1, ndep = ndh - ndl + 1;
	double ***t;

	/* allocate pointers to pointers to rows */
	t = (double ***)malloc((size_t)((nrow + NR_END) * sizeof(float**)));
	t += NR_END;
	t -= nrl;

	/* allocate pointers to rows and set pointers to them */
	t[nrl] = (double **)malloc((size_t)((nrow*ncol + NR_END) * sizeof(double*)));
	t[nrl] += NR_END;
	t[nrl] -= ncl;

	/* allocate rows and set pointers to them */
	t[nrl][ncl] = (double *)malloc((size_t)((nrow*ncol*ndep + NR_END) * sizeof(double)));
	t[nrl][ncl] += NR_END;
	t[nrl][ncl] -= ndl;

	for (j = ncl + 1; j <= nch; j++) t[nrl][j] = t[nrl][j - 1] + ndep;
	for (i = nrl + 1; i <= nrh; i++) {
		t[i] = t[i - 1] + ncol;
		t[i][ncl] = t[i - 1][ncl] + ncol * ndep;
		for (j = ncl + 1; j <= nch; j++) t[i][j] = t[i][j - 1] + ndep;
	}

	/* return pointer to array of pointers to rows */
	return t;
}

void free_d3tensor(double ***t, int nrl, int nrh, int ncl, int nch,
	int ndl, int ndh)
	/* free a double f3tensor allocated by f3tensor() */
{
	free((FREE_ARG)(t[nrl][ncl] + ndl - NR_END));
	free((FREE_ARG)(t[nrl] + ncl - NR_END));
	free((FREE_ARG)(t + nrl - NR_END));
}
