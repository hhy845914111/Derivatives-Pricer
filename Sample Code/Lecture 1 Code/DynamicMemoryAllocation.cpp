#include "stdafx.h"
#include "DynamicMemoryAllocation.h"

float **matrix_fp32(int nrow, int ncol)
// allocate a single precision matrix using malloc
{
	int i;
	float **m;
	// allocate pointers to rows 
	m = (float **)malloc((size_t)((nrow + 1) * sizeof(float*)));
	// allocate the necessary memory and set pointers to the rows 
	m[0] = (float *)malloc((size_t)((nrow + 1)*(ncol + 1) * sizeof(float)));
	for (i = 1; i <= nrow; i++) m[i] = m[i - 1] + (ncol + 1);
	// return pointer to array of pointers to rows
	return m;
}

void free_matrix_fp32(float **m)
// free a single precision matrix allocated by matrix_f64() 
{
	free(m[0]);
	free(m);
}
