
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdafx.h"
#include "DynamicMemoryAllocation.h"

void MatrixMultiply(float **a, float **b, float **c, int nrows, int nwa, int ncols) {
	int i, j, k;
	for (i = 0; i < nrows; i++) {
		for (j = 0; j < ncols; j++) {
			c[i][j] = 0.0;
			for (k = 0; k < nwa; k++) c[i][j] += a[i][k] * b[k][j];
		}
	}
}

__global__ void MatrixMultKernel(float *a, float *b, float *c, int nrows, int nwa, int ncols)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < nrows) && (j < ncols)) {
		int k, ijc = i*ncols + j;
		c[ijc] = 0.0;
		for (k = 0; k < nwa; k++) {
			c[ijc] += a[i*nwa + k] * b[k*ncols + j];
		}
	}
}

int main()
{
	int i, j, k;
	int	nrows = 128, nwa = 500, ncols = 256;
	float **a, **b, **c;
	float *h_a, *h_b, *h_c;
	float *d_a, *d_b, *d_c;
	dim3 threadsPerBlock(16, 16);
	double time0, time1, time2, time3;
	struct _timeb timebuffer;

	a = matrix_fp32(nrows, nwa);
	b = matrix_fp32(nwa, ncols);
	c = matrix_fp32(nrows, ncols);

	h_a = (float *)malloc((nrows*nwa) * sizeof(float));
	h_b = (float *)malloc((nwa*ncols) * sizeof(float));
	h_c = (float *)malloc((nrows*ncols) * sizeof(float));

	printf("  Now running matrix multiplication on CPU sequentially \n");

	cudaMalloc((void **)&d_a, (nrows*nwa) * sizeof(float));
	cudaMalloc((void **)&d_b, (nwa*ncols) * sizeof(float));
	cudaMalloc((void **)&d_c, (nrows*ncols) * sizeof(float));

	for (i = 0; i < nrows; i++) {
		for (j = 0; j < nwa; j++) a[i][j] = i+j;
	}
	for (i = 0; i < nwa; i++) {
		for (j = 0; j < ncols; j++) b[i][j] = 2*i + j;
	}

	_ftime64_s(&timebuffer);
	time0 = timebuffer.time + timebuffer.millitm / 1000.0;

	MatrixMultiply(a, b, c, nrows, nwa, ncols);

	_ftime64_s(&timebuffer);
	time1 = timebuffer.time + timebuffer.millitm / 1000.0;
	time1 = time1 - time0;

	printf(" Execution time on CPU for matrix multiplication: %8.3f \n", time1);

	i = 0; j = 0;
	printf(" c[%i][%i] = %f \n", i, j, c[i][j]);
	i = nrows-1; j = ncols-1;
	printf(" c[%i][%i] = %f \n", i, j, c[i][j]);

	printf("\n  Now rerun the calculations on GPU \n");

	for (i = 0; i < nrows; i++) {
		for (j = 0; j < nwa; j++) {
			k = i*nwa + j;
			h_a[k] = a[i][j];
		}
	}
	for (i = 0; i < nwa; i++) {
		for (j = 0; j < ncols; j++) {
			k = i*ncols + j;
			h_b[k] = b[i][j];
		}
	}

	_ftime64_s(&timebuffer);
	time0 = timebuffer.time + timebuffer.millitm / 1000.0;

//	Copy vectors (matrices) h_a and h_b to device memory d_a, d_b
	cudaMemcpy(d_a, h_a, (nrows*nwa) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, (nwa*ncols) * sizeof(int), cudaMemcpyHostToDevice);

	time1 = timebuffer.time + timebuffer.millitm / 1000.0;

	dim3 numBlocks(1 + (nrows + 1) / threadsPerBlock.x, 1 + (ncols + 1) / threadsPerBlock.y);
// Launch a kernel on the GPU with one thread for each element.
	MatrixMultKernel <<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, nrows, nwa, ncols);

	time2 = timebuffer.time + timebuffer.millitm / 1000.0;

	cudaGetLastError();
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, (nrows*ncols)*sizeof(int), cudaMemcpyDeviceToHost);

	time3 = timebuffer.time + timebuffer.millitm / 1000.0;

	time3 = time3 - time2;
	time2 = time2 - time1;
	time1 = time1 - time0;

	printf(" Execution time on GPU for matrix multiplication: %8.3f \n", time2);
	time0 = time1 + time2 + time3;
	printf(" Time for memory copies: %8.3f  %8.3f  and total time for GPU processing %8.3f \n",
		time1, time3, time0);

//	for (i = 0; i < 10; i++) printf(" a, b, c row %4i  %4d %4d %4d \n", i, a[i], b[i], c[i]);
//	i = Size - 1;
//	printf(" a, b, c row %4i  %4d %4d %4d \n", i, a[i], b[i], c[i]);

	i = 0; j = 0; 
	k = i*ncols + j;
	printf(" c[%i][%i] = %f \n", i, j, h_c[k]);
	i = nrows - 1; j = ncols - 1;
	k = i*ncols + j;
	printf(" c[%i][%i] = %f \n", i, j, h_c[k]);

	free_matrix_fp32(a);
	free_matrix_fp32(b);
	free_matrix_fp32(c);

	free(h_a);
	free(h_b);
	free(h_c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cudaDeviceReset();

	return 0;
}
