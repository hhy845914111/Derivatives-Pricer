#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>

void VectorAdd(int *a, int *b, int *c, int n) {
	int i;
	for (i = 0; i < n; i++)
		c[i] = a[i] + b[i];
}

__global__ void VectorAddKernel(int *a, int *b, int *c, int n)
{
    int i = threadIdx.x;
 //   c[i] = a[i] + b[i];
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

int main()
{
    int i, Size = 1024;
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;

	a = (int *)malloc(Size * sizeof(int));
	b = (int *)malloc(Size * sizeof(int));
	c = (int *)malloc(Size * sizeof(int));

	cudaMalloc((void **)&d_a, Size * sizeof(int));
	cudaMalloc((void **)&d_b, Size * sizeof(int));
	cudaMalloc((void **)&d_c, Size * sizeof(int));

	for (i = 0; i < Size; i++) {
		a[i] = i;
		b[i] = 2*i;
		c[i] = 0;
	}

	VectorAdd(a, b, c, Size);

	for (i = 0; i < 10; i++) printf(" a, b, c row %4i  %4d %4d %4d \n", i, a[i], b[i], c[i]);
	i = Size - 1;
	printf(" a, b, c row %4i  %4d %4d %4d \n", i, a[i], b[i], c[i]);

	printf("\n   Now rerun the calculations on GPU \n");

//	Copy vectors a and b to device memory d_a, d_b
	cudaMemcpy(d_a, a, Size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, Size * sizeof(int), cudaMemcpyHostToDevice);
	
// Launch a kernel on the GPU with one thread for each element.
	VectorAddKernel <<<1, Size >>>(d_a, d_b, d_c, Size);

	cudaGetLastError();
	cudaDeviceSynchronize();
	cudaMemcpy(c, d_c, Size*sizeof(int), cudaMemcpyDeviceToHost);

	for (i = 0; i < 10;i++) printf(" a, b, c row %4i  %4d %4d %4d \n", i, a[i], b[i], c[i]);
	i = Size - 1;
	printf(" a, b, c row %4i  %4d %4d %4d \n", i, a[i], b[i], c[i]);

	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cudaDeviceReset();

    return 0;
}