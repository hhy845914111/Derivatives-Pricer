
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "test_utils.h" // personal utils for C++


constexpr auto ARRAY_SIZE = 4096000;
constexpr auto RAND_SEED = 32767;

cudaError_t cuda_vector_mutiply(int *c, const int *const a, const int *const b, size_t size);

void host_vector_multiply(int *c, const int *a, const int *b, size_t size);

__global__ void kernel_multiply(int *c, const int *const a, const int *const b)
{
	c[blockIdx.x] = a[blockIdx.x] * b[blockIdx.x];
}


int main()
{
	srand(RAND_SEED);

	int *a = test_utils::generate_rand_int_arr(ARRAY_SIZE);
	std::cout << "vector a: ";
	test_utils::print_arr(a, 20);

	int *b = test_utils::generate_rand_int_arr(ARRAY_SIZE);
	std::cout << "vector b: ";
	test_utils::print_arr(b, 20);

	int *c = new int[ARRAY_SIZE];
		
    cudaError_t cudaStatus = cuda_vector_mutiply(c, a, b, ARRAY_SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda_vector_mutiply failed!");
        return 1;
    }

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	std::cout << "GPU calculation result: ";
	test_utils::print_arr(c, 20);

	int *d = new int[ARRAY_SIZE];
	std::cout << "CPU calculation result: ";
	host_vector_multiply(d, a, b, ARRAY_SIZE);
	test_utils::print_arr(d, 20);

	std::cout << "Difference between the two: " << test_utils::l2_norm(c, d, ARRAY_SIZE);

	delete[]a;
	delete[]b;
	delete[]c;
	delete[]d;

    return 0;
}


cudaError_t cuda_vector_mutiply(int *c, const int * const a, const int * const b, size_t size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    kernel_multiply<<<size, 1>>>(dev_c, dev_a, dev_b);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

inline void host_vector_multiply(int *c, const int *a, const int *b, size_t size)
{
	size_t count{0};
	while (count < size)
	{
		*c = *a * *b;
		++c;
		++a;
		++b;
		++count;
	}
}