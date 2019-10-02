#pragma once

#include "matrix.h"
#include <cuda_runtime.h>

template <typename T>
inline T* cuda_matrix_malloc(size_t m, size_t n)
{
	T* _tmp(nullptr);

	cudaMalloc((void**)_tmp, m * n * sizeof(T));
	
	return _tmp;
}


template <typename T>
inline void cuda_matrix_free(T*& mat)
{
	cudaFree(mat);

	mat = nullptr;
}

template <typename T, cudaMemcpyKind copy_kind>
inline void cuda_matrix_memcpy(T* to_mat, T* from_mat, size_t m, size_t n)
{
	cudaMemcpy(to_mat, from_mat, m * n * sizeof(T), copy_kind);
}

/* how to get_access to matrix elegently
__global__ void test(double* mat, size_t m, size_t n)
{
	auto mat_get = [mat] (size_t m, size_t n)-> double& { return mat[m + n]; };
	mat_get(2, 3) = 6;
}
*/