#pragma once
#include "pch.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand_kernel.h>

namespace _rand_cuda_
{
	constexpr double a12(1403580), a13n(810728);
	constexpr double a21(527612), a23n(1370589);
	constexpr double m1(4294967087), m2(4294944443);
	constexpr double norm_(2.328306549295728e-10);
}


template <typename ResultType, typename SeedsType, typename... Args>
inline cudaError_t cuda_sim(void(*func)(ResultType*, SeedsType*, size_t, Args...), ResultType *results, SeedsType *seeds, size_t n_seeds, size_t n_results, Args ...args)
{
	SeedsType *d_seeds(nullptr);
	ResultType *d_results(nullptr);
	cudaError_t cuda_status;

	cuda_status = cudaSetDevice(0);
	if (cuda_status != cudaSuccess) {
		goto Error;
	}

	cuda_status = cudaMalloc((void**)&d_seeds, n_seeds * SEEDS_SIZE * sizeof(SeedsType));
	if (cuda_status != cudaSuccess) {
		goto Error;
	}

	cuda_status = cudaMalloc((void**)&d_results, n_results * sizeof(ResultType));
	if (cuda_status != cudaSuccess) {
		goto Error;
	}

	cuda_status = cudaMemcpy(d_seeds, seeds,  n_seeds * SEEDS_SIZE * sizeof(SeedsType), cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		goto Error;
	}

	func <<<n_results / GPU_THREAD_COUNT, GPU_THREAD_COUNT>>> (d_results, d_seeds, n_results, args...);

	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		goto Error;
	}

	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess) {
		goto Error;
	}

	cuda_status = cudaMemcpy(results, d_results, n_results * sizeof(ResultType), cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess) {
		goto Error;
	}

Error:
	cudaFree(d_seeds);
	cudaFree(d_results);

	return cuda_status;
}


__device__ double rand01_cuda(double *seed);

__device__ double randn_cuda(double *seed);

__global__ void sim_exp_cuda(double *result_ar, double *seeds, size_t n_total_sim, double u);

__global__ void sim_knock_out_cuda(
	double *result_ar, double *seeds, size_t n_total_sim,
	OptionType opt_type, double S0, double r, double sigma, double K, double B, double T, unsigned int n);