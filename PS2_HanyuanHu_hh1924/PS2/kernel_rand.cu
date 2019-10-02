#pragma once
#include "pch.h"
#include "kernel_rand.cuh"

#include <stdio.h>
#include <windows.h>

__device__ double rand01_cuda(double *dseed)
{
	using namespace _rand_cuda_;

	int k;
	double p1, p2;

	p1 = a12 * dseed[1] - a13n * dseed[2];
	k = p1 / m1;
	p1 -= k * m1;
	if (p1 < 0.0) p1 += m1;
	dseed[2] = dseed[1]; dseed[1] = dseed[0]; dseed[0] = p1;

	p2 = a21 * dseed[3] - a23n * dseed[5];
	k = p2 / m2;
	p2 -= k * m2;
	if (p2 < 0.0) p2 += m2;
	dseed[5] = dseed[4]; dseed[4] = dseed[3]; dseed[3] = p2;

	if (p1 <= p2) return ((p1 - p2 + m1)*norm_);
	else return ((p1 - p2)*norm_);
}


__device__ double randn_cuda(double *dseed)
{
	return normcdfinvf(rand01_cuda(dseed));
}


__global__ void sim_exp_cuda(double *result_ar, double *seeds, size_t n_total_sim, double u)
{
	size_t loc(blockIdx.x * blockDim.x + threadIdx.x);
	if (loc < n_total_sim)
	{
		result_ar[loc] = -u * log(1 - rand01_cuda(seeds + SEEDS_SIZE * loc));
	}
		
}


__global__ void sim_knock_out_cuda(
	double *result_ar, double *seeds, size_t n_total_sim,
	OptionType opt_type, double S0, double r, double sigma, double K, double B, double T, unsigned int n)
{
	size_t loc(blockIdx.x * blockDim.x + threadIdx.x);
	if (loc < n_total_sim)
	{
		const double dt(T / n);
		double S_a(S0), S_b;
		unsigned int step(0);
		double pay_off;
		sigma /= sqrt(n * T);

		while (step < n)
		{
			S_b = S_a + r * S_a * dt + sigma * S_a * randn_cuda(seeds + loc * SEEDS_SIZE);

			if (S_b > B && opt_type == Call)
			{
				result_ar[loc] = 0.0;
				return;
			}
			else if (S_b < B && opt_type == Put)
			{
				result_ar[loc] = 0.0;
				return;
			}

			S_a = S_b;
			++step;
		}

		if (opt_type == Call)
		{
			pay_off = S_b - K;
		}
		else
		{
			pay_off = K - S_b;
		}

		result_ar[loc] = pay_off > 0 ? pay_off : 0.0;
	}
}
