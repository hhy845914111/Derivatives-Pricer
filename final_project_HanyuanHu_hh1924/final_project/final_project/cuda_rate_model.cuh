#pragma once
#include "cuda_runtime.h"
#include "cuda_rand.cuh"
#include "cpu_rate_model.h"
#include "configure.h"

/*
We use the cananical form of two-factor Vasicek Model to reduce parameters.
*/

namespace OIS_model
{
	constexpr double d_lambda_1 = LAMBDA_1;
	constexpr double d_lambda_21 = LAMBDA_21;
	constexpr double d_lambda_2 = LAMBDA_2;

	constexpr double d_delta_0 = DELTA_0;
	constexpr double d_delta_1 = DELTA_1;
	constexpr double d_delta_2 = DELTA_2;


	__device__ double _cuda_C1_(double tao)
	{
		double exp_1 = exp(-d_lambda_1 * tao);

		if (d_lambda_1 == d_lambda_2)
		{
			double lambda_delta = d_lambda_21 * d_delta_2 / d_lambda_1;
			return (d_delta_1 - lambda_delta) * (1 - exp_1) / d_lambda_1 + lambda_delta * tao * exp_1;
		}
		else
		{
			double lambda_delta = d_lambda_21 * d_delta_2 / d_lambda_2;
			return (d_delta_1 - d_lambda_21 * d_delta_2 / d_lambda_2) * (1 - exp_1) / d_lambda_1 + lambda_delta / (d_lambda_1 - d_lambda_2) * (exp(-d_lambda_2 * tao) - exp_1);
		}
	}

	__device__ double _cuda_C2_(double tao)
	{
		return d_delta_2 * (1 - exp(-d_lambda_2 * tao)) / d_lambda_2;
	}

	__device__ double _cuda_dA_(double tao)
	{
		double C1 = _cuda_C1_(tao), C2 = _cuda_C2_(tao);
		return -0.5 * (C1 * C1 + C2 * C2) + d_delta_0;
	}

	__device__ double _cuda_A_(double tao, unsigned int N = OIS_model::integrate_N)
	{
		double A = 0.0, h = tao / N;

		for (unsigned int i = 0; i < N; ++i)
		{
			A += h / 2 * (_cuda_dA_(tao) + _cuda_dA_(tao + h));
		}

		return A;
	}

	__device__ double cuda_P(double y1_0, double y2_0, double tao)
	{
		return exp(-_cuda_A_(tao) - _cuda_C1_(tao) * y1_0 - _cuda_C2_(tao) * y2_0);
	}

	__device__ double cuda_y(double y1_0, double y2_0, double tao)
	{
		if (tao > 0.0)
		{
			return (_cuda_A_(tao) + _cuda_C1_(tao) * y1_0 + _cuda_C2_(tao) * y2_0) / tao;
		}
		else
		{
			return 0.0;
		}
	}
}


namespace LIBOR_model
{
	__device__ double cuda_y(double y1_0, double y2_0, double tao, double* seeds)
	{
		return OIS_model::cuda_y(y1_0, y2_0, tao) * (1 + LIBOR_model::width * rand_tools::cuda_rand01(seeds) + LIBOR_model::center);
	}
}

