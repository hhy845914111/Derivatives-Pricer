#pragma once
#include "configure.h"
#include "cpu_rand.h"
#include <cmath>

/* 
We use the cananical form of two-factor Vasicek Model to reduce parameters.
*/

namespace OIS_model
{
	static double lambda_1 = LAMBDA_1;
	static double lambda_21 = LAMBDA_21;
	static double lambda_2 = LAMBDA_2;

	static double delta_0 = DELTA_0;
	static double delta_1 = DELTA_1;
	static double delta_2 = DELTA_2;

	constexpr unsigned int integrate_N = INTEGRATE_N;

	inline void set_params(double lambda_1, double lambda_21, double lambda_2, double delta_0, double delta_1, double delta_2)
	{
		lambda_1 = lambda_1;
		lambda_21 = lambda_21;
		lambda_2 = lambda_2;
		delta_0 = delta_0;
		delta_1 = delta_1;
		delta_2 = delta_2;
	}


	inline double _C1_(double tao)
	{
		double exp_1 = exp(-lambda_1 * tao);

		if (lambda_1 == lambda_2)
		{
			double lambda_delta = lambda_21 * delta_2 / lambda_1;
			return (delta_1 - lambda_delta) * (1 - exp_1) / lambda_1 + lambda_delta * tao * exp_1;
		}
		else
		{
			double lambda_delta = lambda_21 * delta_2 / lambda_2;
			return (delta_1 - lambda_21 * delta_2 / lambda_2) * (1 - exp_1) / lambda_1 + lambda_delta / (lambda_1 - lambda_2) * (exp(-lambda_2 * tao) - exp_1);
		}
	}

	inline double _C2_(double tao)
	{
		return delta_2 * (1 - exp(-lambda_2 * tao)) / lambda_2;
	}

	inline double _dA_(double tao)
	{
		double C1 = _C1_(tao), C2 = _C2_(tao);
		return -0.5 * (C1 * C1 + C2 * C2) + delta_0;
	}

	inline double _A_(double tao, unsigned int N = integrate_N)
	{
		double A = 0.0, h = tao / N; 
		
		for (unsigned int i = 0; i < N; ++i)
		{
			A += h / 2 * (_dA_(tao) + _dA_(tao + h));
		}
		
		return A;
	}
	
	inline double P(double y1_0, double y2_0, double tao)
	{
		return exp(-_A_(tao) - _C1_(tao) * y1_0 - _C2_(tao) * y2_0);
	}

	inline double y(double y1_0, double y2_0, double tao)
	{
		if (tao > 0.0)
		{
			return (_A_(tao) + _C1_(tao) * y1_0 + _C2_(tao) * y2_0) / tao;
		}
		else
		{
			return 0.0;
		}
	}
}


namespace LIBOR_model
{
	constexpr double center = CENTER;
	constexpr double width = WIDTH;

	inline double y(double y1_0, double y2_0, double tao, double* seeds)
	{
		return OIS_model::y(y1_0, y2_0, tao) * (1 + width * rand_tools::rand01(seeds) + center);
	}
}
