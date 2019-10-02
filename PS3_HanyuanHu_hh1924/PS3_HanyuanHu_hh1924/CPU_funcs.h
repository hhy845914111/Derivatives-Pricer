#pragma once
#include <utility>

#include "configure.h"
#include "multi_thread_map.h"
#include "utils.h"

inline void explicit_one_step(double* this_mat, size_t m_i, double* last_mat, size_t M, size_t n_i, 
	constants::OptionType op_type, double dt, double r, double vol, double dS, double S_min, double K)
{
	if (m_i >= M)
	{
		return;
	}

	double dsc_factor = 1 / (1 + r * dt), S = S_min + dS * m_i;

	if (m_i == M - 1)
	{
		if (op_type == constants::Call)
		{
			this_mat[m_i] = S - K * pow(dsc_factor, n_i);
		}
		else
		{
			this_mat[m_i] = 0.0;
		}
	}
	else if (m_i == 0)
	{
		if (op_type == constants::Call)
		{
			this_mat[m_i] = 0.0;
		}
		else
		{
			this_mat[m_i] = K * pow(dsc_factor, n_i) - S;
		}
	}
	else
	{
		double vol_m_i_dt = vol * vol * m_i  * m_i * dt, r_m_i_dt = r * m_i * dt;

		this_mat[m_i] = dsc_factor * (0.5 * (-r_m_i_dt + vol_m_i_dt) * last_mat[m_i - 1] + (1 - vol_m_i_dt) * last_mat[m_i] + 0.5 * (r_m_i_dt + vol_m_i_dt) * last_mat[m_i + 1]);
	}
}


inline void early_exercise(double* this_mat, size_t m_i, size_t M, constants::OptionType op_type, double dS, double S_min, double K)
{
	if (m_i >= M)
	{
		return;
	}

	double S = S_min + dS * m_i, itsc = op_type == constants::Call ? S - K : K - S;
	this_mat[m_i] = this_mat[m_i] < itsc ? itsc : this_mat[m_i];
}


inline double get_price_explicit(double S0, double K, double r, double vol, double tao, 
	constants::OptionType op_type, size_t M, size_t N, double S_MAX, double S_MIN)
{
	// malloc matrix
	double *value_mat1 = new double[M], *value_mat2 = new double[M];

	// initialize parameters
	double ds = (S_MAX - S_MIN) / M;
	double dt = tao / N;

	if (N % 2)
	{
		for (size_t i = 0; i < M; ++i)
		{
			if (op_type == constants::Call)
			{
				value_mat1[i] = MAX(ds * i + S_MIN - K, 0.0);
			}
			else
			{
				value_mat1[i] = MAX(K - ds * i + S_MIN, 0.0);
			}
		}
	}
	else
	{
		for (size_t i = 0; i < M; ++i)
		{
			if (op_type == constants::Call)
			{
				value_mat2[i] = MAX(ds * i + S_MIN - K, 0.0);
			}
			else
			{
				value_mat2[i] = MAX(K - ds * i + S_MIN, 0.0);
			}
		}
	}

	// use parallel computing for backward deduction
	for (size_t j = N - 1; j > 0; --j)
	{
		if (j % 2)
		{
			multi_map(parameters::CPU_THREAD_COUNT, M, explicit_one_step, value_mat1, value_mat2, M, N - j, op_type, dt, r, vol, ds, S_MIN, K);
			multi_map(parameters::CPU_THREAD_COUNT, M, early_exercise, value_mat1, M, op_type, ds, S_MIN, K);
		}
		else
		{
			multi_map(parameters::CPU_THREAD_COUNT, M, explicit_one_step, value_mat2, value_mat1, M, N - j, op_type, dt, r, vol, ds, S_MIN, K);
			multi_map(parameters::CPU_THREAD_COUNT, M, early_exercise, value_mat2, M, op_type, ds, S_MIN, K);
		}
	}

	//find the value which starts as S0
	size_t rst_idx = (S0 - S_MIN) / ds;
	double rst = (value_mat2[rst_idx] + value_mat2[rst_idx + 1]) / 2;
	delete[] value_mat1;
	delete[] value_mat2;
	return rst;
}
