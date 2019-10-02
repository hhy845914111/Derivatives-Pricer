#include "configure.h"

#include "cuda_derivatives.cuh"
#include "cuda_rate_model.cuh"

#include "cpu_rand.h"
#include "cuda_rand.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void _cuda_one_sim(double* disc_loc, unsigned int sim_count, double* y1_t_loc, double* y2_t_loc,
	double y1_0, double y2_0, double tao, unsigned int step_count, double* seeds)
{
	
	using namespace OIS_model;

	const unsigned int job_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (job_id >= sim_count)
	{
		return;
	}
	
	double y1_t, y2_t, R = 0.0;
	const double dt = tao / step_count, exp_1 = exp(-d_lambda_1 * dt), exp_2 = exp(-d_lambda_2 * dt);

	seeds += rand_tools::seeds_len * job_id;

	const double vol_1 = sqrt((1 - exp_1 * exp_1) / 2 / d_lambda_1);
	
	double vol_2, lambda_term = 0.0;
	if (d_lambda_1 != d_lambda_2)
	{
		lambda_term = d_lambda_21 / (d_lambda_1 - d_lambda_2);

		vol_2 = sqrt(lambda_term * lambda_term
			* ((1 - exp_1 * exp_1) / 2 / d_lambda_1 + (1 - exp_2 * exp_2) / 2 / d_lambda_2 - 2 / (d_lambda_1 + d_lambda_2) * (1 - exp_1 * exp_2))
			+ (1 - exp_2 * exp_2) / 2 / d_lambda_2);
	}
	else
	{
		vol_2 = sqrt(d_lambda_21 * d_lambda_21 / 2 / d_lambda_1 * ((1 - exp_1 * exp_1) / d_lambda_1 - dt * dt * exp_1 * exp_1)
			+ (1 - exp_1 * exp_1) / 1 / d_lambda_1);
	}
	
	for (unsigned int step_id = 0; step_id < step_count; ++step_id)
	{
		y1_t = exp_1 * y1_0 + rand_tools::cuda_randn(seeds) * vol_1;
		
		if (d_lambda_1 != d_lambda_2)
		{
			y2_t = lambda_term * (exp_1 - exp_2) * y1_0 + exp_2 * y2_0 + rand_tools::cuda_randn(seeds) * vol_2;
		}
		else
		{
			y2_t = -d_lambda_21 * dt * exp_1 * y1_0 + exp_1 * y2_0 + rand_tools::cuda_randn(seeds) * vol_2;
		}
		
		
		y1_0 = y1_t;
		y2_0 = y2_t;

		R += dt * (d_delta_0 + d_delta_1 * y1_t + d_delta_2 * y2_t);
	}
	
	disc_loc[job_id] = exp(-R);
	y1_t_loc[job_id] = y1_t;
	y2_t_loc[job_id] = y2_t;
	
}


__global__ void _cuda_swaption_pay_off(double *sw_p_loc, unsigned int sim_count, double* y1_t_loc, double* y2_t_loc,
	unsigned int t_count, double tenor, derivatives::OptionType op_type, double K, double F, double* seeds)
{
	const unsigned int job_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (job_id >= sim_count)
	{
		return;
	}

	double yt, yt_1, f, p_sum = 0.0, p_integral = 0.0, payoff;
	for (unsigned int t_idx = 1; t_idx < t_count; ++t_idx)
	{
		yt = LIBOR_model::cuda_y(y1_t_loc[job_id], y2_t_loc[job_id], tenor * t_idx, seeds);
		yt_1 = LIBOR_model::cuda_y(y1_t_loc[job_id], y2_t_loc[job_id], tenor * (t_idx - 1), seeds);

		f = (exp((yt * t_idx - yt_1 * (t_idx - 1)) * tenor) - 1.0) / tenor;

		p_sum += OIS_model::cuda_P(y1_t_loc[job_id], y2_t_loc[job_id], tenor * t_idx);
		p_integral += OIS_model::cuda_P(y1_t_loc[job_id], y2_t_loc[job_id], tenor * t_idx) * f;
	}

	if (op_type == derivatives::Call)
	{
		payoff = p_integral / p_sum - K;
	}
	else
	{
		payoff = K - p_integral / p_sum;
	}
	sw_p_loc[job_id] = payoff > 0.0 ? p_sum * payoff * F * tenor : 0.0;
}


double derivatives::cuda_swaption(double y1_0, double y2_0, double tao, derivatives::OptionType op_type, double K,
	unsigned int t_count, double tenor, double F, unsigned int sim_count, unsigned int step_count)
{
	// tao: from now to option expiration
	// t_count: number of tenor
	// tenor: length of one tenor in years
	// F: face value
	// sim_count: how much path to generate
	// step_count: number of steps to simulate from now to expiration
	
	cudaSetDevice(0);
	
	double *d_seeds = nullptr;
	cudaMalloc((void**)&d_seeds, sizeof(double) * sim_count * rand_tools::seeds_len);

	new double[sim_count * rand_tools::seeds_len];
	for (unsigned int sim_id = 0; sim_id < sim_count; ++sim_id)
	{
		rand_tools::rand01_skip_ahead(2 * t_count + 2 * step_count, rand_tools::default_seeds); // total skip for both OIS path generation and LIBOR generation
		cudaMemcpy(d_seeds + sim_id * rand_tools::seeds_len, rand_tools::default_seeds, rand_tools::seeds_len * sizeof(double), cudaMemcpyHostToDevice);
	}

	double *d_disc_loc = nullptr, *d_y1_t_loc = nullptr, *d_y2_t_loc = nullptr;

	cudaMalloc((void**)&d_disc_loc, sizeof(double) * sim_count);
	cudaMalloc((void**)&d_y1_t_loc, sizeof(double) * sim_count);
	cudaMalloc((void**)&d_y2_t_loc, sizeof(double) * sim_count);

	const unsigned int block_count = sim_count / CUDA_THREAD_COUNT + 1;
	_cuda_one_sim<<< block_count, CUDA_THREAD_COUNT >>>(d_disc_loc, sim_count, d_y1_t_loc, d_y2_t_loc, y1_0, y2_0, tao, step_count, d_seeds);
	cudaDeviceSynchronize();


	double* d_sw_payoff = nullptr;
	cudaMalloc((void**)&d_sw_payoff, sizeof(double) * sim_count);

	_cuda_swaption_pay_off<<<block_count, CUDA_THREAD_COUNT>>>(d_sw_payoff, sim_count, d_y1_t_loc, d_y2_t_loc, t_count, tenor, op_type, K, F, d_seeds);
	cudaDeviceSynchronize();

	cudaFree(d_seeds);
	d_seeds = nullptr;

	cudaFree(d_y1_t_loc);
	d_y1_t_loc = nullptr;

	cudaFree(d_y2_t_loc);
	d_y2_t_loc = nullptr;

	double *sw_payoff = new double[sim_count];
	double *disc_loc = new double[sim_count];
	cudaMemcpy(sw_payoff, d_sw_payoff, sizeof(double) * sim_count, cudaMemcpyDeviceToHost);
	cudaMemcpy(disc_loc, d_disc_loc, sizeof(double) * sim_count, cudaMemcpyDeviceToHost);

	cudaFree(d_sw_payoff);
	d_sw_payoff = nullptr;

	cudaFree(d_disc_loc);
	d_disc_loc = nullptr;

	cudaDeviceReset();

	double rst = 0.0;
	for (size_t i = 0; i < sim_count; ++i)
	{
		rst = (i * rst + sw_payoff[i] * disc_loc[i]) / (i + 1);
	}

	delete[] sw_payoff;
	delete[] disc_loc;

	return rst;	
}


__global__ void _dsc_op_pay_off(double* dsc_op_loc, unsigned int sim_count, double y1_0, double y2_0,
	derivatives::OptionType op_type, double K, double F, unsigned int t_count, double tenor, double* seeds)
{
	using namespace OIS_model;

	const unsigned int job_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (job_id >= sim_count)
	{
		return;
	}

	// prepare for rate random term calculation
	const double dt = tenor, exp_1 = exp(-d_lambda_1 * dt), exp_2 = exp(-d_lambda_2 * dt);

	seeds += rand_tools::seeds_len * job_id;

	const double vol_1 = sqrt((1 - exp_1 * exp_1) / 2 / d_lambda_1);

	double vol_2, lambda_term = 0.0;
	
	if (d_lambda_1 != d_lambda_2)
	{
		lambda_term = d_lambda_21 / (d_lambda_1 - d_lambda_2);

		vol_2 = sqrt(lambda_term * lambda_term
			* ((1 - exp_1 * exp_1) / 2 / d_lambda_1 + (1 - exp_2 * exp_2) / 2 / d_lambda_2 - 2 / (d_lambda_1 + d_lambda_2) * (1 - exp_1 * exp_2))
			+ (1 - exp_2 * exp_2) / 2 / d_lambda_2);
	}
	
	else
	{
		vol_2 = sqrt(d_lambda_21 * d_lambda_21 / 2 / d_lambda_1 * ((1 - exp_1 * exp_1) / d_lambda_1 - dt * dt * exp_1 * exp_1)
			+ (1 - exp_1 * exp_1) / 1 / d_lambda_1);
	}


	double value = 0.0, y1_t = y1_0, y2_t = y2_0, f_t, pay_off;
	for (unsigned int t_idx = 1; t_idx < t_count; ++t_idx)
	{
		f_t = LIBOR_model::cuda_y(y1_t, y2_t, tenor, seeds);

		if (op_type == derivatives::Call)
		{
			pay_off = f_t - K > 0.0 ? f_t - K : 0.0;
		}
		else
		{
			pay_off = K - f_t > 0.0 ? K - f_t : 0.0;
		}

		value += OIS_model::cuda_P(y1_0, y2_0, t_idx * tenor) * F * tenor * pay_off;

		// update state vector
		y1_t = exp_1 * y1_0 + rand_tools::cuda_randn(seeds) * vol_1;

		if (d_lambda_1 != d_lambda_2)
		{
			y2_t = lambda_term * (exp_1 - exp_2) * y1_0 + exp_2 * y2_0 + rand_tools::cuda_randn(seeds) * vol_2;
		}
		else
		{
			y2_t = -d_lambda_21 * dt * exp_1 * y1_0 + exp_1 * y2_0 + rand_tools::cuda_randn(seeds) * vol_2;
		}
	}

	dsc_op_loc[job_id] = value;
}


double derivatives::cuda_cap_floor(double y1_0, double y2_0, derivatives::OptionType op_type, double K,
	unsigned int t_count, double tenor, double F, unsigned int sim_count)
{
	cudaSetDevice(0);

	double *d_seeds = nullptr;
	cudaMalloc((void**)&d_seeds, sizeof(double) * sim_count * rand_tools::seeds_len);

	new double[sim_count * rand_tools::seeds_len];
	for (unsigned int sim_id = 0; sim_id < sim_count; ++sim_id)
	{
		rand_tools::rand01_skip_ahead(t_count * 3, rand_tools::default_seeds); // total skip for both OIS path generation and LIBOR generation
		cudaMemcpy(d_seeds + sim_id * rand_tools::seeds_len, rand_tools::default_seeds, rand_tools::seeds_len * sizeof(double), cudaMemcpyHostToDevice);
	}

	double* d_dsc_op_loc = nullptr;
	cudaMalloc((void**)&d_dsc_op_loc, sizeof(double) * sim_count);

	const unsigned int block_count = sim_count / CUDA_THREAD_COUNT + 1;

	_dsc_op_pay_off <<< block_count, CUDA_THREAD_COUNT >>> (d_dsc_op_loc, sim_count, y1_0, y2_0, op_type, K, F, t_count, tenor, d_seeds);
	cudaDeviceSynchronize();

	cudaFree(d_seeds);
	d_seeds = nullptr;

	double* dsc_op_loc = new double[sim_count];
	cudaMemcpy(dsc_op_loc, d_dsc_op_loc, sizeof(double) * sim_count, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(d_dsc_op_loc);
	d_dsc_op_loc = nullptr;

	cudaDeviceReset();

	double rst = 0.0;
	for (size_t i = 0; i < sim_count; ++i)
	{
		rst = (i * rst + dsc_op_loc[i]) / (i + 1);
	}

	delete[] dsc_op_loc;

	return rst;
}