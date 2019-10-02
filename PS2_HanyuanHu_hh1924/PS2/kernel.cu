#include "pch.h" // some constants defined here

#include "cpu_rand.h"
#include "test_utils.h"
#include "kernel_rand.cuh"

#include <utility>
#include <cmath>
#include <vector>

#define NAIVE_SKIP // we can choose to use naive way to skip seeds or use the method suggested in lecture.

double dseeds[SEEDS_SIZE]{ 298193, 104959, 84736, 727366, 94727, 5928384 };

using ThreadPtr = std::unique_ptr<std::thread>;


inline void sim_exp(double *result_ar, double *seeds, size_t tid, size_t sim_count, double u)
{
	double *start_ptr(result_ar + tid * sim_count), *end_ptr(start_ptr + sim_count - 1);
	while (start_ptr <= end_ptr)
	{
		*start_ptr = -u * log(1 - cpu_rand::rand01(seeds));
		++start_ptr;
	}
}


inline void sim_knock_out(
	double *result_ar, double *seeds, size_t tid, size_t sim_count, 
	OptionType opt_type, double S0, double r, double sigma, double K, double B, double T, unsigned int n)
{
	const double dt(T / n);
	double S_a, S_b;
	unsigned int step;
	double pay_off;
	sigma = sigma / sqrt(n * T);

	double *start_ptr(result_ar + tid * sim_count), *end_ptr(start_ptr + sim_count - 1);
	while (start_ptr <= end_ptr)
	{
		S_a = S0;
		step = 0;
		while (step < n)
		{
			S_b = S_a + r * S_a * dt + sigma * S_a * cpu_rand::randn(seeds);
			
			if (S_b > B && opt_type == Call)
			{
				*start_ptr = 0.0;
				goto one_sim_end;
			}
			else if (S_b < B && opt_type == Put)
			{
				*start_ptr = 0.0;
				goto one_sim_end;
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
		
		*start_ptr = pay_off > 0 ? pay_off : 0.0;

	one_sim_end:
		++start_ptr;
	}
}


template <typename SeedsType>
inline SeedsType *generate_seeds_ar(size_t batch_count, size_t batch_size, SeedsType *seeds, SeedsType *location=nullptr)
{
	if (location == nullptr)
	{
		location = new SeedsType[batch_count * SEEDS_SIZE];
	}

	SeedsType *_tmp_ptr(location);
	size_t counter(0);
	while (counter < batch_count)
	{
		memcpy(_tmp_ptr, seeds, SEEDS_SIZE * sizeof(SeedsType));
		cpu_rand::rand01_skip_ahead(batch_size, seeds);
		++counter;
		_tmp_ptr += SEEDS_SIZE;
	}

	return location;
}


int main()
{
	// 1) simulate exponential distribution using CPU
	const size_t batch_size(TOTAL_SIM_NUM / CPU_THREAD_COUNT);
	double *rst_ar(new double[TOTAL_SIM_NUM]);
	double *seeds_ar(generate_seeds_ar(CPU_THREAD_COUNT, batch_size, dseeds));

	std::vector<ThreadPtr> thread_vec(CPU_THREAD_COUNT);
	for (int i = 0; i < CPU_THREAD_COUNT; ++i)
	{
		thread_vec[i].reset(new std::thread(sim_exp, rst_ar, seeds_ar + i * SEEDS_SIZE, i, batch_size, 1.0));
	}

	for (size_t i(0); i < CPU_THREAD_COUNT; ++i)
	{
		thread_vec[i]->join();
	}

	std::cout << "CPU exponential simulation results:" << std::endl;
	test_utils::print_arr(rst_ar, 50);
	if (SAVE) test_utils::to_csv(rst_ar, rst_ar + TOTAL_SIM_NUM - 1, CPU_TO_FILE);

	// bootstrap to calculate mean, variance and their standard error;
	std::unique_ptr<double[]> mean_ar(new double[BOOTSTRAP_COUNT]);
	std::unique_ptr<double[]> std_ar(new double[BOOTSTRAP_COUNT]);
	for (auto bt_idx(0); bt_idx < BOOTSTRAP_COUNT; ++bt_idx)
	{
		generate_seeds_ar(CPU_THREAD_COUNT, batch_size, dseeds, seeds_ar);

		for (int i = 0; i < CPU_THREAD_COUNT; ++i)
		{
			thread_vec[i].reset(new std::thread(sim_exp, rst_ar, seeds_ar + i * SEEDS_SIZE, i, batch_size, 1.0));
		}

		for (size_t i(0); i < CPU_THREAD_COUNT; ++i)
		{
			thread_vec[i]->join();
		}

		mean_ar[bt_idx] = test_utils::mean_arr(rst_ar, TOTAL_SIM_NUM);
		std_ar[bt_idx] = test_utils::std_arr(rst_ar, TOTAL_SIM_NUM);
	}

	
	double mean_stderr = test_utils::std_arr(mean_ar.get(), BOOTSTRAP_COUNT);
	double std_stderr = test_utils::std_arr(std_ar.get(), BOOTSTRAP_COUNT);

	std::cout << "Mean: " << test_utils::mean_arr(mean_ar.get(), BOOTSTRAP_COUNT) << ", Std error: " << mean_stderr << std::endl;
	std::cout << "Std: " << test_utils::mean_arr(std_ar.get(), BOOTSTRAP_COUNT) << ", Std error: " << std_stderr << std::endl;

	// 2) simulate exponential distribution using CPU
	double *seeds_ar2(generate_seeds_ar(TOTAL_SIM_NUM, 1, dseeds));

	cudaError_t error_t = cuda_sim(sim_exp_cuda, rst_ar, seeds_ar2, TOTAL_SIM_NUM, TOTAL_SIM_NUM, 1.0);

	std::cout << "GPU exponential simulation results:" << std::endl;
	test_utils::print_arr(rst_ar, 50);
	if (SAVE) test_utils::to_csv(rst_ar, rst_ar + TOTAL_SIM_NUM - 1, GPU_TO_FILE);


	for (auto bt_idx(0); bt_idx < BOOTSTRAP_COUNT; ++bt_idx)
	{
		generate_seeds_ar(TOTAL_SIM_NUM, 1, dseeds, seeds_ar2);
		error_t = cuda_sim(sim_exp_cuda, rst_ar, seeds_ar2, TOTAL_SIM_NUM, TOTAL_SIM_NUM, 1.0);

		mean_ar[bt_idx] = test_utils::mean_arr(rst_ar, TOTAL_SIM_NUM);
		std_ar[bt_idx] = test_utils::std_arr(rst_ar, TOTAL_SIM_NUM);
	}

	mean_stderr = test_utils::std_arr(mean_ar.get(), BOOTSTRAP_COUNT);
	std_stderr = test_utils::std_arr(std_ar.get(), BOOTSTRAP_COUNT);

	std::cout << "Mean: " << test_utils::mean_arr(mean_ar.get(), BOOTSTRAP_COUNT) << ", Std error: " << mean_stderr << std::endl;
	std::cout << "Std: " << test_utils::mean_arr(std_ar.get(), BOOTSTRAP_COUNT) << ", Std error: " << std_stderr << std::endl;

	mean_ar.release();
	std_ar.release();

	// 3) a. Value a call option with a knock out at a price above the strike.
	// CPU 
	generate_seeds_ar(CPU_THREAD_COUNT, batch_size, dseeds, seeds_ar);
	for (int i = 0; i < CPU_THREAD_COUNT; ++i)
	{
		thread_vec[i].reset(new std::thread(sim_knock_out, rst_ar, seeds_ar + i * SEEDS_SIZE, i, batch_size, Call, 50.0, 0.03, 0.16, 52.0, 60.0, 1.0, 365));
	}

	for (size_t i(0); i < CPU_THREAD_COUNT; ++i)
	{
		thread_vec[i]->join();
	}

	std::cout << "Call option price by CPU: " << test_utils::mean_arr(rst_ar, TOTAL_SIM_NUM) << std::endl;
	
	// GPU
	generate_seeds_ar(TOTAL_SIM_NUM, 1, seeds_ar2);
	error_t = cuda_sim(sim_knock_out_cuda, rst_ar, seeds_ar2, TOTAL_SIM_NUM, TOTAL_SIM_NUM, Call, 50.0, 0.03, 0.16, 52.0, 60.0, 1.0, (unsigned int)365);

	std::cout << "Call option price by GPU: " << test_utils::mean_arr(rst_ar, TOTAL_SIM_NUM) << std::endl;

	// 3) b. Value a put option with a knock out at a price below the strike.

	//CPU
	generate_seeds_ar(CPU_THREAD_COUNT, batch_size, dseeds, seeds_ar);
	for (int i = 0; i < CPU_THREAD_COUNT; ++i)
	{
		thread_vec[i].reset(new std::thread(sim_knock_out, rst_ar, seeds_ar + i * SEEDS_SIZE, i, batch_size, Put, 50.0, 0.04, 0.16, 52.0, 45.0, 1.0, 365));
	}

	for (size_t i(0); i < CPU_THREAD_COUNT; ++i)
	{
		thread_vec[i]->join();
	}

	std::cout << "Put option price by CPU: " << test_utils::mean_arr(rst_ar, TOTAL_SIM_NUM) << std::endl;

	// GPU
	generate_seeds_ar(TOTAL_SIM_NUM, 1, dseeds, seeds_ar2);
	error_t = cuda_sim(sim_knock_out_cuda, rst_ar, seeds_ar2, TOTAL_SIM_NUM, TOTAL_SIM_NUM, Put, 50.0, 0.02, 0.16, 52.0, 45.0, 1.0, (unsigned int)365);

	std::cout << "Put option price by GPU: " << test_utils::mean_arr(rst_ar, TOTAL_SIM_NUM) << std::endl;

	delete[]seeds_ar;
	delete[]seeds_ar2;
	delete[]rst_ar;

    return 0;
}
