#include "CUDA_funcs.cuh"

__global__ void cuda_explicit_one_step(double* this_mat, double* last_mat, size_t M, size_t n_i,
	constants::OptionType op_type, double dt, double r, double vol, double dS, double S_min, double K)
{
	size_t m_i = blockIdx.x * blockDim.x + threadIdx.x;

	if (m_i >= M)
	{
		return;
	}

	double dsc_factor = 1 / (1 + r * dt), S = S_min + dS * m_i;

	if (m_i == M - 1)
	{
		if (op_type == constants::Call)
		{
			this_mat[m_i] = S - K * pow(dsc_factor, (double)n_i);
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
			this_mat[m_i] = K * pow(dsc_factor, (double)n_i) - S;
		}
	}
	else
	{
		double vol_m_i_dt = vol * vol * m_i  * m_i * dt, r_m_i_dt = r * m_i * dt;

		this_mat[m_i] = dsc_factor * (0.5 * (-r_m_i_dt + vol_m_i_dt) * last_mat[m_i - 1] + (1 - vol_m_i_dt) * last_mat[m_i] + 0.5 * (r_m_i_dt + vol_m_i_dt) * last_mat[m_i + 1]);
	}
}

__global__ void cuda_early_exercise(double* this_mat, size_t M, constants::OptionType op_type, double dS, double S_min, double K)
{
	size_t m_i = blockDim.x * blockIdx.x + threadIdx.x;

	if (m_i >= M)
	{
		return;
	}

	double S = S_min + dS * m_i, itsc = op_type == constants::Call ? S - K : K - S;
	this_mat[m_i] = this_mat[m_i] < itsc ? itsc : this_mat[m_i];
}


double cuda_get_price_explicit(double S0, double K, double r, double vol, double tao,
	constants::OptionType op_type, size_t M, size_t N, double S_MAX, double S_MIN)
{
	// malloc matrix on CPU and initialize
	double* value_mat = new double[M];

	cudaSetDevice(0);

	double *d_value_mat1 = nullptr, *d_value_mat2 = nullptr;
	cudaMalloc((void**)&d_value_mat1, sizeof(double) * M);
	cudaMalloc((void**)&d_value_mat2, sizeof(double) * M);

	// initialize parameters
	double ds = (S_MAX - S_MIN) / M;
	double dt = tao / N;

	for (size_t i = 0; i < M; ++i)
	{
		if (op_type == constants::Call)
		{
			value_mat[i] = MAX(ds * i + S_MIN - K, 0.0);
		}
		else
		{
			value_mat[i] = MAX(K - ds * i + S_MIN, 0.0);
		}
	}

	if (N % 2)
	{
		cudaMemcpy(d_value_mat1, value_mat, sizeof(double) * M, cudaMemcpyHostToDevice);
	}
	else
	{
		cudaMemcpy(d_value_mat2, value_mat, sizeof(double) * M, cudaMemcpyHostToDevice);
	}

	// use parallel computing for backward deduction
	for (size_t j = N - 1; j > 0; --j)
	{
		if (j % 2)
		{
			cuda_explicit_one_step << <1 + M / parameters::GPU_THREADS, parameters::GPU_THREADS >> > (d_value_mat1, d_value_mat2, M, N - j, op_type, dt, r, vol, ds, S_MIN, K);
			cudaDeviceSynchronize();
			cuda_early_exercise << <1 + M / parameters::GPU_THREADS, parameters::GPU_THREADS >> > (d_value_mat1, M, op_type, ds, S_MIN, K);
			cudaDeviceSynchronize();
		}
		else
		{
			cuda_explicit_one_step << <1 + M / parameters::GPU_THREADS, parameters::GPU_THREADS >> > (d_value_mat2, d_value_mat1, M, N - j, op_type, dt, r, vol, ds, S_MIN, K);
			cudaDeviceSynchronize();
			cuda_early_exercise << <1 + M / parameters::GPU_THREADS, parameters::GPU_THREADS >> > (d_value_mat2, M, op_type, ds, S_MIN, K);
			cudaDeviceSynchronize();
		}


	}

	cudaMemcpy(value_mat, d_value_mat2, sizeof(double) * M, cudaMemcpyDeviceToHost);

	cudaFree(d_value_mat1);
	cudaFree(d_value_mat2);
	cudaDeviceReset();

	//find the value which starts as S0
	size_t rst_idx = (S0 - S_MIN) / ds;
	double rst = (value_mat[rst_idx] + value_mat[rst_idx + 1]) / 2;

	delete[] value_mat;

	return rst;
}