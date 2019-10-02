#include "CUDA_funcs.cuh"
#include "CPU_funcs.h"


__device__ double cuda_x2P(double x, double y, double rho, double vol)
{
	return exp(x + (2 * rho * exp(0.5 * y)) / vol);
}


__device__ double cuda_P2x(double P, double y, double rho, double vol)
{
	return log(P) - 2 * rho * exp(0.5 * y) / vol;
}



__global__ void cuda_explicit_step1(double* this_mat, double* last_mat, size_t x_i, size_t n_i, size_t M, double r, constants::OptionType op_type,
	double dt, double dy, double Y_MIN, double rho, double vol, double K)
{
	size_t y_i = blockDim.x * blockIdx.x + threadIdx.x;

	if (y_i >= M)
	{
		return;
	}


	// notice that P is a monotonic increasing function of x and y, so we set boundary condition according to the option value, which is a function of P.
	double dsc = 1 / (1 + r * dt), P = 0.0;
	if (y_i == M - 1)
	{
		if (op_type == constants::Call)
		{
			P = cuda_x2P(x_i, y_i * dy + Y_MIN, rho, vol) - K * pow(dsc, (double)n_i);
			MATRIX_GET(this_mat, x_i, y_i, M) = P;
		}
		else
		{
			MATRIX_GET(this_mat, x_i, y_i, M) = 0.0; // put upper bound
		}
	}
	else if (y_i == 0)
	{
		if (op_type == constants::Call)
		{
			MATRIX_GET(this_mat, x_i, y_i, M) = 0.0; // call lower bound
		}
		else
		{
			P = K * pow(dsc, (double)n_i) - cuda_x2P(x_i, y_i * dy + Y_MIN, rho, vol);
			MATRIX_GET(this_mat, x_i, y_i, M) = P;
		}
	}
	else
	{
		double vol_dt_dy = vol * vol * dt / dy / dy;

		MATRIX_GET(this_mat, x_i, y_i, M) = 0.5 * (MATRIX_GET(last_mat, x_i, y_i + 1, M)
			+ MATRIX_GET(last_mat, x_i, y_i - 1, M)) * vol_dt_dy + MATRIX_GET(last_mat, x_i, y_i, M) * (1 - vol_dt_dy);
	}
}


__global__ void cuda_explicit_step2(double* this_mat, double* last_mat, size_t x_i, size_t n_i, size_t M, double r, constants::OptionType op_type,
	double dt, double dy, double Y_MIN, double kappa, double theta, double rho, double vol, double K)
{
	size_t y_i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (y_i >= M)
	{
		return;
	}

	// notice that P is a monotonic increasing function of x and y, so we set boundary condition according to the option value, which is a function of P.
	double dsc = 1 / (1 + r * dt), P = 0.0;
	if (y_i == M - 1)
	{
		if (op_type == constants::Call)
		{
			P = cuda_x2P(x_i, y_i * dy + Y_MIN, rho, vol) - K * pow(dsc, (double)n_i);
			MATRIX_GET(this_mat, x_i, y_i, M) = P;
		}
		else
		{
			MATRIX_GET(this_mat, x_i, y_i, M) = 0.0; // put upper bound
		}
	}
	else if (y_i == 0)
	{
		if (op_type == constants::Call)
		{
			MATRIX_GET(this_mat, x_i, y_i, M) = 0.0; // call lower bound
		}
		else
		{
			P = K * pow(dsc, (double)n_i) - cuda_x2P(x_i, y_i * dy + Y_MIN, rho, vol);
			MATRIX_GET(this_mat, x_i, y_i, M) = P;
		}
	}
	else
	{
		double k_theta = kappa * (theta - Y_MIN - dy * y_i), dt_dy = dt / dy;

		MATRIX_GET(this_mat, x_i, y_i, M) = MATRIX_GET(last_mat, x_i, y_i, M)
			+ MAX(0.0, k_theta) * dt_dy * (MATRIX_GET(last_mat, x_i, y_i + 1, M) - MATRIX_GET(last_mat, x_i, y_i, M))
			- MAX(0.0, -k_theta) * dt_dy * (MATRIX_GET(last_mat, x_i, y_i, M) - MATRIX_GET(last_mat, x_i, y_i - 1, M));
	}
}


__global__ void cuda_implicit_step(double* this_mat, double* last_mat, double* c_vec, size_t n_i, size_t M, double r, constants::OptionType op_type,
	double dt, double dx, double dy, double Y_MIN, double rho, double kappa, double theta, double vol, double K)
{
	size_t y_i = blockDim.x * blockIdx.x + threadIdx.x;

	if (y_i >= M)
	{
		return;
	}

	double dsc = 1 / (1 + r * dt), P = 0.0;

	c_vec += y_i * M;

	double y = Y_MIN + y_i * dy, dt_dx = dt / dx / 2, ey_rho_dt_dx = exp(y) * (1 - rho * rho) * dt / dx / dx;
	double rho_term = rho * exp(0.5 * y) * (kappa * (theta - y) + 0.25 * vol * vol) / vol;

	//the matrix is a constant matrix
	double a_i = -0.5 * ey_rho_dt_dx - (r - 0.5 * exp(y) - rho_term) * dt_dx;
	double b_i = 1 + r * dt + ey_rho_dt_dx;
	double c_i = -0.5 * ey_rho_dt_dx + (r - 0.5 * exp(y) - rho_term) * dt_dx;

	// solve for this column
	c_vec[0] = c_i / b_i;
	FOR_LOOP(i, 1, M - 1)
	{
		c_vec[i] = c_i / (b_i - a_i * c_vec[i - 1]);
	}

	MATRIX_GET(last_mat, 0, y_i, M) = MATRIX_GET(last_mat, 0, y_i, M) / b_i;
	FOR_LOOP(i, 1, M)
	{
		MATRIX_GET(last_mat, i, y_i, M) = (MATRIX_GET(last_mat, i, y_i, M) - a_i * MATRIX_GET(last_mat, i - 1, y_i, M)) / (b_i - a_i * c_vec[i - 1]);
	}

	MATRIX_GET(this_mat, M - 1, y_i, M) = MATRIX_GET(last_mat, M - 1, y_i, M);
	for (long int i = M - 2; i >= 0; --i)
	{
		MATRIX_GET(this_mat, i, y_i, M) = MATRIX_GET(last_mat, i, y_i, M) - c_vec[i] * MATRIX_GET(this_mat, i + 1, y_i, M);
	}
}


__global__ void cuda_early_exercise(double* this_mat, size_t x_i, size_t M, constants::OptionType op_type, double dx, double X_MIN, double dy, double Y_MIN, double rho, double vol, double K)
{
	size_t y_i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (y_i >= M)
	{
		return;
	}

	double P = cuda_x2P(X_MIN + dx * x_i, Y_MIN + dy * y_i, rho, vol), itsc = op_type == constants::Call ? P - K : K - P;
	MATRIX_GET(this_mat, x_i, y_i, M) = MAX(MATRIX_GET(this_mat, x_i, y_i, M), itsc);
}



double cuda_get_price_stochastic_vol(double P0, double y0, double K, double r, double vol, double tao, double theta, double rho, double kappa, constants::OptionType op_type,
	size_t M, size_t N, double P_MIN, double P_MAX, double Y_MIN, double Y_MAX)
{
	cudaSetDevice(0);
	double *d_value_mat0 = nullptr, *d_value_mat1 = nullptr, *d_value_mat2 = nullptr;
	cudaMalloc((void**)d_value_mat0, sizeof(double) * M * M);
	cudaMalloc((void**)d_value_mat1, sizeof(double) * M * M);
	cudaMalloc((void**)d_value_mat2, sizeof(double) * M * M);
	
	double *value_mat2 = new double[M * M];
	double *mat_lst[] = { d_value_mat0, d_value_mat1, d_value_mat2 };

	// initialize parameters
	double dy = (Y_MAX - Y_MIN) / M;
	double dt = tao / N;

	double X_MIN = P2x(P_MAX, Y_MIN, rho, vol), X_MAX = P2x(P_MIN, Y_MAX, rho, vol), dx = (X_MAX - X_MIN) / M;

	double this_x = 0.0, P = 0.0;
	FOR_LOOP(x_i, 0, M)
	{
		this_x = X_MIN + dx * x_i;
		if (op_type == constants::Call)
		{
			FOR_LOOP(y_i, 0, M)
			{
				P = x2P(this_x, Y_MIN + dy * y_i, rho, vol) - K;
				MATRIX_GET(value_mat2, x_i, y_i, M) = MAX(P, 0.0);
			}
		}
		else
		{
			FOR_LOOP(y_i, 0, M)
			{
				P = K - x2P(this_x, Y_MIN + dy * y_i, rho, vol);
				MATRIX_GET(value_mat2, x_i, y_i, M) = MAX(P, 0.0);
			}
		}
	}

	cudaMemcpy(d_value_mat2, value_mat2, sizeof(double) * M * M, cudaMemcpyHostToDevice);

	// use parallel computing for backward deduction
	const size_t block_count = M / parameters::GPU_THREADS + 1;
	double* c_vec = nullptr;
	cudaMalloc((void**)&c_vec, sizeof(double) * M * M);
	for (size_t j = N - 1; j > 0; --j)
	{
		FOR_LOOP(x_i, 0, M)
		{
			//multi_map(parameters::CPU_THREAD_COUNT, M, explicit_step1, mat_lst[1], mat_lst[2], x_i, N - j, M, r, op_type, dt, dy, Y_MIN, rho, vol, K);
			cuda_explicit_step1 <<<block_count, parameters::GPU_THREADS>>> (mat_lst[1], mat_lst[2], x_i, N - j, M, r, op_type, dt, dy, Y_MIN, rho, vol, K);
			cudaDeviceSynchronize();
		}

		FOR_LOOP(x_i, 0, M)
		{
			//multi_map(parameters::CPU_THREAD_COUNT, M, explicit_step2, mat_lst[0], mat_lst[1], x_i, N - j, M, r, op_type, dt, dy, Y_MIN, kappa, theta, rho, vol, K);
			cuda_explicit_step2 <<<block_count, parameters::GPU_THREADS>>> (mat_lst[0], mat_lst[1], x_i, N - j, M, r, op_type, dt, dy, Y_MIN, kappa, theta, rho, vol, K);
			cudaDeviceSynchronize();
		}


		cuda_implicit_step <<<block_count, parameters::GPU_THREADS>>> (mat_lst[2], mat_lst[0], c_vec, N - j, M, r, op_type, dt, dx, dy, Y_MIN, rho, kappa, theta, vol, K);
		cudaDeviceSynchronize();
		//multi_map(parameters::CPU_THREAD_COUNT, M, implicit_step, mat_lst[2], mat_lst[0], c_vec, N - j, M, r, op_type, dt, dx, dy, Y_MIN, rho, kappa, theta, vol, K);

		FOR_LOOP(x_i, 0, M)
		{
			cuda_early_exercise <<<block_count, parameters::GPU_THREADS>>> (mat_lst[2], x_i, M, op_type, dx, X_MIN, dy, Y_MIN, rho, vol, K);
			//multi_map(parameters::CPU_THREAD_COUNT, M, early_exercise, mat_lst[2], x_i, M, op_type, dx, X_MIN, dy, Y_MIN, rho, vol, K);
			cudaDeviceSynchronize();
		}
	}
	cudaFree(c_vec);

	cudaFree(d_value_mat0);
	cudaFree(d_value_mat1);

	cudaMemcpy(value_mat2, d_value_mat2, sizeof(double) * M * M, cudaMemcpyDeviceToHost);
	
	cudaFree(d_value_mat2);

	//find the best fit of start
	double this_diff = 0.0, min_diff = NULL, rst = 0.0;
	this_x = 0.0;
	FOR_LOOP(x_i, 0, M)
	{
		this_x = X_MIN + x_i * dx;
		FOR_LOOP(y_i, 0, M)
		{
			this_diff = abs(x2P(this_x, Y_MIN + y_i * dy, rho, vol) - P0);

			if (min_diff == NULL || min_diff > this_diff)
			{
				rst = MATRIX_GET(value_mat2, x_i, y_i, M);
				min_diff = this_diff;
			}
		}

	}

	delete[] value_mat2;

	cudaDeviceReset();

	return rst;
}