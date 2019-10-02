#pragma once
#include "configure.h"
#include "utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__device__ double cuda_x2P(double x, double y, double rho, double vol);


__device__ double cuda_P2x(double P, double y, double rho, double vol);


__global__ void cuda_explicit_step1(double* this_mat, double* last_mat, size_t x_i, size_t n_i, size_t M, double r, constants::OptionType op_type,
	double dt, double dy, double Y_MIN, double rho, double vol, double K);


__global__ void cuda_explicit_step2(double* this_mat, double* last_mat, size_t x_i, size_t n_i, size_t M, double r, constants::OptionType op_type,
	double dt, double dy, double Y_MIN, double kappa, double theta, double rho, double vol, double K);


__global__ void cuda_implicit_step(double* this_mat, double* last_mat, double* c_vec, size_t n_i, size_t M, double r, constants::OptionType op_type,
	double dt, double dx, double dy, double Y_MIN, double rho, double kappa, double theta, double vol, double K);


__global__ void cuda_early_exercise(double* this_mat, size_t x_i, size_t M, constants::OptionType op_type, double dx, double X_MIN, double dy, double Y_MIN, double rho, double vol, double K);


double cuda_get_price_stochastic_vol(double P0, double y0, double K, double r, double vol, double tao, double theta, double rho, double kappa, constants::OptionType op_type,
	size_t M, size_t N, double P_MIN, double P_MAX, double Y_MIN, double Y_MAX);
