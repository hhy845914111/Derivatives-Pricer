#pragma once

#include "configure.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"


__global__ void cuda_explicit_one_step(double* this_mat, double* last_mat, size_t M, size_t n_i,
	constants::OptionType op_type, double dt, double r, double vol, double dS, double S_min, double K);


__global__ void cuda_early_exercise(double* this_mat, size_t M, constants::OptionType op_type, double dS, double S_min, double K);


double cuda_get_price_explicit(double S0, double K, double r, double vol, double tao,
	constants::OptionType op_type, size_t M, size_t N, double S_MAX, double S_MIN);