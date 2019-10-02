/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
*/

#include "configure.h"
#include "utils.h"
#include "CPU_funcs.h"
#include "CUDA_funcs.cuh"
#include <iostream>

int main()
{
	// On CPU, we spend a lot of time creating threads and destroying threads. So this is slow.
	double vol = 0.1, volvol = 0.1;
	double call_price = get_price_stochastic_vol(10, vol, 2, 0.03, volvol, 0.25, vol, -0.2, 20, constants::Call, 50, 300, 1e-15, 50, -3, 3);
	std::cout << call_price << std::endl;

	double cuda_call_price = cuda_get_price_stochastic_vol(10, vol, 2, 0.03, volvol, 0.25, vol, -0.2, 20, constants::Call, 50, 300, 1e-15, 50, -3, 3);
	std::cout << cuda_call_price;
}
