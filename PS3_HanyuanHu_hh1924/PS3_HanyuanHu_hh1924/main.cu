
#include "configure.h"
#include "multi_thread_map.h"
#include "CPU_funcs.h"
#include "CUDA_funcs.cuh"
#include <iostream>


int main()
{
	double call_price = get_price_explicit(5, 5, 0.3, 0.1, 0.25, constants::Put, 200, 1000, 10, 1e-15);
	
	std::cout << call_price << std::endl;

	double cuda_call_price = cuda_get_price_explicit(5, 5, 0.3, 0.1, 0.25, constants::Put, 200, 1000, 10, 1e-15);

	std::cout << cuda_call_price << std::endl;

	return 0;
}