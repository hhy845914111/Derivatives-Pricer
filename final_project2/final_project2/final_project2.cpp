#include "configure.h"
#include "cpu_rate_model.h"
#include "cpu_derivatives.h"
#include "cuda_derivatives.cuh"
#include <iostream>


int main()
{
	using namespace derivatives;

	double rst;
	
	rst = OIS_model::P(0.1, 0.1, 1);
	std::cout << rst << std::endl;
	
	
	rst = cpu_swaption(0.1, 0.1, 1, OptionType::Call, 0.02, 20, 0.25, 1000, 5000, 200);
	std::cout << rst << std::endl;
	

	rst = cpu_cap_floor(0.1, 0.1, OptionType::Call, 0.02, 20, 0.25, 1000, 5000);
	std::cout << rst << std::endl;
	
	
	rst = cuda_swaption(0.1, 0.1, 1, OptionType::Call, 0.02, 20, 0.25, 1000, 5000, 200);
	std::cout << rst << std::endl;
	

	rst = cuda_cap_floor(0.1, 0.1, OptionType::Call, 0.02, 20, 0.25, 1000, 5000);
	std::cout << rst << std::endl;
	
}
