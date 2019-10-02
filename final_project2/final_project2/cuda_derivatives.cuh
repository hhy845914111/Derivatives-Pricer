#pragma once
#include "cpu_derivatives.h"

namespace derivatives
{
	double cuda_swaption(double y1_0, double y2_0, double tao, derivatives::OptionType op_type, double K,
		unsigned int t_count, double tenor, double F, unsigned int sim_count, unsigned int step_count);

	double cuda_cap_floor(double y1_0, double y2_0, OptionType op_type, double K,
		unsigned int t_count, double tenor, double F, unsigned int sim_count);
}