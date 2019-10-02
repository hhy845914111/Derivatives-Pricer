#pragma once

namespace derivatives
{
	enum OptionType { Call, Put };

	double cpu_swaption(double y1_0, double y2_0, double tao, derivatives::OptionType op_type, double K,
		unsigned int t_count, double tenor, double F, unsigned int sim_count, unsigned int step_count);

	double cpu_cap_floor(double y1_0, double y2_0, OptionType op_type, double K,
		unsigned int t_count, double tenor, double F, unsigned int sim_count);
}