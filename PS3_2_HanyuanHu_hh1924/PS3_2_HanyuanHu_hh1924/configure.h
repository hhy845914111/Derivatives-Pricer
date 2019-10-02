#pragma once

namespace constants
{
	enum OptionType { Call, Put };
}

namespace parameters
{
	constexpr unsigned int CPU_THREAD_COUNT = 8;
	constexpr unsigned int GPU_THREADS = 64;
}
