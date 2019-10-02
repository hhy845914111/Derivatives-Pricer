#pragma once
#include <iostream>
#include <utility>

namespace test_utils
{
	inline int* generate_rand_int_arr(size_t n, int* location=nullptr, int range=100)
	{
		if (!location)
		{
			location = new int[n];
		}
		
		for (int *p(location), *q(location + n); p < q; ++p)
		{
			*p = rand() % range;
		}

		return location;
	}

	template <typename T>
	inline T* arange(T start, T end, T step)
	{
		if (end < start || step < 0)
		{
			throw std::invalid_argument("arange failed.");
		}
		
		T* rst = new T[T((end - start) / step)];

		for (T *p(rst); start <= end; start += step, ++p)
		{
			*p = start;
		}

		return rst;
	}

	template <typename T, std::ostream& ost = std::cout>
	inline void print_arr(const T* start, size_t n)
	{
		for (size_t i = 0; i < n; ++i, ++start)
		{
			ost << *start << ", ";
		}

		ost << std::endl;
	}

	template <typename T>
	inline bool is_ascend(const T *start, size_t n)
	{
		for (const T* p(start), *q(start + n - 1); p < q; ++p)
		{
			if (*p > *(p + 1))
			{
				return false;
			}
		}
		return true;
	}

	template <typename T>
	inline bool is_descend(const T *start, size_t n)
	{
		for (const T *p(start), *q(start + n - 1); p < q; ++p)
		{
			if (*p < *(p + 1))
			{
				return false;
			}
		}
		return true;
	}

	template <typename T>
	inline double l2_norm(const T *a, const T *b, size_t size, T *location=nullptr)
	{
		if (a == nullptr || b == nullptr)
		{
			throw std::invalid_argument("l2_norm failed.");
		}

		size_t count(0);
		double error_sum(0);
		while (count < size)
		{
			error_sum += (*b - *a) * (*b - *a);
			++a;
			++b;
			++count;
		}

		return error_sum;
	}

	template <typename T1, typename T2>
	inline std::ostream& operator <<(std::ostream& ost, const std::pair<T1*, T2*>& pr)
	{
		return ost << "(" << *pr.first << ", " << *pr.second << ")" << std::endl;
	}
}