#pragma once
#include <iostream>
#include <ctime>
#include <string>
#include <fstream>


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

	template <typename return_t, typename... Args>
	inline std::pair<double, long long> timeit(long long repeat, clock_t max_time, return_t(*func_p)(Args...), Args&&... args)
	{
		clock_t cum_time{ 0 }, t0{ clock() };
		func_p(std::forward<Args>(args)...);
		clock_t t1{ clock() };
		cum_time += t1 - t0;

		long long expected_times{ max_time / (t1 - t0) }, iter_left{ expected_times < repeat ? expected_times : repeat }, counter{ 1 };

		while (counter <= iter_left)
		{
			t0 = clock();
			func_p(std::forward<Args>(args)...);
			cum_time += clock() - t0;

			++counter;
		}

		return std::make_pair(cum_time / 1000.0 / iter_left, iter_left);
	}

	template<typename T>
	inline void to_csv(T *start, T *end, std::string file_path, std::string sep=",")
	{
		std::ofstream output_file(file_path);

		output_file << sep << std::endl;
		while (start <= end)
		{
			output_file << std::endl << sep << *start;
			++start;
		}
	}

	template <typename T>
	inline double sum_arr(T start[], size_t n)
	{
		T *tmp_ptr(start);
		double rst(0);
		
		while (tmp_ptr - start <= n - 1)
		{
			rst += *tmp_ptr;
			++tmp_ptr;
		}

		return rst;
	}

	template <typename T>
	inline double mean_arr(T start[], size_t n)
	{
		return sum_arr(start, n) / n;
	}

	template <typename T>
	inline double std_arr(T start[], size_t n, size_t ddof=0)
	{
		T *tmp_ptr(start);
		double rst(0), tmp(0), _mean(sum_arr(start, n) / n);

		while (tmp_ptr - start <= n - 1)
		{
			tmp = *tmp_ptr - _mean;
			rst += tmp * tmp;
			++tmp_ptr;
		}

		return rst / (n - ddof);
	}
}