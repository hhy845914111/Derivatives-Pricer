#pragma once
#include <future>
#include <vector>
#include <chrono>


template <typename ReturnType, typename... Args>
std::vector<ReturnType> multi_map(const unsigned int pool_size, const size_t n_jobs, ReturnType(*func)(size_t, Args...), Args... args)
{
	
	const std::chrono::milliseconds span(1);
	
	std::vector<ReturnType> results(n_jobs);

	std::vector<std::future<std::pair<size_t, ReturnType>>> futures(pool_size);

	size_t this_id = 0, finished_count = 0;
	for (; this_id < pool_size && this_id < n_jobs; ++this_id)
	{
		futures[this_id] = std::async([&] { return std::make_pair(this_id, func(this_id, args...)); });
	}

	while (finished_count < n_jobs)
	{
		for (auto& fut : futures)
		{
			if (fut.wait_for(span) == std::future_status::ready)
			{
				// collect old jobs
				auto rst_pair = fut.get();
				results[rst_pair.first] = rst_pair.second;
				++finished_count;

				// assign new jobs
				if (this_id < n_jobs)
				{
					fut = std::async([&] { return std::make_pair(this_id, func(this_id, args...)); });
					++this_id;
				}
			}
		}
	}

	return results;
}


template <typename ReturnType, typename... Args>
void multi_map(const unsigned int pool_size, const size_t n_jobs, void(*func)(ReturnType*, size_t, Args...), ReturnType* result_loc, Args... args)
{

	const std::chrono::milliseconds span(1);

	std::vector<std::future<void>> futures(pool_size);

	size_t this_id = 0, finished_count = 0;
	for (; this_id < pool_size && this_id < n_jobs; ++this_id)
	{
		futures[this_id] = std::async(func, result_loc, this_id, args...);
	}

	// use main thread as schedular and wait for all jobs to finish
	while (finished_count < n_jobs)
	{
		for (auto& fut : futures)
		{
			if (fut.wait_for(span) == std::future_status::ready)
			{
				// collect old jobs
				++finished_count;

				// assign new jobs
				if (this_id < n_jobs)
				{
					fut = std::async(func, result_loc, this_id, args...);
					++this_id;
				}
			}
		}
	}

}