#include "pch.h"
#include <thread>
#include <iostream>
#include <string>
#include <utility>

const size_t TASK_COUNT = 8;

extern int counter;

class Params
{
public:
	int tid;
	int age;
	std::string name;
	int result;

	Params() : tid{ std::rand() }, age{ std::rand() }, name{ "sfagagad" } {}

	void display() const
	{
		std::cout << tid << std::endl;
		std::cout << age << std::endl;
		std::cout << name << std::endl;
	}

	void save_result(int rst)
	{
		result = rst;
	}

	int get_result() const
	{
		return result;
	}
};


class ThreadTask
{
public:
	virtual void operator ()(Params& params) = 0;
};

class PrintTask: public ThreadTask
{
private:
	static int count;

private:
	int task_id;

public:
	PrintTask() : task_id{ count++ } {}

	virtual void operator ()(Params& params) override
	{
		params.display();
		params.save_result(params.age - 1);

		for (int i{ 0 }; i < 100; ++i) ++counter;
	}
};

int PrintTask::count(0);

int counter{ 0 };

int main()
{
	Params *param_arr = new Params[TASK_COUNT];
	PrintTask *task_arr = new PrintTask[TASK_COUNT];
	//std::thread **thread_arr = new std::thread *[TASK_COUNT];

	std::shared_ptr<std::thread> *thread_arr = new std::shared_ptr<std::thread>[TASK_COUNT];

	for (size_t i = 0; i < TASK_COUNT; ++i)
	{
		thread_arr[i] = std::make_shared<std::thread>(std::thread{ task_arr[i], std::ref(param_arr[i])});
	}

	for (size_t i = 0; i < TASK_COUNT; ++i)
	{
		thread_arr[i]->join();
	}

	delete[]param_arr;
	delete[]task_arr;
	delete[]thread_arr; // this memory leak

	std::cout << counter;

	return 0;
}
