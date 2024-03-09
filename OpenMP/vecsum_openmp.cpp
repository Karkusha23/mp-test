#include <iostream>
#include <string>
#include <chrono>
#include <omp.h>

int* generate_array(int size, int max);

int main(int argc, char* argv[])
{
	int num_threads = argc < 2 ? 5 : std::stoi(argv[1]);
	int arr_size = argc < 3 ? 10 : std::stoi(argv[2]);

	omp_set_num_threads(num_threads);

	int* arr = generate_array(arr_size, 1000);

	int i;
	int sum = 0;

	auto time_start = std::chrono::high_resolution_clock::now();

	#pragma omp parallel for shared(arr) private(i) schedule(static)
	for (i = 0; i < arr_size; ++i)
	{
		sum += arr[i];
	}

	auto time_end = std::chrono::high_resolution_clock::now();

	delete[] arr;

	std::cout << "Total sum: " << sum << std::endl;
	std::cout << "Time elapsed: " << std::chrono::duration<double>(time_end - time_start).count() << " seconds" << std::endl;

	return 0;
}

int* generate_array(int size, int max)
{
	if (size <= 0)
	{
		return nullptr;
	}

	int* arr = new int[size];

	if (!arr)
	{
		return nullptr;
	}

	std::srand(std::time(NULL));
	for (int i = 0; i < size; ++i)
	{
		arr[i] = std::rand() % max;
	}

	return arr;
}