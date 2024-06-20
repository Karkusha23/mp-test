#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <string>

#include <chrono>

int* generate_array(int size, int max);
int* generate_fixed_array(int size, int val);

int atomicAdd(int* address, int val);

__global__ void sumKernel(int* sum, int* arr, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size)
	{
		atomicAdd(sum, arr[i]);
	}
}

int main(int argc, char* argv[])
{
	int block_count = argc < 2 ? 5 : std::stoi(argv[1]);
	int arr_size = argc < 3 ? 10 : std::stoi(argv[2]);

	if (arr_size < block_count)
	{
		std::cout << "Array size must be bigger than number of blocks!" << std::endl;
		return 1;
	}

	//int* arr = generate_fixed_array(arr_size, 1);
	int* arr = generate_array(arr_size, 10);
	
	int* dev_arr;
	int* dev_sum;

	cudaMalloc((void**)&dev_arr, arr_size * sizeof(int));
	cudaMalloc((void**)&dev_sum, sizeof(int));

	cudaMemcpy(dev_arr, arr, arr_size * sizeof(int), cudaMemcpyHostToDevice);
	delete[] arr;

	cudaMemset(dev_sum, 0, sizeof(int));

	auto time_start = std::chrono::high_resolution_clock::now();

    sumKernel<<<block_count, arr_size / block_count + 1>>>(dev_sum, dev_arr, arr_size);
	cudaDeviceSynchronize();

	auto time_end = std::chrono::high_resolution_clock::now();

	int sum;
	cudaMemcpy(&sum, dev_sum, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_arr);
	cudaFree(dev_sum);

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

int* generate_fixed_array(int size, int val)
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

	for (int i = 0; i < size; ++i)
	{
		arr[i] = val;
	}

	return arr;
}