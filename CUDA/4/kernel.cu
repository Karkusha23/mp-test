#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <string>

#include <chrono>

double* generate_matrix(int rows, int cols);

__global__ void matMultKernel(double* C, double* A, double* B, int m, int n, int k)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < m * k)
	{
		for (int j = 0; j < n; ++j)
		{
			C[i] += A[n * (i / k) + j] * B[k * j + i % k];
		}
	}
}

int main(int argc, char* argv[])
{
	int block_count = argc < 2 ? 5 : std::stoi(argv[1]);
	int m = argc < 3 ? 100 : std::stoi(argv[2]);
	int n = argc < 4 ? 100 : std::stoi(argv[3]);
	int k = argc < 5 ? 100 : std::stoi(argv[4]);

	if (m * k < block_count)
	{
		std::cout << "Matrix size must be bigger than number of blocks!" << std::endl;
		return 1;
	}

	double* A = generate_matrix(m, n);
	double* B = generate_matrix(n, k);

	double* dev_A;
	double* dev_B;
	double* dev_C;

	cudaMalloc((void**)&dev_A, m * n * sizeof(double));
	cudaMalloc((void**)&dev_B, n * k * sizeof(double));
	cudaMalloc((void**)&dev_C, m * k * sizeof(double));

	cudaMemcpy(dev_A, A, m * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice);
	delete[] A;
	delete[] B;

	cudaMemset(dev_C, 0, m * k * sizeof(double));

	auto time_start = std::chrono::high_resolution_clock::now();

    matMultKernel<<<block_count, (m * k) / block_count + 1>>>(dev_C, dev_A, dev_B, m, n, k);
	cudaDeviceSynchronize();

	auto time_end = std::chrono::high_resolution_clock::now();

	double* C = new double[m * k];
	cudaMemcpy(C, dev_C, m * k * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);

	/*for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			std::cout << C[k * i + j] << ' ';
		}
		std::cout << std::endl;
	}*/

	delete[] C;

	std::cout << "Time elapsed: " << std::chrono::duration<double>(time_end - time_start).count() << " seconds" << std::endl;

    return 0;
}

double* generate_matrix(int rows, int cols)
{
	if (rows < 1 || cols < 1)
	{
		return nullptr;
	}

	double* arr = new double[rows * cols];

	if (!arr)
	{
		return nullptr;
	}

	std::srand(std::time(NULL));

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			arr[cols * i + j] = -1.0 + double(std::rand() % 20000) / 10000.0;
		}
	}

	return arr;
}