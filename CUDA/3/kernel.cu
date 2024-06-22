#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <string>

#include <chrono>

typedef double (*fPtr)(double, double);

const double XMIN = -10.0;
const double XMAX = 10.0;
const double YMIN = -10.0;
const double YMAX = 10.0;

double foo(double x, double y)
{
	return x * x * y + x * y * y;
}

double* generate_grid(fPtr func, double xmin, double xmax, int xcount, double ymin, double ymax, int ycount);

__global__ void derivativeKernel(double* A, double* B, int xsize, int ysize, double dy, double dy2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < xsize * ysize)
	{
		if (!(i % ysize))
		{
			B[i] = (A[i + 1] - A[i]) / dy;
		}
		else if (i % ysize == ysize - 1)
		{
			B[i] = (A[i] - A[i - 1]) / dy;
		}
		else
		{
			B[i] = (A[i + 1] - A[i - 1]) / dy2;
		}
	}
}

int main(int argc, char* argv[])
{
	int block_count = argc < 2 ? 5 : std::stoi(argv[1]);
	int x_size = argc < 3 ? 100 : std::stoi(argv[2]);
	int y_size = argc < 4 ? 100 : std::stoi(argv[3]);

	int total_size = x_size * y_size;

	if (total_size < block_count)
	{
		std::cout << "Matrix size must be bigger than number of blocks!" << std::endl;
		return 1;
	}

	double* A = generate_grid(foo, XMIN, XMAX, x_size, YMIN, YMAX, y_size);

	double* dev_A;
	double* dev_B;

	cudaMalloc((void**)&dev_A, total_size * sizeof(double));
	cudaMalloc((void**)&dev_B, total_size * sizeof(double));

	cudaMemcpy(dev_A, A, total_size * sizeof(double), cudaMemcpyHostToDevice);
	delete[] A;

	double dy = (YMAX - YMIN) / double(y_size - 1);
	double dy2 = dy * 2.0;

	auto time_start = std::chrono::high_resolution_clock::now();

    derivativeKernel<<<block_count, total_size / block_count + 1>>>(dev_A, dev_B, x_size, y_size, dy, dy2);
	cudaDeviceSynchronize();

	auto time_end = std::chrono::high_resolution_clock::now();

	double* B = new double[total_size];
	cudaMemcpy(B, dev_B, total_size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_A);
	cudaFree(dev_B);

	/*for (int i = 0; i < x_size; ++i)
	{
		for (int j = 0; j < y_size; ++j)
		{
			std::cout << B[y_size * i + j] << ' ';
		}
		std::cout << std::endl;
	}*/

	delete[] B;

	std::cout << "Time elapsed: " << std::chrono::duration<double>(time_end - time_start).count() << " seconds" << std::endl;

    return 0;
}

double* generate_grid(fPtr func, double xmin, double xmax, int xcount, double ymin, double ymax, int ycount)
{
	if (xmin > xmax || ymin > ymax || xcount <= 1 || ycount <= 1)
	{
		return nullptr;
	}

	double* arr = new double[xcount * ycount];

	if (!arr)
	{
		return nullptr;
	}

	double x = xmin;

	double dx = (xmax - xmin) / double(xcount - 1);
	double dy = (ymax - ymin) / double(ycount - 1);

	for (int i = 0; i < xcount; ++i)
	{
		double y = ymin;

		for (int j = 0; j < ycount; ++j)
		{
			arr[ycount * i + j] = func(x, y);

			y += dy;
		}

		x += dx;
	}

	return arr;
}