#include <iostream>
#include <string>
#include <chrono>
#include <omp.h>

double** generate_matrix(int rows, int cols);
double** zero_matrix(int rows, int cols);

int main(int argc, char* argv[])
{
	int num_threads = argc < 2 ? 5 : std::stoi(argv[1]);
	int m = argc < 3 ? 100 : std::stoi(argv[2]);
	int n = argc < 4 ? 100 : std::stoi(argv[3]);
	int k = argc < 5 ? 100 : std::stoi(argv[4]);

	omp_set_num_threads(num_threads);

	double** A = generate_matrix(m, n);
	double** B = generate_matrix(n, k);

	double** C = zero_matrix(m, k);

	int total_size = m * k;

	int i, j, row, col;

	auto time_start = std::chrono::high_resolution_clock::now();

	#pragma omp parallel for shared(A, B, C, k, n, total_size) private(i, j, row, col) schedule(static)
	for (i = 0; i < total_size; ++i)
	{
		row = i / k;
		col = i % k;

		for (j = 0; j < n; ++j)
		{
			C[row][col] += A[row][j] * B[j][col];
		}
	}

	auto time_end = std::chrono::high_resolution_clock::now();

	for (i = 0; i < m; ++i)
	{
		delete[] A[i];
		delete[] C[i];
	}
	delete[] A;
	delete[] C;

	for (i = 0; i < n; ++i)
	{
		delete[] B[i];
	}
	delete[] B;

	std::cout << "Time elapsed: " << std::chrono::duration<double>(time_end - time_start).count() << " seconds" << std::endl;

	return 0;
}

double** generate_matrix(int rows, int cols)
{
	if (rows < 1 || cols < 1)
	{
		return nullptr;
	}

	double** arr = new double*[rows];

	if (!arr)
	{
		return nullptr;
	}

	std::srand(std::time(NULL));

	for (int i = 0; i < rows; ++i)
	{
		arr[i] = new double[cols];

		if (!arr[i])
		{
			delete[] arr;
			return nullptr;
		}

		for (int j = 0; j < cols; ++j)
		{
			arr[i][j] = -1.0 + double(std::rand() % 20000) / 10000.0;
		}
	}

	return arr;
}

double** zero_matrix(int rows, int cols)
{
	if (rows < 1 || cols < 1)
	{
		return nullptr;
	}

	double** arr = new double*[rows];

	if (!arr)
	{
		return nullptr;
	}

	for (int i = 0; i < rows; ++i)
	{
		arr[i] = new double[cols];

		if (!arr[i])
		{
			delete[] arr;
			return nullptr;
		}

		for (int j = 0; j < cols; ++j)
		{
			arr[i][j] = 0.0;
		}
	}

	return arr;
}