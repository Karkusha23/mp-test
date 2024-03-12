#include <iostream>
#include <string>
#include <chrono>
#include <omp.h>

typedef double (*fPtr)(double, double);

const double XMIN = -10.0;
const double XMAX = 10.0;
const double YMIN = -10.0;
const double YMAX = 10.0;

double foo(double x, double y) 
{
	return x * x * y + x * y * y;
}

double** generate_grid(fPtr func, double xmin, double xmax, int xcount, double ymin, double ymax, int ycount);

int main(int argc, char* argv[])
{
	int num_threads = argc < 2 ? 5 : std::stoi(argv[1]);
	int x_size = argc < 3 ? 100 : std::stoi(argv[2]);
	int y_size = argc < 4 ? 100 : std::stoi(argv[3]);

	omp_set_num_threads(num_threads);

	double** A = generate_grid(foo, XMIN, XMAX, x_size, YMIN, YMAX, y_size);

	double** B = new double*[x_size];
	for (int j = 0; j < x_size; ++j)
	{
		B[j] = new double[y_size];
	}

	int total_size = x_size * y_size;
	double dy = (YMAX - YMIN) / double(y_size - 1);
	double dy2 = dy * 2.0;

	int i, x, y;

	auto time_start = std::chrono::high_resolution_clock::now();

	#pragma omp parallel for shared(A, B, total_size, y_size, dy, dy2) private(i, x, y) schedule(static)
	for (i = 0; i < total_size; ++i)
	{
		x = i / y_size;
		y = i % y_size;

		if (!y)
		{
			B[x][y] = (A[x][y + 1] - A[x][y]) / dy;
		}
		else if (y == y_size - 1)
		{
			B[x][y] = (A[x][y] - A[x][y - 1]) / dy;
		}
		else
		{
			B[x][y] = (A[x][y + 1] - A[x][y - 1]) / dy2;
		}
	}

	auto time_end = std::chrono::high_resolution_clock::now();

	for (int j = 0; j < x_size; ++j)
	{
		delete[] A[j];
		delete[] B[j];
	}
	delete[] A;
	delete[] B;

	std::cout << "Time elapsed: " << std::chrono::duration<double>(time_end - time_start).count() << " seconds" << std::endl;

	return 0;
}

double** generate_grid(fPtr func, double xmin, double xmax, int xcount, double ymin, double ymax, int ycount)
{
	if (xmin > xmax || ymin > ymax || xcount <= 1 || ycount <= 1)
	{
		return nullptr;
	}

	double** arr = new double*[xcount];

	if (!arr)
	{
		return nullptr;
	}

	double x = xmin;

	double dx = (xmax - xmin) / double(xcount - 1);
	double dy = (ymax - ymin) / double(ycount - 1);

	for (int i = 0; i < xcount; ++i)
	{
		arr[i] = new double[ycount];

		if (!arr[i])
		{
			delete[] arr;
			return nullptr;
		}

		double y = ymin;

		for (int j = 0; j < ycount; ++j)
		{
			arr[i][j] = func(x, y);

			y += dy;
		}

		x += dx;
	}

	return arr;
}