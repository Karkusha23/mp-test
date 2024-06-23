#include "mpi.h"

#include <iostream>
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

void derivative(double* B, double* A, int count, int start_index, int y_size, double dy, double dy2);

void execute_main_thread(int num_threads, int x_size, int y_size, double dy, double dy2)
{
	int total_size = x_size * y_size;

	int el_per_thread = total_size / num_threads;

	double* A = generate_grid(foo, XMIN, XMAX, x_size, YMIN, YMAX, y_size);

	double* B = new double[total_size];

	auto time_start = std::chrono::high_resolution_clock::now();

	int info[2];

	if (!(total_size % num_threads))
	{
		for (int i = 1; i < num_threads; ++i)
		{
			int j = i * el_per_thread;

			bool left_offset = j % y_size;
			bool right_offset = (j + el_per_thread) % y_size != y_size - 1;

			int offset = j - left_offset;
			int to_send = el_per_thread + left_offset + right_offset;

			info[0] = el_per_thread;
			info[1] = (int(left_offset) << 1) + right_offset;

			MPI_Send(info, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(A + offset, to_send, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
	}
	else
	{
		for (int i = 1; i < num_threads - 1; ++i)
		{
			int j = i * el_per_thread;

			bool left_offset = j % y_size;
			bool right_offset = (j + el_per_thread) % y_size != y_size - 1;

			int offset = j - left_offset;
			int to_send = el_per_thread + left_offset + right_offset;

			info[0] = el_per_thread;
			info[1] = (int(left_offset) << 1) + right_offset;

			MPI_Send(info, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(A + offset, to_send, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}

		int el_left = total_size % num_threads;

		bool left_offset = (total_size - el_left) % y_size;

		int offset = total_size - el_left - left_offset;

		info[0] = el_left;
		info[1] = int(left_offset) << 1;

		MPI_Send(info, 2, MPI_INT, num_threads - 1, 0, MPI_COMM_WORLD);
		MPI_Send(A + offset, total_size - offset, MPI_DOUBLE, num_threads - 1, 0, MPI_COMM_WORLD);
	}

	derivative(B, A, el_per_thread, 0, y_size, dy, dy2);

	if (!(total_size % num_threads))
	{
		for (int i = 1; i < num_threads; ++i)
		{
			MPI_Status status;

			MPI_Recv(B + i * el_per_thread, el_per_thread, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
		}
	}
	else
	{
		MPI_Status status;

		for (int i = 1; i < num_threads - 1; ++i)
		{

			MPI_Recv(B + i * el_per_thread, el_per_thread, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
		}

		int el_left = total_size % num_threads;

		MPI_Recv(B + (total_size - el_left), el_left, MPI_DOUBLE, num_threads - 1, 0, MPI_COMM_WORLD, &status);
	}

	auto time_end = std::chrono::high_resolution_clock::now();

	delete[] A;

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
}

void execute_regular_thread(int processRank, int processNum, int x_size, int y_size, double dy, double dy2)
{
	MPI_Status status;

	int* info = new int[2];

	MPI_Recv(info, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

	bool left_offset = info[1] & 0b1;
	bool right_offset = info[1] & 0b10;

	int b_size = info[0];
	int a_size = b_size + left_offset + right_offset;

	delete[] info;

	double* A = new double[a_size];

	MPI_Recv(A, a_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

	double* B = new double[b_size];

	derivative(B, A + left_offset, b_size, ((x_size * y_size) / processNum) * processRank, y_size, dy, dy2);

	delete[] A;

	MPI_Send(B, b_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
	int processRank, processNum;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
	MPI_Comm_size(MPI_COMM_WORLD, &processNum);

	int x_size = argc < 2 ? 100 : std::stoi(argv[1]);
	int y_size = argc < 3 ? 100 : std::stoi(argv[2]);

	if (x_size * y_size < processNum)
	{
		std::cout << "Array size must be bigger than number of threads" << std::endl;
		MPI_Finalize();
		return 0;
	}

	double dy = (YMAX - YMIN) / double(y_size - 1);
	double dy2 = dy * 2.0;

	if (!processRank)
	{
		execute_main_thread(processNum, x_size, y_size, dy, dy2);
	}
	else
	{
		execute_regular_thread(processRank, processNum, x_size, y_size, dy, dy2);
	}

	MPI_Finalize();

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

void derivative(double* B, double* A, int count, int start_index, int y_size, double dy, double dy2)
{
	for (int i = 0; i < count; ++i)
	{
		int index = start_index + i;

		if (!(index % y_size))
		{
			B[i] = (A[i + 1] - A[i]) / dy;
		}
		else if (index % y_size == y_size - 1)
		{
			B[i] = (A[i] - A[i - 1]) / dy;
		}
		else
		{
			B[i] = (A[i + 1] - A[i - 1]) / dy2;
		}
	}
}