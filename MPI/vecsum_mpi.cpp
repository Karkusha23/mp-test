#include "mpi.h"

#include <iostream>
#include <string>
#include <chrono>

int* generate_array(int size, int max);

int sum(int* arr, int size);

void execute_main_thread(int num_threads, int arr_size)
{
	int el_per_thread = arr_size / num_threads;

	int* arr = generate_array(arr_size, 100);

	auto time_start = std::chrono::high_resolution_clock::now();

	if (!(arr_size % num_threads))
	{
		for (int i = 1; i < num_threads; ++i)
		{
			int j = i * el_per_thread;

			MPI_Send(&el_per_thread, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(arr + j, el_per_thread, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
	}
	else
	{
		for (int i = 1; i < num_threads - 1; ++i)
		{
			int j = i * el_per_thread;

			MPI_Send(&el_per_thread, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&arr[j], el_per_thread, MPI_INT, i, 0, MPI_COMM_WORLD);
		}

		int el_left = arr_size % num_threads;

		MPI_Send(&el_left, 1, MPI_INT, num_threads - 1, 0, MPI_COMM_WORLD);
		MPI_Send(&arr[arr_size - el_left], el_left, MPI_INT, num_threads - 1, 0, MPI_COMM_WORLD);
	}

	int result = sum(arr, el_per_thread);

	for (int i = 1; i < num_threads; ++i)
	{
		int subsum;
		MPI_Status status;

		MPI_Recv(&subsum, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

		result += subsum;
	}

	auto time_end = std::chrono::high_resolution_clock::now();

	delete[] arr;

	std::cout << "Total sum: " << result << std::endl;
	std::cout << "Time elapsed: " << std::chrono::duration<double>(time_end - time_start).count() << " seconds" << std::endl;
}

void execute_regular_thread()
{
	int arr_size;
	MPI_Status status;

	MPI_Recv(&arr_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

	int* arr = new int[arr_size];

	MPI_Recv(arr, arr_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

	int result = sum(arr, arr_size);

	delete[] arr;

	MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
	int processRank, processNum;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
	MPI_Comm_size(MPI_COMM_WORLD, &processNum);

	int arr_size = argc < 2 ? 100 : std::stoi(argv[1]);

	if (arr_size < processNum)
	{
		std::cout << "Array size must be bigger than number of threads" << std::endl;
		MPI_Finalize();
		return 0;
	}

	if (!processRank)
	{
		execute_main_thread(processNum, arr_size);
	}
	else
	{
		execute_regular_thread();
	}

	MPI_Finalize();

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

int sum(int* arr, int size)
{
	int result = 0;

	for (int i = 0; i < size; ++i)
	{
		result += arr[i];
	}

	return result;
}