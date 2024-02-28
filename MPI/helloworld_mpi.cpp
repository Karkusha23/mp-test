#include <iostream>
#include "mpi.h"

int main(int argc, char* argv[])
{
	int ProcessRank, ProcessNum;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcessRank);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcessNum);

	std::cout << "Hello world! from process " << ProcessRank << ". Total processes: " << ProcessNum << std::endl;

	MPI_Finalize();
}