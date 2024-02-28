#include <iostream>
#include <string>
#include <omp.h>

int main(int argc, char* argv[])
{
	omp_set_num_threads(argc < 2 ? 5 : std::stoi(argv[1]));

	#pragma omp parallel
	{
		#pragma omp critical
		{
			std::cout << "Hello world! from thread " << omp_get_thread_num() << ". Total threads: " << omp_get_num_threads() << std::endl;
		}
	}

	return 0;
}