#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>

__global__ void helloWorld()
{
    printf("Hello world! from thread %d\n", threadIdx.x);
}

int main(int argc, char* argv[])
{
    helloWorld<<<1, argc < 2 ? 5 : std::stoi(argv[1])>>>();
    return 0;
}