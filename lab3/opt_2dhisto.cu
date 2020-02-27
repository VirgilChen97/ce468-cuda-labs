#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

#define BIN_COUNT 1024

__global__ void generate_hist(uint32_t* input, uint32_t* global_bins)
{
	int Tid  = blockIdx.x * blockDim.x + threadIdx.x;
	int numThreads = blockDim.x * gridDim.x;
	
    // shared memory to store partial histogram data
	__shared__ int s_Hist[BIN_COUNT];	

	// Clear the buffer before using
	for (int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x) {
		s_Hist[pos] = 0;
	}
	__syncthreads ();

	// Start calculating partial Histogram
	for (int pos = Tid; pos < INPUT_HEIGHT * INPUT_WIDTH; pos += numThreads) {
		//if (s_Hist[input[pos]] < 255) {
			atomicAdd(s_Hist + input[pos], 1);
		//}
	}
	__syncthreads();

	//update global histogram
	for(int pos = threadIdx.x; pos < BIN_COUNT; pos += numThreads) {
		//if(global_bins[threadIdx.x] < 255) {
			atomicAdd(global_bins + pos, s_Hist[pos]);
		//}
	}
}

__global__ void convertTo8(uint32_t* global_bins, uint8_t* device_bins)
{
	int Tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(global_bins[Tid] < 255) {
		device_bins[Tid] = (uint8_t)global_bins[Tid];
	}
	else {
		device_bins[Tid] = (uint8_t)255;
	}	
}

void* AllocateDevice(size_t size)
{
	void *addr;
	cudaMalloc(&addr, size);
	return addr;
}

void MemCpyToDevice(void* dest, void* src, size_t size)
{
	cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

void CopyFromDevice(void* dest, void* src, size_t size)
{
	cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
}

void FreeDevice(void* addr)
{
	cudaFree(addr);
}

void opt_2dhisto(uint32_t* device_input, uint32_t* global_bins, uint8_t* device_bins)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    cudaMemset(global_bins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
    generate_hist<<<16, 1024>>>(device_input, global_bins);
	convertTo8<<<1, 1024>>>(global_bins, device_bins);
	cudaThreadSynchronize();
}

