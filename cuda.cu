#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "hw8.h"
#include <stdio.h>

__global__ void addCalculateKernel(const int *image, int *dest,
		unsigned int part_size, int arr_size , int mod) {
	int tid = threadIdx.x;

	int image_start = part_size * tid;
	int image_end = image_start + part_size + ((tid == THREADS -1)? mod :0);
//	if (image_end > arr_size)
//		image_end = arr_size;
	int dest_start = arr_size * tid;
	int dest_end = dest_start + arr_size;

	// zero srtarting counters
	for (int j = dest_start; j < dest_end; j++) {
		dest[j] = 0;
	}

	for (int j = image_start; j < image_end; j++)
		dest[dest_start + image[j]]++;

}

__global__ void addMergeKernel(int *histogram, int *temp_arrays, int arr_size) {
	int tid = threadIdx.x;
	// zero starting counters
	for (int i = 0; i < RANGE_SIZE; i++) {
		histogram[i] = 0;
	}
	// merge results. each thread summarize one cell in each temp array
	//for (int i = 0; i < RANGE_SIZE; i++)
	for (int i = 0; i < THREADS; i++)
		histogram[tid] += temp_arrays[arr_size * i + tid];
}

int* calculateHistogramm(int *image, unsigned int size, int arr_size) {
	int *dev_image = 0;
	int *dev_dest_hist = 0;
	int *dev_histogram = 0;
	int *histogram;
	myIntArrCalloc(&histogram, RANGE_SIZE);
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	// Allocate GPU buffers for three vectors (two input, one output).
	cudaStatus = cudaMalloc((void**) &dev_image, size * sizeof(int));
	cudaStatus = cudaMalloc((void**) &dev_histogram, RANGE_SIZE * sizeof(int));
	cudaStatus = cudaMalloc((void**) &dev_dest_hist,
			arr_size * sizeof(int) * (arr_size / 4));

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_image, image, size * sizeof(int),
			cudaMemcpyHostToDevice);


	// Launch a kernel on the GPU with one thread for each element.
	addCalculateKernel <<<1, THREADS>>>(dev_image, dev_dest_hist, size/THREADS, arr_size , size%THREADS);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n",
				cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr,
				"cudaDeviceSynchronize returned error code %d after launching addCalculateKernel!\n",
				cudaStatus);
		goto Error;
	}

	addMergeKernel <<<1, THREADS>>>(dev_histogram, dev_dest_hist , arr_size);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n",
				cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr,
				"cudaDeviceSynchronize returned error code %d after launching addCalculateKernel!\n",
				cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(histogram, dev_histogram, RANGE_SIZE * sizeof(int),
			cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	Error: cudaFree(dev_image);
	cudaFree(dev_histogram);
	cudaFree(dev_dest_hist);

	return histogram;
}


