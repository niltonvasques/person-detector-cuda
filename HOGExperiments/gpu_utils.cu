#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "gpu_utils.h"

void gpuMemcpyDeviceToHost( void ** dst, void ** src, size_t size ){
	cudaMemcpy( *dst, *src, size, cudaMemcpyDeviceToHost );
}

__global__ void copyMatrixRow(float *device_matrix, float* linear, size_t pitch, int r){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float* row = (float*)((char*)device_matrix + r * pitch);
	linear[tid] = row[tid];
}
void matrixCpyDeviceToHost( float **matrix, float* device_matrix, size_t pitch, int width, int height ){
	float* device_linear;
	assert( cudaMalloc( (void**)&device_linear, sizeof( float ) * width ) == cudaSuccess );
	for( int i = 0; i < height; i++){
		copyMatrixRow<<< width,1 >>>( device_matrix, device_linear, pitch, i );
		assert( cudaMemcpy( matrix[i], device_linear, width * sizeof( float ), cudaMemcpyDeviceToHost ) == cudaSuccess );
	}

	assert( cudaFree( device_linear ) == cudaSuccess );
}
