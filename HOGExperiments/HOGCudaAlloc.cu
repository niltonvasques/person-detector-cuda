#include "HOGCudaAlloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

void cudaHOGAlloc( float **desc, int descSize, float **hog, int hogSize, float **mag, float **grad, int magGradSize ){
	cudaMalloc( (void**)desc, descSize );
	cudaMalloc( (void**)hog, hogSize );
	cudaMalloc( mag, magGradSize );
	cudaMalloc( grad, magGradSize );
	cudaMemset( mag, 0, magGradSize );
	cudaMemset( grad, 0, magGradSize );	
}

void cudaHOGFree( float **desc, float **hog, float **mag, float **grad ){
	cudaFree( *desc );
	cudaFree( *hog );
	cudaFree( *mag );
	cudaFree( *grad );
}