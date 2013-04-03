#ifndef HOG_CUDA_ALLOC_H
#define HOG_CUDA_ALLOC_H

#define CUDA_DESCRIPTOR_ELEMENT( arr, type, x, y, z, width, height, depth ) ((arr)[ (( (z) * ( (width) * (height) ) + (y) * (width) + (x)) )  ] )

void cudaHOGAlloc( float **desc, int descSize, float **hog, int hogSize, float **mag, float **grad, int magGradSize );
void cudaHOGFree( float **desc, float **hog, float **mag, float **grad );

#endif