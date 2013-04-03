#ifndef GPU_UTILS_H
#define GPU_UTILS_H

void gpuMemcpyDeviceToHost( void ** dst, void ** src, size_t size );
void matrixCpyDeviceToHost( float **matrix, float* device_matrix, size_t pitch, int width, int height );

#endif