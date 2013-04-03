#ifndef HOG_CUDA_H
#define HOG_CUDA_H

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "GpuImage.h"
#include "GpuTypes.h"

/*
	Cuda Functions Declarations
*/
__global__ void cuda_detect( int x, int y );

using namespace std;

namespace Cuda{

struct Image{
	unsigned char *data;
	unsigned char *dev_data;
	int channels;
	Size size;
	Image( unsigned char* data_, Size size_, int channels_ ) : data(data_), size(size_), channels(channels_)
	{
		size_t length = size.width * size.height * channels * sizeof( unsigned char );
		cudaMalloc( &dev_data, length );
		cudaMemcpy( dev_data, data, length, cudaMemcpyHostToDevice );
	}

	__device__ Image(void) : size( 0, 0, true )
	{	}
		
	void release(){
		cudaFree( dev_data );
	}
};

struct Point{
	int x;
	int y;

	Point( void ) 
	{	}
	Point( int x_, int y_ ) : x( x_ ), y( y_ ) 
	{	}
	__device__ Point( int x_, int y_, bool device ) : x( x_ ), y( y_ ) 
	{	}
};

struct BlockData
{
    BlockData() : histOfs(0), imgOffset() {}
    int histOfs;
    Point imgOffset;
};

struct PixData
{
    size_t gradOfs, qangleOfs;
    int histOfs[4];
    float histWeights[4];
    float gradWeight;
};

struct HOGDescriptor{
	int x;
	int y;
};


class HOGCuda
{
	static HOGCuda *instance;

public:
	enum { L2Hys=0 };
    enum { DEFAULT_NLEVELS=64 };
	static HOGCuda *getInstance();

	static void destroyInstance(){
		delete instance;
	}
 
private:
    HOGCuda() : winSize(64,128), blockSize(16,16), blockStride(8,8),
    	cellSize(8,8), nbins(9), derivAperture(1), winSigma(-1),
        histogramNormType(HOGCuda::L2Hys), L2HysThreshold(0.2), gammaCorrection(true), 
        nlevels(HOGCuda::DEFAULT_NLEVELS)
    {  	
		init();
	}

	void computeBlockHistograms( const Device::GpuImage<uchar>& img );
	void computeGradient(const Device::GpuImage<uchar>& img , Device::GpuImage<float>& grad, Device::GpuImage<uchar>& qangle);

	void init();

public:
	void detect( const Device::GpuImage<uchar> &image, vector<Point> &found_locations );
	Device::GpuImage<float> getGrad();

private:
	Size winSize;
    Size blockSize;
    Size blockStride;
    Size cellSize;
	Size win_stride;
	Size cacheStride;
	Size padding;
    int nbins;
    int derivAperture;
    double winSigma;
    int histogramNormType;
    double L2HysThreshold;
    bool gammaCorrection;
    vector<float> svmDetector;
    int nlevels;
	Device::GpuImage<uchar> qangle;
	Device::GpuImage<float> grad;
};

}

#endif