/*
 * =====================================================================================
 *
 *       Filename:  HOGCuda.cu
 *
 *    Description:  Cuda implements of Histogram Oriented of Gradients for Person Detect
 *
 *
 *         Author:  Nilton Vasques
 *        Company:  iVision UFBA
 *		  Created on: Jun 18, 2012
 *
 * =====================================================================================
 */
#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>

#include "HOGCuda.h"
#include "persondetectorwt.tcc"

using namespace Cuda;

template<typename _Tp> static inline _Tp gcd(_Tp a, _Tp b)
{
    if( a < b )
        std::swap(a, b);
    while( b > 0 )
    {
        _Tp r = a % b;
        a = b;
        b = r;
    }
    return a;
}

size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}

template<class T> void uploadConstant(const char* name, const T& value) 
{ 
    //cudaSafeCall( cudaMemcpyToSymbol(name, &value, sizeof(T)) ); 
	cudaMemcpyToSymbol(name, &value, sizeof(T));
}

int power_2up(unsigned int n)
{
    if (n < 1) return 1;
    else if (n < 2) return 2;
    else if (n < 4) return 4;
    else if (n < 8) return 8;
    else if (n < 16) return 16;
    else if (n < 32) return 32;
    else if (n < 64) return 64;
    else if (n < 128) return 128;
    else if (n < 256) return 256;
    else if (n < 512) return 512;
    else if (n < 1024) return 1024;
    return -1; // Input is too big
}

#ifndef div_up
#define div_up(n, grain) (((n) + (grain) - 1) / (grain))
#endif

__constant__ int cnbins;
__constant__ int cblock_stride_x;
__constant__ int cblock_stride_y;
__constant__ int cnblocks_win_x;
__constant__ int cnblocks_win_y;
__constant__ int cblock_hist_size;
__constant__ int cblock_hist_size_2up;
__constant__ int cdescr_size;
__constant__ int cdescr_width;

__constant__ float SVM_DETECTOR[PERSON_WEIGHT_VEC_LENGTH+1];

__device__ void device_compute_gradient(Image img, float *grad, Image *qangle,
                                    Size paddingTL, Size paddingBR, bool gammaCorrection) 
{
	Size gradsize(img.size.width + paddingTL.width + paddingBR.width,
		img.size.height + paddingTL.height + paddingBR.height, true);
}

__global__ void global_detect( Image image, Point *founds, Size winStride, Size cacheStride, Size padding ){
	//Image qangle;
	//float *grad;
	//Size paddedImgSize(image.size.width + padding.width*2, image.size.height + padding.height*2, true);
	//device_compute_gradient( image, grad, &qangle, padding, padding, false );

}


HOGCuda *HOGCuda::instance = 0;

HOGCuda *HOGCuda::getInstance(){
	if( !HOGCuda::instance )
		HOGCuda::instance = new HOGCuda();
	return HOGCuda::instance;
}

void HOGCuda::init(){
	cout << "init()" << endl;
	cudaMemcpyToSymbol( "SVM_DETECTOR", DETECTOR, (PERSON_WEIGHT_VEC_LENGTH+1) * sizeof(double) );

	win_stride = Size( cellSize );
	cacheStride = Size( gcd(win_stride.width, blockStride.width),gcd(win_stride.height, blockStride.height) );
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);

	uploadConstant("cnbins", nbins);
	uploadConstant("cblock_stride_x", blockStride.width);
    uploadConstant("cblock_stride_y", blockStride.height);
    uploadConstant("cnblocks_win_x", 7);
    uploadConstant("cnblocks_win_y", 15);

    int block_hist_size = nbins * 2 * 2;
    uploadConstant("cblock_hist_size", block_hist_size);

    int block_hist_size_2up = power_2up(block_hist_size);    
    uploadConstant("cv::gpu::hog::cblock_hist_size_2up", block_hist_size_2up);

    int descr_width = 7 * block_hist_size;
    uploadConstant("cv::gpu::hog::cdescr_width", descr_width);

    int descr_size = descr_width * 15;
    uploadConstant("cv::gpu::hog::cdescr_size", descr_size);

}

int numPartsWithin(int size, int part_size, int stride) 
{
    return (size - part_size + stride) / stride;
}

Size numPartsWithin(Size size, Size part_size, Size stride) 
{
    return Size(numPartsWithin(size.width, part_size.width, stride.width),
                numPartsWithin(size.height, part_size.height, stride.height));
}

#define FOUND_LENGTH 10
void HOGCuda::detect( const Device::GpuImage<uchar> &image, vector<Point> &found_locations ){

    computeBlockHistograms(image);

	Size wins_per_img = numPartsWithin(image.size, winSize, win_stride);
    //labels.create(1, wins_per_img.area(), CV_8U);

    //hog::classify_hists(win_size.height, win_size.width, block_stride.height, block_stride.width, 
    //                    win_stride.height, win_stride.width, img.rows, img.cols, block_hists.ptr<float>(), 
    //                    detector.ptr<float>(), (float)free_coef, (float)hit_threshold, labels.ptr());

    //labels.download(labels_host);
    //unsigned char* vec = labels_host.ptr();
    //for (int i = 0; i < wins_per_img.area(); i++)
    //{
    //    int y = i / wins_per_img.width;
    //    int x = i - wins_per_img.width * y;
    //    if (vec[i]) 
    //        hits.push_back(Point(x * win_stride.width, y * win_stride.height));
    //}
}

void HOGCuda::computeBlockHistograms(const Device::GpuImage<uchar>& img ){
	grad = Device::GpuImage<float>( img.size, 1 );
	qangle	=	Device::GpuImage<uchar>( img.size, 1 );
	computeGradient(img, grad, qangle);
	printf( " ponteiro %p\n",qangle.ptr(0) );
	//grad.release();
	//qangle.release();
    //size_t block_hist_size = getBlockHistogramSize();
    //Size blocks_per_img = numPartsWithin(img.size(), block_size, block_stride);
    //block_hists.create(1, block_hist_size * blocks_per_img.area(), CV_32F);

    //hog::compute_hists(nbins, block_stride.width, block_stride.height,
    //                   img.rows, img.cols, grad, qangle, (float)getWinSigma(), 
    //                   block_hists.ptr<float>());

    //hog::normalize_hists(nbins, block_stride.width, block_stride.height, img.rows, img.cols, 
    //                     block_hists.ptr<float>(), (float)threshold_L2hys);
}

Device::GpuImage<float> HOGCuda::getGrad(){
	return grad;
}

template <int nthreads, int correct_gamma>
__global__ void compute_gradients_8UC1_kernel(int height, int width, const Device::GpuImage<uchar> img, 
                                              float angle_scale, Device::GpuImage<float> grad, Device::GpuImage<uchar> qangle)
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;

	const unsigned char* row = (const unsigned char*)&img.data[blockIdx.y];

    __shared__ float sh_row[nthreads + 2];

    if (x < width) 
        sh_row[threadIdx.x + 1] = row[x]; 
    else 
        sh_row[threadIdx.x + 1] = row[width - 2];

    if (threadIdx.x == 0)
        sh_row[0] = row[max(x - 1, 1)];

    if (threadIdx.x == blockDim.x - 1)
        sh_row[blockDim.x + 1] = row[min(x + 1, width - 2)];

    __syncthreads();
    if (x < width)
    {
        float dx;

        if (correct_gamma)
            dx = sqrtf(sh_row[threadIdx.x + 2]) - sqrtf(sh_row[threadIdx.x]);
        else
            dx = sh_row[threadIdx.x + 2] - sh_row[threadIdx.x];

        float dy = 0.f;
        if (blockIdx.y > 0 && blockIdx.y < height - 1)
        {
            float a = ((const unsigned char*)&img.data[(blockIdx.y + 1)])[x];
            float b = ((const unsigned char*)&img.data[(blockIdx.y - 1)])[x];
            if (correct_gamma)
                dy = sqrtf(a) - sqrtf(b);
            else
                dy = a - b;
        }
        float mag = sqrtf(dx * dx + dy * dy);

		float ang = (atan2f(dy, dx) + (float)M_PI) * angle_scale - 0.5f;
        int hidx = (int)floorf(ang);
        ang -= hidx;
        hidx = (hidx + cnbins) % cnbins;

        ((uchar2*)&qangle.data[(blockIdx.y)])[x] = make_uchar2(hidx, (hidx + 1) % cnbins);
        //((float2*)&grad.data[(blockIdx.y)])[x] = make_float2(mag * (1.f - ang), mag * ang);
		((float2*)&grad.data[(blockIdx.y)])[x] = make_float2( 255, 255 );
    }
}

void compute_gradients_8UC1(int nbins, int height, int width, const Device::GpuImage<uchar>& img, 
                            float angle_scale, Device::GpuImage<float>& grad, Device::GpuImage<uchar>& qangle, bool correct_gamma)
{
    const int nthreads = 256;

    dim3 bdim(nthreads, 1);
    dim3 gdim(div_up(width, bdim.x), div_up(height, bdim.y));

    if (correct_gamma)
        compute_gradients_8UC1_kernel<nthreads, 1><<<gdim, bdim>>>(
                height, width, img, angle_scale, grad, qangle);
    else
        compute_gradients_8UC1_kernel<nthreads, 0><<<gdim, bdim>>>(
                height, width, img, angle_scale, grad, qangle);
	cudaThreadSynchronize();
    //cudaSafeCall(cudaThreadSynchronize());
}

void HOGCuda::computeGradient(const Device::GpuImage<uchar>& img , Device::GpuImage<float>& grad, Device::GpuImage<uchar>& qangle){
	assert( img.channels == 1 || img.channels == 4 );

	float angleScale = (float)(nbins / M_PI);
    switch (img.channels) {
        case 1:
			compute_gradients_8UC1(nbins, img.size.height, img.size.width, img, angleScale, grad, qangle, false);
            break;
        case 4:
            //compute_gradients_8UC4(nbins, img.rows, img.cols, img, angleScale, grad, qangle, gamma_correction);
            break;
    }
}





