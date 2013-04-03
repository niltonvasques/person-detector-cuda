#include "GpuImage.h"

#define MAX_THREADS 1024

using namespace std;
using namespace Cuda;
using namespace Cuda::Device;

__global__ void cudaNativeBGR2GrayF( GpuImage<float> src, GpuImage<float> dst ){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if( tid < src.size.width * src.size.height ){
		dst.ptr(tid)[0] = ( 2989 * src.ptr(tid*3+2)[0] + 5870 * src.ptr(tid*3+1)[0] + 1140 * src.ptr(tid*3+0)[0] ) / 10000; 
	}
}

__global__ void cuda_convert_to( GpuImage<uchar> src,  GpuImage<float> dst, double beta ){
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < src.size.width * src.size.height * src.channels ){
		*dst.ptr(tid) = src.ptr(tid)[0] * beta;
	}
}

__global__ void cuda_convert_float_to_char( GpuImage<float> src,  GpuImage<uchar> dst, double beta ){
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < src.size.width * src.size.height * src.channels ){
		*dst.ptr(tid) = (uchar) (src.ptr(tid)[0] * beta);
	}
}

__global__ void cudaNativeResizeBilinearGrayF( GpuImage<float> src, GpuImage<float> dst ){

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int h2 = dst.size.height, w2 = dst.size.width;
	int w = src.size.width, h = src.size.height;
	const int i = tid / w2;
	const int j = tid % w2;

	if( tid < h2 * w2 ){

		float A, B, C, D, gray;
		int x, y, index ;
		float x_ratio = ((float)(w-1))/w2 ;
		float y_ratio = ((float)(h-1))/h2 ;
		float x_diff, y_diff, ya, yb ;
		x = (int)(x_ratio * j) ;
		y = (int)(y_ratio * i) ;
		x_diff = (x_ratio * j) - x ;
		y_diff = (y_ratio * i) - y ;
		index = y*w+x ;

		// range is 0 to 255 thus bitwise AND with 0xff
		A = src.ptr(index)[0];
		B = src.ptr(index+1)[0];
		C = src.ptr(index+w)[0];
		D = src.ptr(index+w+1)[0];
            
		// Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
		gray = (float)( A * (1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) + C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)  ) ;

		dst.ptr(tid)[0] = gray ;                                   
	}
}

/*

http://tech-algorithm.com/articles/nearest-neighbor-image-scaling/
public int[] resizeBilinearBGR(int[] pixels, int w, int h, int w2, int h2) {
    int[] temp = new int[w2*h2] ;
    int a, b, c, d, x, y, index ;
    float x_ratio = ((float)(w-1))/w2 ;
    float y_ratio = ((float)(h-1))/h2 ;
    float x_diff, y_diff, blue, red, green ;
    int offset = 0 ;
    for (int i=0;i<h2;i++) {
        for (int j=0;j<w2;j++) {
            x = (int)(x_ratio * j) ;
            y = (int)(y_ratio * i) ;
            x_diff = (x_ratio * j) - x ;
            y_diff = (y_ratio * i) - y ;
            index = (y*w+x) ;                
            a = pixels[index] ;
            b = pixels[index+1] ;
            c = pixels[index+w] ;
            d = pixels[index+w+1] ;

            // blue element
            // Yb = Ab(1-w)(1-h) + Bb(w)(1-h) + Cb(h)(1-w) + Db(wh)
            blue = (a&0xff)*(1-x_diff)*(1-y_diff) + (b&0xff)*(x_diff)*(1-y_diff) +
                   (c&0xff)*(y_diff)*(1-x_diff)   + (d&0xff)*(x_diff*y_diff);

            // green element
            // Yg = Ag(1-w)(1-h) + Bg(w)(1-h) + Cg(h)(1-w) + Dg(wh)
            green = ((a>>8)&0xff)*(1-x_diff)*(1-y_diff) + ((b>>8)&0xff)*(x_diff)*(1-y_diff) +
                    ((c>>8)&0xff)*(y_diff)*(1-x_diff)   + ((d>>8)&0xff)*(x_diff*y_diff);

            // red element
            // Yr = Ar(1-w)(1-h) + Br(w)(1-h) + Cr(h)(1-w) + Dr(wh)
            red = ((a>>16)&0xff)*(1-x_diff)*(1-y_diff) + ((b>>16)&0xff)*(x_diff)*(1-y_diff) +
                  ((c>>16)&0xff)*(y_diff)*(1-x_diff)   + ((d>>16)&0xff)*(x_diff*y_diff);

            temp[offset++] = 
                    0xff000000 | // hardcode alpha
                    ((((int)red)<<16)&0xff0000) |
                    ((((int)green)<<8)&0xff00) |
                    ((int)blue) ;
        }
    }
    return temp 
	}

	Nearest Neighboor Interpolate Resize
	public int[] resizePixels(int[] pixels,int w1,int h1,int w2,int h2) {
		int[] temp = new int[w2*h2] ;
		double x_ratio = w1/(double)w2 ;
		double y_ratio = h1/(double)h2 ;
		double px, py ; 
		for (int i=0;i<h2;i++) {
			for (int j=0;j<w2;j++) {
				px = Math.floor(j*x_ratio) ;
				py = Math.floor(i*y_ratio) ;
				temp[(i*w2)+j] = pixels[(int)((py*w1)+px)] ;
			}
		}
		return temp ;
	}

	Nearest Neighboor Interpolate Resize Improve using integer
	public int[] resizePixels(int[] pixels,int w1,int h1,int w2,int h2) {
		int[] temp = new int[w2*h2] ;
		// EDIT: added +1 to account for an early rounding problem
		int x_ratio = (int)((w1<<16)/w2) +1;
		int y_ratio = (int)((h1<<16)/h2) +1;
		//int x_ratio = (int)((w1<<16)/w2) ;
		//int y_ratio = (int)((h1<<16)/h2) ;
		int x2, y2 ;
		for (int i=0;i<h2;i++) {
			for (int j=0;j<w2;j++) {
				x2 = ((j*x_ratio)>>16) ;
				y2 = ((i*y_ratio)>>16) ;
				temp[(i*w2)+j] = pixels[(y2*w1)+x2] ;
			}                
		}                
		return temp ;
	}
*/

__host__ void gpuColorBGR2GrayF( GpuImage<float> src, GpuImage<float> dst ){

	size_t N = src.size.height * src.size.width;
	int blocks = ( (N + MAX_THREADS-1 ) / MAX_THREADS );
	int threads = MAX_THREADS;

	cudaNativeBGR2GrayF<<< blocks, threads >>>( src, dst );

}

__host__ void convert_to_f( GpuImage<uchar> src,  GpuImage<float> dst, double beta ){

	size_t N = src.size.height * src.size.width * src.channels;
	int blocks = ( (N + MAX_THREADS-1 ) / MAX_THREADS );
	int threads = MAX_THREADS;

	cuda_convert_to<<< blocks, threads >>>( src, dst, beta );
}

__host__ void convert_float_to_char( Cuda::Device::GpuImage<float> src,  Cuda::Device::GpuImage<uchar> dst, double beta ){

	size_t N = src.size.height * src.size.width * src.channels;
	int blocks = ( (N + MAX_THREADS-1 ) / MAX_THREADS );
	int threads = MAX_THREADS;

	cuda_convert_float_to_char<<< blocks, threads >>>( src, dst, beta );
}

namespace Cuda{
	__host__ void gpuResizeBilinearGrayF( GpuImage<float> src, GpuImage<float> dst ){

		size_t N = src.size.height * src.size.width;
		int blocks = ( (N + MAX_THREADS-1 ) / MAX_THREADS );
		int threads = MAX_THREADS;

		cudaNativeResizeBilinearGrayF<<< blocks, threads >>>( src, dst );

	}
}