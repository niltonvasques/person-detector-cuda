#ifndef IMAGE_H
#define IMAGE_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#define GAUSSIAN( x, y, mx, my, sigma ) ( exp(-pow(sqrt( pow(( (x) - (mx) ),2)  + pow(( (y) - (my) ),2) ),2)/( sigma*sigma ) )/ ( sqrt( M_PI * 2 ) * sigma ) )

/* GPU Methods */
void cudaPowerLawGammaCorrection(unsigned char* pixelsSrc,unsigned char* pixelsDst, int width, int height, int channels,double gamma);

void cudaComputeGradients(unsigned char* pixels, int width, int height,float **mag,float **grad ,int channels);
void cudaComputeDescriptors(int bx, int by, float ** device_desc, int descHeight, int descWidth, float **device_mag, float **device_grad,
	int frameW, int numHistBins, int histBinSpace, int blockHeight, int blockWidth,int cellWidth);
void cudaCalculateL2Hys( int numHistBins, int blockHeight, int blockWidth, float **device_desc );
void cudaWriteToVector( float **device_hog, int index , int numHistBins, int blockHeight, int blockWidth, float **device_desc );

void cudaBGR2Gray(unsigned char* bgr, unsigned char *gray, int width, int height);

void cudaComputeHOG(const int imageWidth,const int imageHeight, const int frameW, const int frameH, const int overlap,
					float **mag, float **grad, float** desc, const int descHeight, const int descWidth, const int blockStride, const int numHistBins,
					const int histBinSpace, const int blockHeight, const int blockWidth, const int blockSize, const int cellWidth );

/*CPU Methods */
void computeGradientsCPU( unsigned char *data, int width, int height, float* mag, float* grad, int channels );

//
//void cudaDebugEnabled();
//void cudaDebugDisable();

#endif