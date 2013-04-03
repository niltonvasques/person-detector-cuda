#ifndef CUDA_HOG_H
#define CUDA_HOG_H
#include <iostream>
#include "Classifier.h"

typedef struct DeviceHOG_{
	float *mag;
	float *grad;
	float *descriptor;
	float *hog;
	int *angle;
	int blockWidth;
	int blockHeight;
	int descWidth;
	int	descHeight;
	int	cellWidth;
	int	numHistBins;
	int	histBinSpace;
	int	blockStride;
	int	frameW;
	int	frameH;
	int	imageWidth;
	int	imageHeight;
	int	numBlocksW;
	int	numBlocksH;
	int	blockSize;
	int bWHSize;
	int hogSize;
	int overlap;	
	size_t pitch_descriptor;
	size_t pitch_hog;
} DeviceHOG;

typedef struct{
	double linearbias_;
	double threshold;
	double *scores;
}LINEAR_CLASSIFY_SVM;

struct CudaPoint{
	int x;
	int y;
	CudaPoint( int x_, int y_ ){
		x = x_;
		y = y_;
	}
};



class CudaHOG {


private:
	 int blockWidth, blockHeight, 
         descWidth, descHeight, 
         cellWidth,
         numHistBins, histBinSpace,
         blockStride,
		 frameW, frameH, imageWidth, imageHeight,
         numBlocksW, numBlocksH, blockSize;
	int blocks;
	int threads;
	float *device_desc;
    float *device_mag, *device_grad;
    bool fullCircle;
	float *device_hog;
	unsigned char* gray;
	int hogSize;
	DeviceHOG hogS;
	LINEAR_CLASSIFY_SVM svm;
	bool oneThreadWindow;
	Classifier *classifier;

public:
	unsigned char* host_gray;
	CudaHOG(int imageWidth_, int imageHeight_, int frameW_ = 54,int frameH_ = 108, bool oneThreadWindow_ = false, Classifier *classifier_ = NULL,  bool fullCircle_ = false );
	~CudaHOG();	
	void extractFeatures(unsigned char *data, int channels);
	void getFoundLocations( std::vector<CudaPoint> &founds );
	float *getMagnitudeN();
	float* getGradientN();
	void getHOGVectorN(float **matrix);
	double *getScoresN();
	inline int getHOGVectorSize(){
		return hogSize;
	}

	inline int getWindowsCount(){
		return blocks*threads;
	}

	inline int getBlocksCount(){
		return blocks;
	}

	inline int getThreadsPerBlockCount(){
		return threads;
	}

private:
	void computeGradients(DeviceHOG hog, unsigned char* data, int channels);
};

#endif