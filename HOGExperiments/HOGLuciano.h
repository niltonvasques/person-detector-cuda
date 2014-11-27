#ifndef HOG_H_
#define HOG_H_
#include <iostream>
#include <opencv/cv.h>
#include "HOGCudaAlloc.h"
#include "image.h"
#include "gpu_utils.h"
#include "Classifier.h"

typedef enum{ CPU,GPU } PROCESSING_MODE;
typedef enum{ L2_HYS, L1_SQRT } NORMALYZE_TYPE;

class HOG {

protected:
    int blockWidth, blockHeight, 
         descWidth, descHeight, 
         cellWidth,
         numHistBins, histBinSpace,
         blockStride,
         winW, winH, frameW, frameH, imageWidth, imageHeight,
         numBlocksW, numBlocksH, blockSize;
    float ***desc;
	float *cudaDesc;
    float *mag, *grad;
    bool fullCircle;

	PROCESSING_MODE mode;

public:

	float *hog, *cudaHog;
    int hogSize;
	HOG(int frameWidth, int frameHeight, int imageWidth_, int imageHeight_, PROCESSING_MODE mode_ = GPU,  int _winW = 54,int _winH = 108,int _blockStride = 8, int _descWidth = 16, //in pixel
	           int _descHeight = 16, //in pixel
	           int _blockWidth = 2, //number of cells
	           int _blockHeight = 2, //number of cells 
	           int _numHistBins = 9, 
	           bool _fullCircle = false ) { // default: 0-180 - pedestrian){

		this->frameW		= frameWidth;
		this->frameH		= frameHeight;
		this->imageWidth	= imageWidth_;
		this->imageHeight	= imageHeight_;
		this->winW			= _winW;
		this->winH			= _winH;
		this->blockStride	= _blockStride;
		this->descWidth		= _descWidth;
		this->descHeight	= _descHeight;
		this->blockWidth	= _blockWidth;
		this->blockHeight	= _blockHeight;
		this->numHistBins	= _numHistBins;
		this->fullCircle	= _fullCircle;
		this->mode			= mode_;

		/* Alocate Matrix Gradients And Magnitudes */
		if( fullCircle ) histBinSpace = (int)( 360 / numHistBins );
		else histBinSpace = (int)( 180 / numHistBins );
	
		cellWidth = (int)( (float) descWidth / (float) blockWidth );
	
		
	
		///allocate final feature vector
		numBlocksW = int( (float) ( winW - descWidth + blockStride ) / (float) blockStride );
		numBlocksH = int( (float) ( winH - descHeight + blockStride ) / (float) blockStride );
		blockSize = numHistBins * blockWidth * blockHeight;
		hogSize =  blockSize * numBlocksW * numBlocksH;
		hog = new float[hogSize];
		memset( hog, 0, hogSize * sizeof(float) );
		//allocate temp feature matrix
		if( mode == GPU ){
			int descSize = sizeof(float) * numHistBins * blockHeight * blockWidth;
			cudaHOGAlloc( &cudaDesc, descSize, &cudaHog, hogSize*sizeof(float), &mag, &grad, imageWidth * imageHeight * sizeof(float) );
		}else if( mode == CPU ){
			desc = new float**[numHistBins];
			for( int b = 0; b < numHistBins; b++ ) {
				desc[b] = new float*[blockHeight];
				for( int h = 0; h < blockHeight; h++ ) desc[b][h] = new float[blockWidth];
			}			
			//alocate gradients and magnitudes
			mag = new float[ imageWidth * imageHeight ];
			grad = new float[ imageWidth * imageHeight ];

			memset( mag, 0, imageWidth * imageHeight * sizeof(float) );
			memset( grad, 0, imageWidth * imageHeight * sizeof(float) );
		}

		
	}

	~HOG(){
		if( mode == CPU ){
			for( int b = 0; b < numHistBins; b++ ) {
				for( int h = 0; h < blockHeight; h++ ) delete[] desc[b][h];
					delete[] desc[b];
				}
			delete[] desc;
			desc = NULL;
	
			delete hog;
			delete mag;
			delete grad;
		}else if( mode == GPU ){
			cudaHOGFree( &cudaDesc, &hog, &mag, &grad );
		}
	}

	inline void setProcessMode( PROCESSING_MODE _mode){
		this->mode	= _mode;
	}

	inline float* getMagnitude(){
		if( mode == GPU ){
			float * magnitude = new float[ imageWidth * imageHeight ];
			gpuMemcpyDeviceToHost( (void**) &magnitude, (void**)&mag, sizeof(float) * imageWidth * imageHeight );
			return magnitude;
		}
		return mag;
	}

	inline float* getGradient(){
		if( mode == GPU ){
			float * gradient = new float[ imageWidth * imageHeight ];
			gpuMemcpyDeviceToHost( (void**) &gradient, (void**)&grad, sizeof(float) * imageWidth * imageHeight );
			return gradient;
		}
		return this->grad;
	}

	void computeGradients( unsigned char *data, int channels );
	void computeWindowFeatures( int wx, int wy );
	void computeAllWindowFeatures( int wx, int wy );
	void detect( vector<cv::Rect> &founds, Classifier *svm );
	int extractFeatures( const char *posfile, const char *negfile, const char *trainfile );

private:
	void computeDescriptor( int bx, int by );
	float gaussian( float x, float y, float mx, float my, float sigma );
	void circularInterpBin( float value, int curBin, float *outCoef, int *outInterpBin );
	void normalize( NORMALYZE_TYPE type );
	void calculateL2Hys();
    void calculateL1Sqrt();
	void writeToVector( float *output );

	void resetDesc();    
    void resetHog();
};

#endif
