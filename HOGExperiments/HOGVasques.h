#ifndef HOG_VASQUES_H
#define HOG_VASQUES_H

#include <iostream>
#include <vector>

#include "Detector.h"
#include "gaussian.tcc"

#define M_PI_       3.14159265358979323846

using namespace std;

class Positives{
public:
	int x;
	int y;
	int width;
	int height;

	Positives( int x_, int y_, int width_, int height_ ){
		this->x			= x_;
		this->y			= y_;
		this->width		= width_;
		this->height	= height_;
	}
};

class HOGVasques {
	int cellWidth;
	int cellHeight;
	int blockWidth;
	int blockHeight;
	int nBins;
	int angleRange;
	int blockSpaceStride;
	int detectWindowWidth;
	int detectWindowHeight;
	int hogSize;
	float* gaussianWeight;
public:

	HOGVasques(){
		cellWidth			= 8;
		cellHeight			= 8;
		blockWidth			= 2;
		blockHeight			= 2;
		nBins				= 9;
		angleRange			= 180 / nBins;
		blockSpaceStride	= 8;
		detectWindowWidth	= 64;
		detectWindowHeight	= 128;
		gaussianWeight		= new float[ 256 ];
		//float sigma = 0.5 * blockWidth*cellWidth;
		memcpy( gaussianWeight, GAUSSIAN_WEIGHTS, 256 * sizeof(float) );
		//printf("[ ");
		//for( int i = 0; i < cellHeight*blockHeight; i++){
		//	for(int j = 0; j < cellHeight*blockHeight; j++){
		//		gaussianWeight[i*cellHeight*blockHeight +j] = Gaussian( float(j), float(i), cellWidth/2, cellWidth/2, cellWidth/2 );
		//		printf(" %f ",gaussianWeight[i*cellHeight*blockHeight +j]);
		//	}
		//	printf("\n");
		//}
				//gaussianWeight[i*cellWidth+j] = (1/sqrtf( 2 * M_PI_ * sigma ) ) * 
				//				exp( (float)-(((powf(i*cellWidth*blockWidth,2))+(powf(j*cellHeight*blockHeight,2)))/(2*sigma*sigma)) );
		int numBlocksWidth	= (detectWindowWidth / blockSpaceStride) - 1;
		int numBlocksHeight = (detectWindowHeight / blockSpaceStride) - 1;
		hogSize				= numBlocksWidth*numBlocksHeight*blockHeight*blockWidth*nBins;
	}

	~HOGVasques(){
		delete gaussianWeight;
	}

	int getFeaturesLenght(){
		return hogSize;
	}

	vector<Positives> detect(unsigned char* data, int width, int height, int channels, float *x = NULL);
	void drawOrientationsBins( unsigned char *data, int width, int height,float scale, const float *hog , int gray = 0);
	static float Gaussian( float x, float y, float mx, float my, float sigma );

private:
	void normalizeL2Hys(float *hog);
	void computeGradients(unsigned char* data, int width, int height, int channels, float *mag, float *grad);

};

#endif