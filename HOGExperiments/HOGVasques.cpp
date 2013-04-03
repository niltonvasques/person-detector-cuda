#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <opencv\ml.h>
#include "LinearClassify.h" 
#include "HOGVasques.h"
#include "Scale.h"

//#define REMAP 1
#define DEBUG
using namespace cv;

void HOGVasques::computeGradients(unsigned char* data, int width, int height, int channels, float *mag, float *grad){
	float *dataF = new float [ width * height * channels ];
	//Remap
	for( int i = 0; i < width * height*channels ; i++){
#ifdef REMAP
		int px = data[i];
		dataF[i]= sqrt((float)px);
#else
		dataF[i] = data[i];
#endif
	}
	
	bool fullCircle = false;
	float p1 = 0, p2 = 0, p3 = 0, p4 = 0, dx = 0, dy = 0;
    float curGrad = 0.0, curMag = 0.0;
	double sum = 0;
	if( channels == 1 ){
		for( int i = 1; i < height - 1; i++ ) {
			for( int j = 1; j < width - 1; j++ ) {
				p1 =  dataF[ i * width + j+1 ];
				p2 =  dataF[ i * width + j-1 ];
				p3 =  dataF[ (i+1) * width + j ];
				p4 =  dataF[ (i-1) * width + j ];
		    
				dx = p1 - p2;
				dy = p3 - p4;
				curMag = (float) sqrt( (double)( dx * dx ) + ( dy * dy ) );
				mag[ i * width + j ] = curMag;
				sum += curMag;
				// make sure we don't divide by zero when calculating the gradient orientation
				if( curMag > 0.0 ){ 
					//curGrad = (atan( (float)dy/ (float)dx + M_PI/2)*180)/M_PI;
					//A = (np.arctan(YX + np.pi/2.0)*180)/np.pi
					//curGrad = ( (float) ( (float) 180 * acos( (float) dx / (float) curMag ) ) / (float) M_PI );
					curGrad = atan2( (float)dy, (float)dx ) * 180 / M_PI + 180;
					if( !fullCircle ) curGrad = float( (int) curGrad % 180 ); //if unsigned, then range it over 0-180 (pedestrian)
					grad[ i * width + j ]	= (float)curGrad;
				} else {
					grad[ i * width + j ]	= 0;
				}
			}
		}
	} else {
		float magChannels[3];
		for( int i = 1; i < height - 1; i++ ) {
			for( int j = 1; j < width - 1; j++ ) {
				curMag = 0;
				for( int c = 0; c < channels; c++) {
					p1 =  dataF[ (i*width*channels + j*channels+1)+c];
					p2 =  dataF[(i*width*channels + (j-1)*channels)+c];
					p3 =  dataF[((i+1)*width*channels + j*channels)+c];
					p4 =  dataF[((i-1)*width*channels + j*channels)+c];		    
					dx = p1 - p2;
					dy = p3 - p4;
					magChannels[c] = (float) sqrt( (double)( dx * dx ) + ( dy * dy ) );
					curMag = ( curMag < magChannels[c] ? magChannels[c] : curMag );
				}
				mag[ i * width + j ] = curMag;
				sum +=curMag;
				// make sure we don't divide by zero when calculating the gradient orientation
				if( curMag > 0.0 ){ 
					//curGrad = (atan( (float)dy/ (float)dx + M_PI/2)*180)/M_PI;
					//A = (np.arctan(YX + np.pi/2.0)*180)/np.pi
					//curGrad = ( (float) ( (float) 180 * acos( (float) dx / (float) curMag ) ) / (float) M_PI );
					curGrad = atan2( (float)dy, (float)dx ) * 180 / M_PI + 180;
					if( !fullCircle ) curGrad = float( (int) curGrad % 180 ); //if unsigned, then range it over 0-180 (pedestrian)
					grad[ i * width + j ]	= (float)curGrad;
				}else {
					grad[ i * width + j ]	= 0;
				}
			}
		}
	}
	delete dataF;
	sum /= (width*height);

#ifdef DEBUG
	IplImage *magnitude = cvCreateImage( cvSize( width, height ), IPL_DEPTH_8U, 1 );
	for( int i = 0; i < width*height; i++) magnitude->imageData[i] = mag[i];
	IplImage *magnitudeResize = cvCreateImage( cvSize( width*4, height*4 ), IPL_DEPTH_8U, 1 );
	cvResize( magnitude, magnitudeResize );
	cvShowImage("MAGNITUDE",magnitudeResize);
	cvWaitKey(1);
	cvReleaseImage( &magnitude );
	cvReleaseImage( &magnitudeResize );

	IplImage *edge = cvCreateImage( cvSize( width, height ), IPL_DEPTH_8U, 1 );
	for( int i = 0; i < width*height; i++) edge->imageData[i] = ( mag[i] > sum ? 0 : 255 );
	IplImage *edgeResize = cvCreateImage( cvSize( width*4, height*4 ), IPL_DEPTH_8U, 1 );
	cvResize( edge, edgeResize );
	cvShowImage("EDGE",edgeResize);
	cvWaitKey(1);
	cvReleaseImage( &edge );
	cvReleaseImage( &edgeResize );
#endif
}

vector<Positives> HOGVasques::detect( unsigned char* data, int width, int height, int channels, float *x){
	float *mag	= new float[width*height];
	float *grad = new float[width*height];
	memset( mag, 0, width*height*sizeof(float) ); 
	memset( grad, 0, width*height*sizeof(float) );

	computeGradients( data, width, height, channels, mag, grad );


	float* hog = new float[ hogSize ];
	memset(hog, 0, hogSize * sizeof(float));

	//Calculando os block descriptors

#ifdef DEBUG
	int scale = 4;
	IplImage* src = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, channels);
	for( int i = 0; i < width*height*channels; i++) src->imageData[i] = (char)data[i];
	IplImage* resize_ = cvCreateImage( cvSize(width*scale, height*scale), IPL_DEPTH_8U, channels);
	IplImage* resize;
	cvResize( src, resize_ );
	if( channels == 3 ){
		resize = cvCreateImage( cvSize(width*scale, height*scale), IPL_DEPTH_8U, 1);
		cvCvtColor( resize_, resize, CV_BGR2GRAY );
		cvReleaseImage( &resize_ );
	}else{
		resize = resize_;
	}
	cvReleaseImage( &src );
#endif
	//memset( resize->imageData, 255, resize->width*resize->height);

	int hogOffset = 0;
	for( int i = 0; i < detectWindowHeight - blockSpaceStride; i+=blockSpaceStride ){
		for( int j = 0; j < detectWindowWidth - blockSpaceStride; j+=blockSpaceStride ){
#ifdef DEBUG
			CvPoint x1 = cvPoint( j*scale, i*scale );
			CvPoint x2 = cvPoint( (j+blockSpaceStride*2)*scale, (i+blockSpaceStride*2)*scale );
			cvRectangle( resize, x1, x2, cvScalar(0,255,0) );
#endif
			//Blocks
			for(int by = 0; by < blockHeight*cellHeight; by++){
				for( int bx = 0; bx < blockWidth*cellWidth; bx++){
					int offset = (i+by) * width + j+bx;
					float orientation = grad[ offset ];
					float magnitude = mag[ offset ];
					int bin = orientation / angleRange;
					bin = ( bin >= nBins ? bin - 1: bin < 0 ? bin+nBins : bin );
					float gWeight = gaussianWeight[(by)*cellWidth*blockWidth + bx];

					///* Interpolação Trilinear */
					float weight = magnitude*gWeight;
					//int hogPosition = hogOffset*blockWidth*blockHeight*nBins + ( by/cellHeight * blockWidth * nBins ) + ( bx/cellWidth * nBins ) + bin;
					//hog[ hogPosition ] += (weight);
					float value[3];
					value[0] = (by)+0.5;
					value[1] = (bx)+0.5;
					value[2] = orientation;

					float pi[3];
					for( int m = 0; m < 2; m++ ) pi[m] = 1 - 0.5 +  (value[m] - cellWidth)/cellWidth;
					pi[2] = 4.5 - 0.5 +  (value[2] - 90.0)/angleRange;

					int fi[3];
					int ci[3];
					float delta[3];
					for( int m = 0; m < 3; m++ ){
						fi[m] = floor(pi[m]);
						ci[m] = ceil(pi[m]);
						delta[m] = fi[m] - pi[m];
					}
					bool lowervalid[3];
					bool uppervalid[3];
					bool warp_[3] = { false, false, true };
					int tvbin_[3] = { blockHeight, blockWidth, nBins };
					for( int m = 0; m < 3; m++ ){
						lowervalid[m] = true;
						uppervalid[m] = true;
					}
					for (int m= 0; m< 3; ++m) {
						if (warp_[m]) {
							ci[m] %= tvbin_[m];
							fi[m] %= tvbin_[m];
							if (fi[m] < 0)
								fi[m] += tvbin_[m];
							if (ci[m] < 0)
								ci[m] += tvbin_[m];
						} else {
							if (ci[m] >=tvbin_[m]-0.5 || ci[m] < -0.5)
								uppervalid[m] = false;
							if (fi[m] < -0.5 || fi[m] >=tvbin_[m]-0.5)
								lowervalid[m] = false;
						}
					}
					float c[3];
					for( int m = 0; m < 3; m++) c[m] = 1 - delta[m];

					for (int m= 0; m< 2; ++m) {
						if ((m==0 && !lowervalid[0]) || (m && !uppervalid[0]))
							continue;
						float iwt = m ? delta[0] : c[0];
						int ipos = m ? ci[0] : fi[0];

						for (int n= 0; n< 2; ++n) {
							if ((n==0 && !lowervalid[1]) || (n && !uppervalid[1]))
								continue;
							float jwt = iwt*(n ? delta[1] : c[1]);
							int jpos = n ? ci[1] : fi[1];

							for (int k= 0; k< 2; ++k) {
								if ((k==0 && !lowervalid[2]) || (k && !uppervalid[2]))
									continue;
								float kwt = jwt*(k ? delta[2] : c[2]);
								int kpos = k ? ci[2] : fi[2];
								{
									int hogPosition = hogOffset*blockWidth*blockHeight*nBins + ( ipos * blockWidth * nBins ) + ( jpos * nBins ) + kpos;
									hog[ hogPosition ] += (weight*kwt);
								}
							}
						}
							
					}
				}
			}
			hogOffset++;
		}
	}
#ifdef DEBUG
	drawOrientationsBins( (unsigned char*)resize->imageData, resize->width, resize->height,scale, hog , 255);
	cvShowImage("Region",resize);
	cvWaitKey(1);
	memset(resize->imageData, 0, resize->width*resize->height);
#endif

	normalizeL2Hys( hog );
#ifdef DEBUG
	drawOrientationsBins( (unsigned char*)resize->imageData, resize->width, resize->height,scale, hog , 255);
	cvShowImage("HogNormalize",resize);
	cvWaitKey(1);
	cvReleaseImage( &resize );
#endif

	//float* hog2 = new float[ hogSize ];
	//int numBlocksWidth	= (detectWindowWidth / blockSpaceStride) - 1;
	//for( int i = 0, nBH = 0; i < height - blockSpaceStride; i+=blockSpaceStride, nBH++ ){
	//	for( int j = 0, nBW = 0; j < width - blockSpaceStride; j+=blockSpaceStride, nBW++ ){
	//		for( int b = 0; b < nBins; b++ ) {
	//			for( int y = 0; y < blockHeight; y++ ) {
	//				for( int x = 0; x < blockWidth; x++ ) {
	//					int offset = nBH *numBlocksWidth* nBins* blockHeight* blockWidth+ nBW*nBins*blockHeight*blockWidth+
	//						b*blockHeight*blockWidth + y * blockWidth + x;
	//					int offset2 = nBH *numBlocksWidth* nBins* blockHeight* blockWidth+ nBW*nBins*blockHeight*blockWidth+
	//						y*blockWidth*nBins + x * nBins + b;
	//					hog2[ offset ] = hog[ offset2 ];
	//				}
	//			}
	//		}
	//	}
	//}

	//LinearClassify svm;
	//float score = svm(hog);
	//cout<< "Score: " << score << endl;
	//FILE *fOut = fopen( "E://hog_features.txt","w+");
	//if( fOut != NULL ){
	//	fprintf(fOut, "SCORE: %f\n",score);
	//	for( int i = 0; i < hogSize; i++)
	//		fprintf(fOut, "HOG[%d] - %f\n",i,hog[i]);
	//	fclose(fOut);
	//}
	//delete hog2;

	vector<Positives> founds;

	founds.push_back( Positives(0,0,64,128) );

	//if( x != NULL ){
	//	memcpy( x, hog, hogSize*sizeof(float) );
	//}

	delete mag;
	delete grad;
	delete hog;
	return founds;
}

void HOGVasques::drawOrientationsBins( unsigned char *data, int width, int height, float scale, const float *hog , int gray){
	int hogOffset = 0;
	for( int i = 0; i < detectWindowHeight - blockSpaceStride; i+=blockSpaceStride){
		for( int j = 0; j < detectWindowWidth - blockSpaceStride; j+=blockSpaceStride){
			for( int by = 0; by < blockHeight; by++ ){
				for( int bx = 0; bx < blockWidth; bx++ ){
					int byOffset = i + by*cellHeight;
					int bxOffset = j + bx*cellWidth;
					//Draw Orientations
					float maxBin = 0;
					float maxAngle = 0;
					float maxBin2 = 0;
					float maxAngle2 = 0;
					for( int k = 0; k < nBins; k++){
						int hogPosition = hogOffset*blockWidth*blockHeight*nBins + ( by * blockWidth * nBins ) + ( bx * nBins );
						if( hog[ hogPosition+k ] > maxBin ){
							maxBin2 = maxBin;
							maxAngle2 = maxAngle;
							maxBin = hog[ hogPosition+k ];
							maxAngle = k * angleRange;
						}else if( hog[ hogPosition+k ] > maxBin2){
							maxBin2 = hog[ hogPosition+k ];
							maxAngle2 = k * angleRange;
						}
					}{
					float radianos = (M_PI) / (180.0/(180.0-maxAngle));
					int axisx = cellWidth/2*scale, axisy = cellHeight/2*scale;
					int ny, nx;
					int pontas = 0;
					for( int x = 0; pontas < 2; x++){
						nx = (x * cos (radianos)) + axisx;
						ny = (x * sin (radianos)) + axisy;
						if( nx > 0 && nx < axisx*4 && ny > 0 && ny < axisy*4){
							int offset = (i*scale+ny+by*cellHeight*scale) * width+ j*scale + nx + bx*cellWidth*scale;
							if( offset < width * height )
								data[ offset ] = gray;
						}
						if( nx > axisx*2 ) pontas++, x=0;
						if( nx < 0 ) pontas++;
						if( ny > axisy*2 ) pontas++, x=0;
						if( ny < 0 ) pontas++;
						if( pontas == 1 ) x-=2;
					}}
				}
			}
			hogOffset++;
		}
	}	
}

void HOGVasques::normalizeL2Hys(float *hog) {
    
    float norm = 0.0, eps = 1.0;
	int hogOffset = 0;        
	for( int i = 0; i < detectWindowHeight - blockSpaceStride; i+=blockSpaceStride){
		for( int j = 0; j < detectWindowWidth - blockSpaceStride; j+=blockSpaceStride){
			float norm = 0.0, eps = 1.0;
			int hogPosition = hogOffset*blockWidth*blockHeight*nBins;
			//compute norm
			for( int i = 0; i < nBins * blockHeight * blockWidth; i++ )
				norm += hog[hogPosition + i] * hog[hogPosition + i];
					//L2-norm
			norm = sqrt( norm + eps ); 
    
			if ( !norm ) norm = 1.0;
    
			// Normalize and threshold ...
			for( int i = 0; i < nBins * blockHeight * blockWidth; i++ ){
				hog[hogPosition + i] /= norm;
				if( hog[hogPosition + i] > 0.2 ) hog[hogPosition + i] = 0.2;
			}
    
			norm = 0.0;
			for( int i = 0; i < nBins * blockHeight * blockWidth; i++ ){
				norm += hog[hogPosition + i] * hog[hogPosition + i];
			}
    
			norm = sqrt( norm + eps );
			if ( !norm ) norm = 1.0;
    
			// and normalize again
			for( int i = 0; i < nBins * blockHeight * blockWidth; i++ ){
				hog[hogPosition + i] /= norm;
			}

			hogOffset++;
		}
	}
    
}

#define M_PI_DIV_2_ 2.5066282746310005024
float HOGVasques::Gaussian( float x, float y, float mx, float my, float sigma ) {
    float dist_, dx = x - mx, dy = y - my;
    dist_ =  sqrt( ( dx * dx ) + ( dy * dy ) );
    return exp( -dist_ * dist_ / ( sigma * sigma ) ) / ( M_PI_DIV_2_ * sigma ); 
}

