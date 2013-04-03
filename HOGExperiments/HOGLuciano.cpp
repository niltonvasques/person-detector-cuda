#include "HOGLuciano.h"
#include "HOGVasques.h"

#include <iostream>
#include <cmath>

#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;

#define GAUSSIAN( x, y, mx, my, sigma ) ( exp(-pow(sqrt( pow(( (x) - (mx) ),2)  + pow(( (y) - (my) ),2) ),2)/( sigma*sigma ) )/ ( sqrt( M_PI_ * 2 ) * sigma ) )

#define fmin( a, b ) ( (a)<(b)? (a):(b) )
template<typename T> inline bool isnan(T value){
	return value != value;
}

void HOG::resetDesc() {
	for( int b = 0; b < numHistBins; b++ )
	    for (int y = 0; y < blockHeight; y++)		
	          memset( desc[b][y], 0, sizeof(float) * blockWidth );
}
    
void HOG::resetHog() {
	memset( hog, 0, sizeof(float) * hogSize );
}

void HOG::writeToVector( float *output ) {
	float feat = 0.0;
	
	for( int b = 0; b < numHistBins; b++ ) {
	    for( int y = 0; y < blockHeight; y++ ) {
			for( int x = 0; x < blockWidth; x++ ) {
				feat = desc[b][y][x];
				if( isnan( feat ) ) {
					feat = 0;
				}
				*output++ = feat;
				//printf("%f ", feat );
			}
	    }
	}
}


void HOG::computeGradients( unsigned char* data, int channels ){
	computeGradientsCPU( data, this->imageWidth, this->imageHeight, this->mag, this->grad, channels );
}

/* Compute Gradients in CPU */
void HOG::computeGradientsCPU( unsigned char *data , int width, int height, float* mag, float* grad, int channels ){
	Mat3b img = Mat3b::zeros( height, width );
	img.data = data;
	bool fullCircle = false;
	int frameH = img.rows;
	int frameW = img.cols;
	int p1 = 0, p2 = 0, p3 = 0, p4 = 0, 
         hor = 0, ver = 0;
    float curGrad = 0.0, curMag = 0.0;

	Mat1b image_gray;
	cvtColor(img,image_gray,CV_BGR2GRAY);
	//imshow("Gray",image_gray);
	//waitKey();
    //compute mag and grad    
	for( int i = 1; i < frameH - 1; i++ ) {
		for( int j = 1; j < frameW - 1; j++ ) {
			p1 = (int) image_gray[i][j+1];
			p2 = (int) image_gray[i][j-1];
			p3 = (int) image_gray[i-1][j];
			p4 = (int) image_gray[i+1][j];
		    
			hor = p1 - p2;
			ver = p3 - p4;
			curMag = (float) sqrt( (double)( hor * hor ) + ( ver * ver ) );
			mag[ i * frameW + j ] = (int)curMag;
		    
			// make sure we don't divide by zero when calculating the gradient orientation
			if( curMag > 0.0 ) {
				curGrad = ( (float) ( (float) 180 * acos( (float) hor / (float) curMag ) ) / (float) M_PI_ );
				if( !fullCircle )
					curGrad = float( (int) curGrad % 180 ); //if unsigned, then range it over 0-180 (pedestrian)
				grad[ i * frameW + j ]= (float)curGrad;
			}else {
				grad[ i * frameW + j ]=0;
			}
		}
	}
}

void HOG::computeWindowFeatures( int wx, int wy ){
    int i = 0;
    for( int by = wy; by <= wy + winH - descHeight; by += blockStride ) {
		for( int bx = wx; bx <= wx + winW - descWidth; bx += blockStride ) {
			if( mode == CPU ){
				computeDescriptor( bx, by );
				normalize( L2_HYS );
				writeToVector( &hog[i] );
				i += blockSize;
			}else if ( mode == GPU ){
				//cudaComputeDescriptors( bx, by, &cudaDesc, descHeight, descWidth, 
				//	&mag, &grad, frameW, numHistBins, histBinSpace, blockHeight, blockWidth, cellWidth );
				//cudaCalculateL2Hys( numHistBins, blockHeight, blockWidth, &cudaDesc );
				//cudaWriteToVector( &cudaHog, i, numHistBins, blockHeight, blockWidth, &cudaDesc );
				//i += blockSize;
			}
		}
    }

	if( mode == GPU ){
		//gpuMemcpyDeviceToHost( (void**)&hog, (void**)&cudaHog, hogSize*sizeof(float) );
	}
}

#define OVERLAP			16

void HOG::computeAllWindowFeatures( int wx, int wy ){
}

//===============================================================
//	Method: computeDescriptor
//	Description: compute block descriptor
//	Parameters: points relative to the sliding window
//	output: block descriptor into the feature vector
//===============================================================
void HOG::computeDescriptor( int bx, int by ) {
    
    float curGrad = 0.0, curMag = 0.0, gWeight = 0.0, 
            cellWeight = 0.0, binWeight = 0.0,
            dist = 0.0;
    int i = 0, iBin = 0,
         stepx = 0, stepy = 0,
         mx = 0, my = 0,
         cx = 0, cy = 0,
         dx = 0, dy = 0;
    
    resetDesc();
    for( int y = 0; y < this->descHeight; y++ ) {
		for( int x = 0; x < this->descWidth; x++ ) {
			curGrad		= this->grad[ ((by+y)*this->frameW) + bx+x ]; 
			curMag		= this->mag[ ((by+y)*this->frameW) + bx+x ]; 
			i = int( curGrad / this->histBinSpace );
			if( i >= this->numHistBins ) i = this->numHistBins - 1;
	    
			mx = int( this->descWidth >> 1 );
			my = int( this->descHeight >> 1 );

			gWeight = GAUSSIAN( float(x), float(y), float(mx), float(my), float(mx) );
			stepx = mx;
			stepy = my;
			for( int iy = 0; iy < blockHeight; iy++ ) {
				for( int ix = 0; ix < blockWidth; ix++ ) {
					cx = stepx + stepx * ix;
					cy = stepy + stepy * iy;
					dx = x - cx;
					dy = y - cy;
					dist = sqrt( (float) ( dx * dx ) + ( dy * dy ) );    
					//cell weighting
					cellWeight = 1.0 - fmin( (float) ( (float) dist / (float) cellWidth ), (float) 1.0 );
		    		    
					// histogram bin weighting
					iBin = 0; binWeight = 0;
					circularInterpBin( curGrad - histBinSpace * i, i, &binWeight, &iBin );

					desc[i][iy][ix] += binWeight * cellWeight * (gWeight) * curMag;
					desc[iBin][iy][ix] += (1.0 - binWeight) * cellWeight * (gWeight) * curMag;
				}
			}
		}
    }
}

//===============================================================
//	Method: gaussian
//	Description: compute gaussian weight
//	Parameters:   x 
//		    y 
//		    mx 
//		    my
//		    sigma 
//	Output: gaussian weight
//===============================================================
float HOG::gaussian( float x, float y, float mx, float my, float sigma ) {
    float dist_, dx = x - mx, dy = y - my;

    dist_ =  sqrt( ( dx * dx ) + ( dy * dy ) );
    return exp( -dist_ * dist_ / ( sigma * sigma ) ) / ( sqrt( M_PI_ * 2 ) * sigma ); 
}

//===============================================================
// 	Method: circularInterpBin
// 	Description: interpolated circularly bins into the histogram
// 	Parameters: value - value to be interpolated
// 		     curBin - current HOG bin
//  		     outCoef - interpolation coeficient
//  		     outInterpBin - 
// ===============================================================
void HOG::circularInterpBin( float value, int curBin, float *outCoef, int *outInterpBin ) {
  
    int halfSize = int( histBinSpace >> 1 );
    
    if( value > halfSize ) { // range: (halfSize, binsize]
		*outInterpBin = ( curBin + 1 ) % numHistBins;
		*outCoef = 1.0 - ( ( value - halfSize ) / histBinSpace );
    } else { // range: [0, halfsize]
		*outInterpBin = ( curBin - 1 ) % numHistBins;
		if( *outInterpBin < 0 ) *outInterpBin += numHistBins;
		*outCoef = ( ( value + halfSize ) / histBinSpace );
    }
}

//===============================================================
//	Method: normalize
//	Description: normalize the HOG feature vector
//	Parameters: normType (L2-HYS or L1-SQRT)
// ===============================================================

void HOG::normalize( NORMALYZE_TYPE type ) {
    
    if ( type == NORMALYZE_TYPE::L2_HYS ) calculateL2Hys();    
    if( type == NORMALYZE_TYPE::L1_SQRT ) calculateL1Sqrt();
    
}

//===============================================================
//	Method: calculateL2Hys
//	Description: calculate L2 Hystheresis norm
//	Output: normalized HOG feature
// ===============================================================

void HOG::calculateL2Hys() {
    
    float norm = 0.0, eps = 1.0;
        
    //compute norm
    for( int i = 0; i < numHistBins; i++ )
	for( int j = 0; j < blockHeight; j++ )
	    for( int k = 0; k < blockWidth; k++ )
		norm += desc[i][j][k] * desc[i][j][k];
    //L2-norm
    norm = sqrt( norm + eps ); 
    
    if ( !norm ) norm = 1.0;
    
    // Normalize and threshold ...
    for( int i = 0; i < numHistBins; i++ )
	for( int j = 0; j < blockHeight; j++ )
	    for( int k = 0; k < blockWidth; k++ ) {
	desc[i][j][k] /= norm;
	if( desc[i][j][k] > 0.2 ) desc[i][j][k] = 0.2;
    }
    
    norm = 0.0;
    for( int i = 0; i < numHistBins; i++ )
	for( int j = 0; j < blockHeight; j++ )
	    for( int k = 0; k < blockWidth; k++ )
		norm += desc[i][j][k] * desc[i][j][k];
    
    norm = sqrt( norm + eps );
    if ( !norm ) norm = 1.0;
    
    // and normalize again
    for( int i = 0; i < numHistBins; i++ )
	for( int j = 0; j < blockHeight; j++ )
	    for( int k = 0; k < blockWidth; k++ )
		    desc[i][j][k] /= norm;
    
}

//===============================================================
//	Method: calculateL1Sqrt
//	Description: calculate L1 Sqrt Norm
//	Output: normalized HOG feature
// ===============================================================

void HOG::calculateL1Sqrt() {
    
    float norm = 0.0, eps = 1.0;
    
    //compute norm
    for( int i = 0; i < numHistBins; i++ )
	for( int j = 0; j < blockHeight; j++ )
	    for( int k = 0; k < blockWidth; k++ )
		norm += desc[i][j][k];
    //L1-sqrt
    norm = sqrt( norm + eps ); 
    if ( !norm ) norm = 1;
    
    // normalize
    for( int i = 0; i < numHistBins; i++ )
	for( int j = 0; j < blockHeight; j++ )
	    for( int k = 0; k < blockWidth; k++ )
		    desc[i][j][k] /= norm;
}

//===============================================================
//	Method: extractFeatures
//	Description: extract features (positives and negatives) to be trained
//	Parameters: positive , negative and train files
//	output: positive and negative examples
//===============================================================
int HOG::extractFeatures( const char *posfile, const char *negfile, const char *trainfile ){
    //QFile pos( (const char*) posfile ), neg( (const char*) negfile ), train( (const char*) trainfile );
    //QTextStream streamPos( &pos ), streamNeg( &neg ), streamTrain( &train );
    //QString imagefile;
 //   int numExamples = 0, target = 0;
 //   IplImage* image = cvCreateImage( cvSize( winW, winH ), IPL_DEPTH_8U, 1 );
 //       
 //   qDebug( "Pos file = %s", posfile );
 //   qDebug( "Neg file = %s", negfile );
 //   qDebug( "Train file = %s\n", trainfile );
 //          
 //   train.open( IO_WriteOnly );
 //   
 //   //positive features
 //   if( !pos.open( IO_ReadOnly ) ) return -2;
 //   int index;
 //   while( !pos.atEnd() ) {
	//index = 1;
	//imagefile = streamPos.readLine();
	//image = cvLoadImage( (const char*) imagefile );
	//if( image ) {
	//    if( image->width < winW || image->height < winH )
	//	qDebug( "Image size is smaller than detection windows! Skipping..." );
	//    else {
	//	computeGradients( image );
	//	computeWindowFeatures( 0,0 );
	//	printf("."); fflush( stdout );
	//	
	//	target = 1;
	//	streamTrain << target << " ";
	//	for( int i = 0; i < hogSize; i++ ) {
	//	    streamTrain << index << ":" << hog[i] << " ";
	//	    index++;
	//	}
	//	streamTrain << "\n";
	//	
	//    }
	//    numExamples++;
	//}
 //   }
 //   pos.close();
 //   qDebug("\nPositive feature extraction successfully terminated!\n");
 //   
 //   //negative features
 //   if( !neg.open( IO_ReadOnly ) ) return -3;
 //   while( !neg.atEnd() ) {
	//index = 1;
	//imagefile = streamNeg.readLine();
	//image = cvLoadImage( (const char*) imagefile );
	//if( image ) {
	//    if( image->width < winW || image->height < winH )
	//	qDebug("Image size is smaller than detection windows!");
	//    else {
	//	computeGradients( image );
	//	computeWindowFeatures( 0,0 );
	//	printf("."); fflush( stdout );
	//	
	//	target = -1;
	//	streamTrain << target << " ";
	//	for( int i = 0; i < hogSize; i++ ) {
	//	    streamTrain << index << ":" << hog[i] << " ";
	//	    index++;
	//	}
	//	streamTrain << "\n";
	//	
	//    }
	//    numExamples++;
	//}
 //   }
 //   neg.close();
 //   qDebug("\nNegative feature extraction successfully terminated!\n");
 //   
 //   qDebug("Number of examples = %d", numExamples );
 //   qDebug("Now, please run SVM_LEARN...");
    
    return( 0 );
}
