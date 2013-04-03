#include <iostream>
#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include "HOGLuciano.h"
#include "image.h"
#include "gpu_utils.h"
#include "SlidingOptions.h"

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
	if( this->mode == PROCESSING_MODE::CPU ){
		computeGradientsCPU( data, this->imageWidth, this->imageHeight, this->mag, this->grad, channels );
	}else{
		cudaComputeGradients( data, this->imageWidth, this->imageHeight, &mag, &grad, channels );
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
				cudaComputeDescriptors( bx, by, &cudaDesc, descHeight, descWidth, 
					&mag, &grad, frameW, numHistBins, histBinSpace, blockHeight, blockWidth, cellWidth );
				cudaCalculateL2Hys( numHistBins, blockHeight, blockWidth, &cudaDesc );
				cudaWriteToVector( &cudaHog, i, numHistBins, blockHeight, blockWidth, &cudaDesc );
				i += blockSize;
			}
		}
    }

	if( mode == GPU ){
		gpuMemcpyDeviceToHost( (void**)&hog, (void**)&cudaHog, hogSize*sizeof(float) );
	}
}

int NumberOfWindows_(const cv::Size& image, const cv::Size& window, const cv::Size& step)
{
	const int v = 1 + (image.height - window.height) / step.height;
    const int h = 1 + (image.width - window.width) / step.width;

    return h * v;
}

void HOG::detect( vector<cv::Rect> &founds, Classifier *svm ){
    if( mode == GPU ){
		cv::Size window(54, 108);
		SlidingOptions opts;
		opts.scale_factor = 1;
		opts.window       = cv::Size(window.width, window.height);
		opts.step         = cv::Size(window.width / 8, window.height / 8);
		int nwindows  = NumberOfWindows_(cv::Size( imageWidth, imageHeight), window,opts.step);
		int count = 0;
		for (int i = 0; i < imageHeight - opts.window.height + 1; i += opts.step.height){
			cout << count*100/nwindows << " %" << endl;
			for (int j = 0; j < imageWidth - opts.window.width + 1; j += opts.step.width){
				count++;
				computeWindowFeatures(j,i);
				float score = svm->run( hog, hogSize, LIGHTSVM );
				if ( score > 0 )
				{
					cv::Point ipoint(j / opts.scale_factor, i / opts.scale_factor);
					cv::Point fpoint(ipoint.x + (opts.window.width - 1) / opts.scale_factor,
									 ipoint.y + (opts.window.height - 1) / opts.scale_factor);
					founds.push_back(cv::Rect(ipoint, fpoint));
				}
			}
		}
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
    return exp( -dist_ * dist_ / ( sigma * sigma ) ) / ( sqrt( M_PI * 2 ) * sigma ); 
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