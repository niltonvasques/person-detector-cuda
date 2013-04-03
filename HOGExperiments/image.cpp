#include "image.h"
#include <iostream>
#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;


/* Compute Gradients in CPU */
void computeGradientsCPU( unsigned char *data , int width, int height, float* mag, float* grad, int channels ){
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
				curGrad = ( (float) ( (float) 180 * acos( (float) hor / (float) curMag ) ) / (float) M_PI );
				if( !fullCircle )
					curGrad = float( (int) curGrad % 180 ); //if unsigned, then range it over 0-180 (pedestrian)
				grad[ i * frameW + j ]= (float)curGrad;
			}else {
				grad[ i * frameW + j ]=0;
			}
		}
	}
}