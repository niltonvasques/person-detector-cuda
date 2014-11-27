#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv/cv.h>
#include <vector>

#include "SlidingOptions.h"

//struct DetectorRect{
//	int x,y,width,height;
//	DetectorRect( int x_, int y_, int width_, int height_ ) : x(x_), y(y_), width(width_), height(height_)
//	{	}
//};

class Detector
{

public:

	virtual bool Predict( const cv::Mat3b &im ) = 0;
	virtual void PredictMultiWindow( const cv::Mat3b &im, std::vector<cv::Rect> &founds  ) = 0;
	/** Return number of windows processed **/
	virtual int PredictMultiscale( const cv::Mat3b &im, std::vector<cv::Rect> &founds, SlidingOptions &opts  ){ return 0; };
	virtual bool isDetectorType( string type ) = 0;
	virtual std::string getDetectorName() = 0;
	virtual void options( SlidingOptions &opts ) { };

};

#endif
