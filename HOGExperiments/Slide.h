/*
 * Slide.h
 *
 *  Created on: Mar 17, 2012
 *      Author: grimaldo
 */

#ifndef SLIDE_H_
#define SLIDE_H_


#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "CalcRect.h"
#include "Detector.h"
#include "SlidingOptions.h"

#include "NonMaxSupression.h"
#include "Scale.h"
#include "Window.h"

template <typename RectOutIter>
cv::Mat3b SlideShowSalientRegion(
		const cv::Mat3b& im,
		const cv::Mat1d& sum_imsaliency,
		const SlidingOptions& opts,
		const double threshold,
		RectOutIter out_iter,
		bool paint_regions = true)
{
	SimplePersonDetector detector;
	int nwindows_ignored = 0;

	cv::Mat3b display;
	if (paint_regions) display = cv::Mat3b(im.rows, im.cols, cv::Vec3b(127, 127, 127));

	std::vector<cv::Rect> salient_regions;

	for (int i = 1; i < im.rows - opts.window.height + 1 + 1; i += opts.step.height)
	{
		for (int j = 1; j < im.cols - opts.window.width + 1 + 1; j += opts.step.width)
		{
			double score = CalcRect(sum_imsaliency, i, j, opts.window.width, opts.window.height) / (opts.window.width * opts.window.height);

			if (score >= threshold)
			{
				const int actual_i = i - 1;
				const int actual_j = j - 1;

				cv::Rect roi_rect(actual_j, actual_i, opts.window.width, opts.window.height);
				salient_regions.push_back(roi_rect);
				*out_iter++ = roi_rect;
			}
		}
	}

	if (paint_regions)
	{
		for (int i = 0; i < salient_regions.size(); ++i)
		{
			cv::Mat3b output(display, salient_regions[i]);
			cv::Mat3b(im, salient_regions[i]).copyTo(output);
		}
	}

	//cv::imshow("default", display);
	//cv::waitKey(0);

	return display;
}

template <typename MatchIter>
int SlideSaliencyGuided(
		const cv::Mat3b& im,
		const cv::Mat1d& sum_imsaliency,
		const SlidingOptions& opts,
		const double threshold,
		MatchIter m_it)
{
	SimplePersonDetector detector;
	int nwindows_ignored = 0;

	for (int i = 1; i < im.rows - opts.window.height + 1 + 1; i += opts.step.height)
	{
		for (int j = 1; j < im.cols - opts.window.width + 1 + 1; j += opts.step.width)
		{
			//std::cout << i << " " << j << std::endl;
			double score = CalcRect(sum_imsaliency, i, j, opts.window.width, opts.window.height) / (opts.window.width * opts.window.height);

			if (score >= threshold)
			{
				const int actual_i = i - 1;
				const int actual_j = j - 1;

				cv::Rect roi_rect(actual_j, actual_i, opts.window.width, opts.window.height);
				cv::Mat roi(im, roi_rect);

				if (detector.Predict(roi))
				{
					cv::Point ipoint(actual_j / opts.scale_factor, actual_i / opts.scale_factor);
					cv::Point fpoint(ipoint.x + (opts.window.width - 1) / opts.scale_factor,
									 ipoint.y + (opts.window.height - 1) / opts.scale_factor);
					*m_it++ = cv::Rect(ipoint, fpoint);
				}
			}
			else
			{
				++nwindows_ignored;
			}
		}
	}

	return nwindows_ignored;
}

template <typename MatchIter>
void SlideAll(
		const cv::Mat3b& im,
		const SlidingOptions& opts,
		MatchIter m_it, 
		Detector &detector = SimplePersonDetector())
{

	//if( detector.isDetectorType("CudaPersonDetectorOneThreadBlock") ){
	//	std::vector<cv::Rect> founds;
	//	detector.PredictMultiWindow( im, founds );
	//	for( int i = 0; i < founds.size(); i++ ){
	//		*m_it++ = founds.at(i);
	//	}
	//	return;
	//}

	for (int i = 0; i < im.rows - opts.window.height + 1; i += opts.step.height)
	{
		//boost::progress_timer loop_timer;
		for (int j = 0; j < im.cols - opts.window.width + 1; j += opts.step.width)
		{
			cv::Rect roi_rect(j, i, opts.window.width, opts.window.height);
			cv::Mat roi(im, roi_rect);

			if (detector.Predict(roi))
			{
				cv::Point ipoint(j / opts.scale_factor, i / opts.scale_factor);
				cv::Point fpoint(ipoint.x + (opts.window.width - 1) / opts.scale_factor,
								 ipoint.y + (opts.window.height - 1) / opts.scale_factor);
				*m_it++ = cv::Rect(ipoint, fpoint);
			}
		}
	}
}

int Multiscale( cv::Mat3b& im, std::vector<cv::Rect> &founds, Detector &detector, SlidingOptions &opts ){
	cv::Size image(im.cols, im.rows);;
	cv::Size window(64, 128);

	std::vector<double> scale_level;
	double scale 		= 1;
	double scale_factor = opts.scale_factor;
	cout << opts.scale_factor << endl;
	int scaleCount = 0;	
	// This 128 is the lower limit of the image size for Graph-based Saliency Detector
	while (cvRound(image.width * scale) > std::max(128, window.width)
			&& cvRound(image.height * scale) > std::max(128, window.height))
	{
		scale_level.push_back(scale);
		scale *= scale_factor;
		scaleCount++;
	}
	opts.scales_count = scaleCount;
	double total_time = 0.0;
	int    nwindows   = 0;
	std::vector<cv::Rect> slideall_detections;
	for (std::vector<double>::const_iterator scale_it = scale_level.begin();
			scale_it != scale_level.end();
			++scale_it)
	{
		SlidingOptions ops;
		ops.scale_factor = *scale_it;
		ops.window       = cv::Size(window.width, window.height);
		ops.step         = cv::Size(window.width / 8, window.height / 8);

		cv::Mat3b imresized = Scale(im, *scale_it);
		SlideAll(
			imresized,
			ops,
			std::back_inserter(slideall_detections),
			detector
		);
		nwindows   += NumberOfWindows(imresized,ops);

		imresized.release();
	}
	NonMaxSupressionResults slideall_nms_results = NonMaxSupression(slideall_detections, 1, 0.2);
	for( vector<cv::Rect>::iterator it =  slideall_nms_results.rects.begin(); it != slideall_nms_results.rects.end(); it++ ){
		founds.push_back(*it);
	}
	opts.number_windows = nwindows;
	return nwindows;
}


#endif /* SLIDE_H_ */
