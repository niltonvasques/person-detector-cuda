/*
 * Window.h
 *
 *  Created on: Mar 14, 2012
 *      Author: grimaldo
 */

#ifndef _WINDOW_H_
#define _WINDOW_H_

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>

#include "SlidingOptions.h"

int NumberOfWindows(const cv::Size& image, const cv::Size& window, const cv::Size& step)
{
	const int v = 1 + (image.height - window.height) / step.height;
    const int h = 1 + (image.width - window.width) / step.width;

    return h * v;
}


int NumberOfWindows(const cv::Mat& im, const SlidingOptions& opts)
{
	const int v = 1 + (im.rows - opts.window.height) / opts.step.height;
    const int h = 1 + (im.cols - opts.window.width) / opts.step.width;

    return h * v;
}

template <typename ScaleIter>
int AccumulatedNumberOfWindows(
		const cv::Mat& original,
		ScaleIter it, ScaleIter e,
		const SlidingOptions& opts)
{
	int total = 0;
	cv::Mat im = original.clone();

	cv::Mat im_resized;

	for (; it != e; ++it)
	{
		im = Scale(im, *it);
		total += NumberOfWindows(im, opts);
	}

    return total;
}


#endif /* WINDOW_H_ */
