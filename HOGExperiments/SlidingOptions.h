/*
 * SlidingOptions.h
 *
 *  Created on: Mar 14, 2012
 *      Author: grimaldo
 */

#ifndef SLIDINGOPTIONS_H_
#define SLIDINGOPTIONS_H_

struct SlidingOptions
{
	cv::Size step;
	cv::Size window;
	double   scale_factor;
	int scales_count;
	int number_windows;

	void copyTo( SlidingOptions &opts ){
		opts.step = step;
		opts.window = window;
		opts.scales_count = scales_count;
		opts.number_windows = number_windows;
		opts.scale_factor = scale_factor;
	}
};


#endif /* SLIDINGOPTIONS_H_ */
