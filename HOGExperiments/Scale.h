/*
 * Scale.h
 *
 *  Created on: Mar 14, 2012
 *      Author: grimaldo
 */

#ifndef SCALE_H_
#define SCALE_H_

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


template <typename MatType>
MatType Scale(const MatType& original, double scale_factor)
{
	cv::Size scaled_size = cv::Size(scale_factor * original.cols,
									scale_factor * original.rows);

	MatType original_resized;
	cv::resize(original, original_resized, scaled_size);

	return original_resized;
}


#endif /* SCALE_H_ */
