/*
 * CalcRect.h
 *
 *  Created on: Mar 14, 2012
 *      Author: grimaldo
 */

#ifndef CALCRECT_H_
#define CALCRECT_H_

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

inline double CalcRect(const cv::Mat1d& integral, int y, int x, int w, int h)
{
	w -= 1;
	h -= 1;

	return integral[y - 1][x - 1]
	     - integral[y + h][x - 1]
	     - integral[y - 1][x + w]
	     + integral[y + h][x + w];
}// Diag

#endif /* CALCRECT_H_ */
