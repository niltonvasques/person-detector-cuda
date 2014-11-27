#ifndef NON_MAX_SUPRESSION_H
#define NON_MAX_SUPRESSION_H

#include <iostream>
#include <vector>

#include <opencv/cv.h>

struct DetectionData
{
	std::vector<int> tp;
	std::vector<int> tp_neighbors;
	std::vector<int> fp;
	std::vector<int> fp_neighbors;
	std::vector<int> fn;
};

struct NonMaxSupressionResults
{
	std::vector<cv::Rect> rects;
	std::vector<int> 	  neighbors;
};

NonMaxSupressionResults NonMaxSupression(cv::vector<cv::Rect>& rects, int groupThreshold, double eps)
{
	NonMaxSupressionResults nonmax;
	nonmax.rects = rects;

	std::vector<int> weights;
	cv::groupRectangles(nonmax.rects, nonmax.neighbors, groupThreshold, eps);

	return nonmax;
}

struct FalseAlarmStats
{
	void Add(const cv::Rect& false_alarm, int number_of_neighbors)
	{
		rects.push_back(false_alarm);
		neighbors.push_back(number_of_neighbors);
	}

	std::size_t size() const
	{
		assert(rects.size() == neighbors.size());
		return rects.size();
	}

	std::vector<cv::Rect> rects;
	std::vector<int> neighbors;
};

struct MatchedStats
{
	void Add(const cv::Rect& matched, int number_of_neighbors)
	{
		rects.push_back(matched);
		neighbors.push_back(number_of_neighbors);
	}

	std::size_t size() const
	{
		assert(rects.size() == neighbors.size());
		return rects.size();
	}

	std::vector<cv::Rect> rects;
	std::vector<int> neighbors;
};

struct NonMaxSupressionStats
{
	MatchedStats match;
	FalseAlarmStats falseAlarm;
	std::vector<cv::Rect> misses;
	cv::Mat display;
};

inline NonMaxSupressionStats
Perfomance
	( NonMaxSupressionResults& detections,
	  const std::vector<cv::Rect>& groundtruth,
	  const cv::Mat& original)
{
    double max_size_diff = 1.5;
    double max_pos_diff  = 0.3;

    cv::Mat im = original.clone();

	const std::vector<cv::Rect>& rects = detections.rects;

	std::map<int, bool> missed_map;
	std::vector<cv::Point> gt_center(groundtruth.size(), cv::Point());
	std::vector<double> gt_width(groundtruth.size(), 0.0);
	for (size_t i = 0; i < groundtruth.size(); ++i)
	{
		gt_center[i].x = groundtruth[i].x + (0.5 * groundtruth[i].width);
		gt_center[i].y = groundtruth[i].y + (0.5 * groundtruth[i].height);

		gt_width[i] = 0.5 * sqrt( (double) groundtruth[i].width * groundtruth[i].width
						          + groundtruth[i].height * groundtruth[i].height);

		missed_map[i] = true;
	}

	NonMaxSupressionStats stats;
	int false_alarms = 0;


	for (size_t i = 0; i < rects.size(); ++i)
	{
		cv::Point center( rects[i].x + 0.5 * rects[i].width
						, rects[i].y + 0.5 * rects[i].height);

		double det_width = 0.5 * sqrt( (double) rects[i].width * rects[i].width
			    				       + rects[i].height * rects[i].height);

		unsigned int c;
		for (c = 0; c < groundtruth.size(); ++c)
		{
			double centers_dist = sqrtf( (gt_center[c].x - center.x) * (gt_center[c].x - center.x)
							   	   	   + (gt_center[c].y - center.y) * (gt_center[c].y - center.y) );

#if 0
			std::cout << "(" << "dist=" << centers_dist << " " << "max=" << gt_width[c] * max_pos_diff << ")" << " "
					  << "(" << "width_diff=" << std::fabs(gt_width[i] - det_width) << " " << "max_diff=" << gt_width[c] * max_size_diff << ")" << " "
					  << "\n" << std::endl;
#endif
			if (centers_dist < gt_width[c] * max_pos_diff
				&& det_width < gt_width[c] * max_size_diff
				&& det_width > gt_width[c] / max_size_diff)
			{
				stats.match.Add(rects[i], detections.neighbors[i]);
				missed_map[c] = false;

				cv::rectangle(im, rects[i], CV_RGB(0, 255, 0), 8);
				break;
			}
		}

		if (c == groundtruth.size())
		{
			cv::rectangle(im, rects[i], CV_RGB(255, 0, 0), 8);
			stats.falseAlarm.Add(rects[i], detections.neighbors[i]);
			false_alarms++;
		}
	}

	for (std::map<int, bool>::const_iterator it = missed_map.begin();
		 it != missed_map.end();
		 ++it)
	{
		if (it->second)
		{
			stats.misses.push_back(groundtruth[it->first]);
			cv::rectangle(im, groundtruth[it->first], CV_RGB(127, 127, 127), 8);
		}
	}

	stats.display = im;

#if 0
	cv::imwrite("StepByStep/" + boost::lexical_cast<std::string>(rand()) + ".png", im);
	cv::waitKey(0);

	stats.hitRate     = stats.matches.size();
	stats.missRate    = groundtruth.size() - stats.matches.size();
	stats.falseAlarms = false_alarms;
	assert(stats.missRate >= 0);
#endif

	return stats;
}


#endif
