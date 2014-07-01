/*
 * PupilDetector.h
 *
 *  Created on: Jul 1, 2014
 *      Author: motaz
 */

#ifndef PUPILDETECTOR_H_
#define PUPILDETECTOR_H_
#include <cv.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

class PupilDetector {
public:
	Mat colored_img, img;
	int coarse_pupil_width, padding;
	const float static min_ratio = 0.3;
	const float static pupil_min = 40.0;
	const float static pupil_max = 150.0;
	const float static inital_ellipse_fit_threshhold = 1.8;
	const float static strong_perimeter_ratio_range_max = 1.1;
	const float static strong_perimeter_ratio_range_min = 0.8;

	const float static strong_area_ratio_range_min = .6;
	const float static strong_area_ratio_range_max = 1.1;
	const float static final_perimeter_ratio_range_min = .6;
	const float static final_perimeter_ratio_range_max = 1.2;

	const float static strong_prior = NULL;
	PupilDetector(Mat img);
	Mat calculateHistogram();
};

#endif /* PUPILDETECTOR_H_ */
