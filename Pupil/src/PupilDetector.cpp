/*
 * PupilDetector.cpp
 *
 *  Created on: Jul 1, 2014
 *      Author: motaz
 */

#include "PupilDetector.h"

PupilDetector::PupilDetector(Mat frame) {
	colored_img = frame.clone();
	cvtColor(colored_img, img, COLOR_BGR2GRAY);
	coarse_pupil_width = img.size[0] / 2;
	padding = coarse_pupil_width / 4;
	Mat hist = calculateHistogram();
}

Mat PupilDetector::calculateHistogram() {

	Mat hist;
	float hist_range[] = { 0, 256 };
	int histSize = 256;
	const float* histRangePointer = { hist_range };
	bool uniform = true;
	bool accumulate = false;
	calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRangePointer, uniform,
			accumulate);
	return hist;
}
