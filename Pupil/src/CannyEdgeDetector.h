/*
 * CannyEdgeDetector.h
 *
 *  Created on: Jun 24, 2014
 *      Author: motaz
 */

#ifndef CANNYEDGEDETECTOR_H_
#define CANNYEDGEDETECTOR_H_

#include <cv.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

class CannyEdgeDetector {
public:
	Mat src, edges_detected_img;

	CannyEdgeDetector(Mat ROI);
	Mat& getEdges();
	void applyCanny();
};

#endif /* CANNYEDGEDETECTOR_H_ */
