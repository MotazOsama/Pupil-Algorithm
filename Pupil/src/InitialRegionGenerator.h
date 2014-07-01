/*
 * InitialRegionGenerator.h
 *
 *  Created on: Jun 24, 2014
 *      Author: motaz
 */
#ifndef INTIALREGIONGENERATOR_H_
#define INTIALREGIONGENERATOR_H_

#include <cv.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace cv;

class InitialRegionGenerator {
public:
	Mat gray_img;
	Mat ROI;
	CascadeClassifier eyes_cascade;
	Rect eye;
	InitialRegionGenerator(Mat gray_img);
	Rect detect(Mat frame);
	void showImage(Mat frame);
	Mat& getROI();
};

#endif /* INTIALREGIONGENERATOR_H_ */
