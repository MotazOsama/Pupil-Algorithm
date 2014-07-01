/*
 * IntialRegionGenerator.cpp
 *
 *  Created on: Jun 24, 2014
 *      Author: motaz
 */

#include "InitialRegionGenerator.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cstddef>

using namespace std;

InitialRegionGenerator::InitialRegionGenerator(Mat gray_img) {
	this->gray_img = gray_img;
	String eyes_cascade_name = "haarcascade_mcs_righteye.xml";
	if (!eyes_cascade.load(eyes_cascade_name)) {
		printf("--(!)Error loading haarcascade_mcs_righteye.xml \n");
	}
	this->eye = detect(this->gray_img);
	this->ROI = gray_img(eye).clone();
	imwrite("ROI.png", ROI);
//	showImage(gray_img);
}

void InitialRegionGenerator::showImage(Mat frame) {
	imshow("gray_image.png", frame);
	waitKey(0);
	destroyAllWindows();
}

Rect InitialRegionGenerator::detect(Mat frame) {
	Mat gray_img = frame;
	equalizeHist(gray_img, gray_img);
	std::vector<Rect> eyes;
	this->eyes_cascade.detectMultiScale(gray_img, eyes, 1.1, 2,
			0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	int max_ew = 0;
	int max_eh = 0;
	int max = 0;
	if (eyes.size() != 0) {
		for (size_t i = 0; i < eyes.size(); i++) {
			if (eyes[i].width >= max_ew && eyes[i].height >= max_eh) {
				max_ew = eyes[i].width;
				max_eh = eyes[i].height;
				max = i;
			}

		}
//		rectangle(gray_img, eyes[max], Scalar(255, 255, 0), 2, 1, 0);
		return eyes[max];
	}
	Rect r;
	return r;
}

Mat& InitialRegionGenerator::getROI() {
	return this->ROI;
}
