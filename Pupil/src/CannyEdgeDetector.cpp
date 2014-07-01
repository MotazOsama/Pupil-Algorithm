/*
 * CannyEdgeDetector.cpp
 *
 *  Created on: Jun 24, 2014
 *      Author: motaz
 */

#include "CannyEdgeDetector.h"

CannyEdgeDetector::CannyEdgeDetector(Mat ROI) {
	this->src = ROI;
	this->edges_detected_img = src.clone();
	applyCanny();
}

Mat& CannyEdgeDetector::getEdges() {
	return this->edges_detected_img;
}

void CannyEdgeDetector::applyCanny() {
	blur(src, this->edges_detected_img, Size(7, 7));

	Canny(this->edges_detected_img, this->edges_detected_img, 50, 150, 7);

	imwrite("edges_ouput.png", this->edges_detected_img);
	waitKey(0);
}
