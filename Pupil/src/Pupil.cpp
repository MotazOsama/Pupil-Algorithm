//============================================================================
// Name        : Pupil.cpp
// Author      : Motaz
// Version     : 1.0.0.0
// Copyright   : Your copyright notice
//============================================================================

#include <cv.h>
#include <highgui.h>
#include "InitialRegionGenerator.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat image;
	image = imread("eye.png", 0);
	InitialRegionGenerator IRG(image);
	return 0;
}
