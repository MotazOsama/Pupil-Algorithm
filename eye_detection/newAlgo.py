import numpy as np 
import cv2 as cv
from eye_test import *
MIN_RATIO = 0.3
PUPIL_MIN = 40.0
PUPIL_MAX = 150.0
INITIAL_ELLIPSE_FIT_THRESHOLD = 1.8
STORNG_PERIMETER_RATIO_RANGE =  .8, 1.1
STRONG_AREA_RATIO_RANGE = .6,1.1
FINAL_PERIMETER_RATIO_RANGE = .6, 1.2
STRONG_PRIOR = None

COLORED_IMAGE = getROI(cv.imread("eye.png"))
IMG = cv.cvtColor(COLORED_IMAGE, cv.COLOR_BGR2GRAY)	
COARSE_PUPIL_WIDTH = IMG.shape[0]/2
PADDING = COARSE_PUPIL_WIDTH/4


def algorithm():
	hist = cv.calcHist([IMG],[0],None,[256],[0,256]); 
	bins = np.arange(hist.shape[0])
	spikes = bins[hist[:,0]>40]
	
	pass;

algorithm();