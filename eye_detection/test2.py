import numpy as np
import cv2 as cv

class Constants(object):
	pass




def show():
	cv.imshow("image", secImg)
	cv.waitKey(0)
	cv.destroyAllWindows();
	pass

IMG_TO = cv.imread("eye.png", 1); 
secImg = IMG_TO.copy();

show()