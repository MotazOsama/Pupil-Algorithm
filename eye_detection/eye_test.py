import numpy as np
import cv2 as cv
def getROI(img):
	eye_cascade = cv.CascadeClassifier('haarcascade_mcs_righteye.xml')

	eyes = eye_cascade.detectMultiScale(img)

	max_ew = 0; 
	max_eh = 0; 
	max_ex= 0; 
	max_ey= 0; 
	maxIndex = 0; 
	counter = 0;
	for (ex,ey,ew,eh) in eyes:
		if ((ew > max_ew) & (eh > max_eh)):
			max_eh = eh; 
			max_ew = ew; 
			max_ex = ex; 
			max_ey = ey;
			maxIndex = counter;
			pass
		counter = counter+1;


	eye = eyes[maxIndex];
	ROI = img[eye[1]:eye[1]+max_eh, eye[0]:eye[0]+max_ew];

	# cv.rectangle(img,(max_ex,max_ey),(max_ex+max_ew,max_ey+max_eh),(255,255,0),2)

	# cv.imwrite('ROI.png',ROI)
	return ROI
	pass