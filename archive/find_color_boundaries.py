import cv2
import numpy as np
if __name__ == '__main__':
	# Read image
	im = cv2.imread("img/tst_sticks/7.png")
	# Select ROI
	r = cv2.selectROI(im)
	# Crop image
	imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
	imCrop = cv2.medianBlur(imCrop, 15)
	# Display cropped image
	img_hsv = cv2.cvtColor(imCrop, cv2.COLOR_BGR2HSV)
	
	#red
	print(img_hsv[...,0].min())
	print(img_hsv[..., 0].max())
	
	#green
	print(img_hsv[..., 1].min())
	print(img_hsv[..., 1].max())
	
	#blue
	print(img_hsv[..., 2].min())
	print(img_hsv[..., 2].max())
	
	
	cv2.imshow("Image", imCrop)
	cv2.waitKey(0)
