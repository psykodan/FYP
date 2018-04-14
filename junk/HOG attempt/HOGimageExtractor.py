import cv2
import numpy as np

import os
import sys

#
# This program creates a images for the positive images used for training a HOG classifier 
# By tracking an the desired object with the mouse. The box will be saved as an image for the 
# Object in each image.
#
# Currently only tracks one object
# 
#

ix,iy = -1,-1
# mouse callback function for tracking mouse position on image
def mouse_pos(event,x,y,flags,param):
	global ix, iy			
	if event == cv2.EVENT_MOUSEMOVE:
		ix,iy = x,y

def main():	#User input file and output location

	DIR = input("Enter path to video: ")
	assert os.path.exists(DIR), "Error: Path does not exist at: , "+str(DIR)
	DIR2 = input("Enter path to images: ")
	if not os.path.exists(DIR2):
		check = input("Create directory " + DIR2 + " ? (Y/n): ")
		if(check == "Y" or check == "y" or check == ""):
			os.makedirs(DIR2)
		else:
			DIR2 = input("Enter path to store frames: ")

	posObjectTracker(DIR, DIR2)

def posObjectTracker(dirIn, dirOut):
	
	

	#User specify the window size
	winX = int(input("Enter the width of the tracking window (px): ") or 128)
	winY = int(input("Enter the height of the tracking window (px): ") or 128)
	speed = int(input("Enter the play speed for the tracking (1 = fastest, 200 = v.slow): ") or 125)


	#Load video file
	vidcap = cv2.VideoCapture(dirIn)


	#Name the image by number in accorance with how many files are already present in output dir
	count = len([name for name in os.listdir(dirOut) if os.path.isfile(os.path.join(dirOut, name))])
	success = True
	
	
	cv2.setMouseCallback('image',mouse_pos)
	


	while success:			#loop until vidcap is finished (success = false from vidcap.read())
		success,image = vidcap.read()
		

		if(success == True):
			
			img = image[iy-int(winY/2):iy+int(winY/2),ix-int(winX/2):ix+int(winX/2)] 
			#Convert image to Grayscale for image processing reasons
			#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# save frame as JPEG file
			cv2.imwrite(dirOut + os.sep+ "image%d.jpg" % count, img)     
			count += 1

			
			#Draw a rectangle of the window size around the mouse
			cv2.rectangle(image,(ix-int(winX/2),iy-int(winY/2)),(ix+int(winX/2),iy+int(winY/2)),(255,0,0),5)
			cv2.namedWindow('image')
			cv2.setMouseCallback('image',mouse_pos)
			cv2.imshow('image',image)
			cv2.waitKey(speed)


			
		
			if cv2.waitKey(1) & 0xFF == ord('q'):
				processed = True
				break


		
	cv2.destroyAllWindows()

main()