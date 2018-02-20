import cv2
import numpy as np
import os
import sys

#
# This program creates a description for the positive images used for training a haar cascade classifier 
# By tracking an the desired object with the mouse. The coords are appended to the description for the 
# Object in each image.
#
# Currently only tracks one object
# Works best with a video converted to images of frames
#

ix,iy = -1,-1
# mouse callback function for tracking mouse position on image
def mouse_pos(event,x,y,flags,param):
	global ix, iy			
	if event == cv2.EVENT_MOUSEMOVE:
		ix,iy = x,y

def main():	#User input file and output location

	DIR = input("Enter path to positive images directory: ")
	assert os.path.exists(DIR), "Error: Path does not exist at: , "+str(DIR)
	DIR2 = input("Enter path to store info.dat: ")
	if not os.path.exists(DIR2):
		check = input("Create directory " + DIR2 + " ? (Y/n): ")
		if(check == "Y" or check == "y" or check == ""):
			os.makedirs(DIR2)
		else:
			DIR2 = input("Enter path to store frames: ")

	posObjectTracker(DIR, DIR2)

def posObjectTracker(dirIn, dirOut):
	
	processed = False

	#User specify the window size
	winX = int(input("Enter the width of the tracking window (px): ") or 100)
	winY = int(input("Enter the height of the tracking window (px): ") or 100)
	speed = int(input("Enter the play speed for the tracking (1 = fastest, 200 = v.slow): ") or 125)
	loc = input("Enter path to positive images relative to where haar cascade will be trained from: ")

	while(processed == False):
		


		#Sort files in numerical order
		dirFiles = os.listdir(dirIn)
		dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

		#Show first image to prepare for tracking object with mouse
		img = cv2.imread(dirIn + os.sep + dirFiles[0],cv2.IMREAD_GRAYSCALE)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,'Press "s" to start',(100,500), font, 2,(255,255,255),2,cv2.LINE_AA)
		cv2.imshow('image',img)
		cv2.setMouseCallback('image',mouse_pos)
		

		#Start tracking when "s" is pressed
		if cv2.waitKey(1) & 0xFF == ord('s'):

			#Stream in directory of images
			for file in dirFiles:
				img = cv2.imread(dirIn + os.sep + file,cv2.IMREAD_GRAYSCALE)
				
				#Draw a rectangle of the window size around the mouse
				cv2.rectangle(img,(ix-int(winX/2),iy-int(winY/2)),(ix+int(winX/2),iy+int(winY/2)),(255,0,0),5)
				cv2.namedWindow('image')
				cv2.setMouseCallback('image',mouse_pos)
				cv2.imshow('image',img)
				cv2.waitKey(speed)

				# Write to description file the image name, num of objects, x y of top left of window and size of window
				writeX = ix-int(winX/2)
				writeY = iy-int(winY/2)
				line = loc+ os.sep +file+' 1 ' + str(writeX) + ' ' + str(writeY) + ' ' + str(winX) + ' ' + str(winY) + '\n'
				with open(dirOut + os.sep + 'info.dat','a') as f:
					f.write(line)

				if(file == dirFiles[-1]):
					processed = True

				if cv2.waitKey(1) & 0xFF == ord('q'):
					processed = True
					break


		
	cv2.destroyAllWindows()

main()