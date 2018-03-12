import numpy as np
import cv2
import os


def main():
	
	DIR = str(input("Enter path to video file for dark clothes detection: ") or "/home/daniel/Documents/FYP/FYP/data/CloudyChopSurfFanore/positive/posCloudyChopSurfFanore4")
	assert os.path.exists(DIR), "Error: Path does not exist at: , "+str(DIR)
	
	colourThreshold(DIR)



def colourThreshold(inputFile):


	upper = np.array([255,20,80], dtype=np.uint8)
	lower = np.array([0,0,0], dtype=np.uint8)

	#feed = "/home/daniel/Documents/FYP/FYP/data/ClearLightChopDoolin/positive/posClearLightChopDoolin2"
	feed = inputFile
	cap = cv2.VideoCapture(feed)
	if not cap.isOpened():
		cap.open(feed)

	if cap.isOpened():
		while True:
			ret, img = cap.read()
			
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

			# Threshold the HSV image to get only dark colors
			mask = cv2.inRange(hsv, lower, upper)
			

			# Bitwise-AND mask and original image
			#mask = cv2.bitwise_not(mask)
			#res = cv2.bitwise_and(img,img, mask= mask)

			kernel = np.ones((5,5),np.uint8)
	
			opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
			closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
			res = cv2.bitwise_and(img,img, mask= closing)
			gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

			
			#for x in range(len(gray[0])):
				#for y in range(len(gray)):
					#if(gray[y][x] != 0):
						#if(x != 0 and y !=0 and x != len(gray) and y != len(gray[0])):
							#if(gray[y+1][x+1] != 0 and gray[y-1][x+1] != 0 and gray[y-1][x-1] != 0 and gray[y+1][x-1] != 0):
								#cv2.rectangle(img,(x-50,y-50),(x+50,y+50),(0,255,255),2)
						


			cv2.imshow('frame',res)
			#cv2.imshow('mask',mask)
			#cv2.imshow('res',gray)
			


			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			
		cap.release()
		cv2.destroyAllWindows()
		
	else:
		print ("Failed to open capture device")



main()