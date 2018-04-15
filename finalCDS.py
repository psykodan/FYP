import numpy as np
import cv2
import os


def main():
	
	#Stream input - live feed = ID value, video = path to video
	DIR = str(input("Enter path to video file for casualty detection: ") or "demoVideos/demo1")
	if(DIR.isdigit()):
		DIR = int(DIR)
	else:
		assert os.path.exists(DIR), "Error: Path does not exist at: , "+str(DIR)

	#Set the cascade for use to be our custom Haar cascade classifier
	cascade = cv2.CascadeClassifier("cascade.xml")

	#Set the confidence cut off threshold for the detection method
	threshold = float(input("Enter a threshold value of Haar cascade classification confidence: ") or 3.6)

	#Set the detection buffer size in pixels
	box = int(input("Enter a value for the width of a box for memory check: ") or 40)
	
	casualtyDetection(DIR, cascade, threshold, box)



def casualtyDetection(inputFile, cascade, threshold, box):

	detectionBuffer = []

	#Open feed
	feed = inputFile
	cap = cv2.VideoCapture(feed)
	if not cap.isOpened():
		cap.open(device)


	#kernel for use in morphological transformations
	kernel = np.ones((5,5),np.uint8)

	#Colour filtering upper and lower bounds
	upper = np.array([255,50,100], dtype=np.uint8)
	lower = np.array([0,0,0], dtype=np.uint8)
														

	if cap.isOpened():
		while True:
			ret, img = cap.read()

			#gray stream for haar cascade classifier
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			#hsv stream for colour filtering
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

			#Detection function that uses our Haar cascade classifier
			#Scale factor is a parameter specifying how much the image size is reduced at each image scale
			#Min Neighbors is a parameter specifying how many neighbors each candidate rectangle should have to retain it
			#Output Reject Levels allows for the confidendce weights to be outputted
			bodies = cascade.detectMultiScale3(gray, scaleFactor=100, minNeighbors=100,outputRejectLevels = True)
			detections = bodies[0]
			neighbours = bodies[1]
			weights = bodies[2]


			
			for a in range(len(weights)):
				if(weights[a][0] >= threshold):
					if(len(detectionBuffer)>0):
						for(x,y,w,h) in detectionBuffer:

							#Check for cross over of coordinates within detection buffer size = box
							if(detections[a][0] <= x+box and detections[a][0] >= x-box and detections[a][1] <= y+box and detections[a][1] >= y-box):
								
								#Create a colour region of interest
								roi_hsv = hsv[int(detections[a][1]):int(detections[a][1])+box, int(detections[a][0]):int(detections[a][0])+box]
								
								#Threshold the HSV image to get only dark colors
								mask = cv2.inRange(roi_hsv, lower, upper)
								
								#opening
								erosion1 = cv2.erode(mask,kernel,iterations = 1)
								dilation1 = cv2.dilate(erosion1,kernel,iterations = 1)

								#closing
								dilation2 = cv2.dilate(dilation1,kernel,iterations = 1)
								erosion2 = cv2.erode(dilation2,kernel,iterations = 1)

								#use the mask to find regions of colour specified
								res = cv2.bitwise_and(roi_hsv,roi_hsv, mask= erosion2)

								#create grayscale version for simpler value checking
								HSV2BGR = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
								gray_res = cv2.cvtColor(HSV2BGR, cv2.COLOR_BGR2GRAY)

								#Check if middle of region of interest is not empty
								if(gray_res[int(len(gray_res)/2)][int(len(gray_res[0])/2)] != 0):

									#draw detection rectangle
									cv2.rectangle(img,(detections[a][0],detections[a][1]),(detections[a][0]+detections[a][2],detections[a][1]+detections[a][3]),(0,255,255),2)

								break

						else:
							dummy=1


			cv2.imshow('img',img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
			detectionBuffer = detections


		cap.release()
		cv2.destroyAllWindows()
		
	else:
		print ("Failed to open capture device")



def casualtyDetectionImageStream(dirIn, cascade, threshold, box):


	#Detection method is the same, only the output is written to file and shown visually

	#Sort files in numerical order
	dirFiles = os.listdir(dirIn)
	dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

	processed = False
	detect = False
	detectionBuffer = []

	kernel = np.ones((5,5),np.uint8)
	upper = np.array([255,50,100], dtype=np.uint8)
	lower = np.array([0,0,0], dtype=np.uint8)


	while(processed == False):
		
		for file in dirFiles:
			detect = False
			img = cv2.imread(dirIn + file)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

			# image, reject levels level weights.
			bodies = cascade.detectMultiScale3(gray, scaleFactor=100, minNeighbors=100,outputRejectLevels = True)
			detections = bodies[0]
			neighbours = bodies[1]
			weights = bodies[2]


			
			for a in range(len(weights)):
				if(weights[a][0] >= threshold):
					if(len(detectionBuffer)>0):
						for(x,y,w,h) in detectionBuffer:
							if(detections[a][0] <= x+box and detections[a][0] >= x-box and detections[a][1] <= y+box and detections[a][1] >= y-box):
								roi_hsv = hsv[int(detections[a][1]):int(detections[a][1])+box, int(detections[a][0]):int(detections[a][0])+box]
								# Threshold the HSV image to get only dark colors
								mask = cv2.inRange(roi_hsv, lower, upper)
								

								#opening
								erosion1 = cv2.erode(mask,kernel,iterations = 1)
								dilation1 = cv2.dilate(erosion1,kernel,iterations = 1)

								#closing
								dilation2 = cv2.dilate(dilation1,kernel,iterations = 1)
								erosion2 = cv2.erode(dilation2,kernel,iterations = 1)


								res = cv2.bitwise_and(roi_hsv,roi_hsv, mask= erosion2)
								HSV2BGR = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
								gray_res = cv2.cvtColor(HSV2BGR, cv2.COLOR_BGR2GRAY)


								if(gray_res[int(len(gray_res)/2)][int(len(gray_res[0])/2)] != 0):
									detect=True
									
									#Write a successful detection to file
									line = 'positive/'+file + ' ' + str(detections[a][0]) + ' ' + str(detections[a][1]) + ' ' + str(detections[a][0]+detections[a][2]) + ' ' + str(detections[a][1]+detections[a][3]) + '\n'
									with open('results.txt','a') as f:
										f.write(line)
										f.close()
								break

						else:
							dummy = 1


			if(detect == False):
				#Write the absence of a detection to file
				line = 'positive/'+file + ' NONE\n'
				with open('results.txt','a') as f:
					f.write(line)
					f.close()

			
			detectionBuffer = detections

			if(file == dirFiles[-1]):
				processed = True
				cv2.destroyAllWindows()
	
			if cv2.waitKey(1) & 0xFF == ord('q'):
				processed == True
				break
		break

	cv2.destroyAllWindows()


main()