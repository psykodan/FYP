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
	
	haarDetection(DIR, cascade, threshold)



def haarDetection(inputFile, cascade, threshold):

	#Open feed
	feed = inputFile
	cap = cv2.VideoCapture(feed)
	if not cap.isOpened():
		cap.open(device)

	if cap.isOpened():
		while True:
			ret, img = cap.read()

			#gray stream for haar cascade classifier
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			
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
					cv2.rectangle(img,(detections[a][0],detections[a][1]),(detections[a][0]+detections[a][2],detections[a][1]+detections[a][3]),(0,255,255),2)

					

			cv2.imshow('img',img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		



		cap.release()
		cv2.destroyAllWindows()
		
	else:
		print ("Failed to open capture device")



def haarDetectionImageStream(dirIn, cascade, threshold):

	#Detection method is the same, only the output is written to file and shown visually

	#Sort files in numerical order
	dirFiles = os.listdir(dirIn)
	dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

	processed = False
	detect = False

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
					detect=True
					
					#Write a successful detection to file
					line = 'positive/'+file + ' ' + str(detections[a][0]) + ' ' + str(detections[a][1]) + ' ' + str(detections[a][0]+detections[a][2]) + ' ' + str(detections[a][1]+detections[a][3]) + '\n'
					with open('results.txt','a') as f:
						f.write(line)
						f.close()



			if(detect == False):
				#Write the absence of a detection to file
				line = 'positive/'+file + ' NONE\n'
				with open('results.txt','a') as f:
					f.write(line)
					f.close()


			if(file == dirFiles[-1]):
				processed = True
				cv2.destroyAllWindows()
	
			if cv2.waitKey(1) & 0xFF == ord('q'):
				processed == True
				break
		break

	cv2.destroyAllWindows()

main()