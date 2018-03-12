import numpy as np
import cv2
import os


def main():
	
	DIR = str(input("Enter path to video file for Haar detection: ") or "/home/daniel/Documents/FYP/FYP/data/CloudyChopSurfFanore/positive/posCloudyChopSurfFanore3")
	assert os.path.exists(DIR), "Error: Path does not exist at: , "+str(DIR)

	#cascade = cv2.CascadeClassifier('/home/daniel/Documents/FYP/FYP/haar/final/cascade.xml')
	haar = str(input("Enter path to Haar cascade classifier xml file: ") or '/home/daniel/Documents/FYP/FYP/haar/final/cascade.xml')
	assert os.path.exists(haar), "Error: File does not exist at: , "+str(haar)
	cascade = cv2.CascadeClassifier(haar)

	threshold = float(input("Enter a threshold value of Haar cascade classification confidence: ") or 0)
	
	haarDetection(DIR, cascade, threshold)



def haarDetection(inputFile, cascade, threshold):

	#feed = "/home/daniel/Documents/FYP/FYP/data/ClearLightChopDoolin/positive/posClearLightChopDoolin2"
	feed = inputFile
	cap = cv2.VideoCapture(feed)
	if not cap.isOpened():
		cap.open(device)

	if cap.isOpened():
		while True:
			ret, img = cap.read()
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
			# image, reject levels level weights.
			bodies = cascade.detectMultiScale3(gray, scaleFactor=100, minNeighbors=100,outputRejectLevels = True)
			rects = bodies[0]
			neighbours = bodies[1]
			weights = bodies[2]
			
			#for (x,y,w,h) in rects:
			#	for (a,b,i,j) in oldBodies:

					#different colour for objects detected in close proximity to each other i.e. likely to be a legit detection
			#		if(x >= a-25 and x <=a+25 and y >= b-25 and y <=b+25):
			#			cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
			#			break
			#	else:
			#		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
			
			for a in range(len(weights)):
				if(weights[a][0] >= threshold):
					cv2.rectangle(img,(rects[a][0],rects[a][1]),(rects[a][0]+rects[a][2],rects[a][1]+rects[a][3]),(0,255,255),2)

			cv2.imshow('img',img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			
		cap.release()
		cv2.destroyAllWindows()
		
	else:
		print ("Failed to open capture device")



def haarDetectionImageStream(dirIn, cascade, threshold):

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

			# image, reject levels level weights.
			bodies = cascade.detectMultiScale3(gray, scaleFactor=100, minNeighbors=100,outputRejectLevels = True)
			rects = bodies[0]
			neighbours = bodies[1]
			weights = bodies[2]
			
			
			for a in range(len(weights)):
				if(weights[a][0] >= threshold):
					#cv2.rectangle(img,(rects[a][0],rects[a][1]),(rects[a][0]+rects[a][2],rects[a][1]+rects[a][3]),(0,255,255),2)
					detect=True
					line = 'positive/'+file + ' ' + str(rects[a][0]) + ' ' + str(rects[a][1]) + ' ' + str(rects[a][0]+rects[a][2]) + ' ' + str(rects[a][1]+rects[a][3]) + '\n'
					with open('results.txt','a') as f:
						f.write(line)
						f.close()
				

			if(detect == False):
				line = 'positive/'+file + ' NONE\n'
				with open('results.txt','a') as f:
					f.write(line)
					f.close()

			#cv2.imshow('img',img)


			if(file == dirFiles[-1]):
				processed = True
				cv2.destroyAllWindows()
	
			if cv2.waitKey(1) & 0xFF == ord('q'):
				processed == True
				break
		break

	cv2.destroyAllWindows()


main()