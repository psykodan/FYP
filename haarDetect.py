import numpy as np
import cv2

#this is the cascade we just made. Call what you want
body_cascade = cv2.CascadeClassifier('/home/daniel/Documents/FYP/FYP/haar/try2/cascade.xml')

#feed = "/home/daniel/Documents/FYP/FYP/data/ClearLightChopDoolin/positive/posClearLightChopDoolin2"
feed = "/home/daniel/Documents/FYP/FYP/data/CloudyChopSurfFanore/positive/posCloudyChopSurfFanore4"
cap = cv2.VideoCapture(feed)
cap = cv2.VideoCapture(feed)
if not cap.isOpened():
	cap.open(device)

if cap.isOpened():
	while True:
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# add this
		# image, reject levels level weights.
		#bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50), maxSize=(100,100))
		bodies = body_cascade.detectMultiScale(gray, 400, 100)
		
		# add this
		for (x,y,w,h) in bodies:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

		cv2.imshow('img',img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
	cap.release()
	cv2.destroyAllWindows()
	
else:
	print ("Failed to open capture device")


