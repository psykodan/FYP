import numpy as np
import cv2


haar = str(input("Enter path to Haar cascade classifier xml file: ") or '/home/daniel/Documents/FYP/FYP/haar/final/cascade.xml')
assert os.path.exists(haar), "Error: File does not exist at: , "+str(haar)
cascade = cv2.CascadeClassifier(haar)

cap = cv2.VideoCapture(0)

while 1:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# add this
	# image, reject levels level weights.
	bodies = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,200))

	# add this
	for (x,y,w,h) in bodies:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

	

	cv2.imshow('img',img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()