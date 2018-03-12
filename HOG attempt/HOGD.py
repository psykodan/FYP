import cv2
import pickle
import numpy as np
import os


winStride = (64,64)
padding = (16,16)
meanShift = True
scale = 1.01


img = cv2.imread("/home/daniel/Documents/FYP/FYP/haar/positive/frame194.jpg",0)
hog = cv2.HOGDescriptor((64,64), (8,8), (4,4), (8,8), 9)
svm = cv2.ml.SVM_create()
svm.load('svm.xml')

svm = pickle.load(open("svm.pickle", "rb"))
hog.setSVMDetector( np.array(svm) )
del svm

found, w = hog.detectMultiScale(img, winStride=winStride, useMeanshiftGrouping=meanShift)
		
for (x,y,w,h) in found:
		
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#hog = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())



#feed = "/home/daniel/Documents/FYP/FYP/data/ClearLightChopDoolin/positive/posClearLightChopDoolin2"
#feed = inputFile
#cap = cv2.VideoCapture(feed)
#if not cap.isOpened():
#	cap.open(device)

#if cap.isOpened():
#	while True:
#		ret, img = cap.read()
#		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		#found, w = hog.detectMultiScale(img, winStride=winStride, useMeanshiftGrouping=meanShift)
		
		
		#for (x,y,w,h) in found:
		
		#	cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
		

#		cv2.imshow('img',img)
#		if cv2.waitKey(1) & 0xFF == ord('q'):
#			break
		
#	cap.release()
#	cv2.destroyAllWindows()
	
#else:
#	print ("Failed to open capture device")



