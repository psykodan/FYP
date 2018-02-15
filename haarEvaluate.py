import numpy as np
import cv2
import os

#this is the cascade we just made. Call what you want
body_cascade = cv2.CascadeClassifier('/home/daniel/Documents/FYP/FYP/haar/try1/cascade.xml')
processed = False
DIR ="/home/daniel/Documents/FYP/FYP/haar/positive/"


#Sort files in numerical order
dirFiles = os.listdir(DIR)
dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
img = cv2.imread(DIR + dirFiles[0],cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)


while(processed == False):
	

	if cv2.waitKey(1) & 0xFF == ord('s'):
		for file in dirFiles:
			img = cv2.imread(DIR + file,cv2.IMREAD_GRAYSCALE)
			# image, reject levels level weights.
			bodies = body_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50,50), maxSize=(100,100))
			
			# add this
			for (x,y,w,h) in bodies:
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
				line = 'positive/'+file + ' ' + str(x) + ' ' + str(y) + ' ' + str(x+w) + ' ' + str(y+h) + '\n'
				with open('results.txt','a') as f:
					f.write(line)
					f.close()

			cv2.namedWindow('image')
			cv2.imshow('image',img)
			cv2.waitKey(10)

			if(file == dirFiles[-1]):
				processed = True
				cv2.destroyAllWindows()
	


			if cv2.waitKey(1) & 0xFF == ord('q'):
				processed == True
				break
		break
				
	

cv2.destroyAllWindows()

r = open('results.txt', 'r')
results = r.read()
r.close()

d = open('/home/daniel/Documents/FYP/FYP/haar/try1/info.dat', 'r')
data = d.read()
d.close()

resRows = results.split('\n')
dataRows = data.split('\n')

resFormatted = []
for i in resRows:
	resFormatted.append(i.split(' '))

dataFormatted = []
for j in dataRows:
	dataFormatted.append(j.split(' '))


for x in dataFormatted:
	if(len(x) != 1):
		for y in resFormatted:
			if(y[0] == x[0]):
				if(int(y[-4]) >= (int(x[-4]) - 50) and int(y[-4]) <= (int(x[-4]) + 50) ):
					line = 'MATCH\n'
					with open('check.txt','a') as f:
						f.write(line)
						f.close()

				else:
					line = 'FAIL\n'
					with open('check.txt','a') as f:
						f.write(line)
						f.close()	



