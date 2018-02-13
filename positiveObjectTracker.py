import cv2
import numpy as np
import os

processed = False
DIR ="/home/daniel/Documents/FYP/FYP/haar/positive/"
ix,iy = -1,-1

while(processed == False):
	# mouse callback function
	def mouse_pos(event,x,y,flags,param):
		global ix,iy
		if event == cv2.EVENT_MOUSEMOVE:
			print(x,y)
			ix,iy = x,y


	#Sort files in numerical order
	dirFiles = os.listdir(DIR)
	dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	img = cv2.imread(DIR + dirFiles[0],cv2.IMREAD_GRAYSCALE)
	cv2.imshow('image',img)
	cv2.setMouseCallback('image',mouse_pos)
	
	if cv2.waitKey(1) & 0xFF == ord('s'):
		for file in dirFiles:
			img = cv2.imread(DIR + file,cv2.IMREAD_GRAYSCALE)
			#print(file)
			
			cv2.rectangle(img,(ix-25,iy-25),(ix+25,iy+25),(255,0,0),5)
			cv2.namedWindow('image')
			cv2.setMouseCallback('image',mouse_pos)
			cv2.imshow('image',img)
			cv2.waitKey(50)

			line = 'positive/'+file+' 1 ' + str(ix-25) + ' ' + str(iy-25) + ' ' + str(ix+25) + ' ' + str(iy+25) + '\n'
			with open('info.dat','a') as f:
				f.write(line)

			if(file == dirFiles[-1]):
				processed = True
				cv2.destroyAllWindows()

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	
cv2.destroyAllWindows()