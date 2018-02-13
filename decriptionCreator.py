import cv2
import os

DIR ="/home/daniel/Documents/FYP/FYP/haar/negative/"
def create_neg():

	#Sort files in numerical order
	dirFiles = os.listdir(DIR)
	dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
		
	for img in dirFiles:

		line = "haar/negative/"+img+'\n'
		with open('bg.txt','a') as f:
			f.write(line)

create_neg()