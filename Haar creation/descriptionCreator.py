import cv2
import os
import sys

#
# This program creates a description for the negative images used for training a haar cascade classifier
#

def main():	#User input file and output location

	DIR = str(input("Enter path to negative images directory: ") or "demoImages/neg/")
	assert os.path.exists(DIR), "Error: Path does not exist at: , "+str(DIR)
	DIR2 = str(input("Enter path to store bg.txt: ") or "demoFiles/")
	if not os.path.exists(DIR2):
		check = input("Create directory " + DIR2 + " ? (Y/n): ")
		if(check == "Y" or check == "y" or check == ""):
			os.makedirs(DIR2)
		else:
			DIR2 = input("Enter path to store frames: ")

	create_neg(DIR, DIR2)


def create_neg(dirIn, dirOut):
	#User input location of negative images relative to where haar cascade will be trained from
	loc = input("Enter path to negative images relative to where haar cascade will be trained from: ")
	#Sort files in numerical order
	dirFiles = os.listdir(dirIn)
	dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
		
	#create a txt file that contains the location of the images and the images names
	for img in dirFiles:
		line = loc + os.sep +img+'\n'
		with open(dirOut + os.sep + 'bg.txt','a') as f:
			f.write(line)

main()