import cv2
import os
import sys
#
#	This program inputs a video file and creates JPG images of each frame in Grayscale
#

def main():	#User input file and output location

	DIR = input("Enter path to video file for frame extration: ")
	assert os.path.exists(DIR), "Error: Path does not exist at: , "+str(DIR)
	DIR2 = input("Enter path to store frames: ")
	if not os.path.exists(DIR2):
		check = input("Create directory " + DIR2 + " ? (Y/n): ")
		if(check == "Y" or check == "y" or check == ""):
			os.makedirs(DIR2)
		else:
			DIR2 = input("Enter path to store frames: ")

	frameExtraction(DIR, DIR2)


def frameExtraction(dirIn, dirOut):

	#Load video file
	vidcap = cv2.VideoCapture(dirIn)


	#Name the image by number in accorance with how many files are already present in output dir
	count = len([name for name in os.listdir(dirOut) if os.path.isfile(os.path.join(dirOut, name))])
	success = True



	while success:			#loop until vidcap is finished (success = false from vidcap.read())
		success,image = vidcap.read()
		
		if(success == True):
			#Convert image to Grayscale for image processing reasons
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# save frame as JPEG file
			cv2.imwrite(dirOut + os.sep+ "frame%d.jpg" % count, gray)     
			count += 1
			
main()