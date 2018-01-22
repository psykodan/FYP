import cv2
import os


DIR = '/home/daniel/Documents/FYP/data/'
for subdir, dirs, files in os.walk(DIR):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".MOV"):
            print (filepath)
a = 2
if(a == 1):
	for vid in vids:
		vidcap = cv2.VideoCapture('/home/daniel/Documents/FYP/data/cloudy chop surf fanore/negative video/DSCN3163.MOV')
		success,image = vidcap.read()
		count = 0
		success = True

		while success:
			success,image = vidcap.read()
			print('Read a new frame: ', success)
			cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
			count += 1