import cv2
import os


DIR = '/home/daniel/Documents/FYP/FYP/data/CloudyChopSurfFanore/holdout'
for subdir, dirs, files in os.walk(DIR):
	for file in files:
		#print os.path.join(subdir, file)
		filepath = subdir + os.sep + file

		if file.startswith("neg"):
			print (filepath)
			vidcap = cv2.VideoCapture(filepath)

			#Create negative image directory
			if not os.path.exists(subdir + os.sep + 'negative'):
				os.makedirs(subdir + os.sep + 'negative')

			#Name the image by number
			folder = subdir + os.sep + 'negative'
			count = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
			success = True

			while success:
				success,image = vidcap.read()
				#print('Read a new frame: ', success)
				#Convert image to Grayscale
				if(success == True):
					gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					cv2.imwrite(folder + os.sep+ "frame%d.jpg" % count, gray)     # save frame as JPEG file
					count += 1
			
		

		elif file.startswith("pos"):
			print (filepath)
			vidcap = cv2.VideoCapture(filepath)

			#Create positive image directory
			if not os.path.exists(subdir + os.sep + 'positive'):
				os.makedirs(subdir + os.sep + 'positive')

			#Name the image by number
			folder = subdir + os.sep + 'positive'
			count = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
			success = True

			while success:
				success,image = vidcap.read()
				#print('Read a new frame: ', success)
				#Convert image to Grayscale
				if(success == True):
					gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					cv2.imwrite(folder + os.sep+ "frame%d.jpg" % count, gray)     # save frame as JPEG file
					count += 1

