import cv2
import os


DIR = '/home/daniel/Documents/FYP/data/'
for subdir, dirs, files in os.walk(DIR):
	for file in files:
		#print os.path.join(subdir, file)
		filepath = subdir + os.sep + file

		if file.startswith("neg") & file.endswith(".MOV"): #will add further file extensions if they are used
			print (filepath)
			vidcap = cv2.VideoCapture(filepath)

			#Create negative image directory
			if not os.path.exists(subdir + os.sep + 'negative images'):
				os.makedirs(subdir + os.sep + 'negative images')

			#Name the image by number
			folder = subdir + os.sep + 'negative images'
			count = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
			success = True

			while success:
				success,image = vidcap.read()
				#print('Read a new frame: ', success)
				cv2.imwrite(folder + os.sep+ "frame%d.jpg" % count, image)     # save frame as JPEG file
				count += 1
		
		

		elif file.startswith("pos") & file.endswith(".MOV"): #will add further file extensions if they are used
			print (filepath)
			vidcap = cv2.VideoCapture(filepath)

			#Create positive image directory
			if not os.path.exists(subdir + os.sep + 'positive images'):
				os.makedirs(subdir + os.sep + 'positive images')

			#Name the image by number
			folder = subdir + os.sep + 'positive images'
			count = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
			success = True

			while success:
				success,image = vidcap.read()
				#print('Read a new frame: ', success)
				cv2.imwrite(folder + os.sep+ "frame%d.jpg" % count, image)     # save frame as JPEG file
				count += 1

