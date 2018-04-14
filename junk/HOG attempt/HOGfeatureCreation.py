import cv2 as cv
import numpy as np
import os

SZ=20
bin_n = 16 # Number of bins
affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR


def main():


	DIR = str(input("Enter path to images for HOG descriptor: ") or "/home/daniel/Documents/FYP/FYP/hog/HOG/")
	assert os.path.exists(DIR), "Error: Path does not exist at: , "+str(DIR)

	#Sort files in numerical order
	dirFiles = os.listdir(DIR)
	dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

	images=[]

	#cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

	for x in dirFiles:
		img=cv.imread(DIR + x,0)
		images.append(img)
		#train_cells.append(img)
		#test_cells.append(img)

	#print(images[1][1])

	l = int(len(images)/2)
	train_cells = images[:2500]
	test_cells = images[2500:5000]
	print(train_cells[1])
	cv.imshow("hgjh", train_cells[1])
	cv.waitKey(1000)
	#hogdata = [list(map(hog,row)) for row in train_cells]
	#print(hogdata)
	hogdata = []
	for a in train_cells:
		h = hog(a)
		hogdata.append(h)
	trainData = np.float32(hogdata).reshape(-1,64)
	print(trainData[76])
	print(len(trainData))
	responses = np.repeat(np.arange(10),250)[:,np.newaxis]
	
	svm = cv.ml.SVM_create()
	svm.setKernel(cv.ml.SVM_LINEAR)
	svm.setType(cv.ml.SVM_C_SVC)
	svm.setC(2.67)
	svm.setGamma(5.383)
	svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
	svm.save('svm_data.dat')



	hogdata = []
	for a in test_cells:
		h = hog(a)
		hogdata.append(h)
	testData = np.float32(hogdata).reshape(-1,bin_n*4)
	result = svm.predict(testData)[1]
	mask = result==responses
	correct = np.count_nonzero(mask)
	print(correct*100.0/result.size)

	
def hog(img):
	#Use a Sobel filter to calculate the horizontal and vertical gradients
	gx = cv.Sobel(img, cv.CV_32F, 1, 0)
	gy = cv.Sobel(img, cv.CV_32F, 0, 1)

	#Find magnitude and direction of the gradients
	mag, ang = cv.cartToPolar(gx, gy)

	bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
	bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
	mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)     # hist is a 64 bit vector
	#print(hist)
	return hist


main()