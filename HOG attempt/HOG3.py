import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import re
import pickle

SZ = 64
CLASS_N = 2

# local modules
from common import clock, mosaic

def split2d(img, cell_size, flatten=True):
	h, w = img.shape[:2]
	sx, sy = cell_size
	cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
	cells = np.array(cells)
	if flatten:
		cells = cells.reshape(-1, sy, sx)
	return cells

def load_digits(fn):
	digits_img = cv2.imread(fn, 0)
	digits = split2d(digits_img, (SZ, SZ))
	labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
	return digits, labels

def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11']/m['mu02']
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img

def svmInit(C=0.5, gamma=0.50625):
	model = cv2.ml.SVM_create()
	model.setGamma(gamma)
	model.setC(C)
	model.setKernel(cv2.ml.SVM_RBF)
	model.setType(cv2.ml.SVM_C_SVC)

	return model

def svmTrain(model, samples, responses):
	model.train(samples, cv2.ml.ROW_SAMPLE, responses)
	model.save("svm.xml")
	tree = ET.parse('svm.xml')
	root = tree.getroot()
	# now this is really dirty, but after ~3h of fighting OpenCV its what happens :-)
	SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0] 
	rho = float( root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text )
	svmvec = [float(x) for x in re.sub( '\s+', ' ', SVs.text ).strip().split(' ')]
	svmvec.append(-rho)
	pickle.dump(svmvec, open("svm.pickle", 'wb'))
	return model

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def svmEvaluate(model, digits, samples, labels):
	predictions = svmPredict(model, samples)
	accuracy = (labels == predictions).mean()
	print('Percentage Accuracy: %.2f %%' % (accuracy*100))

	confusion = np.zeros((10, 10), np.int32)
	for i, j in zip(labels, predictions):
		confusion[int(i), int(j)] += 1
	print('confusion matrix:')
	print(confusion)

	vis = []
	for img, flag in zip(digits, predictions == labels):
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		if not flag:
			img[...,:2] = 0
		
		vis.append(img)
	return mosaic(25, vis)


def preprocess_simple(digits):
	return np.float32(digits).reshape(-1, SZ*SZ) / 255.0


def get_hog() : 
	winSize = (64,64)
	blockSize = (8,8)
	blockStride = (4,4)
	cellSize = (8,8)
	nbins = 9
	derivAperture = 1
	winSigma = -1.
	histogramNormType = 0
	L2HysThreshold = 0.2
	gammaCorrection = 1
	nlevels = 64
	signedGradient = True

	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

	return hog
	affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR



if __name__ == '__main__':

	DIR = "/home/daniel/Documents/FYP/FYP/hog/hogPos/"
	dirFiles = os.listdir(DIR)
	dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	dirFiles=dirFiles[:5000]


	horizontal = 0
	images=[]
	img = []
	for x in dirFiles:
		r=cv2.imread(DIR + x, 0)
		if(horizontal < 100):
			if(images == []):
				images=r
				horizontal = horizontal + 1
			
			else:
				images = np.concatenate((images,r), axis=1)
				horizontal = horizontal + 1
				
		if(horizontal >=100):
			if(img == []):
				img = images
				horizontal = 0
				images = []


			else:
				img = np.concatenate((img,images), axis=0)
				horizontal = 0
				images=[]

	cv2.imwrite('out.png', img)

	print('Loading digits from digits.png ... ')
	# Load data.
	digits, labels = load_digits('out.png')

	print('Shuffle data ... ')
	# Shuffle data
	rand = np.random.RandomState(10)
	shuffle = rand.permutation(len(digits))
	digits, labels = digits[shuffle], labels[shuffle]
	
	#print('Deskew images ... ')
	#digits_deskewed = list(map(deskew, digits))
	
	print('Defining HoG parameters ...')
	# HoG feature descriptor
	hog = get_hog();

	print('Calculating HoG descriptor for every image ... ')
	hog_descriptors = []
	for img in digits:
		hog_descriptors.append(hog.compute(img))
	hog_descriptors = np.squeeze(hog_descriptors)

	print('Spliting data into training (90%) and test set (10%)... ')
	train_n=int(0.9*len(hog_descriptors))
	digits_train, digits_test = np.split(digits, [train_n])
	hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
	labels_train, labels_test = np.split(labels, [train_n])
	
	
	print('Training SVM model ...')
	model = svmInit()
	svmTrain(model, hog_descriptors_train, labels_train)

	print('Evaluating model ... ')
	vis = svmEvaluate(model, digits_test, hog_descriptors_test, labels_test)

	cv2.imwrite("digits-classification.jpg",vis)
	cv2.imshow("Vis", vis)
	cv2.waitKey(0)
cv2.destroyAllWindows()