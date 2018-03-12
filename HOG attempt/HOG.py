import cv2
import numpy as np
import os

CLASS_N = 10



def svmInit(C=12.5, gamma=0.50625):
  model = cv2.ml.SVM_create()
  model.setGamma(gamma)
  model.setC(C)
  model.setKernel(cv2.ml.SVM_RBF)
  model.setType(cv2.ml.SVM_C_SVC)
  
  return model

def svmTrain(model, samples, responses):
  model.train(samples, cv2.ml.ROW_SAMPLE, responses)
  return model

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def svmEvaluate(model, digits, samples, labels):
	predictions = svmPredict(model, samples)
	print(predictions)
	print(labels)
	accuracy = (labels == predictions).mean()
	print('Percentage Accuracy: %.2f %%' % (accuracy*100))

	confusion = np.zeros((10, 10), np.int32)
	for i, j in zip(labels, predictions):
		confusion[int(i), int(j)] += 1
	print('confusion matrix:')
	print(confusion)



def main():


	DIR = str(input("Enter path to images for HOG descriptor: ") or "/home/daniel/Documents/FYP/FYP/hog/hog/")
	assert os.path.exists(DIR), "Error: Path does not exist at: , "+str(DIR)

	#Sort files in numerical order
	dirFiles = os.listdir(DIR)
	dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

	images=[]

	#cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

	for x in dirFiles:
		img=cv2.imread(DIR + x,0)
		images.append(img)
		#train_cells.append(img)
		#test_cells.append(img)


	labels = np.repeat(np.arange(CLASS_N), len(images)/CLASS_N)


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
	useSignedGradients = True
	 
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)

	hog_descriptors=[]
	for i in images:
		descriptor = hog.compute(i)
		hog_descriptors.append(descriptor)
	hog_descriptors = np.squeeze(hog_descriptors)
	

	train_n=int(0.9*len(hog_descriptors))
	images_train, images_test = np.split(images, [train_n])
	hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
	labels_train, labels_test = np.split(labels, [train_n])

	print('Training SVM model ...')
	model = svmInit()
	svmTrain(model, hog_descriptors_train, labels_train)

	print('Evaluating model ... ')
	vis = svmEvaluate(model, images_test, hog_descriptors_test, labels_test)
	
	

main()