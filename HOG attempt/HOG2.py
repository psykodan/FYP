import cv2
import numpy as np
import os

SZ=20
bin_n = 16 # Number of bins


affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

## [deskew]
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img
## [deskew]

## [hog]
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist
## [hog]

DIR = "/home/daniel/Documents/FYP/FYP/hog/hog/"
dirFiles = os.listdir(DIR)
dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
dirFiles=dirFiles[:2500]


horizontal = 0
images=[]
img = []
for x in dirFiles:
	r=cv2.imread(DIR + x, 1)
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

#cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

# First half is trainData, remaining is testData
train_cells = images[:2000]
test_cells = images[2000:]

######     Now training      ########################

#deskewed = [map(deskew,row) for row in train_cells]
hogdata = [list(map(hog,row)) for row in train_cells]
#print(hogdata[0])
trainData = np.float32(hogdata).reshape(-1,64)
#print(trainData[0])
responses = np.repeat(np.arange(1),len(trainData))[:,np.newaxis]
print(len(responses))

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')

######     Now testing      ########################

#deskewed = [map(deskew,row) for row in test_cells]
hogdata = [map(hog,row) for row in test_cells]
testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict(testData)[1]

#######   Check Accuracy   ########################
mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)
