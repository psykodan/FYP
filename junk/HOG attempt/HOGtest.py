
import cv2 as cv
import numpy as np
import os
SZ=20
bin_n = 16 # Number of bins
affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img
def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

DIR = "/home/daniel/Documents/FYP/FYP/hog/hogPos/"
dirFiles = os.listdir(DIR)
dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
dirFiles=dirFiles[:5000]


horizontal = 0
images=[]
img = []
for x in dirFiles:
    r=cv.imread(DIR + x, 0)
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

cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
print(len(cells))
# First half is trainData, remaining is testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]
print(len(train_cells))
#deskewed = [list(map(deskew,row)) for row in train_cells]
hogdata = [list(map(hog,row)) for row in train_cells]
print(len(hogdata))
trainData = np.float32(hogdata).reshape(-1,64)
print(len(trainData))
responses = np.repeat(np.arange(10),250)[:,np.newaxis]
print(len(responses))
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(100)
svm.setGamma(5)
svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')
#deskewed = [list(map(deskew,row)) for row in test_cells]
hogdata = [list(map(hog,row)) for row in test_cells]
testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict(testData)[1].ravel()
mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)