import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from advancedHaarDetect import advancedHaarDetectionImageStream


def main():
	
	plotPoints = []

	#DIR ="/home/daniel/Documents/FYP/FYP/data/CloudyChopSurfFanore/holdout/1/positive/"
	DIR = str(input("Enter path to images for Haar detection: ") or "/home/daniel/Documents/FYP/FYP/data/CloudyChopSurfFanore/holdout/3/positive/")
	assert os.path.exists(DIR), "Error: Path does not exist at: , "+str(DIR)

	haar = str(input("Enter path to Haar cascade classifier xml file: ") or '/home/daniel/Documents/FYP/FYP/haar/final/cascade.xml')
	assert os.path.exists(haar), "Error: File does not exist at: , "+str(haar)
	cascade = cv2.CascadeClassifier(haar)

	threshold = float(input("Enter a threshold value of Haar cascade classification confidence: ") or 4.6)
	
	info = str(input("Enter info.dat file: ") or '/home/daniel/Documents/FYP/FYP/data/CloudyChopSurfFanore/holdout/3/info.dat')

	for i in range(0, 101, 5):
		box = i
		advancedHaarDetectionImageStream(DIR, cascade, threshold, box)
		results = processResults(info)
		plotPoints.append([results[0],results[1], box])
		print(plotPoints)
		print(i)
	
		#print(plotPoints)

	x=[]
	y=[]
	b=[]
	for p in plotPoints:
		x.append(p[1])
		y.append(p[0])
		b.append(p[2])
		
	print(x)
	print(y)
	print(b)
	plt.plot(x,y)
	plt.title("Accuracy over TPR for varying box size of advancedHaarDetect")
	plt.xlabel("True Positive Rate")
	plt.ylabel("Accuracy")
	plt.show()

	r = open('BOX.txt',"a")
	r.write("TPR        Accuracy          Box\n")
	for j in range(len(x)):
		r.write(str(x[j]) + " " + str(y[j]) + " " + str(b[j]) +"\n")
	r.close()

def processResults(info):


	fp=0
	fn=0
	tp=0
	tn=0

	p=0
	n=0

	#opening and processing the created results file and the info.dat file that specifies where the objects actually are
	r = open('results.txt', 'r')
	results = r.read()
	r.close()

	d = open(info, 'r')
	data = d.read()
	d.close()

	resRows = results.split('\n')
	dataRows = data.split('\n')

	resFormatted = []
	for i in resRows:
		resFormatted.append(i.split(' '))

	dataFormatted = []
	for j in dataRows:
		dataFormatted.append(j.split(' '))

	images = []

	#go through each image in the info.dat file checking against every correspoding detection in the results file
	for x in dataFormatted:
		if(len(x) != 1):
			images.append(x[0])
			p = p + int(x[1])
			for y in resFormatted:
				if(y[0] == x[0]):

					if(y[1] == "NONE"):
						fn = fn + 1

					elif(int(x[1])==1):
						#Check x y coords of box are within 50 pixels for 1 object	
						if((int(y[1]) >= (int(x[2]) - 50) and int(y[1]) <= (int(x[2]) + 50)) and  (int(y[2]) >= (int(x[3]) - 50) and int(y[2]) <= (int(x[3]) + 50))):

							tp = tp + 1
							
							line = y[0]+ ' FOUND_object_1\n'
							with open('check.txt','a') as f:
								f.write(line)
								f.close()

						else:

							fp = fp + 1

							line =y[0]+ ' FAIL\n'
							with open('check.txt','a') as f:
								f.write(line)
								f.close()	
					elif(int(x[1])==2):
					
						#Check x y coords of box are within 50 pixels for 1 object	
						if((int(y[1]) >= (int(x[2]) - 50) and int(y[1]) <= (int(x[2]) + 50)) and  (int(y[2]) >= (int(x[3]) - 50) and int(y[2]) <= (int(x[3]) + 50))):
							
							tp = tp + 1

							line = y[0]+ ' FOUND_object_1\n'
							with open('check.txt','a') as f:
								f.write(line)
								f.close()

						#Check x y coords of box are within 50 pixels for 1 object	
						elif((int(y[1]) >= (int(x[6]) - 50) and int(y[1]) <= (int(x[6]) + 50)) and  (int(y[2]) >= (int(x[7]) - 50) and int(y[2]) <= (int(x[7]) + 50))):
							
							tp = tp + 1

							line = y[0]+ ' FOUND_object_2\n'
							with open('check.txt','a') as f:
								f.write(line)
								f.close()

						else:

							fp = fp + 1

							line = y[0]+ ' FAIL\n'
							with open('check.txt','a') as f:
								f.write(line)
								f.close()

					elif(int(x[1])==3):

						#Check x y coords of box are within 50 pixels for 1 object	
						if((int(y[1]) >= (int(x[2]) - 50) and int(y[1]) <= (int(x[2]) + 50)) and  (int(y[2]) >= (int(x[3]) - 50) and int(y[2]) <= (int(x[3]) + 50))):
							
							tp = tp + 1

							line = y[0]+ ' FOUND_object_1\n'
							with open('check.txt','a') as f:
								f.write(line)
								f.close()
						
						#Check x y coords of box are within 50 pixels for 1 object	
						elif((int(y[1]) >= (int(x[6]) - 50) and int(y[1]) <= (int(x[6]) + 50)) and  (int(y[2]) >= (int(x[7]) - 50) and int(y[2]) <= (int(x[7]) + 50))):
							
							tp = tp + 1

							line = y[0]+ ' FOUND_object_2\n'
							with open('check.txt','a') as f:
								f.write(line)
								f.close()

						#Check x y coords of box are within 50 pixels for 1 object	
						elif((int(y[1]) >= (int(x[10]) - 50) and int(y[1]) <= (int(x[10]) + 50)) and  (int(y[2]) >= (int(x[11]) - 50) and int(y[2]) <= (int(x[11]) + 50))):
							
							tp = tp + 1

							line = y[0]+ ' FOUND_object_3\n'
							with open('check.txt','a') as f:
								f.write(line)
								f.close()

						else:

							fp = fp + 1

							line = y[0]+ ' FAIL\n'
							with open('check.txt','a') as f:
								f.write(line)
								f.close()

					else:
						print("nothing to see here")

	for z in resFormatted:
		if(len(z) != 1):
			if(z[1]=="NONE" and z[0] not in images):
			
				print(z[0])
				print(z[1])
				tn = tn +1
				n = n +1
			if(z[1] != "NONE" and z[0] not in images):
				print(z[0])
				print(z[1])
				fp = fp +1
				n = n + 1
	#print(dataFormatted)
	print("True positive count: " + str(tp))
	print("False positive count: "+ str(fp))
	print("True negative count: "+ str(tn))
	print("False negative count: "+ str(fn))
	print("Positive count: "+ str(p))
	print("Negative count: "+ str(n))

	if(tp != 0  and p != 0):
		TPR = tp / p
		
	else:
		TPR = 0
		
	if(fp != 0 and n != 0):
		
		FPR = fp / n
	else:
		
		FPR = 0
	#print(TPR)
	#print(FPR)

	accuracy = (tp + tn)/(tp + fp + tn + fn)
	print(accuracy)


	rF = open('resultsFull.txt', 'a')
	rF.write(results)
	rF.write("------------------NEXT RUN--------------------\n")
	rF.close()

	c = open('check.txt','w')
	c.write("")
	c.close()

	r = open('results.txt','w')
	r.write("")
	r.close()

	return accuracy, TPR

main()