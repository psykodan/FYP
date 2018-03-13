import numpy as np
import cv2
import os


from advancedHaarColourThreshold import advancedHaarDetectionColourThresholdImageStream


def main():

	holdoutSets=[]
	holdoutSets.append("/home/daniel/Documents/FYP/FYP/data/CloudyChopSurfFanore/holdout/1/positive/")
	holdoutSets.append("/home/daniel/Documents/FYP/FYP/data/CloudyChopSurfFanore/holdout/2/positive/")
	holdoutSets.append("/home/daniel/Documents/FYP/FYP/data/CloudyChopSurfFanore/holdout/3/positive/")


	haar = "/home/daniel/Documents/FYP/FYP/haar/final/cascade.xml"
	cascade = cv2.CascadeClassifier(haar)

	objectLocations = []
	objectLocations.append("/home/daniel/Documents/FYP/FYP/data/CloudyChopSurfFanore/holdout/1/info.dat")
	objectLocations.append("/home/daniel/Documents/FYP/FYP/data/CloudyChopSurfFanore/holdout/2/info.dat")
	objectLocations.append("/home/daniel/Documents/FYP/FYP/data/CloudyChopSurfFanore/holdout/3/info.dat")

	threshold = 3.6

	box = 40
	#Iterate through confidence thresholds 0 - 5
	for hSetNum in range(len(holdoutSets)):
		advancedHaarDetectionColourThresholdImageStream(holdoutSets[hSetNum], cascade, threshold, box)
		processResults(objectLocations[hSetNum], holdoutSets[hSetNum])
		


def processResults(info, dirIn):


	tp = 0
	fp = 0
	tn = 0
	fn = 0
	p = 0
	n = 0

	#opening and processing the created results file and the info.dat file that specifies where the objects actually are
	r = open('results.txt', 'r')
	results = r.read()
	r.close()
	resRows = results.split('\n')
	resFormatted = []
	for i in resRows:
		resFormatted.append(i.split(' '))


	d = open(info, 'r')
	data = d.read()
	d.close()
	dataRows = data.split('\n')
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
				if(y[0] == x[0]): #if the image from the results file is the same image from info.dat

					if(y[1] == "NONE"): #If the result is a NONE this is a false negative as the info.dat wouldn't have an entry for a negative
						fn = fn + 1

					
					elif(int(x[1])==1):
						#Check x y coords of box are within 50 pixels for 1 object	
						if((int(y[1]) >= (int(x[2]) - 50) and int(y[1]) <= (int(x[2]) + 50)) and  (int(y[2]) >= (int(x[3]) - 50) and int(y[2]) <= (int(x[3]) + 50))):

							tp = tp + 1
							
							
						else:

							fp = fp + 1

											

					elif(int(x[1])==2):
					
						#Check x y coords of box are within 50 pixels for 1 object	
						if((int(y[1]) >= (int(x[2]) - 50) and int(y[1]) <= (int(x[2]) + 50)) and  (int(y[2]) >= (int(x[3]) - 50) and int(y[2]) <= (int(x[3]) + 50))):
							
							tp = tp + 1

							
						#Check x y coords of box are within 50 pixels for 1 object	
						elif((int(y[1]) >= (int(x[6]) - 50) and int(y[1]) <= (int(x[6]) + 50)) and  (int(y[2]) >= (int(x[7]) - 50) and int(y[2]) <= (int(x[7]) + 50))):
							
							tp = tp + 1

							
						else:

							fp = fp + 1

							

					elif(int(x[1])==3):

						#Check x y coords of box are within 50 pixels for 1 object	
						if((int(y[1]) >= (int(x[2]) - 50) and int(y[1]) <= (int(x[2]) + 50)) and  (int(y[2]) >= (int(x[3]) - 50) and int(y[2]) <= (int(x[3]) + 50))):
							
							tp = tp + 1

													
						#Check x y coords of box are within 50 pixels for 1 object	
						elif((int(y[1]) >= (int(x[6]) - 50) and int(y[1]) <= (int(x[6]) + 50)) and  (int(y[2]) >= (int(x[7]) - 50) and int(y[2]) <= (int(x[7]) + 50))):
							
							tp = tp + 1

							
						#Check x y coords of box are within 50 pixels for 1 object	
						elif((int(y[1]) >= (int(x[10]) - 50) and int(y[1]) <= (int(x[10]) + 50)) and  (int(y[2]) >= (int(x[11]) - 50) and int(y[2]) <= (int(x[11]) + 50))):
							
							tp = tp + 1

							
						else:

							fp = fp + 1

							

					else:
						print("nothing to see here")

	for z in resFormatted:
		if(len(z) != 1):
			if(z[1]=="NONE" and z[0] not in images): #If the results file says NONE and there is no entry in the info.dat file then this is a true negative 

				tn = tn +1
				
			if(z[1] != "NONE" and z[0] not in images): #If the results file doesn't say NONE and there is no entry in the info.dat file then this is a false positive that wouldn't have been picked up above

				fp = fp +1
				
	
	count = len([name for name in os.listdir(dirIn) if os.path.isfile(os.path.join(dirIn, name))])

	n = count - len(dataFormatted)

	#Metrics
	if((tp+fp+fn+tn)!= 0):
		accuracy = (tp + tn)/(tp+fp+fn+tn)
	else:
		accuracy = 0

	if((tp+fp) != 0):
		precision = tp/(tp+fp)
	else:
		precision = 0

	if((tp+fn) != 0):
		sensitivity = tp/(tp+fn)
	else:
		sensitivity = 0

	if((tn+fp) != 0):
		specificity = tn/(tn+fp)
	else:
		specificity = 0


	print("True positive count: " + str(tp))
	print("False positive count: "+ str(fp))
	print("True negative count: "+ str(tn))
	print("False negative count: "+ str(fn))
	print("Positive count: "+ str(p))
	print("Negative count: "+ str(n))

	rF = open('resultsFull.txt', 'a')
	rF.write("Confusion Matrix for black Colour Thresholding\n")
	rF.write("              Actual            \n")
	rF.write("                                \n")
	rF.write("          P("+str(p)+")    N("+str(n)+")\n")
	rF.write("P        --------------------\n")
	rF.write("r   P    | ["+str(tp)+"]  |  ["+str(fp)+"]  |\n")
	rF.write("e  ("+str(int(tp + fp))+") -------------------\n")
	rF.write("d   N    |  ["+str(fn)+"]   |  ["+str(tn)+"]   |\n")
	rF.write("   ("+str(int(tn + fn))+")  -------------------\n\n")

	rF.write("Accuracy = " + str(accuracy) + "\n")
	rF.write("Precision = " + str(precision) + "\n")
	rF.write("Sensitivity = " + str(sensitivity) + "\n")
	rF.write("Specificity = " + str(specificity) + "\n")

	rF.write("-------------- NEXT RUN ------------\n")
	rF.close()

	rR = open('resultsRaw.txt', 'a')
	rR.write(str(p) +","+ str(n) +","+ str(tp) +","+ str(tn) +","+ str(fp) +","+ str(fn) +","+ str(accuracy) +","+ str(precision) +","+ str(sensitivity) +","+ str(specificity) +"\n" )
	rR.close()

	r = open('results.txt','w')
	r.write("")
	r.close()

main()