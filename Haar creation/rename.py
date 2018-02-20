import os


#Enter the path to the files i.e. /home/daniel/Documents/FYP/data/cloudy chop surf fanore/positive video/

#The new name if working with the frame extractor should be of the form "neg/pos|Descriptive words of data|.file extension"

path = input("Enter path to data: ")
name = input("Enter name that classifies data (if working with the frame extractor should be of the form neg/pos|Descriptive words of data|.file extension): ")
num = 0
for filename in os.listdir(path):
	os.rename(path + filename, path + name + str(num))
	num += 1