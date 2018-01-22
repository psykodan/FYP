import os

path = input("Enter path to data: ")
name = input("Enter name that classifies data: ")
num = 0
for filename in os.listdir(path):
	os.rename(path + filename, path + name + str(num) + ".MOV")
	num += 1