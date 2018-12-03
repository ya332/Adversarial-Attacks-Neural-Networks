import cv2
import numpy as np
import math
import random
import os, re
from shutil import copyfile
import csv

import classify

INPUT_DIRECTORY = 'testing/'

def attack(inputImage, actualperson, minconf, baselinesuccess):
	# set parameter for width and height of each square of image from which a random pixel will be selected to test
	densityParameter = 24
	# read input image to obtain dimensions
	m = cv2.imread(inputImage,0)
	h,w = np.shape(m)
	# initialize variables for keeping track of results
	classresults = dict()
	bestpixel = []
	bestattack = ""
	testcount = 0
	# baseline confidence for this attack
	baselineconf = minconf
	# targetconf will be used to track classifier's confidence that the image is the actual person
	targetconf = 1
	# iterate through image height and width, with intervals based on densityParameter, essentially picking one pixel from each densityParameter-by-densityParameter square to test
	for i in range(0,h,densityParameter) :
		# use randomized offsets in order to choose a random pixel within each square
		yoffset = random.randint(0,densityParameter-1)
		row = i + yoffset
		for j in range (0,w,densityParameter) :
			xoffset = random.randint(0,densityParameter-1)
			col = j + xoffset
			# bound possible test pixel position based on image size
			if row > 191 :
				row = 191
			if col > 167 :
				col = 167
			testpixel = (row, col)
			# print the index of which pixel test the current one represents
			print(testcount)
			# re-read in the unaltered input image from disk
			m = cv2.imread(inputImage,0)
			filename='attack'+str(testcount)+'.jpg'
			# apply perturbation
			m[i+yoffset][j+xoffset]=(m[i+yoffset][j+xoffset]+128)%256
			# save perturbed image
			cv2.imwrite(filename,m)
			# apply classifier to perturbed image
			labelresults = classify.classify(actualperson, filename)		
			# success indicates whether image was misclassified (1=yes, 0=no)
			success = labelresults[0]
			# targetconf is the classifier's confidence level that the image is the actual person id
			targetconf = labelresults[1]
			print(actualperson + ": " + str(targetconf)+"\n")
			# update minimum confidence
			if (targetconf < minconf) :
				bestpixel = testpixel
				minconf = targetconf
				bestattack = filename
				finalsuccess = success
			# add result to classresults array
			classresults[testpixel] = targetconf
			testcount += 1
	# if we have found a pixel that reduces the confidence level, make a copy of the most successful attack image
	if (minconf < baselineconf) :
		newFileName = actualperson+"-round"+str(round)+".jpg"
		copyfile(bestattack,newFileName)
	# otherwise print that no better result was found and store original input image as newFileName
	else :
		print("No better result was found.")
		newFileName = inputImage
		finalsuccess = baselinesuccess
	# clean up unnecessary attack image files
	for f in os.listdir('.') :
		if re.search("attack*", f) :
			os.remove(os.path.join('.', f))
	return finalsuccess, minconf, newFileName

if __name__ == "__main__":
	# initialize results array
	finalresults = []
	# walk input image directory and sort list of directories alphabetically
	for fFileObj in os.walk(INPUT_DIRECTORY) :
		dirList = fFileObj[1]
		dirList.sort()
		print(dirList)
		break
	# iterate through each directory, each of which represents a person
	for dir in dirList :
		targetImage = os.path.join(INPUT_DIRECTORY, dir, "image0.jpg")
		# extract actual person id from directory name
		actualperson = dir.lower()
		# initialize success variable to 0
		success = 0
		# determine baseline classifier confidence level for the actual person		
		baselineresult = classify.classify(actualperson, targetImage)
		baselinesuccess = baselineresult[0]
		baselineconf = baselineresult[1]	
		print(targetImage + " - Baseline confidence: " + str(baselineconf) + "\n")
		# minconf will be used to keep track of minimum confidence level acheived for target (actual person)
		minconf = baselineconf
		# perform attack and extract results
		result = attack(targetImage, actualperson, minconf, baselinesuccess)
		# success represents whether or not misclassification was achieved
		success = result[0]
		# minconf is minimum confidence acheived for the actual person class
		minconf = result[1]
		# newImage is filename of perturbed image
		newImage = result[2]
		changeconf = minconf - baselineconf
		percentchange = changeconf / baselineconf
		# create array storing attack results and add to array for all image attack results
		imageresult = [targetImage, changeconf, percentchange, success]
		finalresults.append(imageresult)
		print(targetImage + " - Change in confidence: " + str(changeconf) + "\n")
	# write final results to a csv file
	with open('imageAttackAlter1Pixel.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter= ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(['TargetImage', 'ChangeInConfidence', 'PercentChangeInConfidence', 'Success'])
		for i in finalresults :
			writer.writerow(i)