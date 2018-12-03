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
	# read input image
	m = cv2.imread(inputImage,0)
	h,w = np.shape(m)
	# initialize variables for keeping track of results
	classresults = dict()
	bestpixel = ()
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
				finalsuccess = success
				bestattack = filename
			# add result to classresults dictionary
			classresults[testpixel] = targetconf
			testcount += 1
	# if we have found a first pixel that reduces the confidence level
	if (minconf < baselineconf) :
		newFileName1 = actualperson + "-1stpixelround.jpg"
		copyfile(bestattack,newFileName1)
		# now, find second best pixel
		secondbestpixel = ()
		if bestpixel != () :
			classresults.pop(bestpixel)
		nextminconf = 1
		for (k, v) in classresults.items() :
			if v < nextminconf :
				nextminconf = v
				secondbestpixel = k
		# perturb second best pixel on output image from perturbing first best pixel
		m2 = cv2.imread(newFileName1,0)
		m2[secondbestpixel[0]][secondbestpixel[1]]=(m2[secondbestpixel[0]][secondbestpixel[1]]+128)%256
		# save perturbed image after second pixel edit
		newFileName2 = actualperson + "-2ndpixelround.jpg"
		cv2.imwrite(newFileName2,m2)
		# apply classifier to perturbed image
		labelresults2 = classify.classify(actualperson, newFileName2)
		success2 = labelresults2[0]
		targetconf2 = labelresults2[1]
		print("After 2nd pixel change: " + actualperson + ": " + str(targetconf2)+"\n")
		# if second round improved attack, update success and minimum confidence variables and prepare to return second round perturbed image filename
		if targetconf2 < minconf :
			minconf = targetconf2
			finalsuccess = success2
			finalFileName = newFileName2
		# otherwise return the result from the first pixel change
		else :
			print("Altering second best pixel did not improve results.")
			finalFileName = newFileName1
	# if first pixel did not reduce confidence level, then print that no successful attack was found and make a copy of the original inputimage
	else :
		print("No successful attack was found.")
		finalFileName = actualperson + '-NoImprovement.jpg'
		finalsuccess = baselinesuccess
		copyfile(inputImage, finalFileName)
	# clean up unnecessary attack image files
	for f in os.listdir('.') :
		if re.search("attack*", f) :
			os.remove(os.path.join('.', f))
	# return results
	return finalsuccess, minconf, finalFileName

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
		print(targetImage + "- Baseline confidence: " + str(baselineconf) + "\n")
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
		print(targetImage + "- Change in confidence: " + str(changeconf) + "\n")
	# write final results to a csv file		
	with open('imageAttackAlter2Pixel', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter= ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(['TargetImage', 'ChangeInConfidence', 'PercentChangeInConfidence', 'Success'])
		for i in finalresults :
			writer.writerow(i)