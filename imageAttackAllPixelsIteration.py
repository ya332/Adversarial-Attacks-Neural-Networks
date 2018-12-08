import cv2
import numpy as np
import math
import random
import os, re
from shutil import copyfile
import csv

import classify

INPUT_DIRECTORY = 'testing/'

def attack(inputImage, actualperson, minconf, baselinesuccess, maxrounds, initialperturbation, perturbationscale):
	# read image
	m = cv2.imread(inputImage,0)
	h,w = np.shape(m)
	# initialize variable for keeping track of number of perturbations
	perturbationcount = 0
	# baseline confidence for this attack
	baselineconf = minconf
	# set initial perturbation values
	perturbation = initialperturbation
	for iter in range(maxrounds) :
		filename='attack'+str(perturbationcount)+'.jpg'
		# apply perturbation
		for i in range(0,h) :
			for j in range(0,w) :
				# apply perturbation using checkerboard approach, bounding possible pixel values between 0 and 255
				if (j % 2 == 1 and i % 2 == 0) or (j % 2 == 0 and i % 2 == 1) :
					if (m[i][j] + perturbation > 255) :
						m[i][j] = 255
					else :
						m[i][j] = m[i][j] + perturbation
				elif (j % 2 == 0 and i % 2 == 1) or (j % 2 == 1 and i % 2 == 0) :
					if (m[i][j] - perturbation < 0) :
						m[i][j] = 0
					else :
						m[i][j] = m[i][j] - perturbation
		# save perturbed image
		cv2.imwrite(filename,m)
		# apply classifier to perturbed image
		labelresults = classify.classify(actualperson, filename)
		# success indicates whether image was misclassified (1=yes, 0=no)
		success = labelresults[0]
		# targetconf is the classifier's confidence level that the image is the actual person id
		targetconf = labelresults[1]
		# personclass is the id of the person who the classifier classified the image as
		personclass = labelresults[2]
		# print(actualperson + ": " + str(targetconf)+"\n")
		# update minimum confidence
		if (targetconf < minconf) :
			minconf = targetconf
		if success == 1 :
			break
		else :
			# update perturbation amount based on perturbation scale variable
			perturbation = perturbationscale
		# increment count of perturbations
		perturbationcount += 1
	# if after max rounds, we were not able to get classifier to mis-classify the image
	if (success == 0) :
		print("Was not able to get classifier to mis-classify the person.")
	# make a copy of the final perturbed image
	newFileName = actualperson+"-round"+str(perturbationcount)+".jpg"
	copyfile(filename,newFileName)
	# clean up unnecessary files
	for f in os.listdir('.') :
		if re.search("attack*", f) :
			os.remove(os.path.join('.', f))
	return success, minconf, newFileName, perturbationcount, personclass

if __name__ == "__main__":
	# initialize results array
	finalresults = []
	# set test parameters
	maxrounds = 10
	initialperturbation = 10
	perturbationscale = 10	
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
		result = attack(targetImage, actualperson, minconf, baselinesuccess, maxrounds, initialperturbation, perturbationscale)
		# success represents whether or not misclassification was achieved
		success = result[0]
		# minconf is minimum confidence acheived for the actual person class
		minconf = result[1]
		# newImage is filename of perturbed image
		newImage = result[2]
		perturbationcount = result[3]
		totalperturbation = initialperturbation + perturbationcount * perturbationscale
		personclass = result[4]
		changeconf = minconf - baselineconf
		percentchange = changeconf / baselineconf
		# create array storing attack results and add to array for all image attack results		
		imageresult = [targetImage, changeconf, percentchange, success, perturbationcount, totalperturbation, personclass]
		finalresults.append(imageresult)
		print(targetImage + "- Change in confidence: " + str(changeconf) + "\n")
		print("Number of perturbations: " + str(perturbationcount) + ", Total perturbation: " + str(totalperturbation) + ", Classified as: " + personclass + "\n\n")
	# write final results to a csv file		
	with open('imageAttackAlterAllPixelsIteration.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter= ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(['TargetImage', 'ChangeInConfidence', 'PercentChangeInConfidence', 'Success', 'PerturbationCount', 'Total Perturbation', 'ClassifiedAs'])
		for i in finalresults :
			writer.writerow(i)