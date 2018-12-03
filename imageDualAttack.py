import cv2
import numpy as np
import math
import random
import os, re
from shutil import copyfile
import csv

import classify
import imageAttackAlterAllPixels
import imageAttackV2pixel

INPUT_DIRECTORY = 'testing/'

if __name__ == "__main__":
	# initialize results array
	finalresults = []
	for fFileObj in os.walk(INPUT_DIRECTORY) :
		dirList = fFileObj[1]
		dirList.sort()
		print(dirList)
		break
	for dir in dirList :
		targetImage = os.path.join(INPUT_DIRECTORY, dir, "image0.jpg")
		# extract actual person id from directory name
		actualperson = dir.lower()
		success = 0
		# determine baseline classifier confidence level for the actual person		
		baselineresult = classify.classify(actualperson, targetImage)
		baselinesuccess = baselineresult[0]
		baselineconf = baselineresult[1]	
		print(targetImage + "- Baseline confidence: " + str(baselineconf) + "\n")
		# minconf will be used to keep track of minimum confidence level acheived for target (actual person)		
		minconf = baselineconf
		# perform change-best-2-pixels attack
		result = imageAttackV2pixel.attack(targetImage, actualperson, minconf, baselinesuccess)
		# success represents whether or not misclassification was achieved
		success = result[0]
		# minconf is minimum confidence acheived for the actual person class
		minconf = result[1]
		# newImage is filename of perturbed image
		newImage = result[2]
		changeconf = minconf - baselineconf
		percentchange = changeconf / baselineconf
		print(targetImage + " - Change in confidence phase 1: " + str(changeconf) + "\n")
		# now perform alter-all-pixels attack on the output of the change-best-2-pixels attack
		result2 = imageAttackAlterAllPixels.attack(newImage, actualperson, minconf, success)
		success2 = result2[0]
		minconf2 = result2[1]
		finalImage = result2[2]
		changeconf2 = minconf2 - baselineconf
		percentchange2 = changeconf2 / baselineconf
		print(targetImage + " - Change in confidence phase 2: " + str(changeconf2) + "\n\n")
		# create array storing attack results and add to array for all image attack results		
		imageresult = [targetImage, changeconf, percentchange, success, changeconf2, percentchange2, success2]
		finalresults.append(imageresult)
	# write final results to a csv file		
	with open('dualattack.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter= ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(['TargetImage', 'ChangeInConfidencePhase1', 'PercentChangeInConfidencePhase1', 'SuccessPhase1', 'ChangeInConfidencePhase2', 'PercentChangeInConfidencePhase2', 'SuccessPhase2'])
		for i in finalresults :
			writer.writerow(i)