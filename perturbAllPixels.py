import cv2
import numpy as np
import math
import random
import os, re
import label_image2
from shutil import copyfile
import csv

INPUT_DIRECTORY = 'indivpixelchange/'

def attack(inputImage, imageCount, minconf, actualperson):
	# read image
	m = cv2.imread(inputImage,0)
	h,w = np.shape(m)
	# initialize variables for keeping track of results
	testcount = 0
	# baseline confidence for this attack
	baselineconf = minconf
	# targetconf will be used to track classifier's confidence that the image is the actual person
	targetconf = 1
	# print(testcount)
	# print(actualperson)
	filename='attack'+str(testcount)+'.jpg'
	# apply perturbation
	for i in range(0,h) :
		for j in range(0,w) :
			# consider randomizing the perturbation instead of inverting?
			perturbation = random.randint(20,50)
			# apply perturbation, bounding possible pixel values between 0 and 255
			if j % 2 == 0 :
				if (m[i][j] + perturbation > 255) :
					m[i][j] = 255
				else :
					m[i][j] = m[i][j] + perturbation
			elif j % 2 == 1 :
				if (m[i][j] - perturbation < 0) :
					m[i][j] = 0
				else :
					m[i][j] = m[i][j] - perturbation
	# save perturbed image
	cv2.imwrite(filename,m)
	# apply classifier to perturbed image
	labelresults = classify(actualperson, filename)
	# success indicates whether image was misclassified (1=yes, 0=no)
	success = labelresults[0]
	# targetconf is the classifier's confidence level that the image is the actual person id
	targetconf = labelresults[1]
	# print(actualperson + ": " + str(targetconf)+"\n")
	# update minimum confidence
	if (targetconf < minconf) :
		minconf = targetconf
	testcount += 1
	# if we have found a pixel that reduces the confidence level
	if (minconf < baselineconf) :
		# print("best attack: "+str(bestattack))
		# print(bestpixel)
		# print(str(minconf)+"\n")
		newFileName = actualperson + "-all" + ".jpg"
		copyfile(filename,newFileName)
	else :
		print("No better attack was found.")
		newFileName = inputImage
	# clean up unnecessary files
	for f in os.listdir('.') :
		if re.search("attack*", f) :
			os.remove(os.path.join('.', f))
	return success, minconf, newFileName

def classify (actualperson, filename) :
	# apply classifier to perturbed image
	labelresults = label_image2.main(["--graph","/tmp/output_graph.pb","--labels","/tmp/output_labels.txt","--input_layer","Placeholder","--output_layer","final_result","--image",filename])
	# extract the person id for the most likely label
	personclass = labelresults[0][0]
	# check if mis-classified
	if personclass != actualperson :
		# if so, this has been a successful attack
		success = 1
	else : 
		success = 0
	# find classifier's confidence that the image is the actual person (will not be first listed in case of mis-classification)
	for k in range(len(labelresults)) :
		person = labelresults[k][0]
		if person == actualperson :
			targetconf = labelresults[k][1]
	labelresults = [success, targetconf]
	return labelresults

if __name__ == "__main__":
	# initialize count of images to test
	imageCount = 0
	# initialize results array
	results = []
	# walk input file directory to obtain list of directories for each person
	for fFileObj in os.walk(INPUT_DIRECTORY) :
		dirList = fFileObj[1]
		dirList.sort()
		print(dirList)
		break
	for dir in dirList :
		# walk directory for each person to obtain that person's image filename
		for fFileObj2 in os.walk(os.path.join(INPUT_DIRECTORY, dir)) :
			file = fFileObj2[2][0]
			break
		targetImage = os.path.join(INPUT_DIRECTORY, dir, file)
		success = 0
		baselineresult = label_image2.main(["--graph","/tmp/output_graph.pb","--labels","/tmp/output_labels.txt","--input_layer","Placeholder","--output_layer","final_result","--image",targetImage])
		baselineconf = baselineresult[0][1]		
		print(targetImage + "- Baseline confidence: " + str(baselineconf) + "\n")
		minconf = baselineconf
		result0 = attack(targetImage, imageCount, minconf)
		success = result0[0]
		minconf = result0[1]
		newImage = result0[2]
		changeconf = minconf - baselineconf
		percentchange = changeconf / baselineconf
		imageresult = [targetImage, changeconf, percentchange, success]
		results.append(imageresult)
		print(targetImage + "- Change in confidence: " + str(changeconf) + "\n")
		imageCount += 1
		#if imageCount > 4 :
		#	break
	with open('doubleAttack.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter= ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in results :
			writer.writerow(i)