import cv2
import numpy as np
import math
import random
import os, re
import label_image2
from shutil import copyfile
import csv

def attack(inputImage, minconf, maxrounds, initialperturbation, perturbationscale):
	# read image
	m = cv2.imread(inputImage,0)
	h,w = np.shape(m)
	# initialize variables for keeping track of results
	perturbationcount = 1
	# baseline confidence for this attack
	baselineconf = minconf
	# targetconf will be used to track classifier's confidence that the image is the actual person
	targetconf = 1
	# print(perturbationcount)
	m = cv2.imread(inputImage,0)
	# extract actual person id from image filename
	actualperson = inputImage.split('/')[1].lower()
	# print(actualperson)
	# set initial perturbation values
	perturbation = initialperturbation
	for iter in range(maxrounds) :
		filename='attack'+str(perturbationcount)+'.jpg'
		# apply perturbation
		for i in range(0,h) :
			for j in range(0,w) :
				# apply perturbation, bounding possible pixel values between 0 and 255
				if (m[i][j] + perturbation > 255) :
					m[i][j] = 255
				else :
					m[i][j] = m[i][j] + perturbation
		# save perturbed image
		cv2.imwrite(filename,m)
		# apply classifier to perturbed image
		labelresults = classify(actualperson, filename)
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
			perturbation += perturbationscale
		perturbationcount += 1
	# if we have successfully caused the classifier to mis-classify the person
	if (success == 1) :
		newFileName = actualperson+"-round"+str(perturbationcount)+".jpg"
		copyfile(filename,newFileName)
	elif (success == 0) :
		print("Was not able to get classifier to mis-classify the person.")
		newFileName = inputImage
	else :
		print("Invalid success value")
	# clean up unnecessary files
	for f in os.listdir('.') :
		if re.search("attack*", f) :
			os.remove(os.path.join('.', f))
	return success, minconf, newFileName, perturbationcount, personclass

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
	labelresults = [success, targetconf, personclass]
	return labelresults

if __name__ == "__main__":
	# initialize results array
	results = []
	# set test parameters
	maxrounds = 10
	initialperturbation = 20
	perturbationscale = 10
	# determine list of test files
	for fFileObj in os.walk("training/") :
		dirList = fFileObj[1]
		dirList.sort()
		print(dirList)
		break
	for dir in dirList :
		targetImage = os.path.join("training", dir, "image1.jpg")
		success = 0
		baselineresult = label_image2.main(["--graph","/tmp/output_graph.pb","--labels","/tmp/output_labels.txt","--input_layer","Placeholder","--output_layer","final_result","--image",targetImage])
		baselineconf = baselineresult[0][1]		
		print(targetImage + "- Baseline confidence: " + str(baselineconf) + "\n")
		minconf = baselineconf
		result0 = attack(targetImage, minconf, maxrounds, initialperturbation, perturbationscale)
		success = result0[0]
		minconf = result0[1]
		newImage = result0[2]
		perturbationcount = result0[3]
		totalperturbation = initialperturbation + perturbationcount * perturbationscale
		personclass = result0[4]
		changeconf = minconf - baselineconf
		percentchange = changeconf / baselineconf
		imageresult = [targetImage, changeconf, percentchange, success, perturbationcount, totalperturbation, personclass]
		results.append(imageresult)
		print(targetImage + "- Change in confidence: " + str(changeconf) + "\n")
		print("Number of perturbations: " + str(perturbationcount) + ", Total perturbation: " + str(totalperturbation) + ", Classified as: " + personclass + "\n\n")
	with open('imageAttackAlterAllPixels.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter= ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in results :
			writer.writerow(i)