import cv2
import numpy as np
import math
import random
import os, re
import label_image2
from shutil import copyfile

def attack(inputImage, imageCount, minconf, round):
	# set factor for how many pixels to skip in horizontal and vertical directions between tests
	densityFactor = 30
	# read image
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
	for i in range(0,h-densityFactor,densityFactor) :
		yoffset = random.randint(0,densityFactor-1)
		for j in range (0,w-densityFactor,densityFactor) :
			xoffset = random.randint(0,densityFactor-1)
			currentpixel = (i + yoffset, j + xoffset)
			print(testcount)
			m = cv2.imread(inputImage,0)
			# extract actual person id from image filename
			actualperson = inputImage.split('/')[1].lower()
			# print(actualperson)
			filename='attack'+str(testcount)+'.jpg'
			# consider randomizing the perturbation instead of inverting?
			# perturbation = random.randint(64,192)
			# apply perturbation
			m[i+yoffset][j+xoffset]=(m[i+yoffset][j+xoffset]+128)%256
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
				bestpixel = currentpixel
				minconf = targetconf
				bestattack = filename
			# add result to classresults array
			classresults[currentpixel] = targetconf
			testcount += 1
	# if we have found a pixel that reduces the confidence level
	if (minconf < baselineconf) :
		# print("best attack: "+str(bestattack))
		# print(bestpixel)
		# print(str(minconf)+"\n")
		newFileName = actualperson+"-round"+str(round)+".jpg"
		copyfile(bestattack,newFileName)
	else :
		print("No better attack was found.")
		newFileName = inputImage
	for f in os.listdir('.') :
		if re.search("attack*", f) :
			os.remove(os.path.join('.', f))
	return success, minconf, newFileName

def classify (actualperson, filename) :
	# apply classifier to perturbed image
	labelresults = label_image2.main(["--graph","/tmp/output_graph.pb","--labelresults","/tmp/output_labelresults.txt","--input_layer","Placeholder","--output_layer","final_result","--image",filename])
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
	for fFileObj in os.walk("testing/") :
		dirList = fFileObj[1]
		dirList.sort()
		print(dirList)
		break
	for dir in dirList :
		targetImage = os.path.join("testing", dir, "image0.jpg")
		success = 0
		baselineresult = label_image2.main(["--graph","/tmp/output_graph.pb","--labelresults","/tmp/output_labelresults.txt","--input_layer","Placeholder","--output_layer","final_result","--image",targetImage])
		baselineconf = baselineresult[0][1]		
		minconf = baselineconf
		result0 = attack(targetImage, imageCount, minconf, round=0)
		success0 = result0[0]
		minconf0 = result0[1]
		newImage = result0[2]
		changeconf0 = minconf0 - baselineconf
		percentchange0 = changeconf0 / baselineconf
		# second attack round
		result1 = attack(newImage, imageCount, minconf, round=1)
		success1 = result1[0]
		minconf1 = result1[1]
		newImage = result1[2]
		changeconf1 = minconf1 - baselineconf
		percentchange1 = changeconf1 / baselineconf
		imageresult = [targetImage, changeconf1, percentchange1, success1, changeconf0, percentchange0, success0]
		results.append(imageresult)
		print(targetImage + "- Change in confidence: " + str(changeconf1) + "\n")
		imageCount += 1
		#if imageCount > 4 :
		#	break
	print(results)