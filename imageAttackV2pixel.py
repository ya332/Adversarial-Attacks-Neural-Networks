import cv2
import numpy as np
import math
import random
import os, re
import label_image2
from shutil import copyfile

def attack(inputImage, imageCount, minconf, round):
	# set factor for how many pixels to skip in horizontal and vertical directions between tests
	densityFactor = 50
	# read image
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
			# add result to classresults dictionary
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
	# find second best pixel
	secondbestpixel = ()
	if bestpixel != () :
		classresults.pop(bestpixel)
	nextminconf = 1
	for (k, v) in classresults.items() :
		if v < nextminconf :
			nextminconf = v
			secondbestpixel = k
	# perturb second best pixel on output image from perturbing first best pixel
	m2 = cv2.imread(newFileName,0)
	m2[secondbestpixel[0]][secondbestpixel[1]]=(m2[secondbestpixel[0]][secondbestpixel[1]]+128)%256
	# save perturbed image after second pixel edit
	filename2 = actualperson+'-2ndpixelround'+'.jpg'
	cv2.imwrite(filename2,m2)
	# apply classifier to perturbed image
	labelresults2 = classify(actualperson, filename2)
	success2 = labelresults2[0]
	targetconf2 = labelresults2[1]
	print(actualperson + ": " + str(targetconf2)+"\n")
	# update minimum confidence and success value to return if altering the second pixel improved results
	if success2 == 1 :
		success = success2
	if targetconf2 < minconf :
		minconf = targetconf
	else :
		print("Altering second best pixel did not improve results.")
	# clean up unnecessary files
	for f in os.listdir('.') :
		if re.search("attack*", f) :
			os.remove(os.path.join('.', f))
	# return results
	return success, minconf

def classify (actualperson, filename) :
	# apply classifier to perturbed image
	labels = label_image2.main(["--graph","/tmp/output_graph.pb","--labels","/tmp/output_labels.txt","--input_layer","Placeholder","--output_layer","final_result","--image",filename])
	# extract the person id for the most likely label
	personclass = labels[0][0]
	# check if mis-classified
	if personclass != actualperson :
		# if so, this has been a successful attack
		success = 1
	else : 
		success = 0
	# find classifier's confidence that the image is the actual person (will not be first listed in case of mis-classification)
	for k in range(len(labels)) :
		person = labels[k][0]
		if person == actualperson :
			targetconf = labels[k][1]
	labelresults = [success, targetconf]
	return labelresults
			
if __name__ == "__main__":
	# initialize count of images to test
	imageCount = 0
	# initialize results array
	results = []
	for fFileObj in os.walk("CroppedYale/") :
		dirList = fFileObj[1]
		print(dirList)
		break
	for dir in dirList :
		targetImage = os.path.join("CroppedYale", dir, "image0.jpg")
		success = 0
		baselineresult = label_image2.main(["--graph","/tmp/output_graph.pb","--labels","/tmp/output_labels.txt","--input_layer","Placeholder","--output_layer","final_result","--image",targetImage])
		baselineconf = baselineresult[0][1]		
		minconf = baselineconf
		result0 = attack(targetImage, imageCount, minconf, round=0)
		success = result0[0]
		minconf = result0[1]
		changeconf = minconf - baselineconf
		# second attack round
		# minconf = attack(targetImage, imageCount, minconf, round=1)
		result = [targetImage, changeconf, success]
		results.append(result)
		print(targetImage)
		print("Change in confidence: " + str(changeconf) + "\n")
		imageCount += 1
		#if imageCount > 4 :
		#	break
	print(results)