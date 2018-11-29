import cv2
import numpy as np
import math
import random
import os, re
import label_image2
from shutil import copyfile

def attack(inputImage, imageCount, minconf, round):
	densityFactor = 50
	m = cv2.imread(inputImage,0)
	h,w = np.shape(m)
	bestpixel = []
	bestattack = ""
	attackcount = 0
	baselineconf = minconf
	targetconf = 1
	for i in range(0,h-densityFactor,densityFactor) :
		yoffset = random.randint(0,densityFactor-1)
		for j in range (0,w-densityFactor,densityFactor) :
			xoffset = random.randint(0,densityFactor-1)
			# print(attackcount)
			m = cv2.imread(inputImage,0)
			actualperson = inputImage.split('/')[1].lower()
			# print(actualperson)
			filename='attack'+str(attackcount)+'.jpg'
			# consider randomizing the perturbation instead of inverting?
			# perturbation = random.randint(64,192)
			m[i+yoffset][j+xoffset]=(m[i+yoffset][j+xoffset]+128)%256
			cv2.imwrite(filename,m)
			classresults = label_image2.main(["--graph","/tmp/output_graph.pb","--labels","/tmp/output_labels.txt","--input_layer","Placeholder","--output_layer","final_result","--image",filename])
			personclass = classresults[0][0]
			# check if mis-classified
			if personclass != actualperson :
				success = 1
			else : 
				success = 0
			for i in range(len(classresults)) :
				person = classresults[i][0]
				if person == actualperson :
					targetconf = classresults[i][1]
			# print(actualperson + ": " + str(targetconf)+"\n")
			# update minimum confidence
			if (targetconf < minconf) :
				bestpixel = [i,j]
				minconf = targetconf
				bestattack = filename
			attackcount += 1
	# if we have found a pixel that reduces the confidence level
	if (minconf < baselineconf) :
		# print("best attack: "+str(bestattack))
		# print(bestpixel)
		# print(str(minconf)+"\n")
		newFileName = "image"+str(imageCount)+"-round"+str(round)+".jpg"
		copyfile(bestattack,newFileName)
	else :
		print("No better attack was found.")
	for f in os.listdir('.') :
		if re.search("attack*", f) :
			os.remove(os.path.join('.', f))
	return success, minconf


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
		# minconf = attack(targetImage, imageCount, minconf, round=1)
		result = [targetImage, changeconf, success]
		results.append(result)
		# print(targetImage)
		# print("Change in confidence: " + str(changeconf) + "\n")
		imageCount += 1
		#if imageCount > 4 :
		#	break
	print(results)