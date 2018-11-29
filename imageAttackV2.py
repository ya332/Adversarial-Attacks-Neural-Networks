import cv2
import numpy as np
import math
import random
import os, re
import label_image2
from shutil import copyfile

def attack(inputImage, imageCount, minacc, round):
	densityFactor = 40
	m = cv2.imread(inputImage,0)
	h,w = np.shape(m)
	bestpixel = []
	bestattack = ""
	attackcount = 0
	baselineacc = minacc
	for i in range(0,h-densityFactor,densityFactor) :
		yoffset = random.randint(0,densityFactor-1)
		for j in range (0,w-densityFactor,densityFactor) :
			xoffset = random.randint(0,densityFactor-1)
			print(attackcount)
			m = cv2.imread(inputImage,0)
			filename='attack'+str(attackcount)+'.jpg'
			# consider randomizing the perturbation instead of inverting?
			# perturbation = random.randint(64,192)
			m[i+yoffset][j+xoffset]=(m[i+yoffset][j+xoffset]+128)%256
			cv2.imwrite(filename,m)
			results=label_image2.main(["--graph","/tmp/output_graph.pb","--labels","/tmp/output_labels.txt","--input_layer","Placeholder","--output_layer","final_result","--image",filename])
			person=results[0][0]
			acc=results[0][1]
			print(person + ": " + str(acc)+"\n")
			if (acc < minacc) :
				bestpixel = [i,j]
				minacc = acc
				bestattack=filename
			attackcount+=1
	if (minacc < baselineacc) :
		print("best attack: "+str(bestattack))
		print(bestpixel)
		print(str(minacc)+"\n")
		newFileName = "image"+str(imageCount)+"-round"+str(round)+".jpg"
		copyfile(bestattack,newFileName)
	else :
		print("No better attack was found.")
	for f in os.listdir('.') :
		if re.search("attack*", f) :
			os.remove(os.path.join('.', f))
	return minacc


if __name__ == "__main__":
	# initialize count of images to test
	imageCount = 0
	for fFileObj in os.walk("CroppedYale/") :
		dirList = fFileObj[1]
		print(dirList)
		break
	for dir in dirList :
		targetImage = os.path.join("CroppedYale", dir, "image0.jpg")
		baselineresult=label_image2.main(["--graph","/tmp/output_graph.pb","--labels","/tmp/output_labels.txt","--input_layer","Placeholder","--output_layer","final_result","--image",targetImage])
		baselineacc=baselineresult[0][1]		
		minacc = baselineacc
		minacc = attack(targetImage, imageCount, minacc, round=0)
		minacc = attack(targetImage, imageCount, minacc, round=1)
		print(targetImage)
		print("Change in confidence: " + str(baselineacc - minacc) + "\n")
		imageCount += 1
		if imageCount > 4 :
			break