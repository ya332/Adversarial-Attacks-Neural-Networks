import cv2
import numpy as np
import math
import random
import os, re
import label_image2
from shutil import copyfile

def attack(inputImage, round):
	densityFactor = 10
	m = cv2.imread(inputImage,0)
	h,w = np.shape(m)
	bestpixel = []
	minacc = 1
	bestattack = ""
	attackcount = 0
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
			print(str(acc)+"\n")
			if (acc < minacc) :
				bestpixel = [i,j]
				minacc = acc
				bestattack=filename
			attackcount+=1
	print("best attack: "+str(bestattack))
	print(bestpixel)
	print(minacc)
	newFileName = "round"+str(round)+".jpg"
	copyfile(bestattack,newFileName)
	for f in os.listdir('.') :
		if re.search("attack*", f) :
			os.remove(os.path.join('.', f))
	return newFileName


if __name__ == "__main__":
	inputImage = input('Enter image filename to attack')
	round0 = attack(inputImage,round=0)
	round1 = attack(round0,round=1)
	#round2 = attack(round1,round=2)