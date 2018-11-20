import cv2
import numpy as np
import math
import random
import os, re
import label_image2
from shutil import copyfile

def attack(inputImage, round):
    # This variable sets how many columns and rows to skip between pixel tests. The smaller the number, the more pixels will be tested (and the longer the tests will take). I suggest using 20 to start.
    densityFactor = 20
    # read in image and determine height and width
    m = cv2.imread(inputImage,0)
    h,w = np.shape(m)
    # keep track of most successful attack pixel
    bestpixel = []
    # keep track of minimum accuracy achieved by classifer after applying attack
    minacc = 1
    # keep track of filename for best attack output
    bestattack = ""
    # keep track of how many attacks have been attempted
    attackcount = 0
    # iterate through image h by w matrix
    for i in range(0,h-densityFactor,densityFactor) :
        # choose a random offset in the vertical direction (optional)
        yoffset = random.randint(0,densityFactor-1)
        for j in range (0,w-densityFactor,densityFactor) :
            # choose a random offset in the horizontal direction (optional)
            xoffset = random.randint(0,densityFactor-1)
            print(attackcount)
            # re-read fresh copy of original image
            m = cv2.imread(inputImage,0)
            # create filename for output after attack
            filename='attack'+str(attackcount)+'.jpg'
            # consider randomizing the perturbation instead of inverting?
            # perturbation = random.randint(64,192)
            # attack: invert the value of the selected pixel (add 128 mod 256)
            m[i+yoffset][j+xoffset]=(m[i+yoffset][j+xoffset]+128)%256
            # write the output image to disk
            cv2.imwrite(filename,m)
            # apply classifer and record results
            results=label_image2.main(["--graph","/tmp/output_graph.pb","--labels","/tmp/output_labels.txt","--input_layer","Placeholder","--output_layer","final_result","--image",filename])
            # parse results
            person=results[0][0]
            acc=results[0][1]
            # print accuracy
            print(str(acc)+"\n")
            # if accuracy in this iteration represents a new minimum, update minimum variables
            if (acc < minacc) :
                bestpixel = [i,j]
                minacc = acc
                bestattack=filename
            # increment attack count
            attackcount+=1
    print("best attack: "+str(bestattack))
    print(bestpixel)
    print(minacc)
    # make a copy of the image that represented the most successful in this round of attacks
    newFileName = "round"+str(round)+".jpg"
    copyfile(bestattack,newFileName)
    # cleanup - remove all of the other attack output images
    for f in os.listdir('.') :
        if re.search("attack*", f) :
            os.remove(os.path.join('.', f))
    # return most successful attack image filename
    return newFileName


if __name__ == "__main__":
    # prompt for filename of image to attack
    inputImage = input('Enter image filename to attack')
    # perform series of round of attacks - each round introduces a perturbation to a single pixel that is selected from a sample in a greedy approach (that which causes the greatest immediate decrease in accuracy)
    round0 = attack(inputImage,round=0)
    round1 = attack(round0,round=1)
    #round2 = attack(round1,round=2)
