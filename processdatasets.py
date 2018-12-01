import os

directory = 'training/'
for fFileObj in os.walk(directory) :
    dirList = fFileObj[1]
    break
os.chdir(directory)
for dir in dirList :
    for fFileObj2 in os.walk(dir) :
        fileList = fFileObj2[2]
    os.chdir(dir)
    for f in fileList :
        if f == 'image0.jpg' :
            os.remove(f)
    os.chdir('..')
