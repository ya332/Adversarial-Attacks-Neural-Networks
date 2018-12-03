import label_image2

def classify (actualperson, filename) :
	# apply classifier to perturbed image
	labelresults = label_image2.main(["--graph","/tmp/output_graph.pb","--labels","/tmp/output_labels.txt","--input_layer","Placeholder","--output_layer","final_result","--image",filename])
	# extract the person id for the most likely label returned by the classifier
	personclass = labelresults[0][0].lower()
	# check if mis-classified
	if personclass != actualperson :
		# if so, this has been a successful attack
		success = 1
	else : 
		success = 0
	# find classifier's confidence that the image is the actual person (will not be first listed in case of mis-classification)
	for k in range(len(labelresults)) :
		person = labelresults[k][0].lower()
		if person == actualperson :
			targetconf = labelresults[k][1]
	# return whether the attack was successful, the target confidence (for the actual person), and the most likely label returned by the classifier
	labelresults = [success, targetconf, personclass]
	return labelresults