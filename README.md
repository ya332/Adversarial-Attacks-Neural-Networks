# AdversarialAttacks
Repo for collaboration on Image Perturbation and Adversarial Attacks

The attack scripts in this repository are intended to be run against a facial recognition image classifier.

We used a classifier trained on the Yale B Cropped Face Database and using the the Inception V3 architecture.

The classifier was built and trained using TensorFlow and Google Colaboratory. The scripts assume that the classifier model is stored in the /tmp folder with TensorFlow graph file "output_graph.pb" and labels "output_labels.txt".

Each imageAttack script requires that the label_image2.py and classify.py scripts be available in the working directory. The imageDualAttack.py script requires imageAttack2Pixels.py and imageAttackAllPixels.py.
