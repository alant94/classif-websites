# import the necessary packages
from __future__ import print_function
from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from vgg16 import VGG16
import numpy as np
import argparse
import cv2

import os
from os import listdir

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
	help="Path to the input directory")
args = vars(ap.parse_args())

resultList = []

listOfImg = os.listdir(args["dir"])

website_id = args["dir"].split("/")[-1]
finFilename = "/home/alant/python/nirs/image_classif/results/" + website_id + ".txt"
finFile = open(finFilename, 'w')

for img in listOfImg:
	fullPathImg = args["dir"]+"/"+img
	print(fullPathImg)
# load the original image via OpenCV so we can draw on it and display
# it to our screen later
# orig = cv2.imread(args["image"])
	#orig = cv2.imread(fullPathImg) 

# load the input image using the Keras helper utility while ensuring
# that the image is resized to 224x224 pxiels, the required input
# dimensions for the network -- then convert the PIL image to a
# NumPy array
	print("[INFO] loading and preprocessing image...")
	image = image_utils.load_img(fullPathImg, target_size=(224, 224))
	image = image_utils.img_to_array(image)

# our image is now represented by a NumPy array of shape (3, 224, 224),
# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
# pass it through the network -- we'll also preprocess the image by
# subtracting the mean RGB pixel intensity from the ImageNet dataset
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)

# load the VGG16 network
	print("[INFO] loading network...")
	model = VGG16(weights="imagenet")
 
# classify the image
	print("[INFO] classifying image...")
	preds = model.predict(image)
# (inID, label) = decode_predictions(preds)[0]
	P = decode_predictions(preds)
	(imagenetID, label, prob) = P[0][0]

# display the predictions to our screen
	print("ImageNet ID: {}, Label: {}".format(imagenetID, label))
	imgAndClass = []
	imgAndClass.append(img)
	imgAndClass.append(label)
	resultList.append(imgAndClass)

	print(label, file=finFile)

print (resultList)

rawResName = "raw_" + args["dir"].split("/")[-1]
filename = "/home/alant/python/nirs/image_classif/results/" + rawResName + ".txt"
f = open(filename, 'w')

for item in resultList:
	print(item, file=f)

#cv2.putText(orig, "Label: {}".format(label), (10, 30),
#	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# cv2.imshow("Classification", orig)
# cv2.waitKey(0)
