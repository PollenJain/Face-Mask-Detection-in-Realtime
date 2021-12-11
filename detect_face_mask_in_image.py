import argparse
import os 
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model (different from face mask detector) directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# We are detecting whether given image contains a face mask or not :
# 1. Detect if face is there in the image using face detector model
# 2. Only If face is found in the image then detect if face contains mask on it.

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)


print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

image = cv2.imread(args["image"])
orig = image.copy()

(h,w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

# More than one face can be detected in the input image. Looping over each of the face in the image and checking it is found with the desired confidence probablity
print("[INFO] detections.shape[2] = ", str(detections.shape[2]))
for i in range(0, detections.shape[2]):
	print("[INFO] i = ", str(i))
	confidence = detections[0, 0, i, 2]
	if confidence > args["confidence"]:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# ensure the bounding boxes fall within the dimensions of
		# the frame
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
		
		# extract the face ROI, convert it from BGR to RGB channel
		# ordering, resize it to 224x224, and preprocess it
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)
		# pass the face through the model to determine if the face
		# has a mask or not
		(mask, withoutMask) = model.predict(face)[0]
		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask"+str(i) if mask > withoutMask else "No Mask"+str(i)
		color = (0, 255, 0) if label == "Mask"+str(i) else (0, 0, 255)
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
		
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
print(args["image"].split(".")[:-1])
cv2.imwrite(args["image"].split(".")[:-1][0]+"_labelled.jpg", image)