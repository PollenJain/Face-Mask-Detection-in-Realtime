import argparse
import os 
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from FPS import FPS
from WebcamVideoStream import WebcamVideoStream
from timeit import default_timer as timer
import csv

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model (different from face mask detector) directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-n", "--num-frames", type=int, default=200,
	help="# of frames to loop over for FPS test")
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


def detect_and_predict_mask(frame, faceNet, maskNet):
		(h,w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

		print("[INFO] computing face detections...")
		net.setInput(blob)
		detections = net.forward()


		# More than one face can be detected in the input image. Looping over each of the face in the image and checking it is found with the desired confidence probablity
		#print("[INFO] detections.shape[2] = ", str(detections.shape[2]))
		# initialize our list of faces, their corresponding locations,
		# and the list of predictions from our face mask network
		faces = []
		locs = []
		preds = []
		for i in range(0, detections.shape[2]):
			#print("[INFO] i = ", str(i))
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
				face = frame[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				face = np.expand_dims(face, axis=0)
				# pass the face through the model to determine if the face
				# has a mask or not
				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))
				
				(mask, withoutMask) = model.predict(face)[0]
				# determine the class label and color we'll use to draw
				# the bounding box and text
				
				
		# only make a predictions if at least one face was detected
		if len(faces) > 0:
			# for faster inference we'll make batch predictions on *all*
			# faces at the same time rather than one-by-one predictions
			# in the above `for` loop
			preds = maskNet.predict(faces)
		# return a 2-tuple of the face locations and their corresponding
		# locations
		return (locs, preds)		
		

def main():
	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	# cap = cv2.VideoCapture(0)
	vs = WebcamVideoStream(src=0).start()
	fps = FPS().start() #Notes the start time
	width = 440

	with open("consumer_thread.csv", 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["Thread Frame #",  "Time spent in reading the frame (seconds) from queue", "Time spent performing inference on the frame (seconds)"])
		# loop over the frames from the video stream
		#while True:
		while fps._numFrames < args["num_frames"]:
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 400 pixels
			# Capture frame-by-frame
			start = timer()
			frame = vs.readFromQueue()
			end = timer()
			# if frame is not None then there was atleast one frame in queue 
			# when read from the queue and returned. Else queue was empty.
			if frame is not None:
				# update the FPS counter
				fps.update()
				consumerThreadFrameNumber = fps._numFrames
				consumerThreadTimeTakenToReadThisFrame = (end-start)
				print("[INFO] Consumer Thread : Time taken to read frame number", consumerThreadFrameNumber, "from queue is", consumerThreadTimeTakenToReadThisFrame,"seconds")	
				height = frame.shape[0]
				dim = (width, height)
				frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
				# detect faces in the frame and determine if they are wearing a
				# face mask or not
				startInferenceTime = timer()
				(locs, preds) = detect_and_predict_mask(frame, net, model)
				endInferenceTime = timer()
				consumerThreadTimeTakenToPerformInference = (endInferenceTime-startInferenceTime)
				print("[INFO] Consumer Thread : Time taken to performing inference on consumed frame number", consumerThreadFrameNumber, "is", consumerThreadTimeTakenToPerformInference,"seconds")
				writer.writerow([consumerThreadFrameNumber,  consumerThreadTimeTakenToReadThisFrame, consumerThreadTimeTakenToPerformInference])
				for (box, pred) in zip(locs, preds):
					# unpack the bounding box and predictions
					(startX, startY, endX, endY) = box
					(mask, withoutMask) = pred
					label = "Mask" if mask > withoutMask else "No Mask"	
					color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
					# include the probability in the label
					#label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
					# display the label and bounding box rectangle on the output
					# frame
					cv2.putText(frame, label, (startX, startY - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
					cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
					print("Showing frame")
					# show the output frame
					cv2.imshow("Output", frame)
					#cv2.destroyAllWindows()
					#key = cv2.waitKey(10) & 0xFF
					
				key = cv2.waitKey(1) & 0xFF
				# if the `q` key was pressed, break from the loop
				if key == ord("q"):
					break
			

	fps.stop()
	vs.stop()

	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))		

	cv2.destroyAllWindows()
	
