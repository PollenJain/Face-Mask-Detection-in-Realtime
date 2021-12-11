import cv2
from threading import Thread
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

class WebcamVideoStream_and_mask:
	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
		print("[INFO] loading face detector model...")
		prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
		weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
		self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

		print("[INFO] loading face mask detector model...")
		self.maskNet = load_model("mask_detector.model")
		self.confidence = 0.5
		
		
	def start(self):
		# start the thread to read frames from the video stream. Once, start method gets called from the driver function, the update method will be called by the run() of Thread class. update() gets executed in a separate thread.
		Thread(target=self.update, args=()).start()
		return self
	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
			height = self.frame.shape[0]
			width = 400
			dim = (width, height)
			self.frame = cv2.resize(self.frame, dim, interpolation = cv2.INTER_AREA)
			(locs, preds) = self.detect_and_predict_mask()
			for (box, pred) in zip(locs, preds):
				# unpack the bounding box and predictions
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred
				label = "Mask" if mask > withoutMask else "No Mask"	
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
				# include the probability in the label
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
				# display the label and bounding box rectangle on the output
				# frame
				cv2.putText(self.frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(self.frame, (startX, startY), (endX, endY), color, 2)
				
				
	def read(self):
		# return the frame most recently read
		#In order to access the most recently polled frame  from the stream , we’ll use the read method
		return self.frame
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
		
	
	def detect_and_predict_mask(self):
		(h,w) = self.frame.shape[:2]
		blob = cv2.dnn.blobFromImage(self.frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

		print("[INFO] computing face detections...")
		self.faceNet.setInput(blob)
		detections = self.faceNet.forward()


		# More than one face can be detected in the input image. Looping over each of the face in the image and checking it is found with the desired confidence probablity
		print("[INFO] detections.shape[2] = ", str(detections.shape[2]))
		# initialize our list of faces, their corresponding locations,
		# and the list of predictions from our face mask network
		faces = []
		locs = []
		preds = []
		for i in range(0, detections.shape[2]):
			#print("[INFO] i = ", str(i))
			confidence = detections[0, 0, i, 2]
			if confidence > self.confidence:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
				
				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = self.frame[startY:endY, startX:endX]
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
				
				(mask, withoutMask) = self.maskNet.predict(face)[0]
				# determine the class label and color we'll use to draw
				# the bounding box and text
				
				
			# only make a predictions if at least one face was detected
			if len(faces) > 0:
				# for faster inference we'll make batch predictions on *all*
				# faces at the same time rather than one-by-one predictions
				# in the above `for` loop
				preds = self.maskNet.predict(faces)
			# return a 2-tuple of the face locations and their corresponding
			# locations
		return (locs, preds)	
	