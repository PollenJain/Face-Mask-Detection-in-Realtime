import argparse
import os 
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from FPS import FPS
from WebcamVideoStream_and_mask import WebcamVideoStream_and_mask

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model (different from face mask detector) directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
args = vars(ap.parse_args())


# We are detecting whether given image contains a face mask or not :
# 1. Detect if face is there in the image using face detector model
# 2. Only If face is found in the image then detect if face contains mask on it.

	
# initialize the video stream and allow the camera sensor to warm up
# grab a pointer to the video stream and initialize the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream_and_mask(src=0).start()
fps = FPS().start() #Notes the start time
width = 440

# loop over the frames from the video stream
while fps._numFrames < args["num_frames"]:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	# Capture frame-by-frame
	frame = vs.read()
	# show the output frame
	cv2.imshow("Output", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	
	
	# update the FPS counter
	fps.update()
	
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup

cv2.destroyAllWindows()
vs.stop()
