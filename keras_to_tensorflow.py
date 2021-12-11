from tensorflow import keras

import tensorflow
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_keras_model", type=str,
	default="face_mask_detector_model.h5",
	help="path to trained face mask detector keras model in h5 format")
ap.add_argument("-f", "--output_model", type=str,
	default="face_mask_detector_model.pb",
	help="path to store .pb model in a directory")
args = vars(ap.parse_args())

#model = keras.models.load_model(args["input_keras_model"])

#from keras.models import load_model => Can't use it since our keras model is generated using tensorflow.keras and not keras directly
from tensorflow.keras.models import load_model
from keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)
model = load_model(args["input_keras_model"])
print(model.outputs)
print(model.inputs)
session = K.get_session()
graph = session.graph
