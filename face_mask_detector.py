#Run : python face_mask_detector.py --dataset "./dataset"
import argparse
from myPaths import get_image_path_list
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="path to input dataset", required=True)
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to store face mask detector model")
args = vars(ap.parse_args())

INIT_LR = 1e-4
EPOCHS = 1
BS = 32

print("[INFO] loading images ...")
imagePaths = []
imagePaths = get_image_path_list(args["dataset"])
#print("imagePaths", imagePaths)
labels = []
data = []
for imagePath in imagePaths:
	#print(imagePath.split(os.path.sep))
	label = imagePath.split("/")[-2]
	#print(label)
	labels.append(label) #On Windows, path separator is '\', where as on Ubuntu it is '/'
	image = load_img(imagePath, target_size=(224,224))
	image = img_to_array(image)
	image = preprocess_input(image)
	data.append(image)

data = np.array(data, dtype="float32")
labels = np.array(labels)
		
#labelEncoder = LabelEncoder()
#integerLabels = labelEncoder.fit_transform(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.30, stratify=labels, random_state=42)

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
	
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)
	
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")
model.save('face_mask_detector_model.h5')
