import numpy as np
import pickle
import cv2, os
from glob import glob
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import to_categorical as np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
	# Find the first available gesture image
	for gesture_folder in glob('gestures/*'):
		if os.path.isdir(gesture_folder):
			for image_file in glob(os.path.join(gesture_folder, '*.jpg')):
				img = cv2.imread(image_file, 0)
				if img is not None:
					return img.shape
	# Return a default or raise an error if no images are found
	raise FileNotFoundError("No gesture images found to determine size.")

def get_num_of_classes():
	# return len(glob('gestures/*'))
	# The above line is not robust if gesture folders are not named sequentially from 0.
	# For example, if we have folders 0, 1, 3, it will return 3, but we need 4 classes.
	gesture_folders = glob('gestures/*')
	if not gesture_folders:
		return 0
	# Assuming folder names are integers representing class labels
	max_g_id = max([int(os.path.basename(folder)) for folder in gesture_folders if os.path.basename(folder).isdigit()])
	return max_g_id + 1

image_x, image_y = get_image_size()

def cnn_model(image_x, image_y):
	num_of_classes = get_num_of_classes()
	model = Sequential()
	model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
	model.add(Conv2D(32, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
	model.add(Conv2D(64, (5,5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_of_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	filepath="cnn_model_keras2.keras"
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint1]
	#from keras.utils import plot_model
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model, callbacks_list

def train():
	image_x, image_y = get_image_size()
	with open("train_images", "rb") as f:
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open("val_images", "rb") as f:
		val_images = np.array(pickle.load(f))
	with open("val_labels", "rb") as f:
		val_labels = np.array(pickle.load(f), dtype=np.int32)

	num_of_classes = get_num_of_classes()
	train_images = np.array(train_images, dtype=np.float32) / 255.0
	val_images = np.array(val_images, dtype=np.float32) / 255.0
	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
	val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
	train_labels = np_utils(train_labels, num_classes=num_of_classes)
	val_labels = np_utils(val_labels, num_classes=num_of_classes)

	print(val_labels.shape)

	model, callbacks_list = cnn_model(image_x, image_y)
	model.summary()
	model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=15, batch_size=500, callbacks=callbacks_list)
	scores = model.evaluate(val_images, val_labels, verbose=0)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))
	#model.save('cnn_model_keras2.h5')

train()
train()
K.clear_session();
