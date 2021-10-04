from fetchData import *
import tensorflow as tf
import numpy as np

def plotEvolution(evolution): # plot a graph that shows the evolution of the model's accuracy based on the number of epochs
	import matplotlib.pyplot as plt

	plt.plot(evolution.history['val_accuracy']) # plot the accuracy
	plt.title('model accuracy')
	plt.ylabel('accuracy') # create y axis
	plt.xlabel('epoch') # create x axis
	plt.legend(['training'], loc='best')
	plt.show() # show the graph


def modifyData(train_images, train_labels, validation_images, validation_labels):

	from sklearn.model_selection import train_test_split

	from tensorflow.keras.utils import to_categorical

	# reshape the images to (32, 32, 1) to show that it has only one color channel
	train_images = train_images.reshape(train_images.shape[0], 32, 32, 1)
	validation_images = validation_images.reshape(validation_images.shape[0], 32, 32, 1)

	train_images = train_images.astype('float32')
	validation_images = validation_images.astype('float32')

	# normalize the values of the pixels
	train_images /= 255.0
	validation_images /= 255.0

	# one-hot-encode the labels
	train_labels = to_categorical(train_labels, 9)
	validation_labels = to_categorical(validation_labels, 9)

	return train_images, validation_images, train_labels, validation_labels

def define_model(): # create the network

	from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D
	from tensorflow.keras.layers import MaxPooling2D, BatchNormalization

	model = tf.keras.Sequential()

	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1))) # first convolutional layer that takes in the input
	model.add(BatchNormalization()) # normalize the input data
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization()) # normalize the input data
	model.add(MaxPooling2D((2, 2))) # add a pooling layer to get the best features of the images
	model.add(Dropout(0.2)) # drop some neurons


	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization()) # normalize the input data
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization()) # normalize the input data
	model.add(MaxPooling2D((2, 2))) # add a pooling layer to get the best features of the images
	model.add(Dropout(0.3)) # drop some neurons



	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization()) # normalize the input data
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization()) # normalize the input data
	model.add(MaxPooling2D((2, 2))) # add a pooling layer to get the best features of the images
	model.add(Dropout(0.4)) # drop some neurons


	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization()) # normalize the input data
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization()) # normalize the input data
	model.add(MaxPooling2D((2, 2))) # add a pooling layer to get the best features of the images
	model.add(Dropout(0.4)) # drop some neurons


	model.add(Flatten()) # flatten the data for the dense layer

	model.add(Dense(2048, activation='relu')) # dense layer
	model.add(BatchNormalization()) # normalize the input data
	model.add(Dropout(0.5)) # drop some neurons

	model.add(Dense(2048, activation='relu')) # dense layer
	model.add(BatchNormalization()) # normalize the input data
	model.add(Dropout(0.5)) # drop some neurons

	model.add(Dense(9, activation='softmax')) # each node will have the probability for the respective class

	print(model.summary())

	opt = tf.keras.optimizers.SGD(lr = 0.01, momentum = 0.9)
	model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])



def train(train_images, train_labels, validation_images, validation_labels):
	cnn = define_model() # create network

	evolution = cnn.fit(train_images, train_labels, epochs = 250, batch_size = 128, validation_data = (validation_images, validation_labels)) # train
	plotEvolution(evolution) # plot the accuracy evolution

	cnn.save("cnnModels/cnn2") # save the model

def ConfusionMatrix(validation_images, validation_labels):

	cnn = tf.keras.models.load_model("cnnModels/cnn2") # load pretrained network
	predictions = cnn.predict(validation_images) # make predictions
	predictions = [p.argmax() for p in predictions] # get the best probability for every image

	validation_labels = [p.argmax() for p in validation_labels] # from one-hot-encoding to labels

	from sklearn.metrics import confusion_matrix

	return confusion_matrix(predictions, validation_labels) # return the matrix

def launch(validation_images, validation_labels): # make submission

	cnn = tf.keras.models.load_model("cnnModels/cnn2") # load pretrained model

	test_image_paths, test_images = fetch_test_data() # get the test data
	test_images = test_images.reshape(test_images.shape[0], 32, 32, 1) # reshape it
	test_images = test_images.astype('float32')
	test_images /= 255.0 # normalize

	predictions = cnn.predict(test_images) # make predictions
	predictions = [p.argmax() for p in predictions] # get the best probability for each image

	with open("submission.txt", "w") as w: # create submission file
		w.write("id,label\n")
		for i in range(len(predictions)):
			w.write(test_image_paths[i][5 : ] + "," + str(predictions[i]) + "\n")

train_images, train_labels = fetch_train_data() # get training data
validation_images, validation_labels = fetch_validation_data() # get validation data
train_images, validation_images, train_labels, validation_labels = modifyData(train_images, train_labels, validation_images, validation_labels) # reshape and normalize the data

# print(ConfusionMatrix(validation_images, validation_labels))

# train(train_images, train_labels, validation_images, validation_labels)
# launch(validation_images, validation_labels)