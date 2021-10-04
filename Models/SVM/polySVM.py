from fetchData import *

def train():

	from sklearn.model_selection import train_test_split
	from sklearn import svm
	import pickle
	import numpy as np

	train_images, train_labels = fetch_train_data() # get the training data


	train_images = np.reshape(train_images, (-1, 1024)) # flatten it

	polySVM = svm.SVC(kernel="poly")
	polySVM.fit(train_images, train_labels) # train the classifier

	pickle.dump(polySVM, open("svmModels/poly.sav", "wb")) # save the model

	print("Training is done")

def ConfusionMatrix(): # print confusion matrix for a pretrained polynomial SVM on the validation data
	validation_images, validation_labels = fetch_validation_data() # get the validation data

	import pickle

	polySVM = pickle.load(open("svmModels/poly.sav", "rb")) # load the trained model

	predictions = polySVM.predict(validation_images.reshape(-1, 1024)) # make predictions

	from sklearn.metrics import confusion_matrix

	return confusion_matrix(predictions, validation_labels) # return the matrix


def launch(): # make submission

	import pickle
	import numpy as np
	import PIL.Image

	image_paths, images = fetch_test_data() # get test data

	images = np.reshape(images, (-1, 1024)) # flatten it


	polySVM = pickle.load(open("svmModels/poly.sav", "rb")) # load the trained model
	predicitions = polySVM.predict(images) # make predictions

	with open("submissions/polySVM.txt", "w") as w: # create submission file
		w.write("id,label\n")
		for i in range(len(predicitions)):
			w.write(image_paths[i][5 : ] + "," + predicitions[i] + "\n")

train()
launch()
# print(ConfusionMatrix())

# Accuracy: 0.71920
