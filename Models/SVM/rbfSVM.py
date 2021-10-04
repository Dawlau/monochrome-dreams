from fetchData import *

def train():

	from sklearn.model_selection import train_test_split
	from sklearn import svm
	import pickle
	import numpy as np

	train_images, train_labels = fetch_train_data() # get training data

	train_images = np.reshape(train_images, (-1, 1024)) # flatten it


	rbfSVM = svm.SVC(kernel="rbf")
	rbfSVM.fit(train_images, train_labels) # train model

	pickle.dump(rbfSVM, open("svmModels/rbf.sav", "wb")) # save it

	print("Training is done")

def ConfusionMatrix(): # print confusion matrix for a pretrained knn on the validation data
	validation_images, validation_labels = fetch_validation_data() # get the validation data

	import pickle

	rbfSVM = pickle.load(open("svmModels/rbf.sav", "rb")) # load the pretrained model

	predictions = rbfSVM.predict(validation_images.reshape(-1, 1024)) # make predictions

	from sklearn.metrics import confusion_matrix

	return confusion_matrix(predictions, validation_labels) # return the matrix

def launch(): # make submission

	import pickle
	import numpy as np
	import PIL.Image

	image_paths, images = fetch_test_data() # get the test data

	images = np.reshape(images, (-1, 1024)) # flatten it

	rbfSVM = pickle.load(open("svmModels/rbf.sav", "rb")) # load the pretrained model
	predicitions = rbfSVM.predict(images) # make predictions

	with open("submissions/rbfSVM.txt", "w") as w: # create submission file
		w.write("id,label\n")
		for i in range(len(predicitions)):
			w.write(image_paths[i][5 : ] + "," + predicitions[i] + "\n")

# print(ConfusionMatrix())
train()
launch()

# Accuracy 0.75600
