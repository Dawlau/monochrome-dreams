from fetchData import *

def kfold_cross_validation(x, y, k): # performs a kfold-cross-validation on the input x, output y and knn classifier with k neighbours

	from sklearn.model_selection import cross_val_score
	from sklearn.neighbors import KNeighborsClassifier
	import numpy as np

	knn = KNeighborsClassifier(n_neighbors=k) # define classifier
	cross_validation = cross_val_score(knn, x, y, cv=5) # perform validation

	print("Done with k: " + str(k))

	return k, np.mean(cross_validation), np.max(cross_validation) # return the k, average accuracy and maximum accuracy

def get_best_ks(images, labels): # find the best number of neighbours for a knn classifier

	models = [] # list of tuple of the form (k, average accuracy, maximum accuracy)
	for k in range(1, 51):
		models.append(kfold_cross_validation(images, labels, k)) # perform validation

	maxk = 0 # k with the best average accuracy
	maxavg = 0 # best average accuracy
	globalmax = 0 # best maximum score
	bestk = 0 # k with the maximum globalmax
	print("K   AVG   MAX")
	for x in models: # iterate over the list and find the values maxk, maxavg, globalmax, bestk
		k = x[0]
		avg = x[1]
		Max = x[2]

		if avg > maxavg:
			maxavg = avg
			maxk = k

		if Max > globalmax:
			globalmax = Max
			bestk = k

		print(k, round(avg, 4), Max)

	# print the final answers
	print("best on average is: k=" + str(maxk) + " with avg: " + str(maxavg))
	print("best maximum is: k=" + str(bestk) + " with max: " + str(globalmax))


def ConfusionMatrix(knn): # print confusion matrix for a pretrained knn on the validation data
	validation_images, validation_labels = fetch_validation_data() # get the validation data

	predictions = knn.predict(validation_images.reshape(-1, 1024)) # make predictions

	from sklearn.metrics import confusion_matrix

	return confusion_matrix(predictions, validation_labels) # return the matrix

def launch(knn): # make a submission
	test_image_paths, test_images = fetch_test_data() # get the test data

	predictions = knn.predict(test_images.reshape(-1, 1024)) # make predictions

	with open("submission.txt", "w") as w: # create the submission file
		w.write("id,label\n")
		for i in range(len(predictions)):
			w.write(test_image_paths[i][5 : ] + "," + predictions[i] + "\n")

train_images, train_labels = fetch_train_data() # get training data
train_images = train_images.reshape(-1, 1024) # flatten the images

# get_best_ks(train_images, train_labels)
# k = 10 is the best from 1 to 50

# train and predict

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10) # train classifier with the best k
knn.fit(train_images, train_labels)

# print(ConfusionMatrix(knn))

launch(knn) # make submission