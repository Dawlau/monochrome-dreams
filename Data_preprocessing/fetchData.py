def fetch_train_data():

	import PIL.Image
	import numpy as np

	with open("train.txt", "r") as r:
		lines = [line.strip("\n").split(",") for line in r.readlines()] # read the names and labels of images in the training data
		imagePaths = ["train/" + line[0] for line in lines] # get the path to the images
		labels = [line[1] for line in lines] # get the labels
		images = np.asarray([ np.asarray(PIL.Image.open(path)) for path in imagePaths]) # read images as np arrays

	return images, labels


def fetch_validation_data():

	import PIL.Image
	import numpy as np

	with open("validation.txt", "r") as r:
		lines = [line.strip("\n").split(",") for line in r.readlines()] # read the names and labels of images in the validation data
		imagePaths = ["validation/" + line[0] for line in lines] # get the path to the images
		labels = [line[1] for line in lines] # get the labels
		images = np.asarray([ np.asarray(PIL.Image.open(path)) for path in imagePaths]) # read images as np arrays

	return images, labels

def fetch_test_data():

	import PIL.Image
	import numpy as np

	with open("test.txt", "r") as r:
		lines = [line.strip("\n") for line in r.readlines()] # read the names of images in the test data
		imagePaths = ["test/" + line for line in lines] # get the paths
		images = np.asarray([ np.asarray(PIL.Image.open(path)).reshape(-1) for path in imagePaths]) # read images as np arrays
	return imagePaths, images