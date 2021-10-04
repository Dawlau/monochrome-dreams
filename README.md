# Documentation for Monochrome Dreams Classification

```
Blahovici Andrei, group 241, University Bucharest
Kaggle competition: https://www.kaggle.com/c/ai-unibuc-24-22-2021
```
## I. Data fetching

I implemented a module for data fetching which implements 3 functions for fetching the
images as numpy arrays (each image is a (32, 32) shaped array) and their respective labels as a list.
Since the test data does not have labels and the output is in the format (image_name, label), the
functions that fetch the test images return the images as numpy arrays and their respective full path
in the project. (see the module fetchData.py)

## II. Choosing classifiers for testing.


### 1) K-Nearest Neighbours (KNN)

The first classifier I trained was a KNN since it is very good at finding similarities and it is
really fast to train. First of all, I flattened every image in the train data set by converting the images
from (32, 32) shaped arrays to (1024) shaped arrays. Second of all, I wrote a function that finds the
best k for the classifier. This function iterates over the range [1, 51) and performs a kfold-cross-
validation (with 5 folds) for a classifier with a given k, that is trained on the training data and prints
for every k the average and maximum accuracy. At the end the function prints the k that has the best
accuracy on average and the k with the maximum accuracy for a fold. It turns out the best k is 10.

After finding the k that performs the best on the training data I trained another classifier with
k = 10 and created the file for submission based on the predictions this classifier made. Final
accuracy on the test data is 0.47440 (based on the final score in the competition).



### 2) Polynomial and Radial basis function Support Vector Machines


I will discuss about these two classifiers together since my approach was the same for both.

First of all I fetched the training data and flattened it. Afterwards, I used the data to train the
classifiers and the result was surprising. The polynomial SVM had a final accuracy of 0.71733
whereas the radial basis function SVM had a final accuracy of 0.74320.


### 3) Convolutional Neural Network

Since convolutional neural networks are so good at finding patterns in images I decided to
give them a shot.
First of all we need to reshape and normalize the data. Since the value of each pixel is from
0 to 255 I will reshape each image to the shape (32, 32, 1) (it just add one dimension to the image to
show that it has only a color channel) and then divide the value of each pixel by 255 to avoid sparse
data. After that I One-hot-encoded the train and validation labels so the network will know that
misclassifying 8 with 2 and 3 with 4 is equally bad.

I have tried a total of 3 Convolutional neural networks.

#### a) CNN1

The first convolutional neural network I implemented was composed of a total of 5 layers. The structure can be found in the CNN1.py file. This model was trained on 30 epochs with a batch size of 128.

```
The final accuracy on the test data set is 0.84906.
```

#### b) CNN2

The second convolutional neural network that I trained was deeper than the first one and
also used Dropout for dropping unuseful neurons from every layer. The structure can be found in the CNN2.py file. This model was trained on 100 epochs with a batch size of 128.

```
The final accuracy on the test data set is 0.87920.
```

#### c) CNN3

The last convolutional neural network that I trained was much deeper than the previous ones because I thought I was not getting enough features from the previous ones. The structure can be found in the CNN3.py file. This model was trained on 250 epochs with a batch size of 128.

```
The final accuracy on the test data set is 0.88560 (which is the best one).
```