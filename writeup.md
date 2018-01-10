# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[random5]: ./writeup/random_5.png "Random 5"
[counts_by_category]: ./writeup/counts_by_category.png "Counts by category"
[selfdata1]: ./self-data/1.jpg "Traffic Sign 1"
[selfdata2]: ./self-data/2.jpg "Traffic Sign 2"
[selfdata3]: ./self-data/3.jpg "Traffic Sign 3"
[selfdata4]: ./self-data/4.jpg "Traffic Sign 4"
[selfdata5]: ./self-data/5.jpg "Traffic Sign 5"

## Rubric Points
(Reference: https://review.udacity.com/#!/rubrics/481/view)
### Dataset Exploration

Please refer to the [output file](./Traffic_Sign_Classifier.html) for a summary and visualizations of the dataset.

### Design and Test a Model Architecture

#### Preprocessing

* Training, validation and test sets are normalized with the mean value of the training set
* Training sets and labels are shuffled

### Model Architecture

The model uses a modifled version of LeNet with slightly more neurons in some of the layers to accomodate increase of output dimension from 10 to 43. 

In addition, dropout is applied to fully connected hidden layers.

### Model Training

The model is trained with a learning rate of 0.001 and universal dropout rate of 0.35. Training is done with 30 epochs with batch size 128.

### Solution Approach

During the parameter tuning process training accuracies remained high (>0.98) while validation accuracies is relatively low, which suggests overfitting to training data. Thus, I focused most of my effort on closing the gap between training and validation error. 

After applying dropout regularization, training input normalization, I managed to get validaiton set accuracy to 0.989 and test set accuracy to 0.953.

## Test a Model on New Images

### Acquiring New Images

I acquired 5 german traffic sign images on Google:

![selfdata1] 
![selfdata2] 
![selfdata3] 
![selfdata4]
![selfdata5]

### Performance on New Images

The trained model correctly predicts 4 out of the 5 images. The misclassified image is the pedastrian crossing image.

### Model Certainty - Softmax Probabilities

I use `tf.nn.softmax()` to calculate probabilities from output logits on five of my new images.

Results for all images exhibit very high probability for the top pick (>0.9) while probabilities for 2nd, 3rd and 4th categories are very small.


### Visualize the Neural Network's State with Test Images

I use a modified version of the given `outputFeatureMap()` to display feature maps for both of my two convolutional layers. 

It appears that the first conv layer detects features such as simple vertical, horizontal and diagonal edges and regions, whereas the second conv layer highlights specific regions.



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

* [Writeup](https://github.com/settinghead/CarND-Traffic-Sign-Classifier-P2/blob/master/writeup.md)
* [Project code](https://github.com/settinghead/CarND-Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34,799`
* The size of the validation set is `4,410`
* The size of test set is `12,630`
* The shape of a traffic sign image is `32 * 32 * 3`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

Here are 5 randomly sampled images from the training data set.

![random5]

Here is a barchart plot for number of examples by each category.

![counts_by_category]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized training, validation and test sets with the mean value of the training set. Normalization makes the search space less skewed toward particular dimension(s), thus making it easier for the model to train.

Training sets and labels are also shuffled together. Shuffling ensures that each minibatch receives a broad representation of the dataset, which allows the model to generalize better.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is a slightly expanded version of LeNet.

I picked LeNet because it is generally capable of classifying images of similar sizes.

I increased the capacity from the original LeNet, because the original network was constructed to work on the 10-class MNIST dataset, whereas the trasffic sign dataset has 43 classes and contains more complex shapes than digits. Therefore, more filters are required for capturing features.

I also applied dropout to the first two fully connected layers because it is an effective regularization technique.



| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x48        	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48 				|
| Flatten		|  outputs 1200         									|
| Fully connected		| droput = 0.45, outputs:    320     									|
| RELU					|												|
| Fully connected		| droput = 0.45, outputs:    240     									|
| RELU					|												|
| Fully connected		| droput = 0.45, outputs:    43     									|
| Softmax				|         									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model is trained with a learning rate of 0.001, 15 epochs and a batch size 128.

`0.001` is a typical range for training such kind of networks.

Batch size `128` is chosen because bigger batch size would not fit the GPU memory and might also reduce the model's learning ability whereas smaller batch sizes would taker longer to run.

Number of epoches is chosen such that the network stops training when validation error no longer decreases, which prevents overfitting. 

In addition, a dropout rate of 0.35 is chosen because we want a small enough number for regularization to be effective but not too small that it impedes learning. 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of `0.994`
* validation set accuracy of `0.990` 
* test set accuracy of `0.962`

I started off with LeNet because it achives high performance on MNIST datasets, which is also a low-resolution dataset that has a fixed number of classes. In comparison, the traffic sign dataset appears to be an upgraded version which requires the network to recognize slightly more complex features. Therefore, LeNet seems like a good starting point.

On the first try, Training accuracy remained near 0.95 and the validation accuracy stayed around 0.93. There was a small gap between training and validation accuracies, which suggests overfitting.

Next, I tried to vary number epochs (range 10-100), learning rate (0.0001-0.0002) and batch size (64-256). However validation accuracy failed to achieve any higher number.

I then applied dropout of rate 0.35 and also doubled filter size in conv layers and the number of nodes in fully-connected layers. The result is training and validation accuracy both improves and are at similar level (0.994 and 0.990 respectively).

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

I acquired 5 german traffic sign images on Google:

![selfdata1] 
![selfdata2] 
![selfdata3] 
![selfdata4]
![selfdata5]

The first four images are difficult to classfy because of background noise. The fifth has a slighly skewed angle. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The trained model correctly predicts 4 out of the 5 images. The misclassified image is the pedastrian crossing image.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		| Yield   									| 
| Left turn     			| Left turn 										|
| Stop Sign					| Stop Sign											|
| 50 km/h	      		| 50 km/h					 				|
| Pedestrian Crossing			| Keep Left      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The resulting accuracy is lower than that on the testing set, however it is inconclusive because of very small size of dataset (5).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook.

I use `tf.nn.softmax()` to calculate probabilities from output logits on five of my new images.

Results for all images exhibit very high probability for the top pick (near 1.0) while probabilities for 2nd, 3rd and 4th categories are very small.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield   									| 
| 1.0     				| Left Turn										|
| 1.0					| Stop Sign											|
| 1.0	      			| 50 km/h					 				|
| 1.0				    | Keep Left      							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I use a modified version of the given `outputFeatureMap()` to display feature maps for both of my two convolutional layers. 

It appears that the first conv layer detects features such as simple vertical, horizontal and diagonal edges and regions, whereas the second conv layer highlights specific regions.

Generated feature map images can be found in the notebook as well as in the HTML report.