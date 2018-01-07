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
