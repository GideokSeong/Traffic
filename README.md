# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./Output/1.PNG "Visualization"
[image2]: ./Output/2.PNG "New Images"
[image3]: ./Output/3.PNG "Classification for new images"
[image4]: ./Output/4.PNG "Accuracy for new images"
[image5]: ./Output/5.PNG "Top five classification for new images"
[image6]: ./Output/samples.png "All labels"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?
![alt text][image1]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![alt text][image6]

### Design and Test a Model Architecture

#### 1. Pre-process images

###### Shuffle arrays or sparse matrices in a consistent way
X_train, y_train = shuffle(X_train, y_train)

###### Split arrays or matrices into random train and test subsets
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

###### Normalizes the data between 0.1 and 0.9 instead of 0 to 255    
def normalize(data):
    return data / 255 * 0.8 + 0.1
    
#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| Input 400, output 120      					|
| RELU					|												|
| Fully connected		| Input 120, output 84      					|
| RELU					|												|
| Fully connected		| Input 84, output 43       					|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an 
EPOCHS = 20
BATCH_SIZE = 128
Leaning rate = 0.001.
As an optimizer I used tf.train.AdamOptimizer(learning_rate = rate) which is pre-defined function, to use this optimizer 
First, I found cross_entropy using tf.nn.softmax_cross_entropy_with_logits function, then followed by loss_operation using tf.reduce_mean(cross_entropy).

My final model result was:

* validation set accuracy of 0.973 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
 ![alt text][image2] 
 
 #### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

 ![alt text][image3] 
 
 The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. To classify about new images, it seems not simply because unlike the data given this project, new images found on the Internet include other factors such as cloud and other objects, so it seems to make classification harder than tests I did above.
 
 ![alt text][image4] 
 #### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 
Only three images are correctly classified which means it does not look accurate perfectly. When you see the results, name called NewImages/5.jpg and NewImages/2.jpg are failed to be classified but other images are correctly classified. Images which is correctly classified showed 100 % classification about their image. To summarize, 3 out of 5 images are correctly classified, so final accuracy is 60 %.

 ![alt text][image5]
 
 ### Conclusion
 
Through this project I became familiar using TensorFlow and felt far easier to implement convolutional neural network. 
Also, I got to know what factors can have an affect on the accuracy such as epochs, batch size, learning rate and so on.
Even there are a lot of attributes which can affect the accuracy such as arranging model architecture.
Finally, using the model I implemented and pre-defined functions such as tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation) could do classification among many traffic signs.
