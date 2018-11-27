
# Writeup
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Behavioral Cloning Project

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Result
### Video Implementation  

<center>

| [![Output video](https://img.youtube.com/vi/PSUVOkrUgmE/0.jpg)](https://www.youtube.com/watch?v=PSUVOkrUgmE "Output video") |
|:--:|
| *Project Video* |

</center>

The dashcam view can be found [here](https://www.youtube.com/watch?v=UcZlseZxFuQ "Output video").

As can be seen, the car sways from side to side and the ride isn't quite smooth. There is no logic of jerk-minimization or any finer grained control embedded in the model.

### Model Architecture and Training Strategy

#### 1. Model architecture

1. The model is losely based on a combination of Nvidia architecture and LenNet
2. Nvidia architecture uses strided convolutions to reduce the spatial size of the feture maps whereas I used the pooling layers as in LeNet
3. The size of the fully connected layers in the model is also not as bigger as in the Nvidia architecture as I had limited GPU memory and partly because simulator images have lot less variability than the real world street images that Nvidia model was desinged to handle.
4. I used early stopping (based on validation loss) to select the best model.

#### 2. Attempts to reduce overfitting

The model contains dropout layers in order to reduce overfitting (`model.py` lines 88-108). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 125-128). The model training stops when validation loss plateaus. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 108).

#### 4. Appropriate training data

Training data was collected as follows:

1. 1 lap of careful driving in clockwise direction
2. 1 lap of careful driving in anti-clockwise direction
3. Few instances of recovery from veering off the lane center

The steering angle correction factor of 0.2 was obtained after several iterations of trail and error and observing the side-to-side wobbling of the car in autonomous mode.
For details about how I created the training data, see the next section. 

### Files Submitted & Code Quality

My project includes the following files:

* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 

Using the Udacity provided simulator and the `drive.py` file, the car can be driven autonomously around the track by executing 
```python drive.py model.h5```

[//]: # (Image References)

[image1]: ./examples/center_2018_01_14_10_06_27_928.jpg "Center Lane Driving"
[image2]: ./examples/center_2018_01_14_13_44_19_230.jpg "Recovery 1"
[image3]: ./examples/center_2018_01_14_13_44_16_873.jpg "Recovery 2"
[image4]: ./examples/center_2018_01_14_13_44_17_800.jpg "Recovery 3"
[image5]: ./examples/center_2018_01_14_13_44_18_031.jpg "Recovery 3"
[image6]: ./examples/center_2018_01_14_13_37_36_312_normal.jpg "Normal Image"
[image7]: ./examples/center_2018_01_14_13_37_36_312.jpg "Flipped Image"
[image8]: ./examples/placeholder_small.png "Flipped Image"



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I pretty much followed the progression the coursework. 

1. First I trained a random model (it takes a random action), to make sure that I could run the script end to end.
2. Then I trained a linear regression model with a single layer.
3. I then added convolutional layers (much in the fashion of LeNet). Progressively, model started to do better.
4. I then added input normalization Lambda layer, tanh activation in the last layer to restrict the output to (-1, 1) range, dropout to reduce overfitting. The goal was to get training and validation errors close to each other (training error will almost always be slightly lower than the validation error, however a big difference in the two indicates overfitting). I played around with the model architecture and observed the error curves and tweaked hyperparameters such as number of hidden layers, dropout rate, number of output feature maps of convolutional layers and settled on final model architecture.

Of course, data augmentation (flipping and using data from the left and right cameras) also helps generalizes the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 88-108) consisted of a convolution neural network with the following layers and layer sizes:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 63, 318, 32)       896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 31, 159, 32)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 29, 157, 64)       18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 78, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 76, 128)       73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 38, 128)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 36, 256)        295168    
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 2, 18, 256)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 9216)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               921700    
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_4 (Dropout)          (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 1,315,687
Trainable params: 1,315,687
Non-trainable params: 0
```

#### 3. Creation of the Training Set & Training Process

Training data was collected as follows:

1. 1 lap of careful driving in clockwise direction
2. 1 lap of careful driving in anti-clockwise direction
3. Few instances of recovery from veering off the lane center

The steering angle correction factor of 0.15 was obtained after several iterations of trail and error and observing the side-to-side wobbling of the car in autonomous mode.

Here is an example image of center lane driving:

![Center Lane Driving][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to the center lane should it veer off it. These images show what a recovery looks like starting from the first image:

![Recovery 1][image2]
![Recovery 2][image3]
![Reconver 3][image4]
![Reconver 4][image5]

To augment the data sat, I also flipped images and angles thinking that this would lead to better generalization and the model will be able to do well on unforeseen scenarios. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 24115 number of data points. I then preprocessed this data by normalizing the input and cropping top 75 and bottom 50 rows of the image so as to only include relevant parts in the resultant image.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was determined by the early stopping criterion. I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Discussion
1. The car sways from side to side. I used mouse (& keyboard) to control the car (did not an access to a joystick), as a result the car swayed from side to side while collecting data in a training data lap.
2. The model is too simplistic to be of any use in a real world scenario. End-to-end Deep Learning based control is a pipe-dream at this point and Deel Learning is mostly used in perception part of the pipeline.
