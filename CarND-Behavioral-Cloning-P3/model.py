
# coding: utf-8

# In[1]:


import csv
import cv2
import numpy as np
import sys


# In[2]:


data_dir = sys.argv[1]


# In[3]:


lines = []
reader = csv.reader(open(data_dir + '/driving_log.csv'))
for line in reader:
    lines.append(line)


# ## Training Data
# 
# In the following code, snippet, I create the training set.
# The training data was collected as follows:
# 
# 1. 1 lap of careful driving in clockwise direction
# 2. 1 lap of careful driving in anti-clockwise direction
# 3. Few instances of recovery from veering off the lane center
# 
# The steering angle correction factor of 0.15 was obtained after several iterations of trail and error and observing the side-to-side wobbling of the car in autonomous mode.
# 
# ## Data Augmentation
# 1. For every image, the flipped image is also added to the training set
# 2. Images from driving anti-clockwise direction were also added to the training set
# 
# ## Data Preprocessing
# 1. I did not feel the need to grayscale the images as there is useful color information in the images that I would like the model to train on
# 2. I cropped the top 70 pixes (skyline and trees) and bottom 25 pixels (hood of the car) to reduce noise in the training data

# In[4]:


images = []
measurements = []
for line in lines:
    for i in range(3):
        path = line[i]
        fname = path.split('/')[-1]
        image = cv2.imread(data_dir + '/IMG/' + fname)
        images.append(image)
        measurement = float(line[3])
        if i==1:
            measurement += .2
        elif i==2:
            measurement -= .2
        measurements.append(measurement)
        images.append(cv2.flip(image, 1))
        measurements.append(measurement*-1)


# In[5]:


X_train = np.array(images)
y_train = np.array(measurements)


# ## Model
# 1. The model is losely based on a combination of Nvidia architecture and LenNet
# 2. Nvidia architecture uses strided convolutions to reduce the spatial size of the feture maps whereas I used the pooling layers as in LeNet
# 3. The size of the fully connected layers in my model is also not as bigger as in the Nvidia architecture as I was dealing with limited GPU memory and partly because simulator images have lot less variability than the real world street images that Nvidia model was desinged to handle.
# 4. I use early stopping (based on validation loss) to select the best model.

# In[6]:


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPool2D, Cropping2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dropout(.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse', optimizer='adam')
    return model


# In[7]:


import keras.backend as K

K.set_learning_phase(True)
model = build_model()
model.summary()


# In[8]:


early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
checkpoint_callback = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(X_train, y_train, validation_split=.2, shuffle=True, epochs=40, callbacks=[early_stopping_callback, checkpoint_callback])


# In[9]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# In[10]:


weights = model.get_weights()
for i in range(len(weights)):
    print(weights[i].shape)

