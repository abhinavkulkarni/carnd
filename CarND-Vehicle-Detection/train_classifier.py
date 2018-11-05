
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from features import *
import time


# In[2]:


# Read the data
data_dir = '../vehicle-data/'

cars = glob.glob(data_dir + 'vehicles/*/*g')
notcars = glob.glob(data_dir + 'non-vehicles/*/*g')


# In[3]:


orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32


# In[4]:


X = []
y = []
class_weight = {0:0, 1:1}
for dataset in [cars, notcars]:
    for img_name in dataset:
        img = cv2.imread(img_name)
        img = convert_color(img, conv='BGR2YCrCb')
        img = img.astype(np.float32)/255
        features = get_window_features(img, orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins)

        f = next(features)
        X.append(f)
        if dataset==cars:
            label = 1
        else:
            label = 0
        y.append(label)
        class_weight[1-label] += 1


# In[5]:


X = np.vstack(X)
y = np.array(y)


# In[6]:


# Random shuffle data
ind = list(range(len(y)))
from random import shuffle
shuffle(ind)
X = X[ind]
y = y[ind]


# In[7]:


X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)


# In[8]:


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)


# In[9]:


parameters = {'C': [.1, .5, 1, 5, 10]}

svc = LinearSVC()
svc.class_weight = class_weight
svc = GridSearchCV(svc, parameters, cv=3)
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


# In[10]:


print(svc)


# In[11]:


# Save classifier and scaler
import pickle

params = {}
params['svc'] = svc
params['scaler'] = X_scaler
params['orient'] = orient
params['pix_per_cell'] = pix_per_cell
params['cell_per_block'] = cell_per_block
params['spatial_size'] = spatial_size
params['hist_bins'] = hist_bins

with open("params/svc_pickle.p", "wb" ) as f:
    f.write(pickle.dumps(params))


# In[ ]:




