
# coding: utf-8

# In[1]:


import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from features import *


# In[2]:


with open('params/svc_pickle.p', 'rb') as f:
    params = pickle.loads(f.read())

cell_per_block = params['cell_per_block']
hist_bins = params['hist_bins']
orient = params['orient']
pix_per_cell = params['pix_per_cell']
X_scaler = params['scaler']
spatial_size = params['spatial_size']
svc = params['svc']


# In[3]:


for img_name in ['./examples/car.png', './examples/notcar.png']:
    img = mpimg.imread(img_name)
    original_img = np.copy(img)
    fig, axes = plt.subplots(ncols=4, nrows=2)
    fig.set_size_inches(10, 6)
    axes[0][0].imshow(original_img)
    axes[0][0].set(title='Original image', xticks=[], yticks=[])
    img = convert_color(img, conv='RGB2YCrCb')
    for i in range(3):
        axes[0][i+1].imshow(img[:, :, i], cmap='Greys_r')
        axes[0][i+1].set(title=['Y', 'Cr', 'Cb'][i] + ' channel', xticks=[], yticks=[])
    
    axes[1][0].imshow(original_img)
    axes[1][0].set(title='Original image', xticks=[], yticks=[])
    for i in range(3):
        _, hog_image = get_hog_features(img[:, :, i], orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=True)
        axes[1][i+1].imshow(hog_image, cmap='Greys_r')
        axes[1][i+1].set(title=['Y', 'Cr', 'Cb'][i] + ' channel HOG', xticks=[], yticks=[])
    plt.show()
    name = img_name.split('.png')[0] + '-features.png'
    fig.savefig(name)


# In[4]:


img = mpimg.imread('test_images/test2.jpg')
ystart, ystop = int(img.shape[0]*.5), int(img.shape[0]*.95)
img_cropped = img[ystart:ystop, :, :]
img_cropped = convert_color(img_cropped, conv='RGB2YCrCb')
img_cropped = img_cropped.astype(np.float32)/255
fig, axes = plt.subplots(ncols=2, nrows=2)
fig.set_size_inches(14, 8)
axes[0][0].imshow(img)
axes[0][0].set(title='Original image', xticks=[], yticks=[])
for scale in [1, 2, 3]:
    draw_img = np.copy(img)
    gen = find_cars(img_cropped, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    for bbox in gen:
        bbox[0] = (bbox[0][0], ystart + bbox[0][1])
        bbox[1] = (bbox[1][0], ystart + bbox[1][1])
        cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 255, 0), 3)
    i, j = (0, 1) if scale==1 else (1, 0) if scale==2 else (1, 1)
    axes[i][j].imshow(draw_img)
    axes[i][j].set(title='Scale '+str(scale), xticks=[], yticks=[])
plt.show()
# fig.savefig('./examples/scales.png')


# In[ ]:




