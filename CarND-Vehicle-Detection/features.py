
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from skimage.feature import hog


# In[2]:


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'BGR2RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[3]:


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features


# In[4]:


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# In[5]:


def get_window_features(img, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
        
    ch1 = img[:,:,0]
    ch2 = img[:,:,1]
    ch3 = img[:,:,2]
        
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
            
    window = 64
    cells_per_step = 2
    cells_per_window = window//pix_per_cell
    
    # Window moves cells_per_step = 2 cells at a time
    cells_per_x = ch1.shape[1] // pix_per_cell
    cells_per_y = ch1.shape[0] // pix_per_cell

    for i in range(0, cells_per_y-cells_per_window+1, cells_per_step):
        for j in range(0, cells_per_x-cells_per_window+1, cells_per_step):
            subimg_hog1 = hog1[i, j].ravel()
            subimg_hog2 = hog2[i, j].ravel()
            subimg_hog3 = hog3[i, j].ravel()
            hog_features = np.hstack((subimg_hog1, subimg_hog2, subimg_hog3))
            
            x = j*pix_per_cell
            y = i*pix_per_cell
            subimg = img[y:y+window, x:x+window]
            
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            
            features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            yield features


# In[6]:


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    scaled_img = img
    if scale!=1:
        scaled_img = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)))
    
    cells_per_step = 2
    
    window = 64
    
    if scale==1:
        y_steps_start, y_steps_end = 0, 4
        ymin, ymax = y_steps_start*cells_per_step*pix_per_cell, y_steps_end*cells_per_step*pix_per_cell+window
        xmin, xmax = int(.2*scaled_img.shape[1]), int(.8*scaled_img.shape[1])
        scaled_img = scaled_img[ymin:ymax, xmin:xmax]
    elif scale==2:
        y_steps_start, y_steps_end = 0, 4
        ymin, ymax = y_steps_start*cells_per_step*pix_per_cell, y_steps_end*cells_per_step*pix_per_cell+window
        xmin, xmax = int(0*scaled_img.shape[1]), int(1*scaled_img.shape[1])
        scaled_img = scaled_img[ymin:ymax, xmin:xmax]
    elif scale==3:
        y_steps_start, y_steps_end = 2, 4
        ymin, ymax = y_steps_start*cells_per_step*pix_per_cell, y_steps_end*cells_per_step*pix_per_cell+window
        xmin, xmax = int(0*scaled_img.shape[1]), int(1*scaled_img.shape[1])
        scaled_img = scaled_img[ymin:ymax, xmin:xmax]
    
#     scaled_img = scaled_img[:int(4*scale*pix_per_cell*cells_per_step)]
    gen = get_window_features(scaled_img, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    
    cells_per_window = window//pix_per_cell
    
    # Window moves cells_per_step = 2 cells at a time
    cells_per_x = scaled_img.shape[1] // pix_per_cell
    cells_per_y = scaled_img.shape[0] // pix_per_cell

    for i in range(0, cells_per_y-cells_per_window+1, cells_per_step):
        for j in range(0, cells_per_x-cells_per_window+1, cells_per_step):
            f = next(gen)
            f = X_scaler.transform(f)
            label = svc.predict(f)
#             label = np.argmax(svc.predict_proba(f), axis=1)
            if label==1:    
                y1, x1 = i*pix_per_cell, j*pix_per_cell
                y2, x2 = y1+window-1, x1+window-1
                
                y1, y2 = y1+ymin, y2+ymin
                x1, x2 = x1+xmin, x2+xmin
                
                y1, x1 = int(y1*scale), int(x1*scale)
                y2, x2 = int(y2*scale), int(x2*scale)
                
                bbox = [(x1, y1), (x2, y2)]
                yield bbox

