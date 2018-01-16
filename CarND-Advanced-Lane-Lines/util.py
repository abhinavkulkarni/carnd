
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[2]:


def undistort(image, mtx, dist):
    # Undistort the image
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist


# In[3]:


def get_sobel_mask(gray, ksize=5, thresh=(20, 100), orient='x'):
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel>=thresh[0]) & (scaled_sobel<=thresh[1])] = 1
    return abs_sobel, sbinary


# In[4]:


def sobel_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    abs_sobel_x, sbinary_x = get_sobel_mask(gray, thresh=(15, 255), orient='x')
    abs_sobel_y, sbinary_y = get_sobel_mask(gray, thresh=(0, 150), orient='y')
    angle_rad = np.arctan2(abs_sobel_y, abs_sobel_x)
    
    thresh_angle = np.radians((30, 90))
    sbinary_angle = np.zeros_like(sbinary_x)
    sbinary_angle[(angle_rad>=thresh_angle[0]) & (angle_rad<=thresh_angle[1])] = 1
    
    mask = np.zeros_like(sbinary_angle)
    mask[(sbinary_x==1) & (sbinary_y==1) & (sbinary_angle==1)] = 1
    return mask


# In[ ]:


def color_thresholding(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    h = hls[..., 0]
    l = hls[..., 1]
    s = hls[..., 2]
    
    mask_yellow = np.zeros_like(s)
    mask_yellow[(h<=30) & (s>=50)] = 1
    
    mask_white = np.zeros_like(l)
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    mask_white[(s<50)&(r>=150)&(g>=150)] = 1
    mask_color = cv2.bitwise_or(mask_yellow, mask_white)
    return mask_color


# In[ ]:


def region_of_interest(image, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """        
    #defining a blank mask to start with
    mask = np.zeros_like(image)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# In[ ]:


def crop_top_image(image):
    ysize, xsize = image.shape[:2]
    vertices = [[[0, ysize], [xsize, ysize], [xsize, int(.625*ysize)], [0, int(.625*ysize)]]]
    vertices = np.array(vertices, dtype=np.int32)
    
    masked_image = region_of_interest(image, vertices)
    return masked_image


# In[ ]:


def warp(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


# In[ ]:


class Lane:
    def __init__(self):
        self.right_fit = None
        self.left_fit = None
        
    def find_curvature(self, y_eval):
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        left_fit_cr = np.polyfit(self.lefty*ym_per_pix, self.leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.righty*ym_per_pix, self.rightx*xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    def find_offset(self, y_eval, xsize):
        left_fitx = self.left_fit[0]*y_eval**2 + self.left_fit[1]*y_eval + self.left_fit[2]
        right_fitx = self.right_fit[0]*y_eval**2 + self.right_fit[1]*y_eval + self.right_fit[2]
        midpoint_fitx = (left_fitx+right_fitx)/2
        midpoint_image = xsize/2
        xm_per_pix = 3.7/700
        self.offset = (midpoint_image-midpoint_fitx)*xm_per_pix
                
    def find_lines(self, mask_warped):
        if self.right_fit is None and self.left_fit is None:
            self.init_lane_lines(mask_warped)
            self.find_curvature(mask_warped.shape[0])
            self.find_offset(*mask_warped.shape)

        ploty = np.linspace(0, mask_warped.shape[0]-1, mask_warped.shape[0])
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
                    
        margin = 100
        y_ = ploty.astype('int')
        x_ = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        x_ = x_.astype('int')
        x_ = np.maximum(x_, 0)
        bitmask = np.zeros_like(mask_warped)
        x_low = np.maximum(x_-margin, 0)
        x_high = np.minimum(x_+margin, mask_warped.shape[1])
        for i in range(len(y_)):
            bitmask[y_[i], x_low[i]:x_high[i]] = 1
            
        mask = cv2.bitwise_and(mask_warped, bitmask)
        left_lane_inds = mask.nonzero()
        nonzero_rows_left = len(np.sum(mask, axis=1).nonzero()[0])
        
        x_ = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        x_ = x_.astype('int')
        x_ = np.maximum(x_, 0)
        bitmask = np.zeros_like(mask_warped)
        x_low = np.maximum(x_-margin, 0)
        x_high = np.minimum(x_+margin, mask_warped.shape[1])
        for i in range(len(y_)):
            bitmask[y_[i], x_low[i]:x_high[i]] = 1
        mask = cv2.bitwise_and(mask_warped, bitmask)
        right_lane_inds = mask.nonzero()
        nonzero_rows_right = len(np.sum(mask, axis=1).nonzero()[0])
                               
        # Extract left and right line pixel positions
        # Fit a second order polynomial to each
        minpix = 50
        if len(left_lane_inds[0])>=minpix*9 and nonzero_rows_left>mask_warped.shape[0]//3:
            self.leftx = left_lane_inds[1]
            self.lefty = left_lane_inds[0]
            self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        if len(right_lane_inds[0])>=minpix*9 and nonzero_rows_right>mask_warped.shape[0]//3:
            self.rightx = right_lane_inds[1]
            self.righty = right_lane_inds[0] 
            self.right_fit = np.polyfit(self.righty, self.rightx, 2)
        
        self.find_curvature(mask_warped.shape[0])
        self.find_offset(*mask_warped.shape)
        return
            
    def init_lane_lines(self, mask_warped):    
        # Take a histogram of the bottom half of the image
        histogram = np.sum(mask_warped[mask_warped.shape[0]//2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9

        # Set height of windows
        window_height = np.int(mask_warped.shape[0]/nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = mask_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = mask_warped.shape[0] - (window+1)*window_height
            win_y_high = mask_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy>=win_y_low) & (nonzeroy<win_y_high) & 
                              (nonzerox>=win_xleft_low) & (nonzerox<win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy>=win_y_low) & (nonzeroy<win_y_high) & 
                               (nonzerox>=win_xright_low) & (nonzerox<win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds] 
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)
        
        return

