__author__ = 'zhengxiaoyu'

import cv2
from skimage.morphology import skeletonize
from skimage import draw
import numpy as np
from scipy.ndimage import gaussian_filter,median_filter
img_rgb = cv2.imread("your_image")
row = img_rgb.shape[0]
col = img_rgb.shape[1]
image =  cv2.split(img_rgb)[1]

from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage import exposure

image = gaussian_filter(image, 0.1)

seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image
dilated = reconstruction(seed, mask, method='dilation')
cv2.imwrite('1.jpg',image)
cv2.imwrite('2.jpg',dilated)
new_img = image - dilated
from skimage.util import img_as_ubyte
cv2.imwrite('3.jpg',new_img)
img = cv2.imread('3.jpg',0)
clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(3,3))
cl1 = clahe.apply(img)
cv2.imwrite('4.jpg',cl1)
kernel = np.ones((3,3),np.uint8)
closing= cv2.morphologyEx(cl1, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('5.jpg', closing)
closing= cl1*1.3 - closing
cv2.imwrite('6.jpg', closing)
closing = gaussian_filter(closing,0.01)
cv2.imwrite('7.jpg',closing)
for x in range(row):
    for y in range(col):
        if closing[x,y]<20:
            closing[x,y] = 255
        else:
            closing[x,y] = 0
cv2.imwrite('8.jpg',closing)
kernel = np.ones((2,2),np.uint8)
closing= cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel)

cv2.imwrite('9.jpg',closing)
closing = gaussian_filter(closing,0.1)
cv2.imwrite('result.jpg',closing)