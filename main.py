import os
import cv2
import numpy as np
from skimage import data, util
from skimage.measure import label, regionprops

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from roipoly import RoiPoly


train_path = 'data/img/train'
mask_path = 'data/mask/train'

################################################
# TRAIN
################################################
def segment_and_write_masks(train_path, mask_path):
    for filename in os.listdir(train_path):
        if filename.endswith(('.png', '.jpg')):
            img = cv2.imread(os.path.join(train_path,filename))
            img = util.img_as_ubyte(img)

            fig = plt.figure()
            plt.imshow(img, interpolation='nearest', cmap="Greys")
            plt.colorbar()
            plt.title("left click: line segment         right click or double click: close region")
            plt.show(block=False)

            # Let user draw first ROI
            roi = RoiPoly(color='r', fig=fig)
            mask = roi.get_mask(img[:,:,0])
            mask = mask.astype(np.uint8) * 255
            # save binary mask to file
            cv2.imwrite(os.path.join(mask_path, filename), mask)

# import image
filenames = os.listdir(train_path)
train_files, test_files = train_test_split(filenames, test_size=0.2)

# train model
for filename in train_files:
    # assemble into set of pixels X
            
