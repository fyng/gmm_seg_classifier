import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from skimage import data, util
from skimage.measure import label, regionprops
from sklearn.model_selection import train_test_split

from gmm_seg_classifier import GaussianModel, GaussianMixtureModel
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


# make some utils
def read_img_to_array(fp):
    # read image file into 8-bit encoded image
    img = cv2.imread(fp)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = util.img_as_ubyte(img)
    return img

def read_mask_to_array(fp):
    # read image file into binary mask
    img = cv2.imread(fp)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = util.img_as_ubyte(img)
    _, binary_mask = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    return binary_mask > 0


# import image
filenames = os.listdir(train_path)
train_files, test_files = train_test_split(filenames, test_size=0.2)

X_cone = []
X_bg = []
for filename in train_files:
    # assemble into set of pixels X
    if filename.endswith(('.png', '.jpg')):
        img = read_img_to_array(os.path.join(train_path,filename))
        mask = read_mask_to_array(os.path.join(mask_path, filename))
        cone = img[mask]
        background = img[np.invert(mask)]

        X_cone.append(cone)
        X_bg.append(background)
X_cone = np.concatenate(X_cone)
X_bg = np.concatenate(X_bg)


cone_gaussian_model = GaussianModel()
bg_gaussian_model = GaussianModel()

cone_gaussian_model.fit(X_cone)
bg_gaussian_model.fit(X_bg)

# TODO: plot the gaussians
print(cone_gaussian_model.mean_())
print(cone_gaussian_model.variance_())
print(bg_gaussian_model.mean_())
print(bg_gaussian_model.variance_())

n_cone_pixel = X_cone.shape[0]
n_bg_pixel = X_bg.shape[0]

# p_cone = n_cone_pixel / (n_cone_pixel + n_bg_pixel)



# temp = np.array([200, 50, 50])
temp = np.array([120, 128, 150])

p_cone = cone_gaussian_model.predict_cached(temp)
p_bg = bg_gaussian_model.predict_cached(temp)


print(p_bg)
print(p_cone)



    