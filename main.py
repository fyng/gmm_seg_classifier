import cv2, os
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import util
from roipoly import RoiPoly
from gmm_seg_classifier import GaussianModel, GaussianMixtureModel
# suppress depreciation warning from numpy 1.25.0
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def read_img_to_array(fp):
    # read image file into 8-bit encoded image
    img = cv2.imread(fp)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = util.img_as_ubyte(img)
    return img

# image processing
def classify_image_pixels(image, p_cone):
    W, H, _ = image.shape
    mask = np.zeros((W,H))
    for x in range(W):
        for y in range(H):
            pixel = image[x,y,:]
            p_x_cone = cone_gaussian_model.predict(pixel)
            p_x_bg = bg_gaussian_model.predict(pixel)
            p_x = ((p_x_cone * p_cone) + p_x_bg * (1 - p_cone))
            p_cone_x = (p_x_cone * p_cone) / p_x
            mask[x,y] = p_cone_x[0]
    return mask 


def draw_triangle_around_positive_pixels(image, thresh, prior):
    mask = classify_image_pixels(image, prior)
    #dilation and erosion to clean up
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = mask
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
    binary_mask = mask_cleaned > thresh

    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    triangles = []
    for cnt in contours:
        # approximate the contour to reduce the number of points
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)

        x, y, w, h = cv2.boundingRect(cnt)
        t = np.array([
            [x + w // 2, y], # top
            [x, y + h], # right
            [x + w, y + h] # left
        ])

        # geometry constraint
        w_h_ratio = 7.5 / 17
        if math.isclose((w / h), w_h_ratio, rel_tol=0.28):
            triangles.append(t)
            cv2.drawContours(image, [t], 0, (0, 255, 0), 2)  # Green, tickness 2

    return image, triangles

# geometry processing
def triangle_geometry(triangle):
    centroid = np.mean(triangle, axis=0).squeeze().astype(int)
    x = centroid[0]
    y = centroid[1]

    f = 50 # hand-calculated focal length
    h = np.abs(triangle[0][1] - triangle[1][1])
    w = np.abs(triangle[1][0] - triangle[2][0])
    # average of distance estimates from height and width
    d_1 = f * 17 / h
    d_2 = f * 7.5 / w
    d = (d_1 + d_2) / 2

    return x, y, d


if __name__ == "__main__":
    # FIXME: change to appropriate folder
    folder = "data/test/"

    cone_gaussian_model = GaussianModel()
    bg_gaussian_model = GaussianModel()

    cone_gaussian_model.load('pretrained_model/cone_gaussian_model.json')
    bg_gaussian_model.load('pretrained_model/bg_gaussian_model.json')

    n_cone_pixel = 63866
    n_bg_pixel = 9536134
    prior = n_cone_pixel / (n_cone_pixel + n_bg_pixel)
    thresh = 0.99

    for filename in os.listdir(folder):
        print(filename) 
        img = read_img_to_array(os.path.join(folder,filename))
        img_overlay, triangles = draw_triangle_around_positive_pixels(img, thresh, prior)
        
        x = None
        y = None
        d = 0
        for t in triangles:
            x, y, d = triangle_geometry(t)
            # draw blue dot at centroid
            cv2.circle(img_overlay, (x,y), radius=4, color=(0, 0, 255), thickness=-1)
            print(f'({x},{y}), distance = {d:.3f}')

        img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
        cv2.imshow(f'{filename}: distance = {d:.3f}' ,img_overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()