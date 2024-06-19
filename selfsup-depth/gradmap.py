import cv2
import sys
import time

import torch
import numpy as np

def get_gradient_map(image, magthr=32, ksize=3):

    if type(image) == torch.Tensor:
        if len(image.shape)==3 and image.shape[0]==1:
            image = image[0]
        image = (image * 255).byte().numpy()

    # Compute the gradients using Sobel operator
    if len(image.shape) == 3:
        grad_x = np.max(np.stack([
            cv2.Sobel(image[:, :, 0], cv2.CV_64F, 1, 0, ksize=ksize),
            cv2.Sobel(image[:, :, 1], cv2.CV_64F, 1, 0, ksize=ksize),
            cv2.Sobel(image[:, :, 2], cv2.CV_64F, 1, 0, ksize=ksize)
        ]), 0)
        grad_y = np.max(np.stack([
            cv2.Sobel(image[:, :, 0], cv2.CV_64F, 0, 1, ksize=ksize),
            cv2.Sobel(image[:, :, 1], cv2.CV_64F, 0, 1, ksize=ksize),
            cv2.Sobel(image[:, :, 2], cv2.CV_64F, 0, 1, ksize=ksize)
        ]), 0)
    else:
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    # Compute the gradient magnitude and direction (angle)
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    # Normalize the magnitude to range [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = magnitude.astype(np.uint8)

    # Histogram equalization (adaptive)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    magnitude = clahe.apply(magnitude)

    # Kill weak pixels
    magnitude[ magnitude < magthr ] = 0

    # Create an HSV image where hue represents the direction
    hsv = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle / 2  # OpenCV uses H: [0, 180]
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = magnitude  # Gradient magnitude

    # Convert HSV to RGB
    rgb_result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    ### Show
    ##cv2.imshow("gradmap", rgb_result)
    ##cv2.waitKey(0)

    # We're done
    return rgb_result

def get_edge_enhancer(l, r, max_disp=255, max_amp=64):
    # l and r are HxW np.uint8 arrays representing grayscale input images
    # get_gradient_map computes an np.uint8 array of shape HxWx3 that represents a color image where edges are very clearly seen in color
    gml = get_gradient_map(l)
    gmr = get_gradient_map(r)

    COL_THR = 64
    end_idx = gml.shape[1] - 1
    F0 = torch.from_numpy(gml).permute(2, 0, 1).float() # values from 0 to 255
    F1 = torch.from_numpy(gmr).permute(2, 0, 1).float() # values from 0 to 255
    D = 3*255 + torch.zeros((gml.shape[0], gml.shape[1], max_disp), dtype=torch.float32)
    for i in range(0, max_disp):
        #
        f0 = F0[:, :, i:end_idx]
        f1 = F1[:, :, 0:end_idx-i]
        #
        v = (torch.max(f0, 0)[0] > COL_THR) & (torch.max(f1, 0)[0] > COL_THR)
        #
        D[:, i:end_idx, i][v] = torch.max(torch.abs(f0 - f1), 0)[0][v]

    scores = torch.ones((gml.shape[0], gml.shape[1], max_disp), dtype=torch.float32)
    scores[D < COL_THR] = max_amp

    return scores.numpy()


#
#
#

DEMO_STATE = {}

def coldist(a, b):
    s = np.max(np.abs(a.astype(np.int16)-b))
    return s

def click_ev(ev, x, y, flags, params):
    if ev != cv2.EVENT_LBUTTONDOWN:
        return

    toshow = DEMO_STATE["rm"].copy()

    print(np.max(DEMO_STATE["R"][y, x, :]))
    for disparity in range(0, DEMO_STATE["R"].shape[2]):
        if DEMO_STATE["R"][y, x, disparity] > 1:
            X = x - disparity
            cv2.circle(toshow, (X, y), 3, (0, 0, 255), -1)

    cv2.imshow("rm - colorsim", toshow)

def demo():
    mode = cv2.IMREAD_GRAYSCALE
    limg = cv2.imread(sys.argv[1], mode)
    rimg = cv2.imread(sys.argv[2].replace("image_2", "image_3"), mode)

    t = time.time()
    R = get_edge_enhancer(limg, rimg)
    print("* get_edge_enhancer time: ", time.time() - t)

    lm = get_gradient_map(limg)
    rm = get_gradient_map(rimg)

    DEMO_STATE["lm"] = lm
    DEMO_STATE["rm"] = rm
    DEMO_STATE["R"] = R

    cv2.imshow("lm", lm)
    cv2.setMouseCallback("lm", click_ev)
    cv2.waitKey(0)

if __name__ == "__main__":
    demo()