# Adaptive Binarization Technique included described by 
# Basilios Gatos, Ioannis Pratikakis, and Stavros J. Perantonis
import os
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import gaussian, convolve2d
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_float
import math

def process_image(filename):
    original_image = cv2.imread(filename)

    morph = original_image.copy()

    morph = cv2.bilateralFilter(morph, 2, 100, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

    image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

    channel_height, channel_width, _ = image_channels[0].shape

    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

    image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)


    im = imageSkewNormalization(filename, image_channels)

    cv2.imwrite("cropped.png", im)

    new_image = cv2.imread("cropped.png")
    grey_image = greyscaleImage(new_image)
    grey_image = contrastAdjustImage(grey_image)
    # filtered_image = cv2.adaptiveThreshold(
    #     grey_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 11)
    ret, filtered_image = cv2.threshold(grey_image,220,255, cv2.THRESH_BINARY)

    
    kernel = np.zeros((1, 1), np.uint8)
    filtered_image = cv2.dilate(filtered_image, kernel, iterations=1)

    filtered_image = sharpenImage(filtered_image)

    image_resize = cv2.resize(
                    filtered_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("C_A image", image_resize)
    cv2.waitKey(0)

def sharpenImage(image):
    # sharpen to enhance definition of edges
    kernel = np.array([[-1, -1, -1, -1, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, 2, 8, 2, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, -1, -1, -1, -1]])/8.0
    sharp_image = cv2.filter2D(image, -1, kernel)
    return sharp_image


def contrastAdjustImage(image):
    adaptive_histogram_equalizer = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(2, 2))
    histogram_equalizer = adaptive_histogram_equalizer.apply(image)
    return histogram_equalizer


def greyscaleImage(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grey_image


def smoothingImage(grey_image):
    # Edge-preserving, and noise-reducing smoothing filter
    blur = cv2.bilateralFilter(grey_image, 9, 75, 75)
    return blur

def imageSkewNormalization(filename, image):
    
    deskewed_image = cv2.imread(filename)
    image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(image)

    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.ones_like(image) * 255

    boxes = []

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            hull = cv2.convexHull(contour)
            cv2.drawContours(mask, [hull], -1, 0, -1)
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda box: box[0])
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))

    mask = cv2.bitwise_not(mask)

    # image_resize = cv2.resize(
    #     mask, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("mask Img", image_resize)
    # cv2.waitKey(0)

    # Take a sequence of 1-D arrays and stack them as columns to make a single 2-D array
    XYcoordinates = np.column_stack(np.where(mask > 0))

    # Create bounding box that contains all the coordinates
    angle = cv2.minAreaRect(XYcoordinates)[-1]

    if(angle < -45):
        angle = (angle + 90) * -1
    else:
        angle = angle * -1

    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    r_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(deskewed_image, r_matrix, (width, height),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask, r_matrix, (width, height),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # image_resize = cv2.resize(
    #     rotated, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("Rotated Img", image_resize)
    # cv2.waitKey(0)

    cropped = crop_to_table(rotated, mask)
                             
    return cropped

def crop_to_table(deskewed_image, mask):
    original_image = deskewed_image

    # cv2.imshow("Crop1", mask)
    # cv2.waitKey(0)

    # Take a sequence of 1-D arrays and stack them as columns to make a single 2-D array
    XYcoordinates = np.column_stack(np.where(mask > 0))

    rect = cv2.boundingRect(XYcoordinates)
    y,x,h,w = rect
    cropped = original_image[y:y+h+5, x-5:x+w+5].copy()
    return cropped

def background_estimation(image, thresh_image):
    background_image = image.copy()

    top = 0
    bottom = 0

    padding = cv2.copyMakeBorder(image.copy(),10,10,10,10,cv2.BORDER_CONSTANT)
    thresh = cv2.copyMakeBorder(thresh_image.copy(),10,10,10,10,cv2.BORDER_CONSTANT)
    rows, cols = image.shape

    for i in range(rows - 20):
        for j in range(cols - 20):
            k = thresh[i+10,j+10]
            if(k == 255):
                x = i
                y = j
                for val in range(20):
                    top += (image[x,y] * (1 - thresh[x,y]))
                    bottom += (1 - thresh[x,y])
                    x += 1
                    y += 1
                background_image[i+10,j+10] = top/bottom
                top = 0
                bottom = 0
            # else:
            #     thresh[i,j] = 0
    cv2.imshow("thresh", background_image)
    cv2.waitKey(0)
    return background_image
    

def combining_forground_and_background(og_image, bg_image, thresh_image):
    rows, cols = og_image.shape
    combined = og_image.copy()

    delta = 0
    top = 0
    bottom = 0

    for i in range(rows):
            for j in range(cols):
                if(og_image[i,j] > bg_image[i,j]):
                    top -= og_image[i,j] - bg_image[i,j]
                else:
                    top += (bg_image[i,j] - og_image[i,j])
                
                bottom += thresh_image[i,j]
    
    delta=top/bottom

    for i in range(rows):
        for j in range(cols):

            if(og_image[i,j] > bg_image[i,j]):
                val = 0
            else:
                val = (bg_image[i,j] - og_image[i,j])/255

            if(val > (delta*0.6)):
                combined[i,j] = 255
            else:
                combined[i,j] = 0
    
    return combined


def binarization(filename):
    image = cv2.imread(filename, 0)

    image = cv2.bilateralFilter(image, 2, 100, 100)
    thresh_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    bg_image = background_estimation(image, thresh_image)
    
    final_image = combining_forground_and_background(image, bg_image, thresh_image)
    cv2.imshow("The image", final_image)
    cv2.waitKey(0)


    

  
    
  
