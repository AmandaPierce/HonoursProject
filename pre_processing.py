# Adaptive Binarization Technique included described by 
# Basilios Gatos, Ioannis Pratikakis, and Stavros J. Perantonis
import os
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import gaussian, convolve2d
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
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

    cv2.imwrite("images/testing.jpg", im)

    #cv2.imwrite(filename, im)

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
    rows, cols = padding.shape

    for i in range(rows - 20):
        for j in range(cols - 20):
            k = thresh[i+10,j+10]
            if(k == 0):
                x = i
                y = j
                for val in range(20):
                    top += (padding[x,y] * (1 - thresh[x,y]))
                    bottom += (1 - thresh[x,y])
                    x += 1
                    y += 1
                background_image[i,j] = top/bottom
                top = 0
                bottom = 0
            # else:
            #     thresh[i,j] = 0
    # cv2.imshow("thresh", background_image)
    # cv2.waitKey(0)
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
    thresh_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # cv2.imshow("AAAAAAAAAAAAAA", thresh_image)
    # cv2.waitKey(0)
    bg_image = background_estimation(image, thresh_image)
    
    final_image = combining_forground_and_background(image, bg_image, thresh_image)
    final_image = cv2.bitwise_not(final_image)
    cv2.imwrite("images/testing.jpg", final_image)
    # cv2.imshow("The image", final_image)
    # cv2.waitKey(0)


def scaleImage(image, dimension=2048):

    # Gets the width and height of the image
    height, width = image.shape[:2]

    # Return if scale is already 1 or smaller
    if max(width, height) <= dimension:
        return 1.0, image
    else:
        # Get scale value and rescale image
        scale = 1.0 * (dimension / max(width, height))
        scaled_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return scale, scaled_image


def BorderRemovalAlgorithm(input_path):
    original_image = cv2.imread(input_path)
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    height, width =image.shape[:2]

    third_width = int(width/3)

    segment_x = int(third_width/10)
    segment_y = int((height/5))

    x = third_width
    y = 0
    # average_num_of_pixels = 0
    # i = 0
    cv2.imshow("AAA", image)
    cv2.waitKey(0)

    #Left margin
    while x > segment_x:
        val = x - segment_x
        while y <= (height - segment_y):
            segment_image = image[y:y + segment_y, val:x].copy()
            nzCount = cv2.countNonZero(segment_image)
            section_height, section_width = segment_image.shape[:2]
            totalPixels = section_width * section_height
            ratio = nzCount/totalPixels
            # cv2.imshow("Segmented image", segment_image)
            # cv2.waitKey(0)
            print(ratio)
            if(round(ratio,2) > 0.95):
                x_val = x - segment_x
                y_val = y
                while y_val < (y + segment_y):
                    if(image[y_val, x_val] == 0):
                        while ((image[y_val, x_val] != 255) and (x_val <= x)):
                            x_val += 1

                    while(x_val >= 0):
                        image[y_val, x_val] = 255
                        x_val -= 1
                    y_val += 1
                    x_val = x - segment_x
            y += segment_y
        x -= segment_x
        y = 0

    x = (third_width * 2)
    y = 0

    #Right margin
    while x < (width - segment_x):
        val = x + segment_x
        while y <= (height - segment_y):
            segment_image = image[y:y + segment_y, x:val].copy()
            nzCount = cv2.countNonZero(segment_image)
            section_height, section_width = segment_image.shape[:2]
            totalPixels = section_width * section_height
            ratio = nzCount/totalPixels
            print(ratio)
            # cv2.imshow("Segmented image", segment_image)
            # cv2.waitKey(0)
            if(round(ratio,2) > 0.98):
                x_val = x
                y_val = y
                while y_val < (y + segment_y):
                    if(image[y_val, x_val] == 0):
                        while (image[y_val, x_val] != 255) and (x_val <= x + segment_x):
                            x_val += 1

                    while(x_val < width):
                        image[y_val, x_val] = 255
                        x_val += 1
                    y_val += 1
                    x_val = x
            y += segment_y
        x += segment_x
        y = 0

    cv2.imshow("AAA", image)
    cv2.imwrite("images/testing.png", image)
    cv2.waitKey(0)

    # while x < (third_width * 2):
    #     segment_image = image[0:height, x:x + segment].copy()
    #     nzCount = cv2.countNonZero(segment_image)
    #     # print(nzCount)
    #     section_height, section_width = segment_image.shape[:2]
    #     totalPixels = section_width * section_height
    #     # print(totalPixels)
    #     # print(totalPixels - nzCount)
    #     ratio = (totalPixels-nzCount)/totalPixels
    #     print(ratio)
    #     average_num_of_pixels += ratio
    #     i += 1
    #     x = x + segment
    #     cv2.imshow("AAA", segment_image)
    #     cv2.waitKey(0)
    
    # x = 0
    # average_ratio = average_num_of_pixels/i

    # segment_image = image[0:height, x:x + segment].copy()
    # nzCount = cv2.countNonZero(segment_image)
    # section_height, section_width = segment_image.shape[:2]
    # totalPixels = section_width * section_height
    # ratio = (totalPixels-nzCount)/totalPixels
    # print(ratio)
    # cv2.imshow("segment should be removed", segment_image)
    # cv2.waitKey(0)

    # while x < third_width:
    #     segment_image = image[0:height, x:x + segment].copy()
    #     nzCount = cv2.countNonZero(segment_image)
    #     section_height, section_width = segment_image.shape[:2]
    #     totalPixels = section_width * section_height
    #     ratio = nzCount/totalPixels
    #     print(ratio)
    #     cv2.imshow("segment should be removed", segment_image)
    #     cv2.waitKey(0)
    #     if(ratio < (average_ratio - 0.1)):
    #         cv2.imshow("segment should be removed", segment_image)
    #         cv2.waitkey(0)
    #     x = x + segment
    