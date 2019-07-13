from PIL import Image
import cv2
import os
import glob
import numpy as np
import imutils
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
from characterSegmentation import individualSegmentation
import math


def segment_table_cells():

    image = cv2.imread("images/testing.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # thresh_image = cv2.adaptiveThreshold(
    #     grey_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    image = cv2.bitwise_not(image)

    mask = np.zeros(image.shape, np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    close_small_gaps = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    div = np.float32(image) / close_small_gaps

    result = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))

    # threshold = cv2.adaptiveThreshold(
    #     result, 255, 0, cv2.THRESH_BINARY, 19, 2)

    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))

    derivativex = cv2.Sobel(result, cv2.CV_16S, 1, 0)

    derivativex = cv2.convertScaleAbs(derivativex)
    cv2.normalize(derivativex, derivativex, 0, 255, cv2.NORM_MINMAX)
    ret, close = cv2.threshold(
        derivativex, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernelx, iterations=2)

    contour, hierarchy = cv2.findContours(
        close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        if h / w > 5:
            cv2.drawContours(close, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)

    close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, None, iterations=2)

    closex = close.copy()

    kernel_clean = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, np.array(image).shape[1] // 8))

    closex = cv2.erode(closex, kernel_clean, iterations=1)

    closex = cv2.dilate(closex, kernel_clean, iterations=1)

    image_resize = cv2.resize(closex, None, fx=0.25,
                              fy=0.25, interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Vertical lines identified", image_resize)
    cv2.waitKey(0)

    kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))

    derivativey = cv2.Sobel(result, cv2.CV_16S, 0, 2)

    derivativey = cv2.convertScaleAbs(derivativey)

    cv2.normalize(derivativey, derivativey, 0, 255, cv2.NORM_MINMAX)

    # ret, close = cv2.threshold(
    #     derivativey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernely)

    contour, hierarchy = cv2.findContours(
        close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        if w / h > 5:
            cv2.drawContours(close, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)

    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, None, iterations=2)
    closey = close.copy()

    kernel_clean = cv2.getStructuringElement(
        cv2.MORPH_RECT, (np.array(image).shape[1] // 8, 1))

    closey = cv2.erode(closey, kernel_clean, iterations=1)

    closey = cv2.dilate(closey, kernel_clean, iterations=1)

    image_resize = cv2.resize(closey, None, fx=0.25,
                              fy=0.25, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Horizontal lines identified", image_resize)
    cv2.waitKey(0)

    alpha = 0.5

    beta = 1.0 - alpha

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    final_image = cv2.addWeighted(
        closex, alpha, closey, beta, 0.0)

    final_image = cv2.erode(~final_image, kernel, iterations=2)

    (thresh, final_image) = cv2.threshold(
        final_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    image_resize = cv2.resize(final_image, None, fx=0.25,
                              fy=0.25, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Final Table", image_resize)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(
        final_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    (contours, boundingBoxes) = sort_contours(contours)

    grey_image = cv2.bitwise_not(image)

    i = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 10:
            i += 1
            final_new_img = grey_image[y:y+h, x:x+w]
            cv2.imwrite('images/final' + str(i) + '.png', final_new_img)


def sort_contours(contours):
    i = 1
    reverse = False

    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))

    return (contours, boundingBoxes)


def performCharacterSegmentation():

    data_path = os.path.join("images/cropped/", '*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        image = cv2.imread(f1)
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        grey_image = cv2.morphologyEx(grey_image, cv2.MORPH_OPEN, kernel)
        grey_image = cv2.morphologyEx(grey_image, cv2.MORPH_CLOSE, kernel)
        grey_image = cv2.bitwise_not(grey_image)

        grey_image2 = cv2.threshold(
            grey_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        XYcoordinates = np.column_stack(np.where(grey_image2 > 0))
        angle = cv2.minAreaRect(XYcoordinates)[-1]
        if (angle < -45):
            angle = (angle + 90) * -1
        else:
            angle = angle * -1

        (height, width) = image.shape[:2]
        center = (width // 2, height // 2)
        r_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, r_matrix, (width, height),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(f1, rotated)
        individualSegmentation(f1)
    return

def identify_table(filename):
    image = cv2.imread(filename)
    color_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tmp_image = np.copy(image)
    image = cv2.bitwise_not(image)

    # cv2.imshow("image", image)
    # cv2.waitKey(0)

    horizontal_lines = np.copy(image)
    vertical_lines = np.copy(image)

    height, width = image.shape[:2]
    horizontal_length = int(width/15)
    vertical_length = int(height/15)

    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_length,1))

    horizontal_lines = cv2.erode(horizontal_lines, horizontal_structure)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_structure)

    # cv2.imshow("image", horizontal_lines)
    # cv2.waitKey(0)

    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length))

    vertical_lines = cv2.erode(vertical_lines, vertical_structure)
    vertical_lines = cv2.dilate(vertical_lines, vertical_structure)

    # cv2.imshow("image", vertical_lines)
    # cv2.waitKey(0)

    horizontal_lines = cv2.bitwise_not(horizontal_lines)
    vertical_lines = cv2.bitwise_not(vertical_lines)
    image = cv2.bitwise_not(image)

    # cv2.imshow("image", horizontal_lines)
    # cv2.waitKey(0)

    mask_image = cv2.bitwise_and(horizontal_lines, vertical_lines)
    # cv2.imshow("mask", mask_image)
    # cv2.waitKey(0)
    mask_image = cv2.bitwise_not(mask_image)

    cnts, _ = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (cnts, _) = sort_contours(cnts)

    average_table_cell_height = 0
    average_table_cell_width = 0
    smallest_width = width

    i = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h > 10 and h < height/3 and w > 10 and w < width:
            average_table_cell_height += h
            average_table_cell_width += w
            if w < smallest_width:
                smallest_width = w
            i += 1
    
    average_table_cell_height = int(average_table_cell_height/i)
    average_table_cell_width = int(average_table_cell_width/i)

    print(smallest_width)

    horizontal_lines = cv2.bitwise_not(np.copy(image))
    vertical_lines =cv2.bitwise_not(np.copy(image))

    horizontal_length = int(smallest_width)
    vertical_length = average_table_cell_height

    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_length, 1))

    horizontal_lines = cv2.erode(horizontal_lines, horizontal_structure)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_structure)
    # cv2.imshow("image", horizontal_lines)
    # cv2.waitKey(0)

    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length))

    vertical_lines = cv2.erode(vertical_lines, vertical_structure)
    vertical_lines = cv2.dilate(vertical_lines, vertical_structure)

    # cv2.imshow("image", vertical_lines)
    # cv2.waitKey(0)

    horizontal_lines = cv2.bitwise_not(horizontal_lines)
    vertical_lines = cv2.bitwise_not(vertical_lines)
    image = cv2.bitwise_not(image)

    # cv2.imshow("image", horizontal_lines)
    # cv2.waitKey(0)

    mask_image = cv2.bitwise_and(horizontal_lines, vertical_lines)
    # cv2.imshow("mask2", mask_image)
    # cv2.waitKey(0)

    mask_image = cv2.bitwise_not(mask_image)
    tmp_image = cv2.bitwise_not(tmp_image)

    final_image = cv2.subtract(tmp_image, mask_image)
    # final_image = cv2.subtract(horizontal_lines, final_image)
    cv2.imshow("sub", final_image)
    cv2.waitKey(0)

    final_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_image = cv2.erode(final_image, final_structure)
    final_image = cv2.dilate(final_image, final_structure)

    final_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    final_image = cv2.dilate(final_image, final_structure, iterations=1)

    final_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (smallest_width, 1))
    final_image = cv2.dilate(final_image, final_structure, iterations=1)

    cv2.imshow("dilated", final_image)
    cv2.waitKey(0)

    edges = cv2.Canny(final_image, 50, 100)
    
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts, _) = contours.sort_contours(cnts)

    average_character_height = 0
    number_of_cnts = 0

    for c in cnts:
        orig = color_image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
    
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        # cv2.imshow("contours", orig)
        # cv2.waitKey(0)

        (top_l, top_r, bottom_r, bottom_l) = box

        dist = math.sqrt((top_r[0] - bottom_r[0])**2 + (top_r[1] - bottom_r[1])**2)

        number_of_cnts += 1
        average_character_height += dist
    
    average_character_height = average_character_height/number_of_cnts

    print(average_character_height)

    

