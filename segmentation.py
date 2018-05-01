from PIL import Image
import cv2
import os
import subprocess
import sys
import glob
import numpy as nump
from characterSegmentation import individualSegmentation

def segmentTableCells(png):

    image = Image.open("images/preprocessed_image.png")
    width, height = image.size

    crop_image = image.crop((0, 1065, width, height))
    crop_image.save("images/justNumbers.png")

    image = cv2.imread("images/justNumbers.png")
    image_crop = cv2.imread("images/justNumbers.png")
    # image = cv2.GaussianBlur(image, (7, 7), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = nump.zeros(gray.shape, nump.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    div = nump.float32(gray) / close

    result = nump.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))

    threshold = cv2.adaptiveThreshold(result, 255, 0, 1, 19, 2)
    _, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))

    dx = cv2.Sobel(result, cv2.CV_16S, 1, 0)
    dx = cv2.convertScaleAbs(dx)
    cv2.normalize(dx, dx, 0, 255, cv2.NORM_MINMAX)
    ret, close = cv2.threshold(dx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernelx, iterations=1)

    _, contour, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        if h / w > 5:
            cv2.drawContours(close, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)
    close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, None, iterations=2)
    closex = close.copy()

    kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    dy = cv2.Sobel(result, cv2.CV_16S, 0, 2)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy, dy, 0, 255, cv2.NORM_MINMAX)
    ret, close = cv2.threshold(dy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernely)

    _, contour, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        if w / h > 5:
            cv2.drawContours(close, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)

    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, None, iterations=2)
    closey = close.copy()

    result = cv2.bitwise_and(closex, closey)

    _, contour, hierarchy = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    vertices_multi = []
    for x in range(0, 7):
        vertices_multi.append([])
    yvals = []
    for cnt in contour:
        mom = cv2.moments(cnt)
        if mom["m00"] != 0:
            x = int(mom["m10"] / mom["m00"])
            y = int(mom["m01"] / mom["m00"])
        else:
            x, y = 0, 0

        # result = cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
        if 55 <= x <= 85:
            yvals.append(y)
            vertices_multi[0].append((x, y))
            # result = cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
        elif 190 <= x <= 215:
            vertices_multi[1].append((x, y))
            # result = cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
        elif 505 <= x <= 530:
            vertices_multi[2].append((x, y))
            # result = cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
        elif 840 <= x <= 860:
            vertices_multi[3].append((x, y))
            # result = cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
        elif 965 <= x <= 990:
            vertices_multi[4].append((x, y))
            # result = cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
        elif 1265 <= x <= 1320:
            vertices_multi[5].append((x, y))
            # result = cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
        elif 1550 <= x <= 1600:
            vertices_multi[6].append((x, y))
            # result = cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

    '''elif 175 <= x <= 215:
          elif 673 <= x <= 690:
          elif 373 <= x <= 390'''

    reordered_vertices = []
    for x in range(0, 7):
        reordered_vertices.append([])

    row_counter = 0
    for row in vertices_multi:
        init = False
        for column in row:
            for y in yvals:
                if y - 10 <= column[1] <= y + 20:
                    init = True
            if init:
                reordered_vertices[row_counter].insert(0, column)
                init = False
            else:
                del column
        row_counter = row_counter + 1

    performCellSegmentation(image_crop, reordered_vertices)
    # iterateTest(image, reordered_vertices)
    performCharacterSegmentation()
    return

def iterateTest(image, vertices):
    for row in vertices:
        for column in row:
            print(column)
            crop = cv2.circle(image, column, 10, (0, 255, 255), -1)
            crop = cv2.resize(crop, None, fx=0.20, fy=0.20, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("B", crop)
            cv2.waitKey(0)
    return

def performCellSegmentation(image, vertices):

    for x in range(0, 3):
        count_vals = 1
        for val in vertices[x][:-1]:
            top_left = val
            bottom_right = vertices[x+1][count_vals]
            img = Image.open("images/justNumbers.png")
            # crop_image = img.crop((top_left[0] - 5, top_left[1] - 5, bottom_right[0] + 5, bottom_right[1] + 5))
            crop_image = img.crop((top_left[0] + 6, top_left[1] + 6, bottom_right[0] - 5, bottom_right[1] - 5))
            crop_image.save("images/" + str(x) + "_" + str(count_vals - 1) + ".png")
            count_vals = count_vals + 1

def performCharacterSegmentation():
    image = cv2.imread("images/1_0.png")
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = nump.ones((3, 3), nump.uint8)
    grey_image = cv2.morphologyEx(grey_image, cv2.MORPH_OPEN, kernel)
    grey_image = cv2.morphologyEx(grey_image, cv2.MORPH_CLOSE, kernel)
    grey_image = cv2.bitwise_not(grey_image)

    grey_image2 = cv2.threshold(grey_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    XYcoordinates = nump.column_stack(nump.where(grey_image2 > 0))
    angle = cv2.minAreaRect(XYcoordinates)[-1]
    if (angle < -45):
        angle = (angle + 90) * -1
    else:
        angle = angle * -1

    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    r_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, r_matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite("testing.png", rotated)
    individualSegmentation()
    return
