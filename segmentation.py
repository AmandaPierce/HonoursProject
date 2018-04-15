from PIL import Image
import cv2
import os
import subprocess
import sys
import glob
import numpy as nump

# a line has to have at least 300 continuous pixels to be considered a line
HORIZONTAL_THRESHHOLD = 500
VERTICAL_THRESHOLD = 500

def segmentTableCells(png):


    image = cv2.imread("images/preprocessed_image.png")
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
    centroids = []
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
            centroids.append((x, y))

        if 175 <= x <= 215:
            centroids.append((x, y))
            # result = cv2.circle(image, (x, y), 10, (0, 255, 255), -1)


        if 505 <= x <= 530:
            centroids.append((x, y))
            # result = cv2.circle(image, (x, y), 10, (0, 255, 255), -1)


        if 840 <= x <= 860:
            centroids.append((x, y))
            # result = cv2.circle(image, (x, y), 10, (0, 255, 255), -1)


        if 373 <= x <= 390:
            centroids.append((x, y))
            # result = cv2.circle(image, (x, y), 10, (0, 255, 255), -1)


        if 673 <= x <= 690:
            centroids.append((x, y))
            # result = cv2.circle(image, (x, y), 10, (0, 255, 255), -1)


        if 965 <= x <= 990:
            centroids.append((x, y))
           # result = cv2.circle(image, (x, y), 10, (0, 255, 255), -1)


        if 1265 <= x <= 1300:
            centroids.append((x, y))
           # result = cv2.circle(image, (x, y), 10, (0, 255, 255), -1)


        if 1550 <= x <= 1600:
            centroids.append((x, y))
           # result = cv2.circle(image, (x, y), 10, (0, 255, 255), -1)

    for x in centroids:
        x1 = x
        init = False
        for y in yvals:
            if y - 10 <= x1[1] <= y + 20:
                init = True

        if init == True:
            result = cv2.circle(image, (x[0], x[1]), 10, (0, 255, 0), -1)
        else:
            del x

    result = cv2.resize(result, None, fx=0.20, fy=0.20, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("A", result)
    cv2.waitKey(0)


def performCharacterSegmentation(image, t_cells, row, col):

    current_cell = image.crop(t_cells[row][col])
    current_cell = current_cell.point(lambda p: p > 200 and 255)
    hist = current_cell.histogram()
    background = None
    if hist[0] > hist[255]:
        background = 0
    else:
        background = 255

    pixels = current_cell.load()

    x1, y1 = 0, 0
    x2, y2 = current_cell.size
    x2, y2 = x2 - 1, y2 - 1
    while pixels[x1, y1] != background:
        x1 += 1
        y1 += 1

    while pixels[x2, y2] != background:
        x2 -= 1
        y2 -= 1

    current_cell = current_cell.crop((x1, y1, x2, y2))
    current_cell.save("images/cellSegmented.png", "PNG")

    test = cv2.imread("images/cellSegmented.jpg")
    cv2.imshow(test)
    cv2.waitKey(0)
    return
