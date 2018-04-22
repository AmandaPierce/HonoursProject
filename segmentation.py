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
    image_crop = cv2.imread("images/preprocessed_image.png")
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
    vertices = []
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
            vertices.append((x, y))
        elif 175 <= x <= 215:
            vertices.append((x, y))
        elif 505 <= x <= 530:
            vertices.append((x, y))
        elif 840 <= x <= 860:
            vertices.append((x, y))
        elif 373 <= x <= 390:
            vertices.append((x, y))
        elif 673 <= x <= 690:
            vertices.append((x, y))
        elif 965 <= x <= 990:
            vertices.append((x, y))
        elif 1265 <= x <= 1300:
            vertices.append((x, y))
        elif 1550 <= x <= 1600:
            vertices.append((x, y))

    reordered_vertices = []
    for x in vertices:
        x1 = x
        init = False
        for y in yvals:
            if y - 10 <= x1[1] <= y + 20:
                init = True
        if init == True:
            result = cv2.circle(image, (x[0], x[1]), 10, (0, 255, 0), -1)
            reordered_vertices.append(x)
            init = False
        else:
            del x

    result = cv2.resize(result, None, fx=0.20, fy=0.20, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("A", result)
    cv2.waitKey(0)

    vertices = []
    while reordered_vertices:
        vertices.append(reordered_vertices.pop())

    temp = []
    reordered_vertices = []
    for x in vertices:
        if 1550 <= x[0] <= 1600:
            temp.append(x)
            for i in range(len(temp)):
                min_val = min(temp[i:])
                min_index = temp[i:].index(min_val)
                temp[i + min_index] = temp[i]
                temp[i] = min_val
            while temp:
                reordered_vertices.append(temp.pop(0))
        else:
            temp.append(x)

    performCellSegmentation(image_crop, reordered_vertices)
    # iterateTest(image, reordered_vertices)
    return

def iterateTest(image, vertices):

    crop = cv2.circle(image, vertices[0], 10, (0, 255, 255), -1)
    print(vertices.pop(0))
    crop = cv2.resize(crop, None, fx=0.20, fy=0.20, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("B", crop)
    cv2.waitKey(0)
    iterateTest(image, vertices)
    return


def performCellSegmentation(image, vertices):

    last = True
    if not vertices:
        return
    else:
        top_left = vertices[0]
        vertices.pop(0)
        crop = cv2.circle(image, top_left, 10, (0, 255, 255), -1)
        top_right = vertices[0]
        if 1550 <= top_right[0] <= 1600:
            vertices.pop(0)
        else:
            last = False
        crop = cv2.circle(image, top_right, 10, (0, 255, 255), -1)

        bottom_left = None

        for x in vertices:
            if (top_left[0] - 15) <= x[0] <= (top_left[0] + 15):
                bottom_left = x
                crop = cv2.circle(image, bottom_left, 10, (0, 255, 255), -1)
                break

        bottom_right = None

        for x in vertices:
            if last == False:
                last = True
                continue

            if (top_right[0] - 15) <= x[0] <= (top_right[0] + 15):
                bottom_right = x
                crop = cv2.circle(image, bottom_right, 10, (0, 255, 255), -1)
                break

        img = Image.open("images/preprocessed_image.png")

        #print(top_left[0])
        #print(bottom_right)
        crop_image = img.crop((top_left[0] + 6, top_left[1] + 6, bottom_right[0] - 6, bottom_right[1] - 6))
        crop_image.save(str(top_left[0]) + str(bottom_right[0]) + ".png")
        crop = cv2.resize(crop, None, fx=0.20, fy=0.20, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("B", crop)
        cv2.waitKey(0)

        return performCellSegmentation(image, vertices)

def performCharacterSegmentation(image, t_cells):

    return
