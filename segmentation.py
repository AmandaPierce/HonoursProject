from PIL import Image
import cv2
import os
import glob
import numpy as nump
from characterSegmentation import individualSegmentation


def segment_table_cells():

    image = cv2.imread("images/preDemoimage0.png")
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh_image = cv2.adaptiveThreshold(
        grey_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    grey_image = cv2.bitwise_not(thresh_image)

    mask = nump.zeros(grey_image.shape, nump.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    close_small_gaps = cv2.morphologyEx(grey_image, cv2.MORPH_CLOSE, kernel)

    div = nump.float32(grey_image) / close_small_gaps

    result = nump.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))

    threshold = cv2.adaptiveThreshold(
        result, 255, 0, cv2.THRESH_BINARY, 19, 2)

    contours, hierarchy = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
        cv2.MORPH_RECT, (1, nump.array(grey_image).shape[1] // 8))

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

    ret, close = cv2.threshold(
        derivativey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
        cv2.MORPH_RECT, (nump.array(grey_image).shape[1] // 8, 1))

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

    grey_image = cv2.bitwise_not(grey_image)

    i = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 10:
            i += 1
            final_new_img = grey_image[y:y+h, x:x+w]
            cv2.imwrite('images/final' + str(i) + '.png', final_new_img)


def sort_contours(contours):
    i = 1
    reverse = True

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
        kernel = nump.ones((3, 3), nump.uint8)
        grey_image = cv2.morphologyEx(grey_image, cv2.MORPH_OPEN, kernel)
        grey_image = cv2.morphologyEx(grey_image, cv2.MORPH_CLOSE, kernel)
        grey_image = cv2.bitwise_not(grey_image)

        grey_image2 = cv2.threshold(
            grey_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        XYcoordinates = nump.column_stack(nump.where(grey_image2 > 0))
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
