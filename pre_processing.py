# Adaptive Binarization Technique included described by 
# Basilios Gatos, Ioannis Pratikakis, and Stavros J. Perantonis
import os
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import gaussian

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


    im = imageSkewNormalization(image_channels)

    im = cv2.resize(
        im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("The filtered image", im)
    cv2.waitKey(0)

    cropPdfToTable(filename, im)

def greyscaleImage(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grey_image

def imageSkewNormalization(image):
    
    deskewed_image = image
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

    image_resize = cv2.resize(
        mask, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("mask Img", image_resize)
    cv2.waitKey(0)

    # Take a sequence of 1-D arrays and stack them as columns to make a single 2-D array
    XYcoordinates = np.column_stack(np.where(mask > 0))

    # Create bounding box that contains all the coordinates
    angle = cv2.minAreaRect(XYcoordinates)[-1]

    print(angle)

    if(angle < -45):
        angle = (angle + 90)
    else:
        angle = angle * -1

    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    r_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(deskewed_image, r_matrix, (width, height),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def scaleImage(image, dimension=2048):

    # Gets the width and height of the image
    width, height = image.size

    # Return if scale is already 1 or smaller
    if max(width, height) <= dimension:
        return 1.0, image
    else:
        # Get scale value and rescale image
        scale = 1.0 * dimension / max(width, height)
        scaled_image = image.resize(
            (int(width * scale), int(height * scale)), Image.ANTIALIAS)
        return scale, scaled_image


def dilateForComponents(edges, val, iterations):
    kernel = np.zeros((val, val), dtype=np.uint8)
    kernel[(val - 1) // 2, :] = 1
    dilated_image = cv2.dilate(edges / 255, kernel, iterations=iterations)

    kernel = np.zeros((val, val), dtype=np.uint8)
    kernel[:, (val - 1) // 2] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)

    return dilated_image


def findComponents(edges):
    # dilate image and return contours of components
    count = 17
    n = 1
    while count > 16:
        n += 1
        dilated_image = dilateForComponents(edges, val=3, iterations=n)
        dilated_image = np.uint8(dilated_image)
        contours, hierarchy = cv2.findContours(
            dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    return contours


def BoxContoursInfo(contours, edges):
    contour_box = []
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        contour_image = np.zeros(edges.shape)
        cv2.drawContours(contour_image, [i], 0, 255, -1)

        contour_box.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': np.sum(edges * (contour_image > 0))/255
        })
    return contour_box


def cropArea(image):
    x1, y1, x2, y2 = image
    return max(0, x2 - x1) * max(0, y2 - y1)


def findBestComponentSubset(contours, edges):

    contour_box = BoxContoursInfo(contours, edges)
    contour_box.sort(key=lambda x: -x['sum'])
    total = np.sum(edges) / 255
    area = edges.shape[0] * edges.shape[1]

    cont = contour_box[0]
    del contour_box[0]
    crop = cont['x1'], cont['y1'], cont['x2'], cont['y2']
    covered_sum = cont['sum']

    while covered_sum < total:
        changed = False
        recall = 1.0 * covered_sum / total
        prec = 1 - 1.0 * cropArea(crop) / area
        f1 = 2 * (prec * recall / (prec + recall))

        for i, c in enumerate(contour_box):
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            x11, y11, x21, y21 = crop
            x12, y12, x22, y22 = this_crop
            new_crop = min(x11, x12), min(
                y11, y12), max(x21, x22), max(y21, y22)
            new_sum = covered_sum + c['sum']
            new_recall = 1.0 * new_sum / total
            new_prec = 1 - 1.0 * cropArea(new_crop) / area
            new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)
            remaining_frac = c['sum'] / (total - covered_sum)
            new_area_frac = 1.0 * cropArea(new_crop) / cropArea(crop) - 1

            if new_f1 > f1 or (remaining_frac > 0.25 and new_area_frac < 0.15):
                crop = new_crop
                covered_sum = new_sum
                del contour_box[i]
                changed = True
                break

        if not changed:
            break

    return crop


def cropInBorder(crop, pad_px, bx1, by1, bx2, by2):
    x1, y1, x2, y2 = crop
    x1 = max(x1 - pad_px, bx1)
    y1 = max(y1 - pad_px, by1)
    x2 = min(x2 + pad_px, bx2)
    y2 = min(y2 + pad_px, by2)
    return crop


def UnionCropped(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


def IntersectCropped(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)


def ExpandContour(crop, contours, edges, b_contour, pad_px=15):

    b_x1, b_y1, b_x2, b_y2 = 0, 0, edges.shape[0], edges.shape[1]

    if b_contour is not None and len(b_contour) > 0:
        cont = BoxContoursInfo([b_contour], edges)[0]
        b_x1, b_y1, b_x2, b_y2 = cont['x1'] + \
            5, cont['y1'] + 5, cont['x2'] - 5, cont['y2'] - 5

    crop = cropInBorder(crop, pad_px, b_x1, b_y1, b_x2, b_y2)

    contour_box = BoxContoursInfo(contours, edges)
    changed = False
    for c in contour_box:
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        this_area = cropArea(this_crop)
        int_area = cropArea(IntersectCropped(crop, this_crop))
        new_crop = cropArea(UnionCropped(crop, this_crop))
        if 0 < int_area < this_area and crop != new_crop:
            changed = True
            crop = new_crop

    if changed:
        return ExpandContour(crop, contours, edges, b_contour, pad_px)
    else:
        return crop


def cropPdfToTable(input_path, reworked_image):

    image_without_mod = Image.open(input_path)
    # scale, image = scaleImage(image_without_mod)

    # Canny edge detection of image
    edges = cv2.Canny(np.asarray(reworked_image), 100, 200)

    # Find all contours of image
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find borders of image (if not pdf)
    borders = []
    area = edges.shape[0] * edges.shape[1]
    for i, j in enumerate(contours):
        # x and y is top-left coordinate
        x, y, width, height = cv2.boundingRect(j)
        if width * height > 0.5 * area:
            borders.append((i, x, y, x + width - 1, y + height - 1))

    borders.sort(
        key=lambda i_x1_y1_x2_y2: (i_x1_y1_x2_y2[3] - i_x1_y1_x2_y2[1])*(i_x1_y1_x2_y2[4] - i_x1_y1_x2_y2[2]))

    b_contour = None
    if len(borders):
        b_contour = contours[borders[0][0]]
        contour_image = np.zeros(edges.shape)
        r = cv2.minAreaRect(b_contour)
        degrees = r[2]
        # Use bounding box if not close to right angle else use rectangle
        if min(degrees % 90, 90 - (degrees % 90)) <= 10.0:
            box = cv2.boxPoints(r)
            box = np.int0(box)
            cv2.drawContours(contour_image, [box], 0, 255, -1)
            cv2.drawContours(contour_image, [box], 0, 0, 4)
        else:
            x1, y1, x2, y2 = cv2.boundingRect(b_contour)
            cv2.rectangle(contour_image, (x1, y1), (x2, y2), 255, -1)
            cv2.rectangle(contour_image, (x1, y1), (x2, y2), 0, 4)

        edges = np.minimum(contour_image, edges)

    edges = 255 * (edges > 0).astype(np.uint8)

    contours = findComponents(edges)

    cropped_image = findBestComponentSubset(contours, edges)

    cropped_image = ExpandContour(cropped_image, contours, edges, b_contour)

    # cropped_image = [int(i / scale) for i in cropped_image]

    final_image = image_without_mod.crop(cropped_image)
   
    final_image.save("test1111.png")

