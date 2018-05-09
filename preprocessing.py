# cropImageToTextPortion adopeted from http://www.danvk.org/2015/01/07/finding-blocks-of-text-in-an-image-using-python-opencv-and-numpy.html
# All methods and functions that have to do with preprocessing the image
from PIL import Image
from convertPDFtoJpeg import convertImage
import argparse
import cv2
import os
import numpy as nump
from scipy.ndimage.filters import rank_filter


def startImagePreprocessing():
    getImage()
    return


def getImage():
    agp = argparse.ArgumentParser()
    agp.add_argument("-i", "--image", required=True, help="Enter image -i --image where --image is your image")
    argument = vars(agp.parse_args())
    if (argument["image"].endswith(".pdf")):
        pdf = argument["image"]
        jpg = convertImage(pdf)
        # image = cv2.imread(jpg)
        # print(pdf)
        # print(jpg)
        # For cropping to just the table
        png = jpg.replace('.jpg', '.png')
        cropImageToTextPortion(jpg, png)
        image = cv2.imread(png)
        image_resize = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Image from pdf", image_resize)
        cv2.waitKey(0)
        final_image = greyscaleImage(image)
        image_resize = cv2.resize(final_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Gray scaled image", image_resize)
        cv2.waitKey(0)
        final_image = contrastAdjustImage(final_image)
        image_resize = cv2.resize(final_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Contrast Adjusted image", image_resize)
        cv2.waitKey(0)
        final_image = smoothingImage(final_image)
        image_resize = cv2.resize(final_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Smoothed image", image_resize)
        cv2.waitKey(0)
        final_image = threshholdImage(final_image)
        image_resize = cv2.resize(final_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("thresholded image", image_resize)
        cv2.waitKey(0)
        final_image = sharpenImage(final_image)
        image_resize = cv2.resize(final_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("sharpened image", image_resize)
        cv2.waitKey(0)
        final_image = imageSkewNormalization(final_image)
        image_resize = cv2.resize(final_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("skew normalized image", image_resize)
        cv2.waitKey(0)
        # image_resize = cv2.resize(final_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        preprocess_filename = writeImageToDisk(final_image)
        removePixels()
        return preprocess_filename
    else:
       ''' image = cv2.imread(argument["image"])
        final_image = greyscaleImage(image)
        final_image = contrastAdjustImage(final_image)
        final_image = smoothingImage(final_image)
        final_image = threshholdImage(final_image)
        final_image = sharpenImage(final_image)
        final_image = imageSkewNormalization(final_image)
        writeImageToDisk(final_image)'''
       return "AAAA"


def sharpenImage(image):
    # sharpen to enhance definition of edges
    kernel = nump.array([[-1, -1, -1, -1, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, 2, 8, 2, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, -1, -1, -1, -1]])/8.0
    sharp_image = cv2.filter2D(image, -1, kernel)
    return sharp_image


def contrastAdjustImage(image):
    adaptive_histogram_equalizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    histogram_equalizer = adaptive_histogram_equalizer.apply(image)
    return histogram_equalizer

def greyscaleImage(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grey_image


def smoothingImage(grey_image):
    # Edge-preserving, and noise-reducing smoothing filter
    blur = cv2.bilateralFilter(grey_image, 9, 75, 75)
    return blur


def threshholdImage(grey_image):
    grey_image2 = cv2.adaptiveThreshold(grey_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return grey_image2


def writeImageToDisk(grey_image):
    filename = "images/preprocessed_image.png".format(os.getpid())
    # image_resize = cv2.resize(grey_image2, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(filename, grey_image)
    # cv2.waitKey(0)
    return filename


def imageSkewNormalization(image):
    colourflip_image = cv2.bitwise_not(image)
    thresh_image = cv2.adaptiveThreshold(
        colourflip_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Take a sequence of 1-D arrays and stack them as columns to make a single 2-D array
    XYcoordinates = nump.column_stack(nump.where(thresh_image > 0))
    angle = cv2.minAreaRect(XYcoordinates)[-1]
    if(angle < -45):
        angle = (angle + 90) * -1
    else:
        angle = angle * -1

    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    r_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, r_matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    #image_resize = cv2.resize(rotated, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    return rotated


def scaleImage(image, dimension=2048):

    # Gets the width and height of the image
    width, height = image.size

    # Return if scale is already 1 or smaller
    if max(width, height) <= dimension:
        return 1.0, image
    else:
        scale = 1.0 * dimension / max(width, height)  # Get scale value and rescale image
        scaled_image = image.resize((int(width * scale), int(height * scale)), Image.ANTIALIAS)
        return scale, scaled_image


def dilateForComponents(edges, val, iterations):
    kernel = nump.zeros((val, val), dtype=nump.uint8)
    kernel[(val - 1) // 2, :] = 1
    dilated_image = cv2.dilate(edges / 255, kernel, iterations=iterations)

    kernel = nump.zeros((val, val), dtype=nump.uint8)
    kernel[:, (val - 1) // 2] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
    return dilated_image


def findComponents(edges):
    #dilate image and return contours of components
    count = 17
    n = 1
    while count > 16:
        n += 1
        dilated_image = dilateForComponents(edges, val=3, iterations=n)
        dilated_image = nump.uint8(dilated_image)
        _, contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    return contours


def BoxContoursInfo(contours, edges):
    contour_box = []
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        contour_image = nump.zeros(edges.shape)
        cv2.drawContours(contour_image, [i], 0, 255, -1)
        contour_box.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': nump.sum(edges * (contour_image > 0))/255
        })
    return contour_box


def cropArea(image):
    x1, y1, x2, y2 = image
    return max(0, x2 - x1) * max(0, y2 - y1)


def findBestComponentSubset(contours, edges):

    contour_box = BoxContoursInfo(contours, edges)
    contour_box.sort(key=lambda x: -x['sum'])
    total = nump.sum(edges) / 255
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
            new_crop = min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)
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
        b_x1, b_y1, b_x2, b_y2 = cont['x1'] + 5, cont['y1'] + 5, cont['x2'] - 5, cont['y2'] - 5

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


def cropImageToTextPortion(input_path, output_path):

    image_without_mod = Image.open(input_path)
    im22 = cv2.imread(input_path)
    scale, image = scaleImage(image_without_mod)

    # Canny edge detection of image
    edges = cv2.Canny(nump.asarray(image), 100, 200)

    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find borders of image
    borders = []
    area = edges.shape[0] * edges.shape[1]
    for i, j in enumerate(contours):
        # x and y is top-left coordinate
        x, y, width, height = cv2.boundingRect(j)
        if width * height > 0.5 * area:
            borders.append((i, x, y, x + width - 1, y + height - 1))

    borders.sort(
        key=lambda i_x1_y1_x2_y2: (i_x1_y1_x2_y2[3] - i_x1_y1_x2_y2[1]) * (i_x1_y1_x2_y2[4] - i_x1_y1_x2_y2[2]))

    b_contour = None
    if len(borders):
        b_contour = contours[borders[0][0]]
        contour_image = nump.zeros(edges.shape)
        r = cv2.minAreaRect(b_contour)
        degrees = r[2]
        # Use bounding box if not close to right angle else use rectangle
        if min(degrees % 90, 90 - (degrees % 90)) <= 10.0:
            box = cv2.BoxPoints(r)
            box = nump.int0(box)
            cv2.drawContours(contour_image, [box], 0, 255, -1)
            cv2.drawContours(contour_image, [box], 0, 0, 4)
        else:
            x1, y1, x2, y2 = cv2.boundingRect(b_contour)
            cv2.rectangle(contour_image, (x1, y1), (x2, y2), 255, -1)
            cv2.rectangle(contour_image, (x1, y1), (x2, y2), 0, 4)

        edges = nump.minimum(contour_image, edges)

    edges = 255 * (edges > 0).astype(nump.uint8)

    # Remove borders (1px)
    rows = rank_filter(edges, -4, size=(1, 20))
    cols = rank_filter(edges, -4, size=(20, 1))
    edges = nump.minimum(nump.minimum(edges, rows), cols)

    contours = findComponents(edges)

    cropped_image = findBestComponentSubset(contours, edges)

    cropped_image = ExpandContour(cropped_image, contours, edges, b_contour)

    cropped_image = [int(i / scale) for i in cropped_image]

    final_image = image_without_mod.crop(cropped_image)
    final_image.save(output_path)

    return

def removePixels():
    image = cv2.imread("images/preprocessed_image.png")
    grey_image = greyscaleImage(image)
    _, b_and_w = cv2.threshold(grey_image, 127, 255, cv2.THRESH_BINARY_INV)


    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(b_and_w, 4, cv2.CV_32S)
    sizes = stats[1:, -1]
    img2 = nump.zeros(labels.shape, nump.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 50:
            img2[labels == i + 1] = 255

        final_image = cv2.bitwise_not(img2)
    image_resize = cv2.resize(final_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Pixels removed image", image_resize)
    cv2.waitKey(0)
    cv2.imwrite('images/preprocessed_image.png', final_image)
    return
