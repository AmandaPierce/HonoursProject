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
from recognition import image_canvas_centering, predict_character 
import math


def segment_table_cells(table_image, original_image, average_table_cell_height, smallest_width):
    
    table_image = cv2.bitwise_not(table_image)

    cnts, _ = cv2.findContours(
        table_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    (cnts, boundingBoxes) = sort_contours(cnts)

    ordered_cells = sort_cells(cnts)

    k = 0
    for i in ordered_cells:
        for x in i:
            if((x[3] > 5) and (x[3] < (average_table_cell_height + 10)) and (x[2] > 5)):
                k += 1
                final_new_img = original_image[x[1]:x[1]+x[3], x[0]:x[0]+x[2]]
                cv2.imwrite('images/final' + str(k).zfill(5) + '.png', final_new_img)


def sort_contours(contours):
    i = 1
    reverse = False

    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))

    return (contours, boundingBoxes)

def sort_cells(cnts):
    array = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        array.append( (x, y, w, h) )

    array = insertion_sort(array, 1)
    tmp_array = []
    i = 0
    for x in array:
        if not tmp_array:
            tmp_list = []
            tmp_list.append(x)
            tmp_array.insert(i, tmp_list)
            i += 1
        else:
            found = False
            for n in tmp_array:
                for k in n:
                    if (x[1] > (k[1] - 5) and x[1] < (k[1] + 5)):
                        n.append(x)
                        found = True
                        break
                    
                    if(found):
                        break
            
            if not found:
                tmp_list = []
                tmp_list.append(x)
                tmp_array.insert(i, tmp_list)
                i += 1

    for l in tmp_array:
        l = insertion_sort(l, 0)

    return tmp_array

    


def insertion_sort(arr, axis):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key[axis] < arr[j][axis] : 
                arr[j + 1] = arr[j] 
                j -= 1
        arr[j + 1] = key 

    return arr


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

    average_table_cell_height = 0.0
    average_table_cell_width = 0.0
    smallest_width = width

    i = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h > 10 and h < int(height/3) and w > 10 and w < width:
            average_table_cell_height += float(h)
            average_table_cell_width += float(w)
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
    # cv2.imshow("sub", final_image)
    # cv2.waitKey(0)

    mask_image = cv2.bitwise_not(mask_image)
    final_image = cv2.bitwise_not(final_image)

    segment_table_cells(mask_image, final_image, average_table_cell_height, smallest_width)

    mask_image = cv2.bitwise_not(mask_image)
    final_image = cv2.bitwise_not(final_image)

    final_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_image = cv2.erode(final_image, final_structure)
    final_image = cv2.dilate(final_image, final_structure)

    final_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    final_image = cv2.dilate(final_image, final_structure, iterations=1)

    final_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (smallest_width, 1))
    final_image = cv2.dilate(final_image, final_structure, iterations=1)

    # cv2.imshow("dilated", final_image)
    # cv2.waitKey(0)

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

        dist = float(math.sqrt((top_r[0] - bottom_r[0])**2 + (top_r[1] - bottom_r[1])**2))

        number_of_cnts += 1
        average_character_height += dist
    
    average_character_height = float(average_character_height)/float(number_of_cnts)

    print(average_character_height)
    return average_character_height

def character_segmentation(filename, average_character_height):
    tmp_str = ""
    image = cv2.imread(filename)
    image = cv2.bitwise_not(image)
    # cv2.imshow("testing", image)
    # cv2.waitKey(0)
    kernel = np.ones((2, 1), np.uint8)
    image = cv2.erode(image, kernel)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel)
    # cv2.imshow("testing", image)
    # cv2.waitKey(0)

    colour_image = np.copy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

    height, width = image.shape[:2]

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 4, cv2.CV_32S)
    sizes = stats[1:, -1]
    display_image = np.ones((image.shape), np.uint8)

    position = 1

    stats_in_order = []

    for i in range(0, nlabels):
        cv2.rectangle(colour_image, (stats[i][0], stats[i][1]), (stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]), (0, 255, 0), 2)

        # cv2.imshow("colour", colour_image)
        # cv2.waitKey(0)

        stats_in_order.append((stats[i][0], stats[i][1], stats[i][2], stats[i][3]))

        # final_new_img = image[stats[i][1]:stats[i][1] + stats[i][3], stats[i][0]: stats[i][0] + stats[i][2]]

        # h, w = final_new_img.shape[:2]

        # if h > (average_character_height/2 - 10) and w < width - 10 and w > 5:
        #     display_image[labels == i+1] = 255

        #     # cv2.imshow("final", final_new_img)
        #     # cv2.waitKey(0)

    ordered = insertion_sort(stats_in_order, 0)

    for k in ordered:
        
        final_new_img = image[k[1]:k[1] + k[3], k[0]: k[0] + k[2]]

        h, w = final_new_img.shape[:2]

        if h > (float(average_character_height/2) - 5) and w < width - 15 and w > 5:
            display_image[labels == i+1] = 255

            # cv2.imshow("final", final_new_img)
            # cv2.waitKey(0)
    
            thinned_image = thinning_algorithm(final_new_img)
            cv2.imshow("final", thinned_image)
            cv2.waitKey(0)

            segmentation_points = over_character_segmentation(thinned_image, average_character_height)
            # print (segmentation_points)

            current_x = 0
            idx = 0

            if(len(segmentation_points) == 1):
                char_image = final_new_img[0:h, current_x:segmentation_points[idx][0]]
                # print(segmentation_points[idx][0])
                char_image = cv2.bitwise_not(char_image)
                # kernel = np.zeros((3, 1), np.uint8)
                # char_image = cv2.dilate(char_image, kernel)
                # cv2.imshow("AAAA", char_image)
                # cv2.waitKey(0)
                char_image = image_canvas_centering(char_image)
                predicted_char = predict_character(char_image)
                tmp_str += str(predicted_char)
                cv2.imwrite("images/chars/" + filename[7:-4] + "_" + str(position).zfill(3) + ".png", char_image)
                current_x = segmentation_points[idx][0]
                idx += 1
                position += 1
            elif len(segmentation_points) > 0:

                if ((len(segmentation_points) + 1)%2==0):
                    # print("A")
                    while (current_x < w) and (idx < (len(segmentation_points))):
                        # print (w)
                        # print (current_x)
                        char_image = final_new_img[0:h, current_x:segmentation_points[idx][0]]
                        # print(segmentation_points[idx][0])
                        char_image = cv2.bitwise_not(char_image)
                        # kernel = np.zeros((3, 3), np.uint8)
                        # char_image = cv2.dilate(char_image, kernel, iterations=2)
                        # cv2.imshow("AAAA", char_image)
                        # cv2.waitKey(0)
                        char_image = image_canvas_centering(char_image)
                        cv2.imwrite("images/chars/" + filename[7:-4] + "_" + str(position).zfill(3) + ".png", char_image)
                        predicted_char = predict_character(char_image)
                        tmp_str += str(predicted_char)
                        # cv2.imshow("final", char_image)
                        # cv2.waitKey(0)
                        # print(predicted_char)
                        current_x = segmentation_points[idx][0]
                        idx += 1
                        position += 1
                elif ((len(segmentation_points) + 1)%2 != 0):
                    while (current_x < w) and (idx < (len(segmentation_points))):
                        # print (w)
                        # print (current_x)
                        char_image = final_new_img[0:h, current_x:segmentation_points[idx][0]]
                        # print(segmentation_points[idx][0])
                        char_image = cv2.bitwise_not(char_image)
                        # kernel = np.zeros((3, 3), np.uint8)
                        # char_image = cv2.dilate(char_image, kernel, iterations=2)
                        # cv2.imshow("AAAA", char_image)
                        # cv2.waitKey(0)
                        char_image = image_canvas_centering(char_image)
                        cv2.imwrite("images/chars/" + filename[7:-4] + "_" + str(position).zfill(3) + ".png", char_image)
                        predicted_char = predict_character(char_image)
                        tmp_str += str(predicted_char)
                        # cv2.imshow("final", char_image)
                        # cv2.waitKey(0)
                        # print(predicted_char)
                        current_x = segmentation_points[idx][0]
                        idx += 1
                        position += 1
            
            # print (current_x)
            char_image = final_new_img[0:h, current_x:w]
            char_image = cv2.bitwise_not(char_image)
            char_image = image_canvas_centering(char_image)
            predicted_char = predict_character(char_image)
            tmp_str += str(predicted_char)
            cv2.imwrite("images/chars/" + filename[7:-4] + "_" + str(position).zfill(3) + ".png", char_image)
            # cv2.imshow("final", char_image)
            # cv2.waitKey(0)
            # print(predicted_char)
            # kernel = np.zeros((3, 1), np.uint8)
            # char_image = cv2.dilate(char_image, kernel)
            # cv2.imshow("final", char_image)
            # cv2.waitKey(0)
            position += 1
    
    # print(tmp_str)
    return tmp_str

# Zhang-Suen thinning algorithm
def thinning_algorithm(image):
    thinned_image = image.copy()
    changing1 = changing2 = 1
    while changing1 or changing2:
        changing1 = []
        rows, columns = thinned_image.shape[:2]
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = getNeighbours(x, y, thinned_image)
                if (thinned_image[x][y] == 255 and 510 <= sum(n) <= 2040 and transitions(n) == 1 and P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0):
                    changing1.append((x, y))
        for x, y in changing1:
            thinned_image[x][y] = 0

        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = getNeighbours(x, y, thinned_image)
                if (thinned_image[x][y] == 255 and 510 <= sum(n) <= 2040 and transitions(n) == 1 and
                        P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0):
                    changing2.append((x, y))
        for x, y in changing2:
            thinned_image[x][y] = 0

    cv2.imshow("thinned", thinned_image)
    return thinned_image

def getNeighbours(x, y, image):
    x_1, y_1, x_2, y_2 = x - 1, y - 1, x + 1, y + 1
    array = [image[x_1][y], image[x_1][y_2], image[x][y_2], image[x_2][y_2], image[x_2][y], image[x_2][y_1],image[x][y_1], image[x_1][y_1]]
    return array

def transitions(neighbours):
    n = neighbours + neighbours[0:1]
    return sum((n1, n2) == (0, 255) for n1, n2 in zip(n, n[1:]))

def over_character_segmentation(image, average_character_height):
    image_segmented = image.copy()
    image = np.copy(image_segmented)
    # cv2.imshow("final", image_segmented)
    # cv2.waitKey(0)
    vertical = []
    rows, cols = image_segmented.shape[:2]

    for x in range(0, cols-1):
        total = 0
        for y in range(0, (rows-1)):
            total += image_segmented[y][x]
            
        if total == 0 or total == 255:
            if not ((x - (int(average_character_height/2))) < 0) and not ((x +  (int(average_character_height/2))) > cols):
                vertical.append((x,0,x,y))
    
    i = 0
    current = len(vertical)
    while i < (current - 1):
        diff_1 = float("inf")
        diff_2 = float("inf")
        if((i - 1) > 0):
            diff_1 = vertical[i][0] - vertical[i-1][0]

        if(i + 1 < len(vertical)):
            diff_2 = vertical[i + 1][0] - vertical[i][0]
        
        if(diff_1 < diff_2):
            if(diff_1 < 10):
                vertical.pop(i-1)
                current = len(vertical)
            else:
                i += 1
        elif (diff_2 < diff_1):
            if(diff_2 < 10):
                vertical.pop(i)
                current = len(vertical)
            else:
                i += 1   

    # for x in vertical:
    #     cv2.line(image,(x[0],0),(x[0],(rows - 1)),(255,0,0),1)

    # cv2.imshow("line", image)
    # cv2.waitKey(0)

    return vertical