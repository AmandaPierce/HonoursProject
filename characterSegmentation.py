import cv2
import numpy as nump
from PIL import Image

def individualSegmentation():
    original_image = cv2.imread('testing.png')
    kernel = nump.ones((5, 5), nump.uint8)
    grey_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey_image, 0, 255, cv2.THRESH_OTSU)[1]
    bw_original = cv2.bitwise_not(thresh)
    bw_skeleton = thinningAlgorithm(bw_original)
    cv2.imwrite("images/skeleton.png", bw_skeleton)
    im = cv2.imread("images/skeleton.png")
    segmentationOfCharacters(im)

def getNeighbours(x, y, image):
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    array = [image[x_1][y], image[x_1][y1], image[x][y1], image[x1][y1], image[x1][y], image[x1][y_1],
             image[x][y_1], image[x_1][y_1]]
    return array

def transitions(neighbours):
    n = neighbours + neighbours[0:1]
    return sum((n1, n2) == (0, 255) for n1, n2 in zip(n, n[1:]))

def thinningAlgorithm(image):
    thinned_image = image.copy()
    changing1 = changing2 = 1
    while changing1 or changing2:
        changing1 = []
        rows, columns = thinned_image.shape[:2]
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = getNeighbours(x, y, thinned_image)
                if (thinned_image[x][y] == 255 and 510 <= sum(n) <= 2040 and transitions(n) == 1
                        and P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0):
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
    return thinned_image

def segmentationOfCharacters(image):
    image_segmented = image.copy()
    letter_coordinates_vertical = []
    rows, columns = image_segmented.shape[:2]
    add_the_coordinate = False
    for y in range(1, columns - 1):
        total = 0
        for x in range(1, rows - 1):
            total = total + sum(image_segmented[x][y])

        if total > 0 and add_the_coordinate is False:
            letter_coordinates_vertical.append([])
            letter_coordinates_vertical[len(letter_coordinates_vertical) - 1].append(y)
            add_the_coordinate = True
        elif total > 0 and add_the_coordinate:
            letter_coordinates_vertical[len(letter_coordinates_vertical) - 1].append(y)
        elif total <= 0:
            add_the_coordinate = False

    counter = 0
    for x in letter_coordinates_vertical:
        # crop = cv2.line(image, (x[0], 0), (x[len(x) - 1], rows - 1), (255, 255, 0), 1)
        img = Image.open("testing.png")
        crop = img.crop((x[0], 0, x[len(x) - 1], rows - 1))
        crop.save("images/testing_" + str(counter) + ".png")
        counter = counter + 1

    letter_coordinates_horizontal = []
    add_the_coordinate = False
    for x in range(1, rows - 1):
        total2 = 0
        for y in range(1, columns - 1):
            total2 = total2 + sum(image_segmented[x][y])

        if total2 > 0 and add_the_coordinate is False:
            #letter_coordinates_horizontal.append([])
            letter_coordinates_horizontal.append(x)
            add_the_coordinate = True
        elif total2 > 0 and add_the_coordinate:
            letter_coordinates_horizontal.append(x)
        elif total2 <= 0:
            add_the_coordinate = False

    img = Image.open("images/testing_0.png")
    width, height = img.size
    crop = img.crop((0, letter_coordinates_horizontal[0], width - 1, height - 1))
    crop.save("images/testing_0.png")

    img = Image.open("images/testing_1.png")
    width, height = img.size
    crop = img.crop((0, letter_coordinates_horizontal[0], width - 1, height - 1))
    crop.save("images/testing_1.png")


