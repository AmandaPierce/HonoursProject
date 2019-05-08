import cv2
from PIL import Image
import numpy as nump


def identify_and_extract_characters(filename):
    image = cv2.imread(filename)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, b_and_w = cv2.threshold(grey_image, 128, 255,
                                 cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # find the connected components
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        b_and_w, 4, cv2.CV_32S)
    sizes = stats[1:, -1]
    img2 = nump.zeros((image.shape), nump.uint8)

    for i in range(1, nlabels - 1):
        cv2.rectangle(img2, (stats[i][0], stats[i][1]), (stats[i]
                                                         [0] + stats[i][2], stats[i][1] + stats[i][3]), (0, 255, 0), 2)

        cv2.circle(img2, (stats[i][0], stats[i][1]), 5, (0, 255, 255), 2)
        cv2.circle(img2, (stats[i]
                          [0] + stats[i][2], stats[i][1] + stats[i][3]), 5, (255, 255, 255), 2)

        final_new_img = grey_image[stats[i][1]:stats[i][1] +
                                   stats[i][3], stats[i][0]: stats[i][0] + stats[i][2]]
        file_name = filename.split('/')
        cv2.imwrite('images/cropped/char/' + str(i) + '_' +
                    file_name[len(file_name) - 1], final_new_img)
        img2[labels == i + 1] = 255

        cv2.imshow('connected components', final_new_img)
        cv2.waitKey(0)


def individualSegmentation(image_path):
    original_image = cv2.imread(image_path)
    grey_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey_image, 0, 255, cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.bitwise_not(thresh)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)
    bw_skeleton = thinningAlgorithm(thresh)
    cv2.imwrite("images/skeleton.png", bw_skeleton)
    im = cv2.imread("images/skeleton.png")
    segmentationOfCharacters(image_path, im)


def getNeighbours(x, y, image):
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    array = [image[x_1][y], image[x_1][y1], image[x][y1], image[x1][y1], image[x1][y], image[x1][y_1],
             image[x][y_1], image[x_1][y_1]]
    return array


def transitions(neighbours):
    n = neighbours + neighbours[0:1]
    return sum((n1, n2) == (0, 255) for n1, n2 in zip(n, n[1:]))

# Zhang-Suen thinning algorithm


def thinningAlgorithm(image):
    thinned_image = image.copy()
    changing1 = changing2 = 1
    while changing1 or changing2:
        changing1 = []
        rows, columns = thinned_image.shape[:2]
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = getNeighbours(
                    x, y, thinned_image)
                if (thinned_image[x][y] == 255 and 510 <= sum(n) <= 2040 and transitions(n) == 1
                        and P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0):
                    changing1.append((x, y))
        for x, y in changing1:
            thinned_image[x][y] = 0

        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = getNeighbours(
                    x, y, thinned_image)
                if (thinned_image[x][y] == 255 and 510 <= sum(n) <= 2040 and transitions(n) == 1 and
                        P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0):
                    changing2.append((x, y))
        for x, y in changing2:
            thinned_image[x][y] = 0
    return thinned_image


def segmentationOfCharacters(image_file, image):
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
            letter_coordinates_vertical[len(
                letter_coordinates_vertical) - 1].append(y)
            add_the_coordinate = True
        elif total > 0 and add_the_coordinate:
            letter_coordinates_vertical[len(
                letter_coordinates_vertical) - 1].append(y)
        elif total <= 0:
            add_the_coordinate = False

    val_added = []
    counter = 0
    for x in letter_coordinates_vertical:
        # crop = cv2.line(image, (x[0], 0), (x[len(x) - 1], rows - 1), (255, 255, 0), 1)
        img = Image.open(image_file)
        crop = img.crop((x[0] - 5, 0, x[len(x) - 1] + 5, rows - 1))
        print(image_file[15:-4])
        crop.save("images/cropped/" +
                  image_file[15:-4] + "_" + str(counter) + ".png")
        val_added.append("images/cropped/" +
                         image_file[15:-4] + "_" + str(counter) + ".png")
        counter = counter + 1

    letter_coordinates_horizontal = []
    add_the_coordinate = False
    for x in range(1, rows - 1):
        total2 = 0
        for y in range(1, columns - 1):
            total2 = total2 + sum(image_segmented[x][y])

        if total2 > 0 and add_the_coordinate is False:
            # letter_coordinates_horizontal.append([])
            letter_coordinates_horizontal.append(x)
            add_the_coordinate = True
        elif total2 > 0 and add_the_coordinate:
            letter_coordinates_horizontal.append(x)
        elif total2 <= 0:
            add_the_coordinate = False

    for j in val_added:
        img = Image.open(j)
        width, height = img.size
        cropp = img.crop((0, letter_coordinates_horizontal[0] - 5, width - 1,
                          letter_coordinates_horizontal[len(letter_coordinates_horizontal) - 1]))
        cropp.save(j)

    val_added = None
    return
