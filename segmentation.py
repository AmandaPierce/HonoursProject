from PIL import Image
import os
import subprocess
import sys
import glob

# a line has to have at least 300 continuous pixels to be considered a line
HORIZONTAL_THRESHHOLD = 300
VERTICAL_THRESHOLD = 300

def segmentTableCells(png):

    image = Image.open(png)
    pixels = image.load()
    width, height = image.size

    # Get start and end pixels of horizontal line
    horizontal_lines = []
    for y in range(height):
        x1, x2 = (None, None)
        line = 0
        run = 0
        for x in range(width):
            if pixels[x, y] == (0, 0, 0):
                line = line + 1
                if not x1: x1 = x
                x2 = x
            else:
                if line > run:
                    run = line
                    line = 0
        if run > HORIZONTAL_THRESHHOLD:
            horizontal_lines.append((x1, y, x2, y))

    # Get start and end pixels of vertical line
    vertical_lines = []
    for x in range(width):
        y1, y2 = (None, None)
        line = 0
        run = 0
        for y in range(height):
            if pixels[x, y] == (0, 0, 0):
                line = line + 1
                if not y1: y1 = y
                y2 = y
            else:
                if line > run:
                    run = line
                line = 0
        if run > VERTICAL_THRESHOLD:
            vertical_lines.append((x, y1, x, y2))

    # Get top-left and bottom-right coordinates for each column
    cols_coordinates = []
    for i in range(1, len(vertical_lines)):
        if vertical_lines[i][0] - vertical_lines[i - 1][0] > 1:
            cols_coordinates.append(
                (vertical_lines[i - 1][0], vertical_lines[i - 1][1], vertical_lines[i][2], vertical_lines[i][3]))

    # Get top-left and bottom-right coordinates for each row
    rows_coordinates = []
    for i in range(1, len(horizontal_lines)):
        if horizontal_lines[i][1] - horizontal_lines[i - 1][3] > 1:
            rows_coordinates.append((horizontal_lines[i - 1][0],
                                     horizontal_lines[i - 1][1], horizontal_lines[i][2], horizontal_lines[i][3]))

    # Get top-left and bottom-right coordinates for each cell
    table_cells = {}
    for i, row in enumerate(rows_coordinates):
        table_cells.setdefault(i, {})
        for j, col in enumerate(cols_coordinates):
            x1 = col[0]
            y1 = row[1]
            x2 = col[2]
            y2 = row[3]
            table_cells[i][j] = (x1, y1, x2, y2)

    data = []
    for row in range(len(rows_coordinates)):
        data.append(
            [performCharacterSegmentation(image, table_cells, row, col) for col in range(len(cols_coordinates))])


def performCharacterSegmentation(image, t_cells, row, col):



