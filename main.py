from PIL import Image
from preprocessing import startImagePreprocessing
from segmentation import segmentTableCells

if __name__ == '__main__':
    # filename = startImagePreprocessing()
    # print(filename)
    segmentTableCells("A")

