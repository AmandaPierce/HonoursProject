from PIL import Image
from preprocessing import startImagePreprocessing
from segmentation import segmentTableCells
from cnn import buildTheCNN
from recognition import predictCharacters
import os
import glob

if __name__ == '__main__':
    # startImagePreprocessing()
    # segmentTableCells()
    predictCharacters("images/integers/0_1_1.png")

    '''data_path = os.path.join("images/integers/", '*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        predictCharacters(f1)'''


