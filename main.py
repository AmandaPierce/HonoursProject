# from PIL import Image
from preprocessing import startImagePreprocessing
from segmentation import segment_table_cells, performCharacterSegmentation
from characterSegmentation import identify_and_extract_characters
from keras_cnn import train_cnn
from recognition import predictCharacters, predict_character
import os
import glob

if __name__ == '__main__':
    # startImagePreprocessing()
    # segment_table_cells()
    # performCharacterSegmentation()
    # identify_and_extract_characters('images/cropped/final209.png')
    # predictCharacters("images/cropped/char/2_final209.png")
    # train_cnn()
    predict_character()

    '''data_path = os.path.join("images/integers/", '*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        predictCharacters(f1)'''
