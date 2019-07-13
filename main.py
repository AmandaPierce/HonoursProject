from PIL import Image
# from preprocessing import startImagePreprocessing
from pre_processing import process_image, binarization, scaleImage, BorderRemovalAlgorithm
import cv2
from segmentation import segment_table_cells, performCharacterSegmentation, identify_table
# from characterSegmentation import identify_and_extract_characters
# from keras_cnn import train_cnn
# from recognition import predictCharacters, predict_character
import os
import glob

if __name__ == '__main__':
    # image_without_mod = cv2.imread("images/ExampleB.jpg")
    # scale, image = scaleImage(image_without_mod)
    # cv2.imwrite("images/testing.jpg", image)
    # process_image("images/testing.jpg")
    # binarization("images/testing.jpg")
    # BorderRemovalAlgorithm("images/testing.jpg")
    identify_table("images/testing.png")
    # cropPdfToTable("images/testing.jpg", "images/testing.png")
    # startImagePreprocessing()
    # segment_table_cells()
    # performCharacterSegmentation()
    # identify_and_extract_characters('images/cropped/final209.png')
    # predictCharacters("images/cropped/char/2_final209.png")
    # train_cnn()
    # predict_character()

    '''data_path = os.path.join("images/integers/", '*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        predictCharacters(f1)'''
