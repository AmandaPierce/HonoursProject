from PIL import Image
# from preprocessing import startImagePreprocessing
from pre_processing import process_image, binarization, scaleImage, BorderRemovalAlgorithm
import cv2
from segmentation import segment_table_cells, performCharacterSegmentation, identify_table, character_segmentation
from characterSegmentation import identify_and_extract_characters
from keras_cnn import train_cnn
from recognition import predictCharacters, predict_character, imageCanvasCentering
import os
import glob

if __name__ == '__main__':
    # image_without_mod = cv2.imread("images/exampleB.jpg")
    # scale, image = scaleImage(image_without_mod)
    # cv2.imwrite("images/testing.jpg", image)
    # process_image("images/testing.jpg")
    # binarization("images/testing.jpg")
    # BorderRemovalAlgorithm("images/testing.jpg")
    average_character_height = identify_table("images/testing.png")
    data_path = os.path.join("images/", '*g')
    files = glob.glob(data_path)
    final_array = []
    data = []
    curr = 0
    for f1 in files:
        if 'final' in f1:
            val = character_segmentation(f1, average_character_height)
            print(val)
            data.append(val)
    
    
            # data_path2 = os.path.join("images/chars", '*g')
            # files2 = glob.glob(data_path2)

            # for f2 in files2:
            #     if 'final' in f2:
            #         tmp = predict_character(f2)
            #         data.append(tmp)              
            #         os.rename(f2, "images/chars/done" + "_" + str(curr) + ".png")
            #         curr += 1
            
    #             tmp_str = ""
    #             for i in data:
    #                 tmp_str += str(i)

    #         final_array.append(tmp_str)
    #         data = []
    
    # for k in final_array:
    #     print(k)
            
    # cropPdfToTable("images/testing.jpg", "images/testing.png")
    # startImagePreprocessing()
    # segment_table_cells()
    # performCharacterSegmentation()
    # identify_and_extract_characters('images/cropped/final209.png')
    # predictCharacters("images/chars/final00187_1.png")
    # train_cnn()
    # predict_character()

    '''data_path = os.path.join("images/integers/", '*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        predictCharacters(f1)'''
