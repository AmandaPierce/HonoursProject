from PIL import Image
from preprocessing import startImagePreprocessing
from segmentation import segmentTableCells
from cnn import buildTheCNN
from recognition import predictCharacters
if __name__ == '__main__':
    # filename = startImagePreprocessing()
    # print(filename)
    #segmentTableCells("A")
    # buildTheCNN()
    predictCharacters()


