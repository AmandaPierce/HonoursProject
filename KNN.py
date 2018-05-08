# import the necessary packages
from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from PIL import Image, ImageFilter
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2


def imageCanvasCentering():
        im = Image.open("images/testing_1.png").convert('L')
        width = float(im.size[0])
        height = float(im.size[1])
        newImage = Image.new('L', (20, 20), (255))  # creates white canvas of 28x28 pixels

        if width > height:  # check which dimension is bigger
            # Width is bigger. Width becomes 20 pixels.
            nheight = int(round((12.0 / width * height), 0))  # resize height according to ratio width
            if (nheight == 0):  # rare case but minimum is 1 pixel
                nheigth = 1
                # resize and sharpen
            img = im.resize((12, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((20 - nheight) / 2), 0))  # caculate horizontal pozition
            newImage.paste(img, (4, wtop))  # paste resized image on white canvas
        else:
            # Height is bigger. Heigth becomes 20 pixels.
            nwidth = int(round((12.0 / height * width), 0))  # resize width according to ratio height
            if (nwidth == 0):  # rare case but minimum is 1 pixel
                nwidth = 1
            # resize and sharpen
            img = im.resize((nwidth, 12), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((20 - nwidth) / 2), 0))  # caculate vertical pozition
            newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

        newImage.save("images/training_1.png")


imageCanvasCentering()
oriimg = cv2.imread('images/training_1.png')

grey_image = cv2.cvtColor(oriimg, cv2.COLOR_BGR2GRAY)
bw_original = cv2.bitwise_not(grey_image)
bw_original = cv2.resize(bw_original, (20, 20))
cv2.imshow("A", bw_original)
cv2.waitKey(0)

newImage = Image.new('L', (20, 20), (255))


bw_original_test = np.array(bw_original)

img = cv2.imread('images/digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
x = np.array(cells)

train = x[:,:50].reshape(-1,400).astype(np.float32)
test = x[:,50:100].reshape(-1,400).astype(np.float32)
bw_original_test = bw_original_test.reshape(-1,400).astype(np.float32)

k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()


from sklearn.model_selection import train_test_split

'''dataset = datasets.fetch_mldata("MNIST Original")

mnist = datasets.load_digits()
print(np.array(dataset.data))
# take the MNIST data and construct the training and testing split, using 75% of the
# data for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                                  mnist.target, test_size=0.25, random_state=42)

# now, let's take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
                                                                test_size=0.1, random_state=84)

# show the sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))'''

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []

'''# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 30, 2):
    # train the k-Nearest Neighbor classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train, train_labels)

    # evaluate the model and update the accuracies list
    score = model.score(test, test_labels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

# find the value of k that has the largest accuracy
i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                       accuracies[i] * 100))'''

# re-train our classifier using the best k value and predict the labels of the
# test data
model = KNeighborsClassifier(n_neighbors=1)
model.fit(train, train_labels)
predictions = model.predict(test)

# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(test_labels, predictions))

# loop over a few random digits
for i in list(map(int, np.random.randint(0, high=len(test_labels), size=(5,)))):
    # grab the image and classify it
    image = bw_original_test
    prediction = model.predict(image.reshape(1, -1))[0]

    # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
    # then resize it to 32 x 32 pixels so we can see it better
    image = image.reshape((20, 20)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

    # show the prediction
    print("I think that digit is: {}".format(prediction))
    cv2.imshow("Image", image)
    cv2.waitKey(0)

