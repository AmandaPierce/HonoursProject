# Somewhat adopted from https://niektemme.com/2016/02/21/tensorflow-handwriting/

from PIL import Image, ImageFilter
from keras.models import load_model
from keras.datasets import mnist
import pandas as pd
import numpy as np
# import imutils
import cv2


def predictCharacters(filename):
    imageCanvasCentering(filename)
    # centeredIm = cv2.imread(filename)
    # cv2.imshow('center', centeredIm)
    # cv2.waitKey(0)
    return predict_character(filename)
    # print((p[0]))

def image_canvas_centering(image):
    height, width = image.shape[:2]
    new_image = Image.new('L', (28, 28), (255))
    image = Image.fromarray(image)

    if width > height:
        h = int(round((20.0 / width * height), 0))
        if (h == 0):
            h = 1
        image = image.resize((20, h), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        top = int(round(((28 - h) / 2), 0))
        new_image.paste(image, (4, top))
    else:
        w = int(round((20.0 / height * width), 0))
        if (w == 0):
            w = 1

        img = image.resize((w, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        left = int(round(((28 - w) / 2), 0))
        new_image.paste(img, (left, 4))

    return  np.array(new_image)


def imageCanvasCentering(filename):
    im = Image.open(filename).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    new_image = Image.new('L', (28, 28), (255))

    if width > height:
        h = int(round((20.0 / width * height), 0))
        if (h == 0):
            h = 1
        image = im.resize((20, h), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        top = int(round(((28 - h) / 2), 0))
        new_image.paste(image, (4, top))
    else:
        w = int(round((20.0 / height * width), 0))
        if (w == 0):
            w = 1

        img = im.resize((w, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        left = int(round(((28 - w) / 2), 0))
        new_image.paste(img, (left, 4))

    new_image.save(filename)

def predict_character(img):
    model = load_model('my_model.h5')

    # model.summary()

    # img = cv2.imread(filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)

    img = img.reshape(1, 1, 28, 28)
    img = img / 255.0

    #print(img.shape)

    predictions_single = model.predict(img)

    # print(predictions_single[0])

    maxElement = np.amax(predictions_single[0])

    # print('Max element from npy Array : ', maxElement)

    # print(np.argmax(predictions_single[0]))

    array = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
             'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w','x','y','z']

    return array[np.argmax(predictions_single[0])]


# def predict(image_list):

#     def weight_variable(shape):
#         initial = tf.truncated_normal(shape, stddev=0.1)
#         return tf.Variable(initial)

#     def bias_variable(shape):
#         initial = tf.constant(0.1, shape=shape)
#         return tf.Variable(initial)

#     def convolution2d(image, weight):
#         return tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')

#     def max_pool(x):
#         return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#     def cnn_model(img):
#         current_image = tf.reshape(img, [-1, 28, 28, 1])

#         # 32 features for each 5x5 patch
#         weight_conv1 = weight_variable([5, 5, 1, 32])
#         bias_conv1 = bias_variable([32])

#         conv1 = tf.nn.relu(convolution2d(
#             current_image, weight_conv1) + bias_conv1)
#         pool1 = max_pool(conv1)

#         weight_conv2 = weight_variable([5, 5, 32, 64])
#         bias_conv2 = bias_variable([64])

#         conv2 = tf.nn.relu(convolution2d(pool1, weight_conv2) + bias_conv2)
#         pool2 = max_pool(conv2)

#         weight_fully_connected_layer1 = weight_variable([7 * 7 * 64, 1024])
#         bias_fully_connected_layer1 = bias_variable([1024])

#         pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
#         fully_connected_layer1 = tf.nn.relu(
#             tf.matmul(pool2_flat, weight_fully_connected_layer1) + bias_fully_connected_layer1)

#         keep_prob = tf.placeholder(tf.float32)
#         fully_connected_layer1_dropout = tf.nn.dropout(
#             fully_connected_layer1, keep_prob)

#         weight_fully_connected_layer2 = weight_variable([1024, 10])
#         bias_fully_connected_layer2 = bias_variable([10])

#         final_conv = tf.nn.softmax(
#             tf.matmul(fully_connected_layer1_dropout, weight_fully_connected_layer2) + bias_fully_connected_layer2)
#         return final_conv, keep_prob

#     input_image = tf.placeholder(tf.float32, shape=[None, 784])
#     target_output_class = tf.placeholder(tf.float32, shape=[None, 10])

#     current_image = tf.reshape(input_image, [-1, 28, 28, 1])
#     final_conv, keep_prob = cnn_model(current_image)

#     init_op = tf.initialize_all_variables()
#     saver = tf.train.Saver()

#     with tf.Session() as sess:
#         sess.run(init_op)
#         saver.restore(
#             sess, "C:/Users/Amanda/PycharmProjects/jaarProjek/cnn_trainingModel.ckpt")

#         prediction = tf.argmax(final_conv, 1)
#         return prediction.eval(feed_dict={input_image: [image_list], keep_prob: 1.0}, session=sess)
