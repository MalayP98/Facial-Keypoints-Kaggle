import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


def breakPixels(image):
    # takes string input
    pixel = []
    image = image.split()
    for i in range(len(image)):
        pixel.append(int(image[i]))

    pixel = np.array(pixel)

    return pixel


def showImage(image):
    image = breakPixels(image)
    image = image.reshape(96, 96)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")


def getNegativeImage(image):
    image = breakPixels(image)
    image = image - 255
    image = np.abs(image)

    return image


def showNegImage(image):
    image = getNegativeImage(image)
    image = image.reshape(96, 96)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")


def display_keypoints(keypoints):
    i = 0
    x = []
    y = []
    while i < len(keypoints):
        x.append(keypoints[i])
        y.append(keypoints[i + 1])
        i = i + 2

    plt.scatter(x, y, s=10)
    plt.xlim(0, 96)
    plt.ylim(96, 0)


def processed_image(image):
    image = getNegativeImage(image)

    for i in range(len(image)):
        if image[i] >= 150:
            image[i] = 255
        elif image[i] <= 90:
            image[i] = 0
        elif 91 < image[i] < 149:
            image[i] = 0

    image = image.reshape(96, 96)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")


def string2image(image_dataset):
    # takes string dataset of images(entire dataset)
    new_image_dataset = []
    for i in range(len(image_dataset)):
        pixel = breakPixels(image_dataset[i])
        new_image_dataset.append(list(pixel))

    return np.array(new_image_dataset)


def plot_n_images(nrow, ncol, image_dataset, keypoint_dataset, with_keypoints=True):
    number_of_images = nrow*ncol
    fig = plt.figure(figsize=(8,8))
    images = np.random.randint(len(image_dataset), size=number_of_images)

    for i in range(number_of_images):
        pixel = getNegativeImage(image_dataset[images[i]])
        pixel = pixel.reshape(96, 96)
        fig.add_subplot(nrow, ncol,i+1)
        plt.title("image {}".format(i+1))
        plt.axis('off')
        plt.imshow(pixel, cmap=plt.cm.gray_r)
        if with_keypoints:
            display_keypoints(keypoint_dataset[images[i]])

def select_feature( image, feature=None):
    pass

def fill_na(model):
    pass

def testPredicted(image_num, image, keypoint):
    image = image.reshape(96, 96)
    display_keypoints(yTest[image_num])
    plt.imshow(image, cmap=plt.cm.gray_r)
    display_keypoints(keypoint)

dataset = pd.read_csv("/home/malay/PycharmProjects/Facial-Keypoints-Kaggle/facial-keypoints-detection/training.csv")
dataset.isnull().sum()
dataset = dataset.dropna(axis=0)

image_dataset = dataset.iloc[:, -1].values
keypoint_dataset = dataset.iloc[:, :30].values

image_number = 2
showImage(image_dataset[image_number])
showNegImage(image_dataset[image_number])
processed_image(image_dataset[image_number])
display_keypoints(keypoint_dataset[image_number])

new_image_dataset = string2image(image_dataset)
xTrain, xTest, yTrain, yTest = train_test_split(new_image_dataset, keypoint_dataset, test_size=0.2, random_state=0)
xTrain = xTrain.reshape(len(xTrain), 96, 96, 1)
xTest = xTest.reshape(len(xTest), 96, 96, 1)

plot_n_images(4, 4, image_dataset, keypoint_dataset, with_keypoints=False)

model = Sequential()
model.add((Conv2D(32, kernel_size=3, activation='relu', input_shape=(96, 96, 1))))
model.add((MaxPool2D(2,2)))

model.add((Conv2D(64, kernel_size=2, activation='relu')))
model.add((MaxPool2D(2,2)))

model.add((Conv2D(128, kernel_size=2, activation='relu')))
model.add((MaxPool2D(2,2)))

model.add(Flatten())

model.add(Dense(500,  activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(30))

model.compile(loss="mse", optimizer="rmsprop")

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen.fit(xTrain)
model.fit_generator(datagen.flow(xTrain, yTrain, batch_size=32),
                    steps_per_epoch=len(xTrain) / 32, epochs=200)

pred = model.predict(xTest)

testPredicted(5, xTest[5], pred[5])

