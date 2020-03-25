import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def breakPixels(image):
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
    while(i < len(keypoints)):
        x.append(keypoints[i])
        y.append(keypoints[i+1])
        i = i+2

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


data_frame_dataset = pd.read_csv("/home/malay/PycharmProjects/Facial-Keypoints-Kaggle/facial-keypoints-detection/training.csv")
image_dataset = pd.read_csv("/home/malay/PycharmProjects/Facial-Keypoints-Kaggle/facial-keypoints-detection/training.csv").iloc[:, -1]
keypoint_dataset = pd.read_csv("/home/malay/PycharmProjects/Facial-Keypoints-Kaggle/facial-keypoints-detection/training.csv").iloc[:, :30].values

image_number = 2
showImage(image_dataset[image_number])
showNegImage(image_dataset[image_number])
processed_image(image_dataset[image_number])
display_keypoints(keypoint_dataset[image_number])

data_frame_dataset.isnull().sum()





