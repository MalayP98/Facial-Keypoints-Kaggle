import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def breakPixels(image):
    pixel = []
    image = image.split()
    for i in range(len(image)):
        pixel.append(int(image[i]))

    return pixel

def showImage(image):
    image = breakPixels(image)
    image = np.array(image)
    image = image.reshape(96, 96)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")

def show_negativeImage(image):
    image = breakPixels(image)
    image = np.array(image)
    image = image - 255
    image = np.abs(image)
    image = image.reshape(96, 96)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")

def display_keypoints(keypoints):
    i = 0
    while(i < len(keypoints)):
        x = keypoints[i]
        y = keypoints[i+1]
        i = i+2
        plt.scatter(x, y, s=4)
        plt.xlim(0, 96)
        plt.ylim(96, 0)

data_frame_dataset = pd.read_csv("/home/malay/PycharmProjects/facial-keypoints-detection/training.csv")
image_dataset = pd.read_csv("/home/malay/PycharmProjects/facial-keypoints-detection/training.csv").iloc[:, -1]
keypoint_dataset = pd.read_csv("/home/malay/PycharmProjects/facial-keypoints-detection/training.csv").iloc[:, :30].values

image_number = 2
show_negativeImage(image_dataset[image_number])
display_keypoints(keypoint_dataset[image_number])
