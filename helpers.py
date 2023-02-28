import cv2
import numpy as np
import os


def load_images(mode: bool = True):
    # True for training, False for testing
    if mode:
        path = "MNIST - JPG - training"
    else:
        path = "MNIST - JPG - testing"

    images = []
    for i in range(1, 9):
        for j in os.listdir(path + "/" + str(i) + '/'):
            image = cv2.imread(path + "/" + str(i) + "/" + str(j))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.flatten()
            images.append(image)
    return np.asarray(images)


def load_labels(mode: bool = True):
    # True for training, False for testing
    if mode:
        path = "MNIST - JPG - training"
    else:
        path = "MNIST - JPG - testing"

    labels = []
    for i in range(1, 9):
        for j in os.listdir(path + "/" + str(i) + '/'):
            labels.append(i)
    return labels


def get_data(mode: bool = True):
    # True for training, False for testing
    images = load_images(mode)
    labels = load_labels(mode)
    return images, labels


def apply_noise_single_image(image, noise_amount):
    x_shape = 784

    noise = np.random.normal(0.5, scale=noise_amount, size=(x_shape))
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image


def apply_noise_to_data(images, noise_amount):
    noisy_images = []
    for image in images:
        noisy_images.append(apply_noise_single_image(
            image, noise_amount))
    noisy_images = np.asarray(noisy_images)
    return noisy_images


def normalize(data):

    return data / 255


def denormalize(data):
    return data * 255

def evaluate(model, testing, testing_noise, num_output):
    predicted = model.predict(testing_noise)
    indices =np.random.randint(size = num_output, low=0, high=len(testing))
    error = 0
    for i in range(len(testing)):
        error += (predicted - testing) **2
    error /= len(testing)
    for i in indices:
        cv2.imwrite(f"/eval/testing{i}.jpg", testing[i])
        cv2.imwrite(f"/eval/predict{i}.jpg", predicted[i])
        cv2.imwrite(f"/eval/testing_noise{i}.jpg", testing_noise[i])