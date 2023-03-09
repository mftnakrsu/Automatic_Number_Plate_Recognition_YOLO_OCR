import cv2
import os
import keras_ocr
import numpy as np
import pytesseract
import easyocr
import matplotlib.pyplot as plt
from utils.params import Parameters

params = Parameters()


def easyocr_model_load():
    """
    It takes in an image and returns the text in the image
    :return: The text_reader is being returned.
    """

    text_reader = easyocr.Reader(["en"])  # Initialzing the ocr
    return text_reader


def easyocr_model_works(text_reader, images, visualization=False):
    """
    It takes a list of images and returns a list of texts
    
    :param text_reader: The text reader object
    :param images: list of images
    :param visualization: If True, it will show the images with the bounding boxes and the text,
    defaults to False (optional)
    """

    texts = list()
    for i in range(len(images)):
        results = text_reader.recognize(
            images[i]
        )  # reader.recognize sadece recognize, text detection yok
        for (bbox, text, prob) in results:
            texts.append(text)
        if visualization:
            plt.imshow(images[i])
            plt.title("{} Image".format(str(i)))
            plt.show()
    return texts


def pytesseract_model_works(images, visualization=False):
    """
    It takes in a list of images and returns a list of predictions
    
    :param images: list of images
    :param visualization: If True, it will show the image and the predicted text, defaults to False
    (optional)
    """

    tesseract_preds = []
    for img in images:
        tesseract_preds.append(pytesseract.image_to_string(img))

    for i in range(len(images)):
        print(tesseract_preds[i])

        if visualization:
            plt.imshow(images[i])
            plt.title("{} Image".format(str(i)))
            plt.show()
