# Importing the necessary libraries for the program to run.
from ai.ai_model import load_yolov5_model
from ai.ai_model import detection

from helper.params import Parameters
from helper.general_utils import filter_text
from helper.general_utils import save_results

from ai.ocr_model import easyocr_model_load
from ai.ocr_model import easyocr_model_works
from utils.visual_utils import *

import cv2
from datetime import datetime

# Loading the parameters from the params.py file.
params = Parameters()


if __name__ == "__main__":

    # Loading the model and labels from the ai_model.py file.
    model, labels = load_yolov5_model()
    # Capturing the video from the webcam.
    camera = cv2.VideoCapture(0)
    # Loading the model for the OCR.
    text_reader = easyocr_model_load()

    while 1:

        # Reading the video from the webcam.
        ret, frame = camera.read()
        if ret:

            # Detecting the text from the image.
            detected, _ = detection(frame, model, labels)
            # Reading the text from the image.
            resulteasyocr = text_reader.readtext(
                detected
            )  # text_read.recognize() , you can use cropped plate image or whole image
            # Filtering the text from the image.
            text = filter_text(params.rect_size, resulteasyocr, params.region_threshold)
            # Saving the results of the OCR in a csv file.
            save_results(text[-1], "ocr_results.csv", "Detection_Images")
            print(text)
            cv2.imshow("detected", detected)

        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
