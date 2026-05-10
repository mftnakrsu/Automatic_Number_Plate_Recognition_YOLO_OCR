import numpy as np
import csv
import uuid


def filter_text(rectangle_size, ocr_result, region_threshold):
    """
    It takes in the size of the rectangle, the OCR result, and the region threshold. It then calculates
    the length and height of the rectangle. If the length times the height divided by the rectangle size
    is greater than the region threshold, then it appends the result to the plate
    
    :param rectangle_size: the size of the rectangle that we're looking for
    :param ocr_result: the result of the OCR
    :param region_threshold: This is the threshold for the size of the region. If the region is smaller
    than this threshold, it will be ignored
    :return: the text that is found in the image.
    """

    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate


def save_results(text, csv_filename, folder_path):
    """
    It opens a csv file, creates a csv writer object, and writes a row to the csv file.
    
    :param text: the text you want to save
    :param csv_filename: the name of the csv file you want to save the results to
    :param folder_path: the path to the folder where the images are stored
    """

    with open(csv_filename, mode="a", newline="") as f:
        csv_writer = csv.writer(
            f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow([text])
