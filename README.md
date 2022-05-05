# Automatic Number Plate Recognition
![dataset-cover](https://user-images.githubusercontent.com/57320216/166916670-03dfabe1-8c6c-471a-875c-8715354aa957.jpg)

**Automatic Number Plate Recognition (ANPR)** is the process of reading the characters on the plate with various optical character recognition (OCR) methods by separating the plate region on the vehicle image obtained from automatic plate recognition.

## Table of Content
- [Automatic Number Plate Recognition](#automatic-number-plate-recognition)

  * [What will you learn this project ](#what-will-you-learn-this-project)
  * [Dataset](#dataset)
  * [Installation](#installation)
  * [Usage](#usage)
  * [Project architecture](#project-architecture)
  * [Some Result](#some-result)
  * [Source](#source)
  * [Licence](#licence)


## What will you learn this project 
* Custom Object Detection
* Scene Text Detection
* Scene Text Recognation
* Optic Character Recognation
* EasyOCR, PaddleOCR
* Database,CSV format
* Applying project in Real Time
* Flask
## Dataset
The dataset I use for license plate detection:  

https://www.kaggle.com/datasets/andrewmvd/car-plate-detection

## Installation
Clone repo and install requirements.txt in a Python>=3.7.0 environment.

    git clone https://github.com/mftnakrsu/Automatic-number-plate-recognition-YOLO-OCR.git  # clone
    cd Automatic-number-plate-recognition-YOLO-OCR
    pip install -r requirements.txt  # install

## Usage
After the req libraries are installed, you can run the project by main.py.

    python main.py

## Project architecture
The pipeline in the project is as follows:

- Custom object detection with plate extraction using yolov5
- Apply the extracted plate to EasyOCR and PaddleOCR
- Get plate text
- Filter text
- Write Database and CSV format
- Upload to Flask  

![images](https://github.com/mftnakrsu/Automatic-number-plate-recognition-YOLO-OCR/blob/main/imgs/flowchart.png)

## Some Result
## Source  
- https://docs.python.org/3/library/csv.html  
- https://github.com/ultralytics/yolov5  
- https://github.com/PaddlePaddle/PaddleOCR  
- https://github.com/JaidedAI/EasyOCR  
## Licence
