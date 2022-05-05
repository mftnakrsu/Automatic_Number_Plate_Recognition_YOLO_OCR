# Automatic Number Plate Recognition
![bg](https://user-images.githubusercontent.com/57320216/166912628-709313db-345d-4bc8-988c-50cfb8698bbc.jpg)

**Automatic Number Plate Recognition (ANPR)** is the process of reading the characters on the plate with various optical character recognition (OCR) methods by separating the plate region on the vehicle image obtained from automatic plate recognition.

## Table of Content

## What will you learn this project ? 
* Custom Object Detection
* Scene Text Detection
* Scene Text Recognation
* Optic Character Recognation
* EasyOCR, PaddleOCR
* Database,CSV format
* Applying project in Real Time
* Flask

## Install
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

![images](https://github.com/mftnakrsu/Automatic-number-plate-recognition-YOLO-OCR/blob/main/imgs/arcitec.png)

## Some Result
## Source
## Licence
