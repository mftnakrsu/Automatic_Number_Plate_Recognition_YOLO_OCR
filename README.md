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

![images](https://github.com/mftnakrsu/Automatic-number-plate-recognition-YOLO-OCR/blob/main/imgs/flowchart.png)

- Custom object detection with plate extraction using yolov5
- Apply the extracted plate to EasyOCR and PaddleOCR
- Get plate text
- Filter text
- Write Database and CSV format
- Upload to Flask  


## Some Result

* As you can see, first step is detect the plate with using Yolov5. 

![images](https://github.com/mftnakrsu/Automatic-number-plate-recognition-YOLO-OCR/blob/main/imgs/realtime.png)

* After detect plate, apply the ocr. Paddle ocr Easy ocr for recognizing plate.  

![images](https://github.com/mftnakrsu/Automatic-number-plate-recognition-YOLO-OCR/blob/main/imgs/plate_recog.jpg)

* Then write csv or database, when put it all in one.  

![images](https://github.com/mftnakrsu/Automatic-number-plate-recognition-YOLO-OCR/blob/main/imgs/all.png)

* The last step is Flask :) Actually, I didn't have time to integrate all the code in Flask. I just uploaded the yolov5 part. If you do, don't forget to pull request :)  

![images](https://github.com/mftnakrsu/Automatic-number-plate-recognition-YOLO-OCR/blob/main/imgs/flask_test.png)


## Similar work
A streamlit based implementation of Automatic Number Plate Recognition for cars and other vehicles using images or live camera feed.

![Animation](https://user-images.githubusercontent.com/29462447/168389056-9f39b89d-1221-432b-878d-578d9114d466.gif)
![live feed demo](https://user-images.githubusercontent.com/29462447/168389042-c06f3dd2-5047-4138-8c11-07372d63046a.gif)

The entire code for the webapp can be found [here.](https://github.com/prateekralhan/Streamlit-based-Automatic-Number-Plate-Recognition)


## Source  
- https://docs.python.org/3/library/csv.html  
- https://github.com/ultralytics/yolov5  
- https://github.com/PaddlePaddle/PaddleOCR
- https://medium.com/move-on-ai/yolov5-object-detection-with-your-own-dataset-6e3823a8f66b  
- https://github.com/JaidedAI/EasyOCR  
-     https://www.researchgate.net/publication/319198085_License_Number_Plate_Recognition_System_using_Entropy_basedFeatures_Selection_Approach_with_SVM/figures?lo=1&utm_source=google&utm_medium=organic

## Licence
[MIT](https://github.com/mftnakrsu/Automatic-number-plate-recognition-YOLO-OCR/blob/main/LICENSE)

## To Do 
- [ ] use fcaykon pip yolo instead of hardcoded yolo files
- [ ] hugging face
