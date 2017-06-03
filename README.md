# faceAlign
This script lets you align multiple pictures of faces so that the eyes are always in the same place.
It uses openCV to detect the face and eyes and transform the images.


## Installation

- Download the script file (faceAlign.py)
- Put the Haar-Cascades for the face and eyes in the same directory. You can use these: https://github.com/opencv/opencv/tree/master/data/haarcascades
  The cascade files have to be named "haarcascade_frontalface_default.xml" and "haarcascade_eye.xml"
