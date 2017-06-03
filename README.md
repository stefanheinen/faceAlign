# faceAlign
This script lets you align multiple pictures of faces so that the eyes are always in the same place.

It uses openCV to detect the face and eyes and transform the images.


## Installation

- Download the script file (faceAlign.py)
- Put the Haar-Cascades for the face and eyes in the same directory. You can use these: https://github.com/opencv/opencv/tree/master/data/haarcascades

  The cascade files have to be named "haarcascade_frontalface_default.xml" and "haarcascade_eye.xml"
- Install OpenCV 3 (use a virtuelenv if applicable)


## Usage

### Align all images using a reference image:
```
faceAlign.py --align -r referenceImage.jpg --outputDir out image1.jpg image2.jpg image3.jpg
```


### Align all images specifying the eye position:
```
faceAlign.py --align --eyePosition 240 346 390 346 --outputDir out image1.jpg image2.jpg image3.jpg
```


### Finding eye cascade parameters.
When detecting feature with a Haarcascade there are two parameters that can me changed: the scaleFactor and the minimumNeighbours (cf. [here](https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid))
To test for the optimal values you can use the following command:
```
faceAlign.py -fep -feps 1.07 1.2 0.01 -fepm 1 15 1 image1.jpg image2.jpg image3.jpg
```
Where -feps 1.07 1.2 0.01 is the range of scaleFactor values to test. The values are start, end and step.
Same for -fepm which specifies the minimumNeighbours testing range.

This can take a while...

It will give you an output similar to this:
```
****	5 	6 	7 	8 	9 	10 	11 	12 	13 	14 	minNeighbours
1.07 	126 	132 	135 	138 	142 	139 	139 	140 	138 	136
1.08 	133 	136 	139 	139 	139 	136 	137 	138 	138 	136
1.09 	129 	132 	135 	138 	137 	137 	136 	136 	134 	133
1.1 	125 	128 	130 	131 	133 	131 	131 	135 	133 	131
1.11 	123 	137 	136 	137 	137 	135 	136 	135 	132 	129
1.12 	131 	133 	134 	134 	130 	129 	129 	126 	124 	124
1.13 	126 	128 	132 	132 	131 	130 	127 	124 	124 	123
scaleFactor
```
On the x-Axis you have the scaleFactor, on the y-Axis the minNeighbours. The result numbers represent the number of images in which two eyes were detected. Choose the highest number and take the corresponding scaleFactor and minNeighbours value.


You can now use these in the alignment process:
```
faceAlign.py --align --eyeCascadeScaleFactor 1.07 --eyeCascadeMinimumNeighbours 9 -r referenceImage.jpg --outputDir out image1.jpg image2.jpg image3.jpg
```

## License
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License - see the [LICENSE.txt](LICENSE.txt) file for details.
