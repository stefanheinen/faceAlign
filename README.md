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
	1 	10 	11 	12 	13 	14 	2 	3 	4 	5 	6 	7 	8 	9 	minNeighbours
1.07 	0 	2 	2 	3 	2 	2 	0 	2 	2 	2 	2 	1 	1 	2
1.08 	1 	2 	2 	3 	3 	2 	1 	2 	2 	3 	3 	3 	3 	2
1.09 	1 	2 	3 	3 	2 	2 	0 	2 	2 	2 	2 	2 	2 	2
1.1 	1 	2 	2 	3 	3 	3 	2 	2 	2 	2 	2 	2 	3 	3
1.11 	1 	3 	3 	3 	2 	2 	1 	2 	2 	2 	2 	2 	3 	3
1.12 	1 	2 	2 	2 	2 	2 	1 	1 	1 	1 	2 	2 	2 	3
1.13 	1 	2 	2 	2 	2 	3 	0 	1 	1 	1 	2 	3 	3 	3
1.14 	2 	3 	3 	3 	3 	3 	3 	2 	3 	3 	2 	2 	2 	3
1.15 	0 	2 	2 	2 	2 	3 	0 	1 	2 	2 	2 	3 	3 	3
1.16 	3 	2 	3 	3 	3 	3 	2 	2 	2 	2 	2 	2 	2 	2
1.17 	1 	3 	3 	3 	3 	3 	2 	3 	3 	3 	2 	2 	2 	3
1.18 	2 	2 	2 	2 	2 	2 	3 	3 	2 	2 	2 	2 	2 	2
1.19 	1 	2 	3 	3 	3 	3 	1 	2 	3 	3 	2 	2 	2 	2
scaleFactor
```
On the x-Axis you have the scaleFactor, on the y-Axis the minNeighbours. The result numbers represent the number of images in which two eyes were detected. Choose the highest number and take the corresponding scaleFactor and minNeighbours value.


You can now use these in the alignment process:
```
faceAlign.py --align --eyeCascadeScaleFactor 1.14 --eyeCascadeMinimumNeighbours 6 -r referenceImage.jpg --outputDir out image1.jpg image2.jpg image3.jpg
```

## License
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License - see the [LICENSE.txt](LICENSE.txt) file for details.
