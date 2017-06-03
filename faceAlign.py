#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script lets you align multiple pictures of faces so that the eyes are always in the same place.
"""

__author__ = "Stefan Heinen"
__version__ = "0.0.1"
__license__ = "CC BY-NC-SA 4.0"

# TODO: process one image after the other, process unscaled images

import os.path
import logging

import math, cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def blue(s):
    return bcolors.OKBLUE + s + bcolors.ENDC


def red(s):
    return bcolors.FAIL + s + bcolors.ENDC


def green(s):
    return bcolors.OKGREEN + s + bcolors.ENDC


def findEyes(images, cascade, scaleFactor=1.07, minNeighbours=9):
    for i, image in enumerate(images):
        image['eyeCoordinates'] = []
        logging.info('Finding eyes for ({:04d}/{:04d}) "{}")'.format(i, len(images), image['srcPath']))
        faces = image['faceAreas']
        for (x, y, w, h) in faces:
            roi = image['npImageArray'][y:y + h, x:x + w]

            eyes = cascade.detectMultiScale(roi, scaleFactor, minNeighbours)
            for (ex, ey, ew, eh) in eyes:
                exc = x + ex + ew / 2
                eyc = y + ey + eh / 2
                image['eyeCoordinates'].append((exc, eyc))

        logging.info('\t"{} found")'.format(len(image['eyeCoordinates'])))
        image['eyeCoordinates'].sort(key=lambda coord: coord[0])


def testFindEyes(images, cascade, scaleFactorList, minNeighboursList):
    results = []
    for image in images[:]:
        logging.info('image: {}'.format(image['srcPath']))
        for scaleFactor in scaleFactorList:
            logging.info('\tscaleFactor: {:03.2f}'.format(scaleFactor))
            for minNeighbours in minNeighboursList:
                findEyes([image], cascade, scaleFactor, minNeighbours)
                logging.info(
                    '\t\tminNeighbours: {:03.2f}. Found: {}'.format(minNeighbours, len(image['eyeCoordinates'])))
                results.append((scaleFactor, minNeighbours, image['eyeCoordinates']))

    return results


def findFaces(images, cascade, scaleFactor=1.3, minNeighbours=5):
    for i, image in enumerate(images):
        image['faceAreas'] = []
        logging.info('Finding face for ({:04d}/{:04d}) "{}")'.format(i, len(images), image['srcPath']))
        faces = cascade.detectMultiScale(image['npImageArray'], scaleFactor, minNeighbours)
        for (x, y, w, h) in faces:
            image['faceAreas'].append((x, y, w, h))

        logging.info('\t"{} found")'.format(len(image['faceAreas'])))


def findTransforms(images, referenceEyepairCoordinates):
    logging.info('Finding transforms. Reference eyepair position: {}, {}'.format(*referenceEyepairCoordinates))

    def dotproduct(a, b):
        return sum([a[i] * b[i] for i in range(len(a))])

    # Calculates the size of a vector
    def veclength(a):
        return sum([a[i] ** 2 for i in range(len(a))]) ** .5

    # Calculates the angle between two vectors including direction
    def angle(a, b):
        dp = dotproduct(a, b)
        la = veclength(a)
        lb = veclength(b)
        costheta = dp / (la * lb)
        cp = b[1] * a[0] - a[1] * b[0]
        if cp > 0:
            dir = 1
        else:
            dir = -1
        if costheta >= 1:
            return 0
        return dir * math.degrees(math.acos(costheta))

    def findTranslationLeftEye(destPos, srcPos):
        return destPos[0] - srcPos[0], destPos[1] - srcPos[1]

    destEyePair = referenceEyepairCoordinates
    destEye0Eye1Vect = (destEyePair[1][0] - destEyePair[0][0], destEyePair[1][1] - destEyePair[0][1])

    transforms = []
    for i, image in enumerate(images):
        logging.info('Finding transform for ({:04d}/{:04d}) "{}")'.format(i, len(images), image['srcPath']))
        srcEyePair = image['eyeCoordinates']
        translation = findTranslationLeftEye(destEyePair[0], srcEyePair[0])
        srcEye0Eye1Vect = (srcEyePair[1][0] - srcEyePair[0][0], srcEyePair[1][1] - srcEyePair[0][1])
        rotation = angle(destEye0Eye1Vect, srcEye0Eye1Vect)
        scaleFactor = veclength(destEye0Eye1Vect) / veclength(srcEye0Eye1Vect)

        transform = (destEyePair[0], translation, rotation, scaleFactor)
        logging.info('\t"{}")'.format(transform))
        transforms.append(transform)
    return transforms


def applyTransform(image, transform):
    logging.info('Applying transform to "{}")'.format(image['srcPath']))
    rows, cols, colors = image['npImageArray'].shape
    M = np.float32([[1, 0, transform[1][0]], [0, 1, transform[1][1]]])
    dst = cv2.warpAffine(image['npImageArray'], M, (cols, rows))

    rotationM = cv2.getRotationMatrix2D(transform[0], float(transform[2]), float(transform[3]))
    image['npImageArray'] = cv2.warpAffine(dst, rotationM, (cols, rows))
    logging.info('\tDONE')


def loadImages(paths):
    images = []
    for i, path in enumerate(paths):
        logging.info('Loading image ({:04d}/{:04d}) "{}"'.format(i, len(paths), path))
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        im = cv2.resize(im, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
        images.append({'srcPath': path, 'npImageArray': im.copy()})
    return images


def removeImagesNotTwoEyes(images):
    for image in list(images):
        if len(image['eyeCoordinates']) is not 2:
            logging.warning(
                '{} eyes found in image "{}". Deleting.'.format(len(image['eyeCoordinates']), image['srcPath']))
            images.remove(image)


def drawEyePositions(images):
    for imageIndex, image in enumerate(images):
        for eyeCoordinateIndex, eyeCoordinate in enumerate(image['eyeCoordinates']):
            cv2.circle(image['npImageArray'], eyeCoordinate, 2, (0, 0, 255), -1)
            cv2.putText(image['npImageArray'], "Eye {}.{}".format(imageIndex, eyeCoordinateIndex),
                        eyeCoordinate,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)


if __name__ == '__main__':
    import argparse
    scriptPath = os.path.dirname(os.path.realpath(__file__))

    # command line argument magic
    parser = argparse.ArgumentParser(description="This script takes images of faces and tries to align the eyes")
    parser.add_argument('-o', '--outputDir')
    parser.add_argument('images', nargs='+')
    parser.add_argument('-p', '--outputPrefix', default='faceAligned_')
    parser.add_argument('-r', '--referenceImage', help='Image from which to take the eye position.')
    parser.add_argument('-ep', '--eyePosition', nargs=4, help='Where to place the eyes. Requires four coordinates:'
                                                              'x1 y1 x2 y2')

    parser.add_argument('-fc', '--faceCascade', default=os.path.join(scriptPath, 'haarcascade_frontalface_default.xml'),
                        help='The file containing the Haarcascade for detecting the face.')
    parser.add_argument('-ec', '--eyeCascade', default=os.path.join(scriptPath, 'haarcascade_eye.xml'),
                        help='The file containing the Haarcascade for detecting the eyes.')

    parser.add_argument('-dd', '--drawDebug', action="store_true",
                        help='Set to yes to draw debug information into images.')
    parser.add_argument('-ecsf', '--eyeCascadeScaleFactor', default=1.07,
                        help='The scaleFactor parameter for the eye Haarcascade.')
    parser.add_argument('-ecmn', '--eyeCascadeMinimumNeighbours', default=9,
                        help='The minimumNeighbours parameter for the eye Haarcascade.')
    parser.add_argument('-fcsf', '--faceCascadeScaleFactor', default=1.3,
                        help='The scaleFactor parameter for the eye Haarcascade.')
    parser.add_argument('-fcmn', '--faceCascadeMinimumNeighbours', default=5,
                        help='The minimumNeighbours parameter for the eye Haarcascade.')

    parser.add_argument('-fe', '--findEyes', action="store_true", default=False,
                        help='Print the position of detected eyes')
    parser.add_argument('-ff', '--findFaces', action="store_true", default=False,
                        help='Print the position of detected face(s)')
    parser.add_argument('-fep', '--findEyeCascadeParameters', action="store_true", default=False,
                        help='Test different sets of parameters for the eye haarcascade.'
                             'This iterates over possible combinations of "scaleFactors" and "minNeighbours"'
                             'and prints how often 2 eyes were detected in all the input images with each combination.')
    parser.add_argument('-feps', '--findEyeCascadeParametersScaleFactorRange', default=['1.07', '1.2', '0.01'], nargs=3,
                        help='Set the testing range for the scaleFactor parameter. '
                             'Expects 3 floats: start, end and stop')
    parser.add_argument('-fepm', '--findEyeCascadeParametersMinimumNeighboursRange', default=['1', '15', '1'], nargs=3,
                        help='Set the testing range for the minimumNeighbours parameter. '
                             'Expects 3 floats: start, end and stop')
    parser.add_argument('-a', '--align', action="store_true",
                        help='Align all images.')
    parser.add_argument('-av', '--average', action="store_true",
                        help='Generate an average image. Requires -a')

    args = parser.parse_args()

    drawDebug = args.drawDebug
    inputImagePaths = args.images

    eyeC_scaleF = float(args.eyeCascadeScaleFactor)
    eyeC_minNeighbours = int(args.eyeCascadeMinimumNeighbours)
    faceC_scaleF = float(args.faceCascadeScaleFactor)
    faceC_minNeighbours = int(args.faceCascadeMinimumNeighbours)

    # cascade initialization
    faceCascadePath = args.faceCascade
    eyeCascadePath = args.eyeCascade
    face_cascade = cv2.CascadeClassifier(faceCascadePath)
    eye_cascade = cv2.CascadeClassifier(eyeCascadePath)

    testFindEyes_scaleFactorRange = np.arange(float(args.findEyeCascadeParametersScaleFactorRange[0]),
                                              float(args.findEyeCascadeParametersScaleFactorRange[1]),
                                              float(args.findEyeCascadeParametersScaleFactorRange[2]))
    testFindEyes_minimumNeighboursRange = np.arange(int(args.findEyeCascadeParametersMinimumNeighboursRange[0]),
                                                    int(args.findEyeCascadeParametersMinimumNeighboursRange[1]),
                                                    int(args.findEyeCascadeParametersMinimumNeighboursRange[2]))

    if args.referenceImage is not None:
        referenceImage = loadImages([args.referenceImage])[0]
        findFaces([referenceImage], face_cascade)
        if len(referenceImage['faceAreas']) is not 1:
            logging.error('{} faces found in reference image "{}". Aborting.'.format(len(referenceImage['faceAreas']),
                                                                                     referenceImage['srcPath']))
            raise Exception

        findEyes([referenceImage], eye_cascade)
        if len(referenceImage['eyeCoordinates']) is not 2:
            logging.error(
                '{} eyes found in reference image "{}". Aborting.'.format(len(referenceImage['eyeCoordinates']),
                                                                          referenceImage['srcPath']))
            raise Exception

        referenceEyepairCoordinates = referenceImage['eyeCoordinates']

    if args.eyePosition is not None:
        referenceEyepairCoordinates = [(int((args.eyePosition[0])), int(args.eyePosition[1])),
                                       (int((args.eyePosition[2])), int(args.eyePosition[3]))]

    if args.findFaces:
        images = loadImages(inputImagePaths)
        findFaces(images, face_cascade, scaleFactor=faceC_scaleF, minNeighbours=faceC_minNeighbours)

        for i in images:
            print({k: i[k] for k in ['srcPath', 'faceAreas']})

    if args.findEyes:
        images = loadImages(inputImagePaths)
        findFaces(images, face_cascade)
        findEyes(images, eye_cascade, scaleFactor=eyeC_scaleF, minNeighbours=eyeC_minNeighbours)

        for i in images:
            print({k: i[k] for k in ['srcPath', 'eyeCoordinates']})

    if args.findEyeCascadeParameters:
        images = loadImages(inputImagePaths)
        findFaces(images, face_cascade)
        results = testFindEyes(images, eye_cascade, list(testFindEyes_scaleFactorRange), list(testFindEyes_minimumNeighboursRange))

        print "Raw results:"
        print results
        print ""

        # resultsX = []
        # resultsY = []
        # for result in results:
        #     resultsX.append(result[0])
        #     resultsY.append(result[1])

        resultAcc = {}
        for (x, y, n) in results:
            if not str(x) in resultAcc:
                resultAcc[str(x)] = {}
            if not str(y) in resultAcc[str(x)]:
                resultAcc[str(x)][str(y)] = 0
            if len(n) == 2:
                resultAcc[str(x)][str(y)] += 1

        xValues = sorted(resultAcc.iterkeys())
        print '****\t',
        for y in sorted([int(i) for i in resultAcc[xValues[0]]]):
            print y, '\t',
        print 'minNeighbours'

        for x in sorted(resultAcc.iterkeys()):
            print x, '\t',
            for y in sorted([int(i) for i in resultAcc[xValues[0]]]):
                print resultAcc[x][str(y)], '\t',
            print ''
        print 'scaleFactor'

    if args.align:
        if args.outputDir is None:
            parser.error('Please specify an output dir with --outputDir')

        images = loadImages(inputImagePaths)
        findFaces(images, face_cascade, scaleFactor=faceC_scaleF, minNeighbours=faceC_minNeighbours)
        findEyes(images, eye_cascade, scaleFactor=eyeC_scaleF, minNeighbours=eyeC_minNeighbours)

        # if you have more or less than two eyes we have a problem
        removeImagesNotTwoEyes(images)

        if drawDebug:
            drawEyePositions(images)

        transforms = findTransforms(images, referenceEyepairCoordinates)

        for i in range(len(images)):
            applyTransform(images[i], transforms[i])

        for i in range(len(images)):
            outputFilename = args.outputPrefix + str(i) + '.jpg'
            outputPath = os.path.join(args.outputDir, outputFilename)
            cv2.imwrite(outputPath, images[i]['npImageArray'])

        if args.average:
            for i in range(len(images)):
                if i == 0:
                    blendedImage = images[i]['npImageArray']
                else:
                    blendedImage = cv2.addWeighted(images[i]['npImageArray'], 1.0 / (i + 1.0), blendedImage, i / (i + 1.0), 0.0)

            outputFilename = 'Average.jpg'
            outputPath = os.path.join(args.outputDir, outputFilename)
            cv2.imwrite(outputPath, blendedImage)


    exit(0)

    for i in range(len(images)):
        applyTransform(images[i], transforms[i])
        # if k == 0:
        #     blendedImage = transformedImage
        # else:
        #     print k
        #     print 1.0 / (k + 1.0)
        #     print k / (k + 1.0)
        #     blendedImage = cv2.addWeighted(transformedImage, 1.0 / (k + 1.0), blendedImage, k / (k + 1.0), 0.0)

    for i in range(len(images)):
        outputFilename = args.outputPrefix + str(i) + '.jpg'
        outputPath = os.path.join(args.outputDir, outputFilename)
        cv2.imwrite(outputPath, images[i]['npImageArray']);


        # for i in range(1000):
        #     index = i % len(images)
        #     cv2.imshow('transformed {}'.format(index), images[index]['npImageArray'])
        #     cv2.waitKey(200)
        #     cv2.destroyAllWindows()
        # cv2.waitKey(0)


        # # command line argument magic
        # parser = argparse.ArgumentParser(
        #     description="This script takes a directory, scans it recursively and puts the media files into "
        #                 "an html file for viewing")
        # parser.add_argument('-x', '--excludeDirs', nargs='+', default=[])
        # parser.add_argument('-bf', '--blacklistFilePatterns', nargs='+', default=[])
        # parser.add_argument('-wf', '--whitelistFilePatterns', nargs='+', default=['*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]',
        #                                                                           '*.[Pp][Nn][Gg]', '*.[Gg][Ii][Ff]',
        #                                                                           '*.[Mm][Pp]4', '*.[Mm][Oo][Vv]'])
        # parser.add_argument('-t', '--template', default='templates/overview.html.template')
        # parser.add_argument('--allowUserCat', default='no',
        #                     help='Normally all pictures which have a canon user category set are excluded. '
        #                          'Set this to true to include them')
        # parser.add_argument('-iGps', '--interpolateGps', default='no', help='Set to yes if you want to calculate'
        #                                                                     'interpolated gps positions for pictures'
        #                                                                     'without gps data.')
        # parser.add_argument('-f', '--filter', nargs='+', default=[])
        # parser.add_argument('-s', '--sortBy', default='')
        # parser.add_argument('-g', '--grouping', default='allInOne=default')
        # parser.add_argument('-sg', '--sortGroups', default='A#name')
        # parser.add_argument('-gtf', '--groupsToFiles', default='groupname')
        # parser.add_argument('-b', '--breadcrumbs', nargs='+', default=[])
        # parser.add_argument('-p', '--pageTitlePrefix', default='')
        # parser.add_argument('-o', '--outputFolder')
        # parser.add_argument('dir', nargs='+')
        # args = parser.parse_args()
        #
        # if args.outputFolder:
        #     outdir = os.path.abspath(os.path.expanduser(args.outputFolder))
        # else:
        #     outdir = os.path.abspath(os.path.expanduser(args.dir[0]))
        #
        # if args.allowUserCat.lower() in ("yes", "true", "t", "1"):
        #     allowUserCat = True
        # else:
        #     allowUserCat = False
        #
        # if args.interpolateGps.lower() in ("yes", "true", "t", "1"):
        #     interpolateGps = True
        # else:
        #     interpolateGps = False
        #
        # importParameters = {
        #     'excludeDirs': args.excludeDirs,
        #     'whitelistFilePatterns': args.whitelistFilePatterns,
        #     'blacklistFilePatterns': args.blacklistFilePatterns,
        #     'filters': args.filter,
        #     'grouping': args.grouping,
        #     'sortGroups': args.sortGroups,
        #     'groupsToFiles': args.groupsToFiles,
        #     'sortBy': args.sortBy,
        #     'pageTitlePrefix': args.pageTitlePrefix,
        #     'breadcrumbs': args.breadcrumbs,
        #     'interpolateGpsPosition': interpolateGps
        # }
        #
        # createGalleryHtml(args.dir, args.template, outdir, allowUserCat, importParameters=importParameters)
