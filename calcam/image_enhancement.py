'''
* Copyright 2015-2019 European Atomic Energy Community (EURATOM)
*
* Licensed under the EUPL, Version 1.1 or - as soon they
will be approved by the European Commission - subsequent
versions of the EUPL (the "Licence");
* You may not use this work except in compliance with the
Licence.
* You may obtain a copy of the Licence at:
*
* https://joinup.ec.europa.eu/software/page/eupl
*
* Unless required by applicable law or agreed to in
writing, software distributed under the Licence is
distributed on an "AS IS" basis,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied.
* See the Licence for the specific language governing
permissions and limitations under the Licence.
'''

"""
Module for image enhancement to make identifiable / trackable features in an image clearer.
Much of this is based on the ideas and algorithms developed by:
Sam Van Stroud, Jamie McGowan, Augustin Marignier, Emily Nurse & Christian Gutschow
in a collaboration between UKAEA and University College London.
"""
import warnings

import cv2
import numpy as np
from scipy.optimize import curve_fit



def scale_to_8bit(image,cutoff=0.999):

    # If we already have an 8-bit image, don't do anything
    if image.dtype == np.uint8:
        return image

    # If we have a multi-channel image
    if len(image.shape) == 3:

        # Remove any transparency channel if there is one
        if image.shape[2] == 4:
            image = image[:, :, :-1]

        # If it's actually monochrome, throw away the colour
        if (np.diff(image.reshape(image.shape[2],-1),axis=0)==0).all():
            image = image[:,:,0]

    # Do the subtraction & normalisation as float to avoid
    # overflows or quantisation problems.
    image = image.astype(np.float32)
    image[np.isfinite(image) == False] = 0

    sorted = np.sort(image.flatten())
    maxval = sorted[int(sorted.size * cutoff)]
    minval = sorted[int(sorted.size * (1 - cutoff))]

    # Scale the image to its max & min
    image = np.maximum(0,image - minval)
    image = np.minimum(1,image / maxval)

    # Final 8 bit output
    return np.uint8(255 * image)


def enhance_image(image,target_msb=25,target_noise=500,tiles=(20,20),downsample=False,median=False,bilateral=False):
    """
    Enhance details in a given image. Used both for visual enhancement
    in the Calcam GUIs and as a pre-processing step for automatic camera
    movement detection.

    Parameters:
        image (np.ndarray)   : Image to enhance
        target_msb (float)   : Controls contrast enhancement. Higher numbers \
                               give more contrast enhancement.
        target_noise (float) : Controls level of de-noising. Lower values \
                               give more agressive de-noising.
        tiles (tuple of int) : Number of tiles in horizontal and vertical directions \
                               for local histogram equilisation.
        downsample (bool)    : Whether or not to downsample the image by a factor 2 \
                               using cv2.pyrDown(). Using this greatly increases the \
                               success of automatic point detection, but makes the images \
                               look worse by eye due to the lower resolution.

    Returns:

        np.ndarray : The processed image.
    """

    # Make sure we have an 8-bit unisnged int image.
    if image.dtype != np.uint8:
        image = scale_to_8bit(image)


    if len(image.shape) > 2:
        mono = False
        image_lab = cv2.cvtColor(image[:,:,:3],cv2.COLOR_RGB2LAB)
        image = image_lab[:,:,0]
    else:
        mono = True


    if downsample:
        image = cv2.pyrDown(image)
        if not mono:
            image_lab = cv2.resize(image_lab,(image.shape[1],image.shape[0]))
    else:
        target_noise = target_noise * 2

    if median:
        image = cv2.medianBlur(image,ksize=3)

    test_clip_lims = [1.,5.,10.]
    contrast = []

    for cliplim in test_clip_lims:
        contrast.append( local_contrast( cv2.createCLAHE(cliplim, tiles).apply(image), tiles) )

    result = image.copy()
    if max(contrast) - min(contrast) > 0:
        coefs = np.polyfit(contrast,test_clip_lims,2)
        best_cliplim = np.polyval(coefs,target_msb)
        if best_cliplim > 0:
            result = cv2.createCLAHE(best_cliplim, tiles).apply(image)


    if bilateral:
        result = cv2.bilateralFilter(result,d=-1,sigmaColor=25,sigmaSpace=25)

    starting_noise = cv2.Laplacian(image,cv2.CV_64F).var()
    nlm_win_size = int(np.mean([image.shape[0] / 100, image.shape[1] / 100]))

    if starting_noise > target_noise and nlm_win_size > 1 and min(image.shape) > 512:
        test_strengths = [1.,10.,20.,30.]

        noise = []
        for strength in test_strengths:
            noise.append(cv2.Laplacian(cv2.fastNlMeansDenoising(result,h=strength,searchWindowSize=nlm_win_size),cv2.CV_64F).var())

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tanparams,_ = curve_fit(tan_shape,noise,test_strengths,p0=[1/np.max(noise),20,500,10],bounds=([1e-5,1e-4,-1000,-100],[3.14159/(np.max(noise) - np.min(noise)),100,1000,100]))

        best_strength = tan_shape(target_noise,*tanparams)

        if best_strength > 0:
            result = cv2.fastNlMeansDenoising(result,h=best_strength,searchWindowSize=nlm_win_size)
    if mono:
        result = np.tile(result[:,:,np.newaxis],(1,1,3))
    else:
        image_lab[:,:,0] = result
        result = cv2.cvtColor(image_lab,cv2.COLOR_LAB2RGB)

    return result



def local_contrast(image,tilegridsize=(20,20)):
    """
    Return a measure of the local contrast in a given image

    Parameters:
        image (np.ndarray)          : Image to process
        tilegridsize (tuple of int) : Number of tiles in horizontal and \
                                      vertical directions to split the image in to \
                                      for local contrast measurements.

    Returns:

        float: Local contrast parameter
    """
    tile_height = int(np.ceil(image.shape[0] / tilegridsize[1]))
    tile_width = int(np.ceil(image.shape[1] / tilegridsize[0]))
    sb = []

    for i in range(tilegridsize[1]):
        for j in range(tilegridsize[0]):
            sb.append(image[i*tile_height:(i+1)*tile_height,j*tile_width:(j+1)*tile_width].std())

    return np.nanmean(sb)


def tan_shape(x,xscale,yscale,xshift,yshift):
    """
    Tan fit function for use in image enhancement.

    Parameters:

        x (float or array) : Independent variable
        xscale (float)     : Horizontal scaling factor for tan function
        yscale (float)     : Vertical scaling factor for tan function
        xshift (float)     : Horizontal shift distance
        yshift (float)     : Vertical shift distance

    Returns:
        float : -yscale * (np.tan( (x - xshift)*xscale) + yshift)
    """
    return -yscale * (np.tan( (x - xshift)*xscale) + yshift)