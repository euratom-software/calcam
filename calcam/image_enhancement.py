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

def enhance_image(image,target_msb=25,target_noise=500,tiles=(20,20),downsample=False):
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
            image = 255 * image / image.max()
            image = image.astype(np.uint8)


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

    image = cv2.medianBlur(image,ksize=3)

    test_clip_lims = [1.,5.,10.]
    contrast = []
    for cliplim in test_clip_lims:
        contrast.append( local_contrast( cv2.createCLAHE(cliplim, tiles).apply(image), tiles) )

    if max(contrast) - min(contrast) > 0:
        coefs = np.polyfit(contrast,test_clip_lims,2)
        best_cliplim = np.polyval(coefs,target_msb)
        result = cv2.createCLAHE(best_cliplim, tiles).apply(image)
    else:
        result = image.copy()

    #result = cv2.bilateralFilter(result,d=-1,sigmaColor=25,sigmaSpace=25)
    starting_noise = cv2.Laplacian(image,cv2.CV_64F).var()
    nlm_win_size = int(np.mean([image.shape[0] / 100, image.shape[1] / 100]))

    if starting_noise > target_noise and nlm_win_size > 1:
        test_strengths = [1.,10.,20.,30.]

        noise = []
        for strength in test_strengths:
            noise.append(cv2.Laplacian(cv2.fastNlMeansDenoising(result,h=strength,searchWindowSize=nlm_win_size),cv2.CV_64F).var())

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tanparams,_ = curve_fit(tan_shape,noise,test_strengths,p0=[1/np.max(noise),20,500,10],bounds=([1e-5,1e-4,-1000,-100],[3.14159/(np.max(noise) - np.min(noise)),100,1000,100]))

        best_strength = tan_shape(target_noise,*tanparams)

        #plt.plot(noise,test_strengths,'o')
        #nn = np.linspace(5,2000,500)
        #plt.plot(nn, tan_shape(nn,*tanparams))
        #plt.show()

        if best_strength > 0:
            result = cv2.fastNlMeansDenoising(result,h=best_strength,searchWindowSize=nlm_win_size)
    if mono:
        result = np.tile(result[:,:,np.newaxis],(1,1,3))
    else:
        image_lab[:,:,0] = result
        result = cv2.cvtColor(image_lab,cv2.COLOR_LAB2RGB)
    print(result.dtype)
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