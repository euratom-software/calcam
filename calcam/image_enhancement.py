import warnings

import cv2
import numpy as np
from scipy.optimize import curve_fit

def enhance_image(image,target_msb=25,target_noise=500,tiles=20,downsample=False):
    '''
    Pre-process a given image to make it suitable for auto point detection.
    '''

    # OpenCV will require that the image is an unsigned int dtype.
    if image.dtype not in [np.uint8, np.uint16]:
        if image.max() > 255:
            image = image.astype('uint16')
        elif image.max() > 1:
            image = image.astype('uint8')
        else:
            image = (2**16 - 1) * image / image.max()
            image =  image.astype('uint16')


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
        contrast.append( local_contrast( cv2.createCLAHE(cliplim, (tiles, tiles)).apply(image), tiles) )

    coefs = np.polyfit(contrast,test_clip_lims,2)

    best_cliplim = np.polyval(coefs,target_msb)

    result = cv2.createCLAHE(best_cliplim, (tiles, tiles)).apply(image)

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

    return result



def local_contrast(image,tilegridsize=20):
    '''
    Return a measure of the local contrast in a givem image
    '''
    tile_height = int(np.ceil(image.shape[0] / tilegridsize))
    tile_width = int(np.ceil(image.shape[1] / tilegridsize))

    sb = []

    for i in range(tilegridsize):
        for j in range(tilegridsize):
            sb.append(image[i*tile_height:(i+1)*tile_height,j*tile_width:(j+1)*tile_width].std())

    return np.mean(sb)


def tan_shape(x,xscale,yscale,xshift,yshift):
    '''
    Tan fit function for use in image preprocessing
    '''
    return -yscale * (np.tan( (x - xshift)*xscale) + yshift)