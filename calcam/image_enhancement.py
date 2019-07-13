'''
* Copyright 2015-2018 European Atomic Energy Community (EURATOM)
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
import cv2
import numpy as np
import warnings
from scipy.optimize import curve_fit

def enhance_image(image,target_msb=25,target_noise=500,tiles=20,downsample=False):

    if len(image.shape) > 2:
        mono =  False
        image_lab = cv2.cvtColor(image[:,:,:3],cv2.COLOR_RGB2LAB)
        image = image_lab[:,:,0]
    else:
        mono = True

    if downsample:
        image = cv2.pyrDown(image)

    image = cv2.medianBlur(image,ksize=3)

    test_clip_lims = [1.,5.,10.]
    contrast = []
    for cliplim in test_clip_lims:
        contrast.append( local_contrast( cv2.createCLAHE(cliplim, (tiles, tiles)).apply(image), tiles) )

    coefs = np.polyfit(contrast,test_clip_lims,2)

    best_cliplim = np.polyval(coefs,target_msb)

    result = cv2.createCLAHE(best_cliplim, (tiles, tiles)).apply(image)


    #result = cv2.bilateralFilter(result,d=-1,sigmaColor=25,sigmaSpace=25)

    test_strengths = [1.,10.,20.,30.]
    nlm_win_size = 5#int(np.mean([image.shape[0]/tiles,image.shape[1]/tiles]))
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

    if downsample:
        result = cv2.resize(result,(result.shape[1]*2,result.shape[0]*2),interpolation=cv2.INTER_NEAREST)

    if mono:
        return result
    else:
        image_lab[:,:,0] = result
        return cv2.cvtColor(image_lab,cv2.COLOR_LAB2RGB)



def local_contrast(image,tilegridsize=20):

    tile_height = int(np.ceil(image.shape[0] / tilegridsize))
    tile_width = int(np.ceil(image.shape[1] / tilegridsize))

    sb = []

    for i in range(tilegridsize):
        for j in range(tilegridsize):
            sb.append(image[i*tile_height:(i+1)*tile_height,j*tile_width:(j+1)*tile_width].std())

    return np.mean(sb)


def tan_shape(x,xscale,yscale,xshift,yshift):

    return -yscale * (np.tan( (x - xshift)*xscale) + yshift)