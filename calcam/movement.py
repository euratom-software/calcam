import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
import cv2
from deepmatching import deepmatching
from scipy.optimize import curve_fit

from .pointpairs import PointPairs
from .calibration import Calibration, Fitter
from . import config


def adjust_pointpairs(original_pointpairs,homography):
    """
    Given an input PointPairs object and a homography matrix,
    adjust the point pair image positions based on the homography
    and return a new PointPairs object.

    Parameters:

        original_pointpairs (calcam.PointPairs) : Starting point pairs.

        homography (np.ndarray)                 : 3x3 homography matrix.

    Returns:

        calcam.PointPairs : Adjusted PointPairs object.

    """

    if not isinstance(homography,list):
        homography = [homography] * original_pointpairs.n_subviews

    if len(homography) != original_pointpairs.n_subviews:
        raise ValueError('Different number of homographies to sub-views provided!')

    original_impoints = original_pointpairs.image_points

    new_pp = PointPairs()

    for i in range(original_pointpairs.get_n_pointpairs()):
        new_pp.add_pointpair(original_pointpairs.object_points[i],[list(np.squeeze(np.array(x[1] * np.matrix(x[0] + [1]).T))[:2]) for x in zip(original_impoints[i],homography)])

    return new_pp



def find_pointpairs_deepmatching(ref_image,image):
    """
    Given two images, return an estimated homography matrix for the transform
    from the new image to the reference image using DeepMatching. NOTE: DeepMatching
    can segfault if it runs out memory.

    Parameters:

        ref_image (np.ndarray) : Array containing the reference image.

        image (np.ndarray)     : Array containing the new image

        scale_limit (float)    : Limit to apply to DeepMatching's scale

        angle_limit (float)    : Angle limit in degrees for DeepMatching

        score_tol (int)        : Tolerance for DDScore calculation.

    Returns:

        3x3 ndarray      : Homography matrix.

        float            : DDScore which indicates how well the homography appears \
                           to improve the image alignment. 0 = No change; > 0 = improvement, \
                           < 0 = it makes things even worse.
    """


    # Enhance the images and ensure they're in the correct format
    ref_image = preprocess_image(ref_image)
    image = preprocess_image(image)

    #ax0 = plt.subplot(121)
    #ax0.imshow(ref_image)
    #ax1 = plt.subplot(122)
    #ax1.imshow(image)
    #plt.show()


    # Configure and run DeepMatching.
    options = '-nt {:d} -ngh_rad 20'.format(config.n_cpus)

    matches = deepmatching(image,ref_image,options)

    #tilesize_x = int(np.ceil(ref_image.shape[1]/n_tiles[0]))
    #tilesize_y = int(np.ceil(ref_image.shape[0]/n_tiles[1]))

    #inds = []
    #for x in range(n_tiles[0]):
    #    in_x_range = np.logical_and( matches[:,0] >= x*tilesize_x, matches[:,0] < (x+1)*tilesize_x)
    #    for y in range(n_tiles[1]):
    #        in_y_range = np.logical_and( matches[:,1] >= y*tilesize_y, matches[:,1] < (y+1)*tilesize_y)#

    #        scores = matches[:,4].copy()
    #       scores[np.invert(np.logical_and(in_x_range, in_y_range))] = 0
    #        inds.append(np.argmax(scores))

    #print(matches[:,5])
    #order = np.argsort(matches[:,5])
    #matches = matches[inds,:]

    #plt.plot(matches[:,4],'o')
    #plt.show()

    #nc = plt.get_cmap('jet',matches.shape[0])
    #for i in range(matches.shape[0]):
    #    ax0.plot(matches[i,0],matches[i,1],'o',color=nc(i),markersize=(matches[i,4]-3)*5)
    #    ax1.plot(matches[i,2],matches[i,3],'o',color=nc(i),markersize=(matches[i,4]-3)*5)


    return filter_points(matches[:,2:4]*2,matches[:,:2]*2)


def filter_points(ref_points,new_points,n_points=50,err_limit = 10):

    transform = np.matrix(cv2.estimateRigidTransform(new_points, ref_points, fullAffine=True))

    if transform[0,0] is None:
        return np.array([])

    err = []
    for pp in range(ref_points.shape[0]):
            oldpt = np.matrix(np.concatenate((new_points[pp,:], np.ones(1)))).T
            fitted_pt = transform * oldpt
            err.append( (np.sqrt( (ref_points[pp,0] - fitted_pt[0])**2 + (ref_points[pp,1] - fitted_pt[1])**2) )[0,0])

    err = np.array(err)

    order = np.argsort(err)

    ref_points = ref_points[order[:n_points],:]
    new_points = new_points[order[:n_points],:]
    err = err[order[:n_points]]

    ref_points = ref_points[err < err_limit]
    new_points = new_points[err < err_limit]

    return ref_points,new_points


def ddscore(ref_image,image,transform,tol=50):

    # Calculate DDScore
    image_adjusted,mask = warp_image(image,transform)
    diff_before = np.abs(image[mask].astype(int) - ref_image[mask].astype(int))
    diff_after = np.abs(image_adjusted[mask].astype(int) - ref_image[mask].astype(int))
    diffdiff = diff_before - diff_after
    diffdiff = diffdiff[np.abs(diffdiff) > tol]

    if diffdiff.size > 0:
        ddscore = np.count_nonzero(diffdiff > 0) / diffdiff.size
    else:
        ddscore = 0.5

    ddscore = (ddscore - 0.5) * 2

    return ddscore


def find_pointpairs_opticalflow(ref_image,image):
    """
    Given two images, return an estimated homography matrix for the transform
    from the new image to the reference image using DeepMatching. NOTE: DeepMatching
    can segfault if it runs out memory.

    Parameters:

        ref_image (np.ndarray) : Array containing the reference image.

        image (np.ndarray)     : Array containing the new image

        scale_limit (float)    : Limit to apply to DeepMatching's scale

        angle_limit (float)    : Angle limit in degrees for DeepMatching

        score_tol (int)        : Tolerance for DDScore calculation.

    Returns:

        3x3 ndarray      : Homography matrix.

        float            : DDScore which indicates how well the homography appears \
                           to improve the image alignment. 0 = No change; > 0 = improvement, \
                           < 0 = it makes things even worse.
    """



    # Enhance the images and ensure they're in the correct format
    ref_image = preprocess_image(ref_image)
    image = preprocess_image(image)

    #plt.figure()
    #ax0 = plt.subplot(121)
    #ax0.imshow(ref_image)
    #ax1 = plt.subplot(122)
    #ax1.imshow(image)
    #plt.show()

    ref_points = cv2.goodFeaturesToTrack(ref_image[:,:,0],maxCorners=100,qualityLevel=0.1,minDistance=50).astype(np.float32)

    # Run optical flow

    new_points,status,err = cv2.calcOpticalFlowPyrLK(ref_image,image,ref_points,np.zeros(ref_points.shape,dtype=ref_points.dtype))

    #dists = np.sqrt(np.sum((np.squeeze(new_points) - np.squeeze(ref_points)) ** 2, axis=1))

    #status[dists > dists.mean() + 2*dists.std()] = 0

    #ax0.plot(ref_points[status == 0,0],ref_points[status == 0,1],'ro')
    #ax1.plot(new_points[status == 0,0], new_points[status == 0, 1], 'ro')



    new_points = new_points[status == 1,:]
    ref_points = ref_points[status == 1,:]




    #ax0.plot(ref_points[:,0],ref_points[:,1],'go')
    #ax1.plot(new_points[:,0],new_points[:,1],'go')

    #plt.show()


    # Homography based on DeepMatching points
    #homography,mask = cv2.findHomography(new_points,ref_points)

    # Calculate DDScore
    #image_adjusted,mask = warp_image(image,homography)
    #diff_before = np.abs(image[mask].astype(int) - ref_image[mask].astype(int))
    #diff_after = np.abs(image_adjusted[mask].astype(int) - ref_image[mask].astype(int))
    #diffdiff = diff_before - diff_after
    #diffdiff = diffdiff[np.abs(diffdiff) > score_tol]

    #if diffdiff.size > 0:
    #    ddscore = np.count_nonzero(diffdiff > 0) / diffdiff.size
    #else:
    #    ddscore = 0.5

    #ddscore = (ddscore - 0.5) * 2

    # Adjust homography matrix to account for image scaling.
    #homography[0,2] = 2 * homography[0,2]
    #homography[1,2] = 2 * homography[1,2]

    return filter_points(2*ref_points,2*new_points)


def warp_image(original_image,transform):
    """
    Warp an image based on a given matrix.
    A rather thin OpenCV wrapper.

    Parameters:

        original_image (np.ndarray)
    """

    if transform.shape[0] == 2:
        transform_ = np.matrix(np.zeros((3,3)))
        transform_[:2,:] = transform
        transform_[2,2] = 1.
        transform = transform_

    warped_im = cv2.warpPerspective(original_image,transform,dsize=original_image.shape[1::-1])

    mask_im = cv2.warpPerspective(original_image*0,transform,dsize=original_image.shape[1::-1],borderValue=255)
    mask_im = mask_im == 0

    return warped_im,mask_im


def adjust_calibration(calibration,image,coords='Display',on_fail='warn'):

    adjusted_calib = copy.deepcopy(calibration)
    adjusted_calib.filename = None

    original_image = calibration.get_image(coords=coords)

    homography,ddscore = get_homography_opticalflow(original_image,image)
    homography = np.linalg.inv(homography)
    if ddscore < 0:
        if on_fail == 'raise':
            raise Exception('Image movement detection failed (DDScore = {:.2f})'.format(ddscore))
        elif on_fail == 'warn':
            print('WARNING: Image movement detection failed (DDScore = {:.2f}). No correction will be made.'.format(ddscore))
            return calibration

    newpp = adjust_pointpairs(calibration.pointpairs, homography)

    fitter = Fitter()
    fitter.set_fitflags_strings(calibration.view_models[0].fit_options)
    fitter.set_image_shape(calibration.geometry.get_display_shape())
    fitter.set_pointpairs(newpp)

    adjusted_calib.set_pointpairs(newpp,history=[calibration.history['pointpairs'][0],'Auto-updated based on image homography'])
    adjusted_calib.set_image(image,src='Changed using adjust_calibration()')
    adjusted_calib.set_fit(0,fitter.do_fit())

    return adjusted_calib



def preprocess_image(image,target_msb=25,target_noise=500,tiles=20,maintain_scale=False):

    if len(image.shape) > 2:
        image = cv2.cvtColor(image[:,:,:3],cv2.COLOR_RGB2LAB)[:,:,0]

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

    res = np.tile(result[:,:,np.newaxis],(1,1,3))

    if maintain_scale:
        res = cv2.resize(res,(res.shape[1]*2,res.shape[0]*2),interpolation=cv2.INTER_NEAREST)

    return res



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



def get_transform_gui(ref,new_im):

    if isinstance(ref,Calibration):
        ref_im = ref.get_image(coords='Display')
    else:
        ref_im = ref

    try:
        from .gui import qt_wrapper as qt
        from .gui.image_alignment_dialog import  ImageAlignDialog
    except:
        raise Exception('Calcam GUI module not available. This function requires the GUI!')

    app = qt.QApplication([])
    dialog = ImageAlignDialog(None, ref_im, new_im, app=app)
    retcode = dialog.exec_()

    if retcode == qt.QDialog.Accepted:
        return dialog.transform
    else:
        return None