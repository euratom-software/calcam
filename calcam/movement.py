import copy

import numpy as np
import cv2
from deepmatching import deepmatching

from .pointpairs import PointPairs
from .calibration import Fitter
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
    original_points = np.array(original_pointpairs.image_points)
    new_points = np.squeeze(cv2.perspectiveTransform(original_points,homography))

    new_pp = PointPairs()

    for i in range(original_pointpairs.get_n_pointpairs()):
        new_pp.add_pointpair(original_pointpairs.object_points[i],[new_points[i,:]])

    return new_pp


def get_homography_deepmatching(ref_image,image,scale_limit=None,angle_limit=None,im_scale=1.,score_tol=50):
    """
    Given two images, return an estimated homography matrix for the transform
    from the new image to the reference image using DeepMatching. NOTE: DeepMatching
    can segfault if it runs out memory.

    Parameters:

        ref_image (np.ndarray) : Array containing the reference image.

        image (np.ndarray)     : Array containing the new image

        scale_limit (float)    : Limit to apply to DeepMatching's scale

        angle_limit (float)    : Angle limit in degrees for DeepMatching

        im_scale (float)       : Scale factor of these input images relative to \
                                 the homography's final application. For example if \
                                 the input images have been downsampled to half their \
                                 original size, set im_scale to 0.5.

        score_tol (int)        : Tolerance for DDScore calculation.

    Returns:

        3x3 ndarray      : Homography matrix.

        float            : DDScore which indicates how well the homography appears \
                           to improve the image alignment. 0 = No change; > 0 = improvement, \
                           < 0 = it makes things even worse.
    """

    # DeepMatching requires HxWx3 images, so create RGB from single channel or
    # remove alpha channel as necessary.
    if len(ref_image.shape) == 2:
        ref_image = np.tile(ref_image[:,:,np.newaxis],(1,1,3))
    if ref_image.shape[2] == 1:
        ref_image = np.tile(ref_image, (1, 1, 3))
    elif ref_image.shape[2] == 4:
        ref_image = ref_image[:,:,:3]

    if len(image.shape) == 2:
        image = np.tile(image[:,:,np.newaxis],(1,1,3))
    if image.shape[2] == 1:
        image = np.tile(image, (1, 1, 3))
    elif image.shape[2] == 4:
        image = image[:,:,:3]

    # A spot of pre-processing
    ref_image = cv2.medianBlur(cv2.pyrDown(ref_image),5)
    image = cv2.medianBlur(cv2.pyrDown(image),5)
    im_scale = im_scale * 0.5

    # Configure and run DeepMatching.
    options = '-nt {:d}'.format(config.n_cpus)
    if scale_limit is not None:
        options = options + ' -max_scale {:.2f}'.format(scale_limit)
    if angle_limit is not None:
        options = options + ' -rot_range -{:.1f} {:.1f}'.format(angle_limit,angle_limit)

    matches = deepmatching(ref_image,image,options)

    # Homography based on DeepMatching points
    homography,mask = cv2.findHomography(matches[:,2:4],matches[:,:2])

    # Calculate DDScore
    image_adjusted = warp_image(image,homography)
    mask = np.logical_and( np.logical_and(ref_image > 0, image > 0), image_adjusted > 0)
    diff_before = np.abs(image[mask].astype(int) - ref_image[mask].astype(int))
    diff_after = np.abs(image_adjusted[mask].astype(int) - ref_image[mask].astype(int))
    diffdiff = diff_before - diff_after
    diffdiff = diffdiff[np.abs(diffdiff) > score_tol]

    if diffdiff.size > 0:
        ddscore = np.count_nonzero(diffdiff > 0) / diffdiff.size
    else:
        ddscore = 0.5

    ddscore = (ddscore - 0.5) * 2

    # Adjust homography matrix to account for image scaling.
    homography[0,2] = homography[0,2] / im_scale
    homography[1,2] = homography[1,2] / im_scale

    return homography,ddscore


def warp_image(original_image,homography):
    """
    Warp an image based on a homography matrix.
    A rather thin OpenCV wrapper.

    Parameters:

        original_image (np.ndarray)
    """
    return cv2.warpPerspective(original_image,homography,dsize=original_image.shape[1::-1])


def adjust_calibration(calibration,image,coords='Display',on_fail='warn'):

    adjusted_calib = copy.deepcopy(calibration)
    adjusted_calib.filename = None

    ref_image = calibration.get_image(coords=coords)

    homography,ddscore = get_homography_deepmatching(ref_image,image)

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


def adjust_image(calibration,image,coords='Display',on_fail='warn'):

    ref_image = calibration.get_image(coords=coords)

    homography,ddscore = get_homography_deepmatching(ref_image, image)

    if ddscore < 0:
        if on_fail == 'raise':
            raise Exception('Image movement detection failed (DDScore = {:.2f})'.format(ddscore))
        elif on_fail == 'warn':
            print('WARNING: Image movement detection failed (DDScore = {:.2f}). No correction will be made.'.format(ddscore))
            return image

    return warp_image(image,homography)