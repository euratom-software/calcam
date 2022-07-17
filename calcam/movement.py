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
Module for camera movement detection and compensation.
Much of this is based on the ideas and algorithms developed by:
Sam Van Stroud, Jamie McGowan, Augustin Marignier, Emily Nurse & Christian Gutschow
in a collaboration between UKAEA and University College London.
"""
import copy
import json

import numpy as np
import cv2

from .pointpairs import PointPairs
from .calibration import Calibration, Fitter
from . import misc
from .image_enhancement import enhance_image, scale_to_8bit




def filter_points(ref_points, new_points, n_points=50, err_limit=10):
    """
    Given sets of auto-detected corresponding points in two images,
    filter the list of points to get rid of obviously bad points.

    Parameters:
        ref_points (np.ndarray) : Nx2 array of detected x,y coordinates in the reference image
        new_points (np.ndarray) : Nx2 array of detected corresponding x,y coordinates in the movedd image
        n_points (int)          : The function will return at most this many points
        err_limit (float)       : Limit on the re-projection error above which points are always discounted. \
                                  This is a distance in pixels, default is 10.

    Returns:
        np.ndarray  : Nx2 array of filtered x,y coordinates in the reference image
        np.ndarray  : Nx2 array of filtered x,y coordinates in the moved image
    """
    transform = np.matrix(cv2.cv2.estimateAffinePartial2D(new_points, ref_points)[0])

    if transform[0,0] is None:
        return np.array([]),np.array([])

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



def find_pointpairs(ref_image,image):
    """
    Auto-detect matching points in a pair of images using sparse Optical Flow.

    Parameters:

        ref_image (np.ndarray) : Array containing the reference image.
        image (np.ndarray)     : Array containing the moved image

    Returns:

        np.ndarray : Nx2 matrix of x,y image coordinates in the reference image
        np.ndarray : Nx2 matrix of corresponding coordinates in the moved image.

    """
    # Enhance the images and ensure they're in the correct format
    ref_image = enhance_image(ref_image,downsample=True,median=True,bilateral=True)
    image = enhance_image(image,downsample=True,median=True,bilateral=True)

    # Does this work in all cases?
    ref_image = ref_image.astype(np.uint8)
    image = image.astype(np.uint8)

    ref_points = cv2.goodFeaturesToTrack(ref_image[:,:,0].astype(np.float32),maxCorners=100,qualityLevel=0.1,minDistance=50).astype(np.float32)

    # Run optical flow
    new_points,status,err = cv2.calcOpticalFlowPyrLK(ref_image,image,ref_points,np.zeros(ref_points.shape,dtype=ref_points.dtype))

    new_points = new_points[status == 1,:]
    ref_points = ref_points[status == 1,:]

    if np.count_nonzero(status) > 3:

        return filter_points(2*ref_points,2*new_points)

    else:

        return np.array([]),np.array([])


def detect_movement(ref,moved):
    """
    Attempt to auto-detect image movement between two images and return a
    MovementCorrection object representing the movement. If the movement cannot
    be successfully determined, raises calcam.movement.DetectionFailedError

    Parameters:

        ref (np.ndarray or calcam.Calibration)    : Reference image or calibration to align to. This can \
                                                    be either an array containg a refrence image, or a \
                                                    calcam calibrationn containing an image.

        moved (np.ndarray or calcam.Calibration) : Moved image or calibration to align. This can \
                                                    be either an array containg a refrence image, or a \
                                                    calcam calibrationn containing an image.


    Returns:

        MovementCorrection : Movement correction object representing the movement correction.

    """
    if isinstance(ref,Calibration):
        ref_im = ref.get_image(coords='Display')
    else:
        ref_im = ref

    if isinstance(moved,Calibration):
        moved_im = moved.get_image(coords='Display')
    else:
        moved_im = moved

    if ref_im.shape[:2] != moved_im.shape[:2]:
        raise ValueError('Moved image has different dimensions ({:d}x{:d}) to reference image ({:d}x{:d})! The two images must have the same dimensions.'.format(moved_im.shape[1],moved_im.shape[0],ref_im.shape[1],ref_im.shape[0]))

    ref_points, new_points = find_pointpairs(ref_im, moved_im)

    if ref_points.shape[0] == 0:
        raise DetectionFailedError('Could not auto-detect a good set of matching points in these two images. Consider using manual movement correction instead.')

    m = np.matrix(cv2.cv2.estimateAffinePartial2D(new_points, ref_points)[0])

    if m[0, 0] is None:
        raise DetectionFailedError('Could not determine image movement from automatically. Consider using manual movement correction instead.')

    mov_correction = MovementCorrection(m, ref_im.shape[:2], ref_points, new_points,'Auto-generated by {:s} on {:s} at {:s}'.format(misc.username, misc.hostname, misc.get_formatted_time()))

    if mov_correction.get_ddscore(ref_im,moved_im) >= 0:
        return mov_correction
    else:
        raise DetectionFailedError('Could not determine image movement automatically. Consider using manual movement correction instead.')


def manual_movement(ref,moved,correction=None,parent_window=None):
    '''
    Determine camera movement (semi-)manually using a GUI tool.
    See the :doc:`gui_movement` GUI doc page for the GUI user guide.

    Paremeters:

        ref (np.ndarray or calcam.Calibration)    : Reference image or calibration to align to. This can \
                                                    be either an array containg a refrence image, or a \
                                                    calcam calibrationn containing an image.

        mmoved (np.ndarray or calcam.Calibration) : Moved image or calibration to align. This can \
                                                    be either an array containg a refrence image, or a \
                                                    calcam calibrationn containing an image.

        correction (MovementCorrection)           : Existing movement correction object to start from.

        parent_window (QWidget)                   : If being called from a QT GUI window class, a reference to \
                                                    the parent window must be passed. If it is not, the parent \
                                                    window might irrecoverably freeze when the movement dialog is closed.

    Returns:

        MovementCorrection or NoneType : Either the image transofmration, or None is the user does not define one.

    '''

    try:
        from . import gui
    except Exception:
        from . import no_gui_reason
        raise Exception('Cannot start movement correction GUI because the Calcam GUI module not available - {:s}'.format(no_gui_reason))

    if isinstance(ref,Calibration):
        ref_im = ref.get_image(coords='Display')
    else:
        ref_im = scale_to_8bit(ref)

    if isinstance(moved,Calibration):
        moved_im = moved.get_image(coords='Display')
    else:
        moved_im = scale_to_8bit(moved)

    if ref_im.shape[:2] != moved_im.shape[:2]:
        raise ValueError('Moved image has different dimensions ({:d}x{:d}) to reference image ({:d}x{:d})! The two images must have the same dimensions.'.format(moved_im.shape[1],moved_im.shape[0],ref_im.shape[1],ref_im.shape[0]))

    if correction is not None:
        correction.warp_moved_to_ref(moved_im)

    if parent_window is None:
        retcode,dialog = gui.open_window(gui.ImageAlignDialog,ref_im,moved_im,correction)
    else:
        dialog = gui.ImageAlignDialog(None,parent_window,ref_im,moved_im,correction)
        retcode = dialog.exec()

    if retcode == gui.qt.QDialog.Accepted:
        return dialog.transform
    else:
        return correction


def phase_correlation_movement(ref_im,moved_im):

    ref_x = np.linspace(0,ref_im.shape[1],4)
    ref_y = np.linspace(0,ref_im.shape[0],4)
    ref_x,ref_y = np.meshgrid(ref_x,ref_y)
    ref_x = ref_x.flatten()
    ref_y = ref_y.flatten()

    ref_points = np.hstack((ref_x[:,np.newaxis],ref_y[:,np.newaxis]))

    if len(ref_im.shape) == 3:
        ref_im = cv2.cvtColor(ref_im,cv2.COLOR_RGB2GRAY)

    if len(moved_im.shape) == 3:
        moved_im = cv2.cvtColor(moved_im,cv2.COLOR_RGB2GRAY)

    movxy, c = cv2.phaseCorrelate(np.float32(ref_im), np.float32(moved_im))

    new_points = np.zeros_like(ref_points)
    new_points[:,0] = ref_points[:,0] + movxy[0]
    new_points[:,1] = ref_points[:,1] + movxy[1]

    m = np.matrix(cv2.cv2.estimateAffinePartial2D(new_points, ref_points)[0])

    mov_correction = MovementCorrection(m, ref_im.shape[:2], ref_points, new_points,'Auto-generated by {:s} on {:s} at {:s}'.format(misc.username, misc.hostname, misc.get_formatted_time()))

    return mov_correction


def update_calibration(calibration, moved_image, mov_correction, image_src=None, coords='Display'):
    """
    Update a given calibration to account for image movement. This currently only supports
    point pair fitting calibrations, not manual alignment calibrations.

    Parameters:

        calibration (calcam.Calibration)    : Calibration to update.
        moved_image (np.ndarray)            : Moved image
        mov_correction (MovementCorrection) : Movement correction object representing the movement between \
                                              the original calibrated image and the moved image.
        image_src (string)                  : Human-readable description of where the moved image comes from, \
                                              for data provenance tracking.
        coords (string)                     : 'Display' or 'Original', whether the movement correction  and moved image \
                                              are in the calibration's display or original image orientation.


    Returns:

        calcam.Calibration : Updated calibration object for the moved image

    """

    if calibration._type != 'fit':
        raise ValueError('This function only supports point fitting calibrations at the moment!')

    if coords.lower() == 'display':
        calib_shape = tuple(calibration.geometry.get_display_shape()[::-1])
    elif coords.lower() == 'original':
        calib_shape = tuple(calibration.geometry.get_original_shape()[::-1])
    else:
        raise ValueError('"coords" argument must be either "Display" or "Original"!')

    movement_shape = tuple(mov_correction.im_shape)
    im_shape = tuple(moved_image.shape[:2])

    if not calib_shape == movement_shape == im_shape:
        raise ValueError('Calibration image display shape ({:d} x {:d}), moved image shape ({:d} x {:d}) and movement correction image shape ({:d} x {:d} must be the same!'.format(calib_shape[1],calib_shape[0],im_shape[1],im_shape[0],movement_shape[1],movement_shape[0]))

    new_calib = copy.deepcopy(calibration)

    # Update image and subview mask
    subview_mask = calibration.get_subview_mask(coords='Display')
    subview_names = calibration.subview_names
    subview_mask = mov_correction.warp_ref_to_moved(subview_mask,interp='nearest',fill_edges=True)[0].astype(np.int8)
    if image_src is None:
        image_src = 'Updated by {:s} on {:s} at {:s}'.format(misc.username,misc.hostname,misc.get_formatted_time())

    new_calib.set_image(moved_image, image_src, coords='Display', transform_actions=calibration.geometry.get_transform_actions(), subview_mask=subview_mask, pixel_aspect=calibration.geometry.pixel_aspectratio, subview_names=subview_names, pixel_size=calibration.pixel_size, offset=calibration.geometry.offset)

    # Update point pairs
    old_pp = calibration.pointpairs
    new_pp = PointPairs()

    pos_lim = np.array(calibration.geometry.get_display_shape()) - 0.5
    for i in range(old_pp.get_n_pointpairs()):
        pp = []
        for subview in range(calibration.n_subviews):
            if old_pp.image_points[i][subview] is not None:
                if coords.lower() == 'display':
                    new_coords = mov_correction.ref_to_moved_coords(*old_pp.image_points[i][subview])
                elif coords.lower() == 'original':
                    orig_coords = calibration.geometry.display_to_original_coords(*old_pp.image_points[i][subview])
                    new_coords = mov_correction.ref_to_moved_coords(*orig_coords)
                    new_coords = calibration.geometry.original_to_display_coords(*new_coords)

                if np.all( np.array(new_coords) >= 0) and np.all(np.array(new_coords) < pos_lim):
                    pp.append(new_coords)
                else:
                    pp.append(None)
            else:
                pp.append(None)

        if all(p is None for p in pp):
            continue
        else:
            new_pp.add_pointpair(old_pp.object_points[i], pp)

    new_calib.set_pointpairs(new_pp, history=[calibration.history['pointpairs'][0], 'Updated based on movement correction by {:s} on {:s} at {:s}'.format(misc.username,misc.hostname,misc.get_formatted_time())])

    for subview in range(calibration.n_subviews):
        fitter = Fitter()
        fitter.set_fitflags_strings(calibration.view_models[subview].fit_options)
        fitter.set_image_shape(calibration.geometry.get_display_shape())
        fitter.set_pointpairs(new_pp, subview)
        for _,ipp in calibration.intrinsics_constraints:
            fitter.add_intrinsics_pointpairs(ipp,subview)

        new_calib.set_fit(subview, fitter.do_fit())

    return new_calib


class MovementCorrection:
    '''
    Class to represent a geometric transform from a moved image
    to a reference image. This type of object is returned by image movement
    correction related functions.

    Parameters:

        matrix (np.matrix)        : 2x3 or 3x3 Affine or projective transform matrix
        im_shape (tuple)          : Image array dimensions (rows,cols) to which this transform applies
        ref_points (np.ndarray)   : Nx2 array containing coordinates of points on the reference image
        moved_points (np.ndarray) : Nx2 array containing coordinates of corresponding points on the moved image
        src (string)              : Human-readable description of how the transform was created.
    '''

    def __init__(self,matrix,im_shape,ref_points,moved_points,src):

        if matrix.shape[0] == 2:
             mat_ = np.matrix(np.zeros((3, 3)))
             mat_[:2, :] = matrix
             mat_[2, 2] = 1.
             self.matrix = mat_
        else:
            self.matrix = matrix

        self.im_shape = im_shape
        self.ref_points = ref_points
        self.moved_points = moved_points
        self.history = src

    @property
    def translation(self):
        '''
        x, y translation in pixels to go from the moved to reference image.
        '''
        return self.matrix[0, 2], self.matrix[1, 2]

    @property
    def rotation(self):
        '''
        Image rotation to go from the moved to reference image in degrees clockwise.
        '''
        return 180 * np.arctan2(self.matrix[1, 0], self.matrix[0, 0]) / 3.14159

    @property
    def scale(self):
        '''
        Scale factor to go from moved to reference image.
        '''
        return np.sqrt(self.matrix[1, 0] ** 2 + self.matrix[0, 0] ** 2)

    def warp_moved_to_ref(self, image, interp='linear', fill_edges=False):
        '''
        Warp a moved image to align with the reference one. Note: this can also be used
        on binned or englarged images, with respect to the one used for the original movement
        correction determination, provided they are scaled proportionally i.e. have the same
        aspect ratio as the originals.

        Parameters:
            image (np.ndarray) : Moved image to warp
            interp (string)    : Interpolation method to use, can be 'linear' or 'nearest'
            fill_edges (bool)  : Whether to fill the warped image edges with a repetition \
                                 of the edge pixels from the image (if True), or leave un-filled \
                                 images edges as 0 value (if False; this is the default).


        Returns:
            np.ndarray : Two ndarrays: the warped image, and a boolean mask the same shape \
            as the image indicating which pixels contain valid image data (True) and which do not (False).
        '''

        scaley = image.shape[0] / self.im_shape[0]
        scalex = image.shape[1] / self.im_shape[1]

        if np.abs(1 - scalex / scaley) > 0.005:
            raise Exception('The provided image is the wrong shape!')

        mat = self.matrix.copy()
        mat[0, 2] = mat[0, 2] * scalex
        mat[1, 2] = mat[1, 2] * scalex

        if interp == 'linear':
            flags = cv2.INTER_LINEAR
        elif interp == 'nearest':
            flags = cv2.INTER_NEAREST
        else:
            raise ValueError('Interpolation method must be "linear" or "nearest"!')

        if fill_edges:
            bm = cv2.BORDER_REPLICATE
        else:
            bm = cv2.BORDER_CONSTANT

        warped_im = cv2.warpPerspective(image, mat, dsize=image.shape[1::-1], borderMode=bm, flags=flags)

        mask_im = cv2.warpPerspective(image * 0, mat, dsize=image.shape[1::-1], borderValue=255)
        mask_im = mask_im == 0

        return warped_im,mask_im

    def warp_ref_to_moved(self, image, interp='linear',fill_edges=False):
        '''
        Warp a reference-aligned image to align with the moved one. Note: this can also be used
        on binned or englarged images, with respect to the one used for the original movement
        correction determination, provided they are scaled proportionally i.e. have the same
        aspect ratio as the originals.

        Parameters:
            image (np.ndarray) : Image to warp
            interp (string)    : Interpolation method to use, can be 'linear' or 'nearest'

        Returns:

            np.ndarray : Two ndarrays: the warped image, and a boolean mask the same shape \
            as the image indicating which pixels contain valid image data (True) and which do not (False).
        '''

        scaley = image.shape[0] / self.im_shape[0]
        scalex = image.shape[1] / self.im_shape[1]

        if np.abs(1 - scalex / scaley) > 0.005:
            raise Exception('The image aspect ratio is not correct for this movement correction!')

        mat = self.matrix.copy()
        mat[0, 2] = mat[0, 2] * scalex
        mat[1, 2] = mat[1, 2] * scalex

        if interp == 'linear':
            flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        elif interp == 'nearest':
            flags = cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP
        else:
            raise ValueError('Interpolation method must be "linear" or "nearest"!')

        if fill_edges:
            bm = cv2.BORDER_REPLICATE
        else:
            bm = cv2.BORDER_CONSTANT

        warped_im = cv2.warpPerspective(image, mat, dsize=image.shape[1::-1], borderMode=bm, flags=flags)

        mask_im = cv2.warpPerspective(image * 0, mat, dsize=image.shape[1::-1], borderValue=255)
        mask_im = mask_im == 0

        return warped_im,mask_im

    def get_ddscore(self,ref_im,moved_im,tol=50):
        '''
        Get the DDScore for this movement correction when applied to the given image pair.
        DDScore is a score developed by Van-Stroud et.al. which indicates how much better
        the alignment of two input images after application of the correction.
        The score ranges from -1 to 1, where negative values indicate the alignment gets worse,
        0 is no change or undetermined, and positive values are an improvement.

        Parameters:

            ref_im (np.ndarray)   : Reference aligned image

            moved_im (np.ndarray) : Moved image

            tol (float)           : Tolerance

        Returns:

            float : DDscore in the range -1,1

        '''

        if len(ref_im.shape) > 2:
            ref_im = cv2.cvtColor(ref_im[:, :, :3], cv2.COLOR_RGB2LAB)[:,:,0]

        if len(moved_im.shape) > 2:
            moved_im = cv2.cvtColor(moved_im[:, :, :3], cv2.COLOR_RGB2LAB)[:,:,0]

        image_adjusted, mask = self.warp_moved_to_ref(moved_im)
        diff_before = np.abs(moved_im[mask].astype(int) - ref_im[mask].astype(int))
        diff_after = np.abs(image_adjusted[mask].astype(int) - ref_im[mask].astype(int))
        diffdiff = diff_before - diff_after
        diffdiff = diffdiff[np.abs(diffdiff) > tol]

        if diffdiff.size > 0:
            ddscore = np.count_nonzero(diffdiff > 0) / diffdiff.size
        else:
            ddscore = 0.5

        ddscore = (ddscore - 0.5) * 2

        return ddscore

    def moved_to_ref_coords(self,x,y):
        '''
        Given image coordinates in the moved image, return the corresponding
        coordinates in the reference image.

        Parameters:
            x (np.ndarray) : Array of x (horizontal) pixel coordinates in the moved image
            y (np.ndarray) : Array of y (vertucal) pixel coordinates in the moved image

        Returns:
            np.ndarray : Array of x (horizontal) pixel coordinates in the reference image
            np.ndarray : Array of y (vertical) pixel coordinates in the reference image
        '''

        xnew = self.matrix[0,0]*x + self.matrix[0,1]*y + self.matrix[0,2]
        ynew = self.matrix[1, 0] * x + self.matrix[1, 1] * y + self.matrix[1, 2]

        return xnew,ynew

    def ref_to_moved_coords(self,x,y):
        '''
        Given image coordinates in the reference image, return the corresponding
        coordinates in the moved image.

        Parameters:
            x (np.ndarray) : Array of x (horizontal) pixel coordinates in the reference image
            y (np.ndarray) : Array of y (vertucal) pixel coordinates in the reference image

        Returns:
            np.ndarray : Array of x (horizontal) pixel coordinates in the moved image
            np.ndarray : Array of y (vertical) pixel coordinates in the moved image
        '''

        matrix = np.linalg.inv(self.matrix)

        xnew = matrix[0,0]*x + matrix[0,1]*y + matrix[0,2]
        ynew = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]

        return xnew,ynew

    def save(self,filename):
        '''
        Save the correction to a given filename.

        Paremeters:

            filaneme (string) : Filename to which to save the correction. The file extension is .cmc; \
                                if this is not included in the given filename it is added.
        '''

        if not filename.endswith('.cmc'):
            filename = filename.split('.')[0] + '.cmc'

        with open(filename,'w') as outfile:
            json.dump({'transform_matrix':self.matrix.tolist(),'im_array_shape':self.im_shape,'ref_points':self.ref_points.tolist(),'moved_points':self.moved_points.tolist(),'history':self.history},outfile,indent=4)

    @classmethod
    def load(cls,filename):
        '''
        Load a movement correction from a .cmc file on disk.

        Parameters:

            filename (string) : File name to load from.
        '''

        with open(filename,'r') as readfile:
            loaded_dict = json.load(readfile)

        return cls(np.matrix(loaded_dict['transform_matrix']),loaded_dict['im_array_shape'],np.array(loaded_dict['ref_points']),np.array(loaded_dict['moved_points']),loaded_dict['history'])


class DetectionFailedError(Exception):
    pass
