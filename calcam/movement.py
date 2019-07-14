import copy
import json

import numpy as np
import cv2
try:
    from deepmatching import deepmatching
except:
    deepmatching = None

from .pointpairs import PointPairs
from .calibration import Calibration, Fitter
from . import config, misc
from .image_enhancement import enhance_image



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

    if deepmatching is None:
        raise Exception('DeepMatching python interface not available!')

    # Enhance the images and ensure they're in the correct format
    ref_image = enhance_image(ref_image,downsample=True)
    image = enhance_image(image,downsample=True)

    # Configure and run DeepMatching.
    options = '-nt {:d} -ngh_rad 20'.format(config.n_cpus)

    matches = deepmatching(image,ref_image,options)

    return filter_points(matches[:,2:4]*2,matches[:,:2]*2)


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



def find_pointpairs_opticalflow(ref_image,image):
    """
    Auto-detect matching points in a pair of images using Optical Flow.

    Parameters:

        ref_image (np.ndarray) : Array containing the reference image.
        image (np.ndarray)     : Array containing the moved image

    Returns:

        np.ndarray : Nx2 matrix of x,y image coordinates in the reference image
        np.ndarray : Nx2 matrix of corresponding coordinates in the moved image.

    """
    # Enhance the images and ensure they're in the correct format
    ref_image = enhance_image(ref_image,downsample=True)
    image = enhance_image(image,downsample=True)

    ref_points = cv2.goodFeaturesToTrack(ref_image[:,:,0],maxCorners=100,qualityLevel=0.1,minDistance=50).astype(np.float32)

    # Run optical flow
    new_points,status,err = cv2.calcOpticalFlowPyrLK(ref_image,image,ref_points,np.zeros(ref_points.shape,dtype=ref_points.dtype))

    new_points = new_points[status == 1,:]
    ref_points = ref_points[status == 1,:]

    return filter_points(2*ref_points,2*new_points)


def detect_movement(ref,moved):
    '''
    Attempt to auto-detect image movement between two images and return a
    MovementCorrection object representing the movement.

    Paremeters:

        ref (np.ndarray or calcam.Calibration)    : Reference image or calibration to align to. This can \
                                                    be either an array containg a refrence image, or a \
                                                    calcam calibrationn containing an image.

        mmoved (np.ndarray or calcam.Calibration) : Moved image or calibration to align. This can \
                                                    be either an array containg a refrence image, or a \
                                                    calcam calibrationn containing an image.

        correction (MovementCorrection)           : Existing movement correction object to start from.

    Returns:

        MovementCorrection : Movement correction object representing the movement correction.

    '''
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

    ref_points, new_points = find_pointpairs_opticalflow(ref_im, moved_im)

    if ref_points.shape[0] == 0:
        raise DetectionFailedError('Could not auto-detect a good set of matching points in these two images. Consider using manual movement correction instead.')

    m = np.matrix(cv2.cv2.estimateAffinePartial2D(new_points, ref_points)[0])

    if m[0, 0] is None:
        raise DetectionFailedError('Could not determine image movement from auto-detected points. Consider using manual movement correction instead.')

    mov_correction = MovementCorrection(m, ref_im.shape[:2], ref_points, new_points,'Auto-generated by {:s} on {:s} at {:s}'.format(misc.username, misc.hostname, misc.get_formatted_time()))

    if mov_correction.get_ddscore(ref_im,moved_im) > 0:
        return mov_correction
    else:
        raise DetectionFailedError('Could not determine image movement from auto-detected points. Consider using manual movement correction instead.')



def manual_movement(ref,moved,correction=None):
    '''
    Create a movement correction manually using a GUI tool.
    See the online documentation for the GUI user guide.

    Paremeters:

        ref (np.ndarray or calcam.Calibration)    : Reference image or calibration to align to. This can \
                                                    be either an array containg a refrence image, or a \
                                                    calcam calibrationn containing an image.

        mmoved (np.ndarray or calcam.Calibration) : Moved image or calibration to align. This can \
                                                    be either an array containg a refrence image, or a \
                                                    calcam calibrationn containing an image.

        correction (MovementCorrection)           : Existing movement correction object to start from.

    Returns:

        MovementCorrection : Movement correction object representing the movement correction.

    '''

    try:
        from . import gui
    except:
        raise Exception('Calcam GUI module not available. Manual movement correction requires the GUI module!')

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

    if correction is not None:
        correction.warp_moved_to_ref(moved_im)

    retcode,dialog = gui.open_window(gui.ImageAlignDialog,ref_im,moved_im,correction)

    if retcode == gui.qt.QDialog.Accepted:
        return dialog.transform
    else:
        return None


def update_calibration(calibration, moved_image, mov_correction, image_src=None):
    """
    Update a given calibration to account for image movement.

    Parameters:

        calibration (calcam.Calibration)    : Calibration to update.
        moved_image (np.ndarray)            : Moved image
        mov_correction (MovementCorrection) : Movement correction object representing the movement between \
                                              the original calibrated image and the moved image.
        image_src (string)                  : Human-readable description of where the moved image comes from

    Returns:

        calcam.Calibration : Updated calibration object for the moved image

    """

    calib_shape = tuple(calibration.geometry.get_display_shape()[::-1])
    movement_shape = tuple(mov_correction.im_shape)
    im_shape = tuple(moved_image.shape[:2])

    if not calib_shape == movement_shape == im_shape:
        raise ValueError('Calibration image display shape ({:d} x {:d}), moved image shape ({:d} x {:d}) and movement correction image shape ({:d} x {:d} must be the same!'.format(calib_shape[1],calib_shape[0],im_shape[1],im_shape[0],movement_shape[1],movement_shape[0]))

    new_calib = copy.deepcopy(calibration)

    # Update image and subview mask
    subview_mask = calibration.get_subview_mask(coords='Display')
    subview_names = calibration.subview_names
    subview_mask = mov_correction.warp_ref_to_moved(subview_mask,interp='nearest',fill_edges=True)[0].astype(np.uint8)
    if image_src is None:
        image_src = 'Updated by {:s} on {:s} at {:s}'.format(misc.username,misc.hostname,misc.get_formatted_time())

    new_calib.set_image(moved_image, image_src, coords='Display', transform_actions=calibration.geometry.get_transform_actions(), subview_mask=subview_mask, pixel_aspect=calibration.geometry.pixel_aspectratio, subview_names=subview_names, pixel_size=calibration.pixel_size, offset=calibration.geometry.offset)

    # Update point pairs
    old_pp = calibration.pointpairs
    new_pp = PointPairs()

    for i in range(old_pp.get_n_pointpairs()):
        pp = []
        for subview in range(calibration.n_subviews):
            if old_pp.image_points[i][subview] is not None:
                pp.append(mov_correction.ref_to_moved_coords(*old_pp.image_points[i][subview]))
            else:
                pp.append(None)

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
        x, y image translation in pixels.
        '''
        return self.matrix[0, 2], self.matrix[1, 2]

    @property
    def rotation(self):
        '''
        Image rotation in degrees.
        '''
        return 180 * np.arctan2(self.matrix[1, 0], self.matrix[0, 0]) / 3.14159

    @property
    def scale(self):
        '''
        Image scaling.
        '''
        return np.sqrt(self.matrix[1, 0] ** 2 + self.matrix[0, 0] ** 2)

    def warp_moved_to_ref(self, image, interp='linear', fill_edges=False):
        '''
        Warp a moved image to align with the reference one.

        Parameters:
            image (np.ndarray) : Moved image to warp
            interp (string)    : Interpolation method to use, can be 'linear' or 'nearest'
        Returns:

            np.ndarray : Warped image
            np.ndarray : Boolean mask the same shape as the image, indicating which pixels contain
                         valid image data (True) and which do not (False)
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
        Warp a reference-aligned image to align with the moved one.

        Parameters:
            image (np.ndarray) : Image to warp
            interp (string)    : Interpolation method to use, can be 'linear' or 'nearest'

        Returns:

            np.ndarray : Warped image
            np.ndarray : Boolean mask the same shape as the image, indicating which pixels contain
                         valid image data (True) and which do not (False)
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

        x = self.matrix[0,0]*x + self.matrix[0,1]*y + self.matrix[0,2]
        y = self.matrix[1, 0] * x + self.matrix[1, 1] * y + self.matrix[1, 2]

        return x,y

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

        x = matrix[0,0]*x + matrix[0,1]*y + matrix[0,2]
        y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]

        return x,y

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
