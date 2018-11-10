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


"""
Camera model fitting for CalCam using OpenCV
Written by Scott Silburn, Alasdair Wynn & James Harrison 
2015-05-19
"""

import numpy as np
import cv2
import os
import json
import copy

from .io import ZipSaveFile
from scipy.ndimage.measurements import center_of_mass as CoM
from .coordtransformer import CoordTransformer
from .pointpairs import PointPairs

try:
    from .raycast import raycast_sightlines
except ImportError:
    raycast_sightlines = None

import datetime
import socket
import getpass


# This stuff is used for keeping track of calibration history
_user = getpass.getuser()
_host = socket.gethostname()

def _get_formatted_time(timestamp=None):
    
    if timestamp is None:
        when = datetime.datetime.now()
    else:
        when = datetime.datetime.fromtimestamp(timestamp)
    
    return when.strftime('%H:%M on %Y-%m-%d')


# Superclass for camera models.
class ViewModel():


    # Factory method for loading saved view model from a dictionary
    @staticmethod
    def from_dict(coeffs_dict):

        if coeffs_dict['model'] == 'perspective':
            return PerspectiveViewModel(coeffs_dict=coeffs_dict)
        elif coeffs_dict['model'] == 'fisheye':
            return FisheyeeViewModel(coeffs_dict=coeffs_dict)

    # Get the pupil position
    def get_pupilpos(self):

        rotation_matrix = np.matrix(cv2.Rodrigues(self.rvec)[0])
        cam_pos = np.matrix(self.tvec)
        cam_pos = np.array( - (rotation_matrix.transpose() * cam_pos) )

        return np.array([cam_pos[0][0],cam_pos[1][0],cam_pos[2][0]])


    # Get the sight-line direction(s) for given pixel coordinates, as unit vector(s) in the lab frame.
    def get_los_direction(self,x,y):

        if np.shape(x) != np.shape(y):
            raise ValueError("X pixels array and Y pixels array must be the same size!")

        # Flatten everything and create output array        
        oldshape = np.shape(x)
        x = np.reshape(x,np.size(x),order='F')
        y = np.reshape(y,np.size(y),order='F')

        # Get the normalised 2D coordinates including distortion
        x_norm,y_norm = self.normalise(x,y)

        # Normalise them to 3D unit vectors
        vect_length = np.sqrt(x_norm**2 + y_norm**2 + 1)
        x_norm = x_norm / vect_length
        y_norm = y_norm / vect_length
        z_norm = np.ones(x_norm.shape) / vect_length

        # Finally, rotate in to lab coordinates
        rotationMatrix = self.get_cam_to_lab_rotation()

        # x,y and z components of the LOS vectors
        x = rotationMatrix[0,0]*x_norm + rotationMatrix[0,1]*y_norm + rotationMatrix[0,2]*z_norm
        y = rotationMatrix[1,0]*x_norm + rotationMatrix[1,1]*y_norm + rotationMatrix[1,2]*z_norm
        z = rotationMatrix[2,0]*x_norm + rotationMatrix[2,1]*y_norm + rotationMatrix[2,2]*z_norm

        # Return an array the same shape as the input x and y pixel arrays + an extra dimension
        out = np.concatenate([np.expand_dims(x,-1),np.expand_dims(y,-1),np.expand_dims(z,-1)],axis=-1)

        return np.reshape(out,oldshape + (3,),order='F')


    # Get a dictionary of the model coeffs
    def get_dict(self):

        out_dict = {
                        'model': self.model,
                        'reprojection_error':self.reprojection_error,
                        'fx':self.cam_matrix[0,0],
                        'fy':self.cam_matrix[1,1],
                        'cx':self.cam_matrix[0,2],
                        'cy':self.cam_matrix[1,2],
                        'dist_coeffs':list(np.squeeze(self.kc)),
                        'rvec':list(np.squeeze(self.rvec)),
                        'tvec':list(np.squeeze(self.tvec)),
                        'fit_options':self.fit_options
                    }

        return out_dict


    def get_cam_to_lab_rotation(self):

        rotation_matrix = np.matrix(cv2.Rodrigues(self.rvec)[0])

        return rotation_matrix.transpose()





# Class representing a perspective camera model.
class PerspectiveViewModel(ViewModel):

    # Can be initialised either with the output of opencv.CalibrateCamera or from 
    # a dictionary containing the model coefficients
    def __init__(self,cv2_output=None,coeffs_dict=None):

        if cv2_output is None and coeffs_dict is None:
            raise ValueError('Either OpenCV output or coefficient dictionary must be defined!')

        self.model = 'perspective'
        self.fit_options = []

        if cv2_output is not None:

            self.reprojection_error = cv2_output[0]
            self.cam_matrix = cv2_output[1]
            self.kc = cv2_output[2]
            self.rvec = cv2_output[3][0]
            self.tvec = cv2_output[4][0]        

        elif coeffs_dict is not None:
            self.load_from_dict(coeffs_dict)



    # Load from a dicationary
    def load_from_dict(self,coeffs_dict):

        if 'reprojection_error' in coeffs_dict:
            self.reprojection_error = coeffs_dict['reprojection_error']
        else:
            self.reprojection_error = None

        self.cam_matrix = np.zeros([3,3])
        self.cam_matrix[0,0] = coeffs_dict['fx']
        self.cam_matrix[1,1] = coeffs_dict['fy']
        self.cam_matrix[0,2] = coeffs_dict['cx']
        self.cam_matrix[1,2] = coeffs_dict['cy']
        self.cam_matrix[2,2] = 1.

        self.kc = np.zeros([1,len(coeffs_dict['dist_coeffs'])])
        self.kc[0,:] = coeffs_dict['dist_coeffs']
        
        if 'rvec' in coeffs_dict and 'tvec' in coeffs_dict:
            self.rvec = np.zeros([3,1])
            self.rvec[:,0] = coeffs_dict['rvec']
            self.tvec = np.zeros([3,1])
            self.tvec[:,0] = coeffs_dict['tvec']

        if 'fit_options' in coeffs_dict:
            self.fit_options = coeffs_dict['fit_options']
        else:
            self.fit_options = ['']





    # Given pixel coordinates x,y, return the NORMALISED
    # coordinates of the corresponding un-distorted points.
    def normalise(self,x,y):

        if np.shape(x) != np.shape(y):
            raise ValueError("x and y must be the same shape!")

        # Flatten everything and create output array        
        oldshape = np.shape(x)
        x = np.reshape(x,np.size(x),order='F')
        y = np.reshape(y,np.size(y),order='F')

        input_points = np.zeros([x.size,1,2])
        for point in range(len(x)):
            input_points[point,0,0] = x[point]
            input_points[point,0,1] = y[point]

        undistorted = cv2.undistortPoints(input_points,self.cam_matrix,self.kc)

        undistorted = np.swapaxes(undistorted,0,1)[0]

        return np.reshape(undistorted[:,0],oldshape,order='F') , np.reshape(undistorted[:,1],oldshape,order='F')
 



    # Get list of 2D coordinates based on 3D point coordinates
    def project_points(self,points):

        # Check the input points are in a suitable format
        if np.ndim(points) < 3:
            points = np.array([points],dtype='float32')
        else:
            points = np.array(points,dtype='float32')

        output = np.zeros([len(points[0]),2])

        # Do reprojection
        points,_ = cv2.projectPoints(points,self.rvec,self.tvec,self.cam_matrix,self.kc)

        return np.squeeze(points)



    # Un-distort an image based on the calibration
    def undistort_image(self,image):

        return cv2.undistort(image,self.cam_matrix,self.kc)




# Class representing a perspective camera model.
class FisheyeeViewModel(ViewModel):

    # Can be initialised either with the output of opencv.CalibrateCamera or from 
    # a dictionary containing the model coefficients
    def __init__(self,cv2_output=None,coeffs_dict=None):

        if int(cv2.__version__[0]) < 3:
            raise Exception('OpenCV 3.0+ is required for fisheye calibration, you are using {:s}'.format(cv2.__version__))

        if cv2_output is None and coeffs_dict is None:
            raise ValueError('Either OpenCV output or coefficient dictionary must be defined!')

        self.model = 'fisheye'
        self.fit_options = []

        if cv2_output is not None:

            self.reprojection_error = cv2_output[0]
            self.cam_matrix = cv2_output[1]
            self.kc = cv2_output[2]
            self.rvec = cv2_output[3][0][0].T
            self.tvec = cv2_output[4][0][0].T


        elif coeffs_dict is not None:
            self.load_from_dict(coeffs_dict)



    # Load from a dicationary
    def load_from_dict(self,coeffs_dict):

        if 'reprojection_error' in coeffs_dict:
            self.reprojection_error = coeffs_dict['reprojection_error']
        else:
            self.reprojection_error = None

        self.cam_matrix = np.zeros([3,3])
        self.cam_matrix[0,0] = coeffs_dict['fx']
        self.cam_matrix[1,1] = coeffs_dict['fy']
        self.cam_matrix[0,2] = coeffs_dict['cx']
        self.cam_matrix[1,2] = coeffs_dict['cy']
        self.cam_matrix[2,2] = 1.

        self.kc = np.array(coeffs_dict['dist_coeffs'])
        
        self.rvec = np.zeros([3,1])
        self.rvec[:,0] = coeffs_dict['rvec']
        self.tvec = np.zeros([3,1])
        self.tvec[:,0] = coeffs_dict['tvec']

        self.fit_options = coeffs_dict['fit_options']


    # Given pixel coordinates x,y, return the NORMALISED
    # coordinates of the corresponding un-distorted points.
    def normalise(self,x,y):

        if np.shape(x) != np.shape(y):
            raise ValueError("x and y must be the same shape!")

        # Flatten everything and create output array        
        oldshape = np.shape(x)
        x = np.reshape(x,np.size(x),order='F')
        y = np.reshape(y,np.size(y),order='F')

        input_points = np.zeros([x.size,1,2])
        for point in range(len(x)):
            input_points[point,0,0] = x[point]
            input_points[point,0,1] = y[point]

        undistorted = cv2.fisheye.undistortPoints(input_points,self.cam_matrix,self.kc)

        undistorted = np.swapaxes(undistorted,0,1)[0]

        return np.reshape(undistorted[:,0],oldshape,order='F') , np.reshape(undistorted[:,1],oldshape,order='F')
 


    # Get list of 2D coordinates based on 3D point coordinates
    def project_points(self,points):

        # Check the input points are in a suitable format
        if np.ndim(points) < 3:
            points = np.array([points],dtype='float32')
        else:
            points = np.array(points,dtype='float32')

        # Do reprojection
        points,_ = cv2.fisheye.projectPoints(points,self.rvec,self.tvec,self.cam_matrix,self.kc)
        points = np.swapaxes(points,0,1)
                   
        return np.squeeze(points)


    # Un-distort an image based on the calibration
    def undistort_image(self,image):

        return cv2.fisheye.undistortImage(image,self.cam_matrix,self.kc)




class Calibration():
    '''
    Class representing a camera view calibration. 

    A complete Calibration object contains the camera 
    image which was calibrated (if any), the point 
    pairs used for fitting (if applicable), the camera
    model parameters, and metadata about each of these.

    If instantiated with the name of a .ccc file to load,
    the resulting  object represents the calibration 
    contained in that file. If no file name is given, an 
    empty calibration object of a specified type is created.
    
    Parameters:

       load_filename (str) : File name of the calibration to load. If not given, an "empty" \
                             calibration object is created.
       cal_type (str)     :  Required only if load_file is not specified i.e. creating an empty \
                             calibration object. Must be one of "fit", "alignment" or "virtual".\
                             If load_file is provided, this is ignored.
    '''
    def __init__(self,load_filename = None,cal_type = None):

      # Start off with mostly empty properties
        self.image = None
        self.pointpairs = None
        self.cad_config = None
        self.subview_mask = None
        self.n_subviews = 1
        self.subview_names = ['Full Frame']
        self.view_models = []
        self.intrinsics_constraints = []
        self.pixel_size = None
        self.readonly = None
        self.geometry = CoordTransformer()

        # Load calibration from disk if requested.
        if load_filename is not None:
            self._load(load_filename)

        # Otherwise, check the specified cal_type is allowed.
        elif cal_type.lower() not in ['fit','alignment','virtual']:
            raise ValueError('To create a new empty calibration, the "type" argument must be spplied and be "fit","alignment" or "virtual".')
        
        # And initialise empty stuff as appropriate for that cal_type.
        else:
            self._type = cal_type.lower()
            self.history = {}
            if  self._type != 'virtual':
                self.history['image'] = None
            if self._type != 'fit':
                self.history['extrinsics'] = None
                self.history['intrinsics'] = None
            if self._type == 'fit':
                self.history['pointpairs'] = [None,None]
                self.history['intrinsics_constraints'] = []

            if self._type != 'fit':
                self.intrisnics_type = None



    def set_pointpairs(self,pointpairs,src=None,history=None):
        '''
        Add a set of point pairs with the calibration. This replaces
        the existing point pairs, if any.

        Parameters:

            pointpairs (calcam.PointPairs) : Set of point pairs to add.
            src (str)                      : Optional metadata describing where the point pairs are from.
            history (tuple)                : Optional 2-element tuple of strings describing the \
                                             history of the point pairs. The first string describes \
                                             the original source (like src), and the second string \
                                             describes the most recent modification.
        '''
        if pointpairs is not None:
            if pointpairs.n_subviews != self.n_subviews:
                raise ValueError('Provided PointPairs object has a different number of sub-views ({:d}) to this calibration ({:d})!'.format(pointpairs.n_subviews,self.n_subviews))
        else:
            self.history['pointpairs'] = [None,None]

        self.pointpairs = pointpairs

        if history is not None:
            self.history['pointpairs'] = history

        elif self.history['pointpairs'][0] is None and src is None:
            self.history['pointpairs'][0] = 'Created by {:s} on {:s} at {:s}'.format(_user,_host,_get_formatted_time())
        elif src is not None:
            self.history['pointpairs'][0] = src + ' by {:s} on {:s} at {:s}'.format(_user,_host,_get_formatted_time())
            self.history['pointpairs'][1] = None
        else:
            self.history['pointpairs'][1] = 'Last modified by {:s} on {:s} at {:s}'.format(_user,_host,_get_formatted_time())




    def add_intrinsics_constraints(self,image=None,pointpairs=None,calibration=None,im_history=None,pp_history=None,src=None):
        '''
        Associate additional data used as intrinsics constraints to the calibration.

        Parameters:

            image (numpy.ndarray)            : Image data
            pointpairs (calcam.PointPairs)   : Point pairs associated with the image
            calibration (calcam.Calibration) : Calcam calibration
            im_history (str)                 : Description of the image history
            pp_history (tuple)               : 2 element tuple of strings describing the point pair history
            src (str)                        : String describing the source of the intrinsics constraints.
        '''
        if calibration is not None:
            im = self.geometry.original_to_display_image(calibration.get_image(coords='original'))
            pp = self.geometry.original_to_display_pointpairs( calibration.geometry.display_to_original_pointpairs(calibration.pointpairs) )
            self.intrinsics_constraints.append((im,pp))
            self.history['intrinsics_constraints'].append( calibration.history['image'],calibration.history['pointpairs'] )
            self.intrinsics_constraints = self.intrinsics_constraints + calibration.intrinsics_constraints
            self.history['intrinsics_constraints'] = self.history['intrinsics_constraints'] + calibration.history['intrinsics_constraints']

        elif pointpairs is not None:
            self.intrinsics_constraints.append((image,pointpairs))

        if im_history is not None and pp_history is not None:
            self.history['intrinsics_constraints'].append( (im_history,pp_history) )
        elif src is not None:
            self.history['intrinsics_constraints'].append(src + ' by {:s} on {:s} at {:s}'.format(_user,_host,_get_formatted_time()))


    def clear_intrinsics_constraints(self):
        '''
        Remove all intrinsics constraints from the calibration object.
        '''
        self.intrinsics_constraints = []


    def _load(self,filename):
        '''
        Load calibration from a file in to this object.

        Parameters:

            filename (str) : Name of the file to load.
        '''
        self.name = os.path.split(filename)[-1].split('.')[0]

        with ZipSaveFile(filename,'r') as save_file:

            # Load the general information
            try:
                with save_file.open_file('calibration.json','r') as f:
                    meta = json.load(f)
            except IOError:
                raise IOError('"{:s}" does not appear to be a Calcam calibration file!'.format(filename))

            # Load the field mask and set up geometry object
            subview_mask = cv2.imread(os.path.join(save_file.get_temp_path(),'subview_mask.png'))[:,:,0]
            self.geometry = CoordTransformer(transform_actions=meta['image_transform_actions'],paspect=meta['orig_paspect'])
            self.geometry.set_image_shape(subview_mask.shape[1],subview_mask.shape[0],coords='Display')
            
            self.subview_mask = self.geometry.display_to_original_image(subview_mask)
            
            # Load the image. Note with the opencv imread function, it will silently return None if the image file does not exist.
            image = cv2.imread(os.path.join(save_file.get_temp_path(),'image.png'))
            if image is not None:
                if len(image.shape) == 3:
                    if image.shape[2] == 3:
                        image[:,:,:3] = image[:,:,2::-1]
                
                self.image = self.geometry.display_to_original_image(image)        
            
            self.n_subviews = meta['n_subviews']
            self.history = meta['history']
            self.subview_names = meta['subview_names']
            self._type = meta['calib_type']
            self.pixel_size = meta['pixel_size']
            
            if self._type != 'fit':
                self.intrinsics_type = meta['intrinsics_type']

            # Load the primary point pairs
            try:
                with save_file.open_file('pointpairs.csv','r') as ppf:
                    self.pointpairs = PointPairs(ppf)
            except:
                self.pointpairs = None

            # Load fit results
            self.view_models = []
            for nview in range(self.n_subviews):
                try:
                    with save_file.open_file('calib_params_{:d}.json'.format(nview),'r') as f:
                        self.view_models.append(ViewModel.from_dict(json.load(f)))
                except IOError:
                    self.view_models.append(None)
                    
            # Load CAD config
            if 'cad_config.json' in save_file.list_contents():
                with save_file.open_file('cad_config.json','r') as f:
                    self.cad_config = json.load(f)

            # Load any intrinsics constraints
            for i in range( len( [f for f in save_file.list_contents() if f.startswith('intrinsics_constraints') and 'points_' in f] ) ):

                im = cv2.imread(os.path.join(save_file.get_temp_path(),'intrinsics_constraints','im_{:03d}.png'.format(i)))
                if im is not None:
                    if len(im.shape) == 3:
                        if im.shape[2] == 3:
                            im[:,:,:3] = im[:,:,2::-1]
                
                with save_file.open_file(os.path.join('intrinsics_constraints','points_{:03d}.csv'.format(i)),'r') as ppf:
                    pp = PointPairs(ppf)

                self.intrinsics_constraints.append([im,pp])

            # Note: this is only needed for calibration created with Calcam 2.0.0-dev
            # --------------------------------------------------------------------------
            if os.path.join('intrinsics_constraints','intrinsics_calib.ccc') in save_file.list_contents():
                self.add_intrinsics_constraints( calibration = Calibration(os.path.join(save_file.get_temp_path(),'intrinsics_constraints','intrinsics_calib.ccc')) )
            
            if type(self.history) is list:
                old_history = self.history
                self.history = {}
                if  self._type != 'virtual':
                    self.history['image'] = None
                if self._type != 'fit':
                    self.history['extrinsics'] = None
                    self.history['intrinsics'] = None
                if self._type == 'fit':
                    self.history['pointpairs'] = [None,None]
                    self.history['intrinsics_constraints'] = []
                    self.history['fit'] = [None] * self.n_subviews

                if self._type != 'fit':
                    self.intrisnics_type = None

                for event in old_history:
                    if 'Image' in event[3]:
                        self.history['image'] = event[3] + ' by {:s} on {:s} at {:s}'.format(event[1],event[2],_get_formatted_time(event[0]))
                    elif 'Point pairs' in event[3]:
                        if self.history['pointpairs'][0] is None:
                            self.history['pointpairs'][0] = event[3] + ' by {:s} on {:s} at {:s}'.format(event[1],event[2],_get_formatted_time(event[0]))
                        else:
                            self.history['pointpairs'][1] = self.history['pointpairs'][1] = 'Last modified by {:s} on {:s} at {:s}'.format(event[1],event[2],_get_formatted_time(event[0]))
                    elif 'Fit' in event[3]:
                        subview_ind = self.subview_names.index(event[3].split('for ')[1])
                        self.history['fit'][subview_ind] = 'Modified by {:s} on {:s} at {:s}'.format(event[1],event[2],_get_formatted_time(event[0]))

            self.readonly = save_file.is_readonly()



    def set_image(self,image,src,coords='Display',transform_actions = [],subview_mask=None,pixel_aspect=1.,subview_names = [],pixel_size=None):
        '''
        Set the main image associated with the calibration.

        Parameters:

            image(numpy.ndarray)         : Image data array
            src (str)                    : Description of where the image is from
            coords (str)                 : Either ``Display`` or ``Original`` specifying \
                                           the orientation of the provided image array.
            transform_actions (list)     : List of strings describing the \
                                           geometrical transformations between \
                                           original and display coordinates. 
            subview_mask (numpy.ndarray) : Integer array the same size as the image \
                                           where the value in each element specifies what \
                                           sub-view each pixel belongs to.
            subview_names (list)         : List of strings specifying the names of the sub-views.
            pixel_size (float)           : Physical size of the detector pixels in microns.
        '''
        image = image.copy()

        # If the array isn't already 8-bit int, make it 8-bit int...
        if image.dtype != np.uint8:
            # If we're given a higher bit-depth integer, it's easy to downcast it.
            if image.dtype == np.uint16 or image.dtype == np.int16:
                image = np.uint8(image/2**8)
            elif image.dtype == np.uint32 or image.dtype == np.int32:
                image = np.uint8(image/2**24)
            elif image.dtype == np.uint64 or image.dtype == np.int64:
                image = np.uint8(image/2**56)
            # Otherwise, scale it in a floating point way to its own max & min
            # and strip out any transparency info (since we can't be sure of the scale used for transparency)
            else:

                if image.min() < 0:
                    image = image - image.min()

                if len(image.shape) == 3:
                    if image.shape[2] == 4:
                        image = image[:,:,:-1]

                image = np.uint8(255.*(image - image.min())/(image.max() - image.min()))

        if subview_mask is None:
            self.n_subviews = 1
            subview_mask = np.zeros(image.shape[:2],dtype='uint8')
            self.subview_names = ['Image']
        else:
            self.n_subviews = subview_mask.max() + 1
            if len(subview_names) == self.n_subviews:
                self.subview_names = subview_names
            else:
                self.subview_names = []
                for n in range(self.n_subviews):
                    self.subview_names.append('View {:d}'.format(n+1))

        self.geometry = CoordTransformer()

        self.geometry.set_transform_actions(transform_actions)
        self.geometry.set_pixel_aspect(pixel_aspect,relative_to='Original')
        self.geometry.set_image_shape(image.shape[1],image.shape[0],coords=coords)

        if coords.lower() == 'original':
            self.image = image
            self.subview_mask = subview_mask
        else:
            self.image = self.geometry.display_to_original_image(image)
            self.subview_mask = self.geometry.display_to_original_image(subview_mask)

        self.pixel_size = pixel_size

        if 'by' in src and 'on' in src and 'at' in src:
            self.history['image'] = src
        else:
            self.history['image'] = src + ' by {:s} on {:s} at {:s}'.format(_user,_host,_get_formatted_time())



    def set_subview_mask(self,mask,subview_names=None,coords='Original'):
        '''
        Set the mask specifying which sub-view each image pixel belongs to.

        Parameters:

            mask (numpy.ndarray) : Integer array the same size as the calibration's main image.
            subview_names (list) : List of strings specifying the names of the different sub-views
            coords (str)         : Either ``Original`` or ``Display``, specifies what orientation the \
                                   supplied mask is in.
        '''        
        n_subviews = mask.max() + 1

        if subview_names is None:

            if n_subviews == 1:
                subview_names = ['Image']

            else:
                subview_names = ['Sub-view #{:d}'.format(i + 1) for i in range(n_subviews)]
        else:
            if len(subview_names) != n_subviews:
                raise ValueError('The subview mask appears to show {:d} sub-views, but {:d} names were provided!'.format(n_subviews,len(subview_names)))

        mask = mask.copy()

        if coords.lower() == 'original':
            self.subview_mask = mask
        elif coords.lower() == 'display':
            self.subview_mask = self.geometry.display_to_original_image(mask)
            
        self.subview_names = subview_names
        self.n_subviews = n_subviews


    def get_image(self,coords='Display'):
        '''
        Get the image which was calibrated.

        Parameters:

            coords (str)           : Either ``Display`` or ``Original``, \
                                     what orientation to return the image.

        Returns:

            np.ndarray or NoneType : If the calibration contains an image: returns image data array \
                                     with shape (h x w x n) where n is the number of colour \
                                     channels in the image. If the calibration does not contain \
                                     an image, returns None.
        '''
        if self.image is None:
            return None
        else:
            im_out = self.image.copy()
            if coords.lower() == 'display':
                im_out = self.geometry.original_to_display_image(im_out)
           
            return im_out
        
        
    def get_subview_mask(self,coords='Display'):
        '''
        Get an integer array the same shape as the image where
        the value of each element specifies which sub-view
        its corresponding image pixel belongs to.

        Parameters:

            coords (str)           : Either ``Display`` or ``Original``, \
                                     what orientation to return the mask.

        Returns:

            np.ndarray or NoneType : If the calibration contains an image: subview mask \
                                     array with shape (h x w). If the calibration does \
                                     not contain an image, returns None.
        '''
        if self.subview_mask is None:
            return None
        else:
            mask_out = self.subview_mask.copy()
            if coords.lower() == 'display':
                mask_out = self.geometry.original_to_display_image(mask_out)
           
            return mask_out        
      


    def save(self,filename):
        '''
        Save the calibration to a file.

        Parameters:

            filename (str) : File name to save to. If it does not \
                             already end with .ccc, the extension will \
                             be added.
        '''

        if not filename.endswith('.ccc'):
            filename = filename + '.ccc'

        with ZipSaveFile(filename,'w') as save_file:

            # Save the image
            if self.image is not None:
                im_out = self.get_image(coords='Display')
                if len(im_out.shape) == 3:
                    if im_out.shape[2] > 2:
                        im_out[:,:,:3] = im_out[:,:,2::-1]  
                        
                cv2.imwrite(os.path.join(save_file.get_temp_path(),'image.png'),im_out)


            # Save the field mask
            cv2.imwrite(os.path.join(save_file.get_temp_path(),'subview_mask.png'),self.get_subview_mask(coords='display'))

            # Save the general information
            meta = {
                    'n_subviews': int(self.n_subviews),
                    'history':self.history,
                    'pixel_size':self.pixel_size,
                    'orig_paspect':self.geometry.pixel_aspectratio,
                    'image_transform_actions':self.geometry.transform_actions,
                    'subview_names':self.subview_names,
                    'calib_type':self._type
            }

            if self._type != 'fit':
                meta['intrinsics_type'] = self.intrinsics_type

            for nview in range(self.n_subviews):
                if self.view_models[nview] is not None:
                    with save_file.open_file('calib_params_{:d}.json'.format(nview),'w') as f:
                        json.dump(self.view_models[nview].get_dict(),f,indent=4,sort_keys=True)


            with save_file.open_file('calibration.json','w') as f:
                json.dump(meta,f,indent=4,sort_keys=True)


            # Save the primary point pairs (if any)
            if self.pointpairs is not None:
                if self.pointpairs.get_n_points() > 0:
                    with save_file.open_file('pointpairs.csv','w') as ppf:
                        self.pointpairs.save(ppf)

            # Save CAD config
            if self.cad_config is not None:
                with save_file.open_file('cad_config.json','w') as f:
                    json.dump(self.cad_config,f,indent=4,sort_keys=True)

            # Save intrinsics constraints
            save_file.mkdir('intrinsics_constraints')
            for i,intrinsics in enumerate(self.intrinsics_constraints):

                if intrinsics[0] is not None:

                    im_out = intrinsics[0]
                    if len(im_out) == 3:
                        if im_out.shape[2] == 3:
                            im_out[:,:,:3] = im_out[:,:,2::-1]
                    cv2.imwrite(os.path.join(save_file.get_temp_path(),'intrinsics_constraints','im_{:03d}.png'.format(i)),im_out)

                with save_file.open_file(os.path.join(save_file.get_temp_path(),'intrinsics_constraints','points_{:03d}.csv'.format(i)),'w') as ppf:
                    intrinsics[1].save(ppf)


    def set_fit(self,subview,view_model):
        '''
        Add a fitted camera model to the calibration.
        Replaces any existing fit.

        Parameters:

            subview (int)                             : What sub-view index the fit applies to.
            view_model (calcam.calibration.ViewModel) : The fitted camera view model.
        '''
        self.view_models[subview] = view_model
        self.history['fit'][subview] = 'Modified by {:s} on {:s} at {:s}'.format(_user,_host,_get_formatted_time())


    def subview_lookup(self,x,y,coords='Display'):
        '''
        Check which sub-view given pixel coordinates belong to.

        Parameters:

            x,y (float)  : Pixel coordinates to check.
            coords (str) : Either ``Display`` or ``Original``, specifies whether \
                           the provided x and y coordinates are relative to the  \
                           image in display or original orientation. X and Y must \
                           be the same shape.

        Returns:

            np.ndarray   : Array of integers the same shape as input X and Y arrays. \
                           The value of each element specified which sub-view the \
                           corresponding input image position belongs to.
        '''
        if coords.lower() == 'display':
            shape = self.geometry.get_display_shape()
            mask = self.geometry.original_to_display_image(self.subview_mask)
        else:
            shape = self.geometry.get_original_shape()
            mask = self.subview_mask


        good_mask = (x >= -0.5) & (y >= -0.5) & (x < shape[0] - 0.5) & (y < shape[1] - 0.5)
        
        try:
            x[ good_mask == 0 ] = 0
            y[ good_mask == 0 ] = 0

            out = mask[np.round(y).astype('int'),np.round(x).astype('int')]

            out[good_mask == 0] = -1
        except TypeError:

            if good_mask:
                out = mask[np.round(y).astype('int'),np.round(x).astype('int')]
            else:
                out = -1

        return out




    def get_pupilpos(self,subview=None,x=None,y=None,coords='display',):
        '''
        Get the camera pupil position in 3D space.

        Can be used together with get_los_direction to obtain a full
        description of the camera's sight line geometry.

        Parameters:
            
            subview (int)                : Which sub-view to get the pupil position for. \
                                           Only required for calibrations with more than 1 \
                                           sub-view.

            x,y (float or numpy.ndarray) : For calibrations with more than one subview, get the pupil \
                                           position(s) corresponding to these given image pixel coordinates.

            coords (str)                 : Only used if x and y are also given. Either ``Display`` \
                                           or ``Original``, specifies whether the provided x and y are in \
                                           display or original coordinates.

        Returns:

            np.ndarray                  : Camera pupil position in 3D space. If not specifying \
                                          x or y inputs, this will be a 3 element array containing \
                                          the [X,Y,Z] coordinates of the pupil position in metres. \
                                          If using x and y inputs, the output array will be the same \
                                          shape as the x and y input arrays with an additional dimension \
                                          added; the X, Y and Z components are then given along the new \
                                          new array dimension.
        '''
        # If we're given pixel coordinates
        if x is not None or y is not None:

            # Validate the input coordinates
            if x is None or y is None:
                raise ValueError("X pixels and Y pixels must both be specified!")
            if np.shape(x) != np.shape(y):
                raise ValueError("X pixels array and Y pixels array must be the same size!")

            # Convert pixel coords if we need to
            if coords.lower() == 'original':
                x,y = self.geometry.original_to_display_coords(x,y)

            if subview is None:

                # Output should be the same shape as input + an extra length 3 axis
                output = np.zeros(np.shape(x) + (3,))

                # An array the same size as output sepcifying which sub-view calibration to use
                subview_mask = self.subview_lookup(x,y)
                subview_mask = np.tile( subview_mask, [3] + [1]*x.ndim )
                subview_mask = np.squeeze(np.swapaxes(np.expand_dims(subview_mask,-1),0,subview_mask.ndim),axis=0) # Changed to support old numpy versions. Simpler modern version: np.moveaxis(subview_mask,0,subview_mask.ndim-1)

                for nview in range(self.n_subviews):
                    pupilpos = np.tile( self.view_models[nview].get_pupilpos(), x.shape + (1,) )
                    output[subview_mask == nview] = pupilpos[subview_mask == nview]

            else:

                output = np.tile( self.view_models[subview].get_pupilpos() , x.shape + (1,) )

        else:

            if self.n_subviews == 1:
                subview = 0
            elif subview is None:
                raise ValueError('This calibration contains multiple sub-views; either pixel coordinates or subview number must be specified!')

            output = self.view_models[subview].get_pupilpos()


        return output



    def get_cam_matrix(self,subview=None):
        '''
        Get the camera matrix. 

        Parameters:
            
            subview (int) : For calibrations with multiple sub-views, \
                            which sub-view index to return the camera \
                            matrix for.

        Returns:

            np.matrix     : 3x3 camera matrix.

        '''
        if self.n_subviews > 1 and subview is None:
            raise ValueError('This calibration has more than 1 sub-view; sub-view must be specified!')

        elif subview is None:
            subview = 0

        if self.view_models[subview] is not None:
            return np.matrix(self.view_models[subview].cam_matrix)
        else:
            raise Exception('This calibration does not contain a camera model for sub-view #{:d}!'.format(subview))



    def get_cam_roll(self,subview=None):
        '''
        Get the camera roll. This is the angle between the lab 
        +Z axis and the camera's "view up" direction.

        Parameters:
            
            subview (int) : For calibrations with multiple sub-views, \
                            which sub-view index to return the camera \
                            roll for.

        Returns:

            float         : Camera roll in degrees. Positive angles correspond to an anti-clockwise \
                            roll of the camera i.e. clockwise roll of the image.

        '''
        if self.n_subviews > 1 and subview is None:
            raise ValueError('This calibration has more than 1 sub-view; sub-view must be specified!')

        elif subview is None:
            subview = 0

        return self.view_models[subview].get_cam_roll()


    # Get the horizontal and vertical field of view of a given sub-view
    def get_fov(self,subview=None,fullchip=False):
        '''
        Get the camera field of view.

        Parameters:
            
            subview (int)   : For calibrations with multiple sub-views, \
                              which sub-view index to return the field \
                              of view for.
            fullchip (bool) : For calibrations with multiple sub-views, \
                              setting this to True will return the field of \
                              view defined by the camaera model of the specified \
                              sub-view, as if that sub-view covered the whole image.

        Returns:

            tuple           : 2-element tuple containing the full angles of the horizontal and \
                              vertical fields of view (h_fov,v_fov) in degrees.
        '''
        if subview is None and self.n_subviews > 1:
            raise ValueError('This calibration contains multuple sub-views; subview number must be specified!')
        elif subview is None and self.n_subviews == 1:
            subview = 0

        subview_mask = self.geometry.original_to_display_image(self.subview_mask)

        # Calculate FOV by looking at the angle between sight lines at the image extremes
        if fullchip:
            hsize,vsize = self.geometry.get_display_shape()
            vcntr = vsize/2.
            hcntr = hsize/2.
        else:
            vcntr,hcntr = CoM( subview_mask == subview )

        vcntr = int(np.round(vcntr))
        hcntr = int(np.round(hcntr))

        # Horizontal extent of sub-field
        if fullchip:
            h_extent = np.array([0,hsize-1])
        else:
            h_extent = np.argwhere(subview_mask[vcntr,:] == subview)

        # Horizontal FOV
        norm1 = self.normalise(h_extent.min()-0.5,vcntr,subview=subview)
        norm2 = self.normalise(h_extent.max()+0.5,vcntr,subview=subview)
        fov_h = 180*(np.arctan(norm2[0]) - np.arctan(norm1[0])) / 3.141592635

        if fullchip:
            v_extent = np.array([0,vsize-1])
        else:
            v_extent = np.argwhere(subview_mask[:,hcntr] == subview)

        # Vertical field of view
        norm1 = self.normalise(hcntr,v_extent.min()-0.5,subview=subview)
        norm2 = self.normalise(hcntr,v_extent.max()+0.5,subview=subview)
        fov_v = 180*(np.arctan(norm2[1]) - np.arctan(norm1[1])) / 3.141592635

        return fov_h, fov_v



    def get_los_direction(self,x=None,y=None,coords='Display',subview=None):
        '''
        Get unit vectors representing the directions of the camera's sight-lines in 3D space. 

        Can be used together with get_pupilpos to obtain a full description of the camera's sight-line geometry.

        Parameters:

            x,y (array-like of floats) : Image pixel coordinates at which to get the sight-line directions. \
                                         x and y must be the same shape. If not specified, the line of sight direction \
                                         at the centre of every detector pixel is returned.

            coords (str)               : Either ``Display`` or ``Original``, specifies which image orientation the provided x and y \
                                         inputs and/or shape of the returned array correspond to.

            subview (int)              : If specified, forces the use of the camera model from the specified sub-view index. \
                                         If not given, the correct sub-view(s) will be chosen automatically.

        Returns:

            np.ndarray                 : Array of sight-line vectors. If specifying x_pixels and y_pixels, the output array will be \
                                         the same shape as the input arrays but with an extra dimension added. The extra dimension contains \
                                         the [X,Y,Z] components of the sight-line vectors. If not specifying x_pixels and y_pixels, the output \
                                         array shape will be (h x w x 3) where w and h are the image width and height in pixels.
        '''

        # If we're given pixel coordinates
        if x is not None or y is not None:

            if type(x) != np.ndarray or type(y) != np.ndarray:
                if type(x) == list:
                    x = np.array(x)
                    y = np.array(y)
                else:
                    x = np.array([x])
                    y = np.array([y])

            # Validate the input coordinates
            if x is None or y is None:
                raise ValueError("X pixels and Y pixels must both be specified!")
            if np.shape(x) != np.shape(y):
                raise ValueError("X pixels array and Y pixels array must be the same size!")

            # If given original coordinates, convert to display to do the calculation
            if coords.lower() == 'original':
                x,y = self.geometry.original_to_display_coords(x,y)

        # If not given any, generate our own image coordinates covering every pixel.
        else:
            if coords.lower() == 'original':
                shape = self.geometry.get_original_shape()
                x,y = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))
                x,y = self.geometry.original_to_display_coords(x,y)
            else:
                shape = self.geometry.get_display_shape()
                x,y = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))


        # Output should be the same shape as input + an extra length 3 axis
        output = np.zeros(np.shape(x) + (3,))


        if subview is None:

            # An array the same size as output sepcifying which sub-view calibration to use
            subview_mask = self.subview_lookup(x,y)
            subview_mask = np.tile( subview_mask, [3] + [1]*x.ndim )
            subview_mask = np.squeeze(np.swapaxes(np.expand_dims(subview_mask,-1),0,subview_mask.ndim),axis=0) # Changed to support old numpy versions. Simpler modern version: np.moveaxis(subview_mask,0,subview_mask.ndim-1)
            
            for nview in range(self.n_subviews):
                losdir = self.view_models[nview].get_los_direction(x,y)
                output[subview_mask == nview] = losdir[subview_mask == nview]

        else:

            output = self.view_models[subview].get_los_direction(x,y)


        return np.squeeze(output)


    def project_points(self,points_3d,coords='display',check_occlusion_by=None,fill_value=np.nan,occlusion_tol=1e-3):
        '''
        Get the image coordinates corresponding to given real-world 3D coordinates. 

        Optionally can also check whether the 3D points are hidden from the camera's view by part of the CAD model being in the way.

        Parameters:

            points_3d                            : 3D point coordinates, in metres, to project on to the image. Can be EITHER an Nx3 array, where N is the \
                                                   number of 3D points and each row gives [X,Y,Z] for a 3D point, or an array-like of 3 element array-likes,\
                                                   where each 3 element array specifies a 3D point.
            
            coords (str)                         : Either ``Display`` or ``Original``, specifies which image orientation the returned image coordinates \
                                                   should correspond to.

            check_occlusion_by (calcam.CADModel) : If provided, for each projected point the function will check if the point \
                                                   is hidden from the camera's view by part of the provided CAD model.

            fill_value (float)                   : For any 3D points not visible to the camera, the returned image coordinates will be set equal to \
                                                   this value. If set to ``None``, image coordinates will be returned for every 3D point even if the \
                                                   point is outside the camera's field of view or hidden from view.

            occlusion_tol (float)                : Tolerance (in mrtres) to use to check point occlusion. Try increasing this value if having trouble \
                                                   with points being wrongly detected as occluded.  

        Returns:

            list of np.ndarray                  : A list of Nx2 NumPY arrays containing the image coordinates of the given 3D points (N is the number of input 3D points). \
                                                  Each NumPY array corresponds to a single sub-view, so for images without multuiple sub-views this will return a single element \
                                                  list containing an Nx2 array. Each row of the NumPY arrays contains the [X,Y] image coordinates of the corresponding input point. \
                                                  If fill_value is not None, points not visible to the camera have their coordinates set to ``[fill_value, fill_value]``.
        '''
        points_3d = np.array(points_3d)

        # This will be the output
        points_2d = []

        if check_occlusion_by is not None and raycast_sightlines is None:
            raise Exception('VTK not available; cannot check occlusion without VTK.')

        for nview in range(self.n_subviews):

            if self.view_models[nview] is None:
                # Check the input points are in a suitable format
                if np.ndim(points_3d) < 3:
                    dims = (np.shape(points_3d)[0],2)
                else:
                    dims = points.shape[:-1] + (2,)

                p2d = np.zeros(dims)
                p2d[:] = fill_value

            else:

                # Do the point projection for this sub-fov
                p2d =  self.view_models[nview].project_points(points_3d)

                if len(p2d.shape) == 1:
                    p2d = p2d[np.newaxis,:]

                # If we're checking whether the 3D points are actually visible...
                if fill_value is not None:

                    # Create a mask indicating where the projected points are outside their correct
                    # sub-view (including out of the image completely)
                    wrong_subview_mask = self.subview_lookup(p2d[:,0],p2d[:,1]) != nview

                    # If given a CAD model to check occlusion against, compare the distances to the 3D points with
                    # the distance to a surface in the CAD model
                    if check_occlusion_by is not None:

                        point_vectors = points_3d - np.tile( self.view_models[nview].get_pupilpos() , (points_3d.shape[0],1) )
                        point_distances = np.sqrt( np.sum(point_vectors**2,axis= 1))

                        # Ray cast to get the ray lengths
                        ray_lengths = raycast_sightlines(self,check_occlusion_by,p2d[:,0],p2d[:,1],verbose=False,force_subview=nview).get_ray_lengths()

                        # The 3D points are invisible where the distance to the point is larger than the
                        # ray length
                        occluded_mask = ray_lengths < (point_distances - occlusion_tol)

                        p2d[ np.tile(occluded_mask,(1,2))  ] = fill_value

                    p2d[ np.tile(wrong_subview_mask[:,np.newaxis],(1,2)) ] = fill_value


            # If the user asked for things in original coordinates, convert them.
            if coords.lower() == 'original':
                p2d[:,0],p2d[:,1] = self.geometry.display_to_original_coords(p2d[:,0],p2d[:,1])

            points_2d.append(p2d)

        return points_2d



    def undistort_image(self,image,coords='display'):
        '''
        Correct lens distortion a given image from the calibrated camera,
        to give an image with a pure perspective projection.

        Parameters:
            image (np.ndarray) : (h x w x N) array containing the image to be un-distorted,
                                 where N is the number of colour channels.
            coords (str)       : Either ``Display`` or ``Original``, specifies which orientation \
                                 the input image is in.

        Returns:

            np.ndarray         : Image data array the same shape as the input array containing the \
                                 corrected image.
        '''
        if coords.lower() == 'original':
            image = self.geometry.original_to_display_image(image)

        image_out = np.zeros(image.shape)

        subview_mask_display = self.geometry.original_to_display_image(self.subview_mask)
        
        for nview in range(self.n_subviews):

            subview_mask = subview_mask_display == nview

            im_in = image.copy()
            if len(im_in.shape) > 2:
                im_in[ np.tile(subview_mask[:,:,np.newaxis],(1,1,im_in.shape[2])) == 0 ] = 0
            else:
                im_in[ subview_mask == 0 ] = 0

            undistorted =  self.view_models[nview].undistort_image(im_in)

            if len(im_in.shape) > 2:
                image_out[ np.tile(subview_mask[:,:,np.newaxis],(1,1,im_in.shape[2])) ] = undistorted[ np.tile(subview_mask[:,:,np.newaxis],(1,1,im_in.shape[2])) ]
            else:
                image_out[ subview_mask ] = undistorted[ subview_mask ]

        return image_out


    def get_cam_to_lab_rotation(self,subview=None):
        '''
        Get a 3D rotation matrix which will rotate a point in the camera coordinate system
        (see Calcam theory documentation) in to the lab coordinate system's orientation.

        Parameters:
            subview (int): For calibrations with multiple sub-views, specifies which sub-view \
                           to return the rotation matrix for.

        Returns:
            np.matrix    : 3x3 rotation matrix.

        '''
        if subview is None:
            if self.n_subviews > 1:
                raise Exception('This calibration has multiple sub-views; you must specify a pixel location to get_cam_to_lab_rotation!')
            else:
                subview = 0

        return self.view_models[subview].get_cam_to_lab_rotation()


    def normalise(self,x,y,subview=None):
        '''
        Given x and y image pixel coordinates, return corresponding normalised coordinates
        (see Calcam theory documentation).

        Parameters:

            x,y (array-like) : Arrays of x and y pixel coordinates to convert to normalised\
                               coordinates. x and y must be the same shape.
            subview (int)    : If specified, force the calculation to use the view model \
                               from the given sub-view. If not specified, the correct sub-view \
                               is chosen automatically.

        Returns:

            np.ndarray       : Array containing nroamlised coordinates. The output array will be the \
                               same shape as the input x and y arrays with an extra dimension added; \
                               the extra dimension contains the [x_n, y_n] normalised coordinates at the \
                               corresponding input coordinates.

        '''
        if subview is None:
            subview = self.subview_lookup(x,y)
            slic = [slice(None)]*(len(subview.shape)+1)
            slic[-1] = np.newaxis
            subview = np.tile(subview[tuple(slic)],list(np.ones(len(subview.shape),dtype=int)) + [2])

            outp = np.zeros(x.shape + (2,))

            for isubview in range(self.n_subviews):
                if self.view_models[isubview] is not None:
                    outp[subview == isubview] = self.view_models[isubview].normalise(x,y)[subview == isubview]
                else:
                    outp[subview == isubview] = np.nan
        else:
            outp = self.view_models[subview].normalise(x,y)

        return outp



    def set_calib_intrinsics(self,intrinsics_calib):
        '''
        For manual alignment or virtual calibrations: set the camera intrinsics
        of this calibration from an existing calcam calibration.

        Parameters:

            intrinsics_calib (calcam.Calibration) : Calibration from which to import \
                                                    the intrinsics.
        '''
        if self._type == 'fit':
            raise Exception('You cannot modify the intrinsics of a fitted calibration.')
        
        self.view_models = intrinsics_calib.view_models
        self.subview_mask = intrinsics_calib.subview_mask.copy()
        self.geometry = copy.copy(intrinsics_calib.geometry)
        self.pixel_size = intrinsics_calib.pixel_size
        self.intrinsics_constraints = []

        self.view_models[0] = intrinsics_calib.view_models[0]

        if intrinsics_calib._type == 'fit':
            self.intrinsics_constraints.append((intrinsics_calib.get_image(coords='Display'),intrinsics_calib.pointpairs))
            self.intrinsics_constraints = self.intrinsics_constraints + intrinsics_calib.intrinsics_constraints
            self.intrinsics_type = 'calibration'
            self.history['intrinsics'] = intrinsics_calib.history
        else:
            self.intrinsics_type = intrinsics_calib.intrinsics_type
            self.history['intrinsics'] = intrinsics_calib.history['intrinsics']
            self.intrinsics_constraints = intrinsics_calib.intrinsics_constraints



    def set_chessboard_intrinsics(self,view_model,images_and_points,src):
        '''
        For manual alignment or virtual calibrations: set the camera intrinsics
        from a view model obtained from chessboard image fitting.

        Parameters:

            view_model (calcam.calibration.ViewModel) : Calibration from which to import \
                                                    the intrinsics.
        '''
        if self._type == 'fit':
            raise Exception('You cannot modify the intrinsics of a fitted calibration.')

        self.view_models[0] = view_model

        self.subview_mask = np.zeros(images_and_points[0][0].shape[:2],dtype=np.uint8)
        self.geometry = CoordTransformer(orig_x=self.subview_mask.shape[1],orig_y = self.subview_mask.shape[0])

        self.intrinsics_constraints = images_and_points
        self.intrinsics_type = 'chessboard'

        if 'by' in src and 'on' in src and 'at' in src:
            self.history['intrinsics'] = src
        else:
            self.history['intrinsics'] = src + ' by {:s} on {:s} at {:s}'.format(_user,_host,_get_formatted_time())


    def set_pinhole_intrinsics(self,fx,fy,cx=None,cy=None,nx=None,ny=None):
        '''
        For manual alignment or virtual calibrations: set the camera intrinsics
        to be an ideal pinhole camera.

        Parameters:

            fx, fy (float) : Focal length in pixels in the horizontal and vertical directions.
            cx, cy (float) : Position of the perspective projection centre on the image in pixels. \
                             If not given, this is set to the image centre.
            nx, ny (int)   : Number of detector pixels in the horizontal and vertical directions. Must be \
                             given for virtual calibrations but not for any other type.
        '''
        if self._type == 'fit':
            raise Exception('You cannot modify the intrinsics of a fitted calibration.')
        
        elif self._type == 'virtual':
            if (cx is None or cy is None or nx is None or ny is None):
                raise ValueError('All of fx,fy,cx,cy,nx and ny must be specified for a virtual calibration.')
            else:
                self.geometry = CoordTransformer(orig_x=nx,orig_y = ny)
                self.subview_mask = np.zeros([ny,nx],dtype=np.uint8)

        elif self._type == 'alignment':
            shape = self.geometry.get_display_shape()
            if cx is None:
                cx = shape[0]/2
            if cy is None:
                cy = shape[1]/2

        coeffs_dict = {'fx':fx,'fy':fy,'cx':cx,'cy':cy,'dist_coeffs':np.zeros(5)}

        self.view_models = [PerspectiveViewModel(coeffs_dict=coeffs_dict)]

        self.intrinsics_constraints = []

        self.intrinsics_type = 'pinhole'

        self.history['intrinsics'] = 'Set by {:s} on {:s} at {:s}'.format(_user,_host,_get_formatted_time())


    def set_extrinsics(self,campos,upvec,camtar=None,view_dir=None):
        '''
        Manually set the camera extrinsics parameters.

        Parameters:

            campos (array-like)   : 3-element array-like specifying the camera position (X,Y,Z) in metres.
            upvec (array-like)    : 3-element array-like specifying the camera up vector.
            camtar (array-like)   : 3-element array-like specifying a point in 3D space where the camera is pointed to. \
                                    If both camtar and view_dir are provided, camtar is used.
            view_dir (array-like) : 3D vector [X,Y,Z] specifying the camera view direction. If both camtar and view_dir \
                                    are provided, camtar will be used instead.
        '''

        if self._type == 'fit':
            raise Exception('You cannot modify the extrinsics of a fitted calibration.')

        if camtar is not None:
            w = np.squeeze(np.array(camtar) - np.array(campos))
        elif view_dir is not None:
            w = view_dir
        else:
            raise ValueError('Either viewing target or view direction must be specified!')

        w = w / np.sqrt(np.sum(w**2))
        v = upvec / np.sqrt(np.sum(upvec**2))

        u = np.cross(w,v)

        Rmatrix = np.zeros([3,3])
        Rmatrix[:,0] = u
        Rmatrix[:,1] = -v
        Rmatrix[:,2] = w
        Rmatrix = np.matrix(Rmatrix)

        campos = np.matrix(campos)
        if campos.shape[0] < campos.shape[1]:
            campos = campos.T

        self.view_models[0].tvec = np.array(-Rmatrix.T * campos)
        self.view_models[0].rvec = np.array(-cv2.Rodrigues(Rmatrix)[0])

        self.history['extrinsics'] = 'Set by {:s} on {:s} at {:s}'.format(_user,_host,_get_formatted_time())



# The 'fitter' class 
class Fitter:

    def __init__(self,model='perspective'):
        
        self.pointpairs = [None]
        self.set_model(model)

        # Default fit options
        self.fixaspectratio = True
        self.fixk1 = False
        self.fixk2 = False
        self.fixk3 = True
        self.fixk4 = False
        self.fixskew = True
        self.disabletangentialdist=False
        self.fixcc = False


    def set_model(self,model):

        if model == 'perspective':
            self.model = model
        elif model == 'fisheye':
            opencv_major_version = int(cv2.__version__[0])
            if opencv_major_version < 3:
                raise ValueError('Fisheye model fitting requires OpenCV 3 or newer, you have ' + cv2.__version__)
            else:
                self.model = model
        else:
            raise ValueError('Unknown model type ' + model)


    # Based on the current fit parameters, returns the fit flags
    # in the format required by OpenCV fitting functions.
    # Output: fitflags (long int)
    def get_fitflags(self):

        if self.model == 'perspective':
            fitflags = cv2.CALIB_USE_INTRINSIC_GUESS

            if self.fixaspectratio:
                fitflags = fitflags + cv2.CALIB_FIX_ASPECT_RATIO
            if self.fixk1:
                fitflags = fitflags + cv2.CALIB_FIX_K1
            if self.fixk2:
                fitflags = fitflags + cv2.CALIB_FIX_K2
            if self.fixk3:
                fitflags = fitflags + cv2.CALIB_FIX_K3
            if self.disabletangentialdist:
                fitflags = fitflags + cv2.CALIB_ZERO_TANGENT_DIST
            if self.fixcc:
                fitflags = fitflags + cv2.CALIB_FIX_PRINCIPAL_POINT
        
        elif self.model == 'fisheye':

            fitflags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS + cv2.fisheye.CALIB_FIX_SKEW

            if self.fixk1:
                fitflags = fitflags + cv2.fisheye.CALIB_FIX_K1
            if self.fixk2:
                fitflags = fitflags + cv2.fisheye.CALIB_FIX_K2
            if self.fixk3:
                fitflags = fitflags + cv2.fisheye.CALIB_FIX_K3
            if self.fixk4:
                fitflags = fitflags + cv2.fisheye.CALIB_FIX_K4
        
        return fitflags




    # Get a list of human-readable strings saying what fitting options are anebled. 
    # Output: list of strings, each string describes an enabled fit option.
    def get_fitflags_strings(self):
        
        Output = []

        if self.model == 'perspective':

            if self.fixaspectratio:
                Output.append('Fix Fx = Fy')
            if self.fixcc:
                Output.append('Fix CC at ({:.1f},{:.1f})'.format(self.initial_matrix[0,2],self.initial_matrix[1,2]))
            if self.disabletangentialdist:
                Output.append('Disable Tangential Distortion')

        if self.fixk1:    
            Output.append('Disable k1')
        if self.fixk2:    
            Output.append('Disable k2')
        if self.fixk3:    
            Output.append('Disable k3')
        if self.fixk4:
            Output.append('Disable k4')

        return Output


    def set_fitflags_strings(self,fitflags_strings):

        if len(fitflags_strings) != self.nfields:
            raise ValueError('Number of fit flag string lists provided is different to number of fields!')

        # Start off with resetting everything
        self.fixaspectratio = False
        self.fixk1 = False
        self.fixk2 = False
        self.fixk3 = False
        self.fixk4 = False
        self.fixskew = True
        self.disabletangentialdist=False
        self.fixcc = False

        # Set fit flags based on provided strings
        for string in fitflags_strings:
            if string == 'Fix Fx = Fy':
                self.fixaspectratio = True
            elif string.startswith('Fix CC at'):
                coords = string.split('(')[1].split(')')[0].split(',')
                self.fix_cc(True,Cx = float(coords[0]),Cy = float(coords[1]))
            elif string == 'Disable Tangential Distortion':
                self.disabletangentialdist = True
            elif string == 'Disable k1':
                self.fixk1 = True
            elif string == 'Disable k2':
                self.fixk2 = True
            elif string == 'Disable k3':
                self.fixk3 = True
            elif string == 'Disable k4':
                self.fixk4 = True


    def fix_cc(self,fix,field=0,Cx=None,Cy=None):
        if self.model == 'fisheye':
            raise Exception('This option is not available for the fisheye camera model.')

        self.fixcc = fix
        if Cx is not None:
            self.initial_matrix[0,2] = Cx
        if Cy is not None:
            self.initial_matrix[1,2] = Cy

    def fix_k1(self,fix):
        self.fixk1 = fix

    def fix_k2(self,fix):
        self.fixk2 = fix

    def fix_k3(self,fix):
        self.fixk3 = fix

    def fix_k4(self,fix):
        self.fixk4 = fix   

    def fix_k5(self,fix):
        self.fixk5 = fix

    def fix_k6(self,fix):
        self.fixk6 = fix

    def fix_tangential(self,fix):
        self.disabletangentialdist = fix

    def fix_aspect(self,fix):
        self.fixaspectratio = fix


    def get_n_params(self):
        
        # If we're doing a perspective fit...
        if self.model == 'perspective':
            free_params = 15
            free_params = free_params - self.fixk1
            free_params = free_params - self.fixk2
            free_params = free_params - self.fixk3
            free_params = free_params - 2*self.disabletangentialdist
            free_params = free_params - self.fixaspectratio
            free_params = free_params - 2*self.fixcc

        # Or for a fisheye fit...
        elif self.model == 'fisheye':
            free_params = 14
            free_params = free_params - self.fixk1
            free_params = free_params - self.fixk2
            free_params = free_params - self.fixk3
            free_params = free_params - self.fixk4

        return free_params


    # Do a fit, using the current input data and specified fit options.
    # Output: CalibResults instance containing the fit results.
    def do_fit(self):

        # Gather up the image and object points for this field
        obj_points = []
        img_points = []
        for pointpairs,subfield in self.pointpairs:
            obj_points.append([])
            img_points.append([])
            for point in range(len(pointpairs.object_points)):
                    if pointpairs.image_points[point][subfield] is not None:
                        obj_points[-1].append(pointpairs.object_points[point])
                        img_points[-1].append(pointpairs.image_points[point][subfield])
            if len(img_points[-1]) == 0:
                    obj_points.remove([])
                    img_points.remove([])

            obj_points[-1] = np.array(obj_points[-1],dtype='float32')
            img_points[-1] = np.array(img_points[-1],dtype='float32')
            
        obj_points = np.array(obj_points)
        img_points = np.array(img_points)


        # Do the fit!
        if self.model == 'perspective':
            if int(cv2.__version__[0]) == 3:
                fit_output = cv2.calibrateCamera(obj_points,img_points,self.image_display_shape,self.initial_matrix,None,flags=self.get_fitflags())
            else:
                fit_output = cv2.calibrateCamera(obj_points,img_points,self.image_display_shape,self.initial_matrix,flags=self.get_fitflags())

            fitted_model = PerspectiveViewModel(cv2_output = fit_output)
            fitted_model.fit_options = self.get_fitflags_strings()

            return fitted_model

        elif self.model == 'fisheye':

            # Prepare input arrays in the annoying, strange way OpenCV requires them...
            obj_points = np.expand_dims(obj_points,1)
            img_points = np.expand_dims(img_points,1)
            rvecs = [np.zeros((1,1,3),dtype='float32') for i in range(len(self.pointpairs))]
            tvecs = [np.zeros((1,1,3),dtype='float32') for i in range(len(self.pointpairs))]
            fit_output = cv2.fisheye.calibrate(obj_points,img_points,self.image_display_shape,self.initial_matrix, np.zeros(4),rvecs,tvecs,flags = self.get_fitflags())
        
            fitted_model = FisheyeeViewModel(cv2_output = fit_output)
            fitted_model.fit_options = self.get_fitflags_strings()

            return fitted_model


    # Set the primary point pairs to be fit to - these are the ones used for the extrinsics fit.
    # Input: PointPairs, a Calcam PointPairs object containing the points you want to fit.
    # In future, I want to add a similar function for adding additional intrinsics constraints 
    # (e.g. test pattern images taken in the lab)
    def set_pointpairs(self,pointpairs,subview=0):
        # The primary point pairs are always first in the list.
        self.pointpairs[0] = (pointpairs,subview)
        
        # Set up image properties from the image that belongs to these point pairs
        self.image_display_shape = tuple(pointpairs.image_shape)


        # Initialise initial values for fitting
        initial_matrix = np.zeros((3,3))
        initial_matrix[0,0] = 1200    # Fx
        initial_matrix[1,1] = 1200    # Fy
        initial_matrix[2,2] = 1
        initial_matrix[0,2] = self.image_display_shape[0]/2    # Cx
        initial_matrix[1,2] = self.image_display_shape[1]/2    # Cy
        self.initial_matrix = initial_matrix


    def add_intrinsics_pointpairs(self,pointpairs,subview=0):
        
        self.pointpairs.append( (pointpairs,subview) )


    def clear_intrinsics_pointpairs(self):
        del self.pointpairs[1:]



