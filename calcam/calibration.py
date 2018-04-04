'''
* Copyright 2015-2017 European Atomic Energy Community (EURATOM)
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
from .io import ZipSaveFile
from scipy.interpolate import interp1d
from scipy.ndimage.measurements import center_of_mass as CoM

opencv_major_version = int(cv2.__version__[0])

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
        rotationMatrix = np.matrix(cv2.Rodrigues(self.rvec)[0]).transpose()

        # x,y and z components of the LOS vectors
        x = rotationMatrix[0,0]*x_norm + rotationMatrix[0,1]*y_norm + rotationMatrix[0,2]*z_norm
        y = rotationMatrix[1,0]*x_norm + rotationMatrix[1,1]*y_norm + rotationMatrix[1,2]*z_norm
        z = rotationMatrix[2,0]*x_norm + rotationMatrix[2,1]*y_norm + rotationMatrix[2,2]*z_norm

        # Return an array the same shape as the input x and y pixel arrays + an extra dimension
        out = np.stack([x,y,z],axis=-1)

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
                        'cam_matrix':self.cam_matrix,
                        'dist_coeffs':list(np.squeeze(self.kc)),
                        'rvec':list(np.squeeze(self.rvec)),
                        'tvec':list(np.squeeze(self.tvec))
                    }

        return out_dict






# Class representing a perspective camera model.
class PerspectiveViewModel(ViewModel):

    # Can be initialised either with the output of opencv.CalibrateCamera or from 
    # a dictionary containing the model coefficients
    def __init__(self,cv2_output=None,coeffs_dict=None):

        if cv2_output is None and coeffs_dict is None:
            raise ValueError('Either OpenCV output or coefficient dictionary must be defined!')

        self.model = 'perspective'

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
        
        self.rvec = np.zeros([3,1])
        self.rvec[:,0] = coeffs_dict['rvec']
        self.tvec = np.zeros([3,1])
        self.tvec[:,0] = coeffs_dict['tvec']



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

        return cv2.undistort(im,self.cam_matrix,self.kc)




# Class representing a perspective camera model.
class FisheyeeViewModel(ViewModel):

    # Can be initialised either with the output of opencv.CalibrateCamera or from 
    # a dictionary containing the model coefficients
    def __init__(self,cv2_output=None,coeffs_dict=None):

        if cv2_output is None and coeffs_dict is None:
            raise ValueError('Either OpenCV output or coefficient dictionary must be defined!')

        self.model = 'fisheye'

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

        return cv2.fisheye.undistortImage(im,self.cam_matrix,self.kc)



# Class to represent a camera calibration.
class Calibration():

    def __init__(self,load_file = None):

        self.images = []
        self.n_subviews = 0
        self.subview_lookup = None
        self.pointpairs = []

        self.chessboard_images = []

        if cv2_output is not None:

            self.reprojection_error = cv2_output[0]
            self.cam_matrix = cv2_output[1]
            self.kc = cv2_output[2]
            self.rvec = cv2_output[3][0][0].T
            self.tvec = cv2_output[4][0][0].T

        elif coeffs_dict is not None:
            self.load_from_dict(load_file)


    # Get the pupil position for given pixels or named sub-view.
    def get_pupilpos(self,x=None,y=None,coords='display',subview=None):

        # If we're given pixel coordinates
        if x is not None or y is not None:

            # Validate the input coordinates
            if x is None or y is None:
                raise ValueError("X pixels and Y pixels must both be specified!")
            if np.shape(x) != np.shape(y):
                raise ValueError("X pixels array and Y pixels array must be the same size!")

            # Convert pixel coords if we need to
            if coords.lower() == 'original':
                x,y = self.images[0].transform.original_to_display_coords(x,y)

            # Output should be the same shape as input + an extra length 3 axis
            output = np.zeros(np.shape(x) + (3,))

            # An array the same size as output sepcifying which sub-view calibration to use
            subview = np.tile( self.suvbiew_lookup(x,y), np.shape(x) + (3,) )

            for nview in range(self.n_subviews):
                pupilpos = np.tile( view_models[nview].get_pupilpos(), x.shape + (1,) )
                output[subview == nview] = pupilpos[subview == nview]

        else:

            if self.n_subview == 1:
                subview = 0
            elif subview is None:
                raise ValueError('This calibration contains multiple sub-views; either pixel coordinates or subview number must be specified!')

            output = self.view_models[subview].get_pupilpos()


        return output


    # Get the horizontal and vertical field of view of a given sub-view
    def get_fov(self,subview=None):

        if subview is None and self.n_subviews > 1:
            raise ValueError('This calibration contains multuple sub-views; subview number must be specified!')
        elif subview is None and self.n_subviews == 1:
            subview = 0

        # Calculate FOV by looking at the angle between sight lines at the image extremes
        vcntr,hcntr = CoM( self.images[0].subviewmask == subview )

        vcntr = int(np.round(vcntr))
        hcntr = int(np.round(hcntr))

        # Horizontal extent of sub-field
        h_extent = np.argwhere(self.images[0].subview_mask[vcntr,:] == subview)

        # Horizontal FOV
        norm1 = self.normalise(h_extent.min()-0.5,vcntr)
        norm2 = self.normalise(h_extent.max()+0.5,vcntr)
        fov_h = 180*(np.arctan(norm2[0]) - np.arctan(norm1[0])) / 3.141592635

        v_extent = np.argwhere(self.images[0].subviewmask[:,hcntr] == subview)

        # Y FOV
        norm1 = self.normalise(xcntr,fieldpoints.min()-0.5,field=field)
        norm2 = self.normalise(xcntr,fieldpoints.max()+0.5,field=field)
        fov_v = 180*(np.arctan(norm2[1]) - np.arctan(norm1[1])) / 3.141592635

        return fov_h, fov_v


    # Get line-of-sight directions
    def get_los_direction(self,x=None,y=None,coords='Display'):

        # If we're given pixel coordinates
        if x is not None or y is not None:

            # Validate the input coordinates
            if x is None or y is None:
                raise ValueError("X pixels and Y pixels must both be specified!")
            if np.shape(x) != np.shape(y):
                raise ValueError("X pixels array and Y pixels array must be the same size!")

            # If given original coordinates, convert to display to do the calculation
            if coords.lower() == 'original':
                x,y = self.images[0].transform.original_to_display_coords(x,y)

        # If not given any, generate our own image coordinates covering every pixel.
        else:
            if coords.lower() == 'original':
                shape = self.image.get_original_shape()
                x,y = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))
                x,y = self.image.transform.original_to_display_coords(x,y)
            else:
                shape = self.image.get_display_shape()
                x,y = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))


        # Output should be the same shape as input + an extra length 3 axis
        output = np.zeros(np.shape(x) + (3,))

        # An array the same size as output sepcifying which sub-view calibration to use
        subview = np.tile( self.suvbiew_lookup(x,y), np.shape(x) + (3,) )

        for nview in range(self.n_subviews):
            losdir = self.view_models[nview].get_los_direction(x,y)
            output[subview == nview] = losdir[subview == nview]

        return output


    def project_points(self,points_3d,coords='display',check_occlusion_by=None,fill_value=np.nan):

        # If we have more than 1 sub-view, we need a fill value
        if self.n_subviews > 1 and fill_value is None:
            raise Exception('Cannot return coordinates for hidden points because this fit has multiple sub-fields!')

