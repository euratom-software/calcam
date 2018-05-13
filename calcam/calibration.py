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
import os
import json

from .io import ZipSaveFile
from scipy.ndimage.measurements import center_of_mass as CoM
from .coordtransformer import CoordTransformer
from .pointpairs import PointPairs


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

        return cv2.fisheye.undistortImage(im,self.cam_matrix,self.kc)



# Class to represent a camera calibration.
class Calibration():

    def __init__(self,load_file = None):

        self.image = None
        self.pointpairs = None

        self.cad_config = None

        self.geometry = CoordTransformer()

        self.subview_mask = None
        self.n_subviews = 1

        self.view_models = []
        self.intrinsics_constraints = []

        self.history = ['']

        self.subview_names = []

        self.pixel_size = None

        if load_file is not None:
            self.load(load_file)


    def set_pointpairs(self,pointpairs):

        if pointpairs is not None:
            if pointpairs.n_subviews != self.n_subviews:
                raise ValueError('Provided PointPairs object has a different number of sub-views ({:d}) to this calibration ({:d})!'.format(pointpairs.n_subviews,self.n_subviews))

        self.pointpairs = pointpairs


    def load(self,filename):

        self.name = os.path.split(filename)[-1].split('.')[0]

        with ZipSaveFile(filename,'r') as save_file:

            # Load the image
            self.image = cv2.imread(os.path.join(save_file.get_temp_path(),'image.png'))
            if self.image is not None:
                if len(self.image.shape) == 3:
                    if self.image.shape[2] == 3:
                        self.image[:,:,:3] = self.image[:,:,2::-1]

            # Load the field mask
            self.subview_mask = cv2.imread(os.path.join(save_file.get_temp_path(),'subview_mask.png'))[:,:,0]

            # Load the general information
            with save_file.open_file('calibration.json','r') as f:
                meta = json.load(f)

            self.geometry = CoordTransformer(meta['image_transform_actions'],meta['orig_x'],meta['orig_y'],meta['orig_paspect'])
            self.n_subviews = meta['n_subviews']
            self.history = meta['history']
            self.subview_names = meta['subview_names']
            self.calib_type = meta['calib_type']
            self.pixel_size = meta['pixel_size']
            
            # Load the primary point pairs
            with save_file.open_file('pointpairs.csv','r') as ppf:
                self.pointpairs = PointPairs(ppf)

            # Load fit results
            self.view_models = []
            for nview in range(self.n_subviews):
                with save_file.open_file('calib_params_{:d}.json'.format(nview),'r') as f:
                    self.view_models.append(ViewModel.from_dict(json.load(f)))

            # Load CAD config
            if 'cad_config.json' in save_file.list_contents():
                with save_file.open_file('cad_config.json','r') as f:
                    self.cad_config = json.load(f)

            # Load any intrinsics constraints
            for i in range( (len( [f for f in save_file.list_contents() if f.startswith('intrinsics_constraints')] ) - 1) // 2 ):

                im = cv2.imread(os.path.join(save_file.get_temp_path(),'intrinsics_constraints','im_{:03d}.png'.format(i)))
                if im is not None:
                    if len(im.shape) == 3:
                        if im.shape[2] == 3:
                            im[:,:,:3] = im[:,:,2::-1]
                
                with save_file.open_file(os.path.join('intrinsics_constraints','points_{:03d}.csv'.format(i)),'r') as ppf:
                    pp = PointPairs(ppf)

                self.intrinsics_constraints.append([im,pp])


    def set_image(self,image,coords='Display',transform_actions = [],subview_mask=None,pixel_aspect=1.,subview_names = [],pixel_size=None):

        self.image = image.copy()

        # If the array isn't already 8-bit int, make it 8-bit int...
        if self.image.dtype != np.uint8:
            # If we're given a higher bit-depth integer, it's easy to downcast it.
            if self.image.dtype == np.uint16 or self.image.dtype == np.int16:
                self.image = np.uint8(self.image/2**8)
            elif self.image.dtype == np.uint32 or self.image.dtype == np.int32:
                self.image = np.uint8(self.image/2**24)
            elif self.image.dtype == np.uint64 or self.image.dtype == np.int64:
                self.image = np.uint8(self.image/2**56)
            # Otherwise, scale it in a floating point way to its own max & min
            # and strip out any transparency info (since we can't be sure of the scale used for transparency)
            else:

                if self.image.min() < 0:
                    self.image = self.image - self.image.min()

                if len(self.image.shape) == 3:
                    if self.image.shape[2] == 4:
                        self.image = self.image[:,:,:-1]

                self.image = np.uint8(255.*(self.image - self.image.min())/(self.image.max() - self.image.min()))

        if subview_mask is None:
            self.n_subviews = 1
            self.subview_mask = np.zeros(image.shape[:2],dtype='uint8')
            self.subview_names = ['Image']
        else:
            self.subview_mask = subview_mask
            self.n_subviews = subview_mask.max() + 1
            if len(subview_names) == self.n_subviews:
                self.subview_names = subview_names
            else:
                self.subview_names = []
                for n in range(self.n_subviews):
                    self.subview_names.append('View {:d}'.format(n+1))

        self.geometry = CoordTransformer()

        self.geometry.set_transform_actions(transform_actions)
        self.geometry.pixel_aspectratio = pixel_aspect

        if coords.lower() == 'original':
            self.geometry.x_pixels = self.image.shape[1]
            self.geometry.y_pixels = self.image.shape[0]
            self.image = self.geometry.original_to_display_image(self.image)
        else:
            shape = self.geometry.display_to_original_shape(self.image.shape[1::-1])
            self.geometry.x_pixels = shape[0]
            self.geometry.y_pixels = shape[1]

        self.pixel_size = pixel_size


    def save(self,filename):

        with ZipSaveFile(filename,'w') as save_file:

            # Save the image
            if self.image is not None:
                im_out = self.image
                if len(im_out) == 3:
                    if im_out.shape[2] == 3:
                        im_out[:,:,:3] = im_out[:,:,2::-1]
                cv2.imwrite(os.path.join(save_file.get_temp_path(),'image.png'),im_out)


            # Save the field mask
            cv2.imwrite(os.path.join(save_file.get_temp_path(),'subview_mask.png'),self.subview_mask)

            # Save the general information
            meta = {
                    'n_subviews': self.n_subviews,
                    'history':self.history,
                    'pixel_size':self.pixel_size,
                    'orig_x':self.geometry.x_pixels,
                    'orig_y':self.geometry.y_pixels,
                    'orig_paspect':self.geometry.pixel_aspectratio,
                    'image_transform_actions':self.geometry.transform_actions,
                    'subview_names':self.subview_names,
                    'calib_type':self.calib_type
            }

            for nview in range(self.n_subviews):
                if self.view_models[nview] is not None:
                    with save_file.open_file('calib_params_{:d}.json'.format(nview),'w') as f:
                        json.dump(self.view_models[nview].get_dict(),f,indent=4,sort_keys=True)

            with save_file.open_file('calibration.json','w') as f:
                json.dump(meta,f,indent=4,sort_keys=True)


            # Save the primary point pairs (if any)
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




    def subview_lookup(self,x,y):

        good_mask = (x >= -0.5) & (y >= -0.5) & (x < self.geometry.get_display_shape()[0] - 0.5) & (y < self.geometry.get_display_shape()[1] - 0.5)
        
        x[ good_mask == 0 ] = 0
        y[ good_mask == 0 ] = 0

        out = self.subview_mask[np.round(y).astype('int'),np.round(x).astype('int')]

        out[good_mask == 0] = -1

        return out



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
                x,y = self.geometry.original_to_display_coords(x,y)

            if subview is None:

                # Output should be the same shape as input + an extra length 3 axis
                output = np.zeros(np.shape(x) + (3,))

                # An array the same size as output sepcifying which sub-view calibration to use
                subview_mask = self.subview_lookup(x,y)
                subview_mask = np.tile( subview_mask, [3] + [1]*x.ndim )
                subview_mask = np.moveaxis(subview_mask,0,subview_mask.ndim-1)

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


    # Get the horizontal and vertical field of view of a given sub-view
    def get_fov(self,subview=None):

        if subview is None and self.n_subviews > 1:
            raise ValueError('This calibration contains multuple sub-views; subview number must be specified!')
        elif subview is None and self.n_subviews == 1:
            subview = 0

        # Calculate FOV by looking at the angle between sight lines at the image extremes
        vcntr,hcntr = CoM( self.subview_mask == subview )

        vcntr = int(np.round(vcntr))
        hcntr = int(np.round(hcntr))

        # Horizontal extent of sub-field
        h_extent = np.argwhere(self.subview_mask[vcntr,:] == subview)

        # Horizontal FOV
        norm1 = self.normalise(h_extent.min()-0.5,vcntr)
        norm2 = self.normalise(h_extent.max()+0.5,vcntr)
        fov_h = 180*(np.arctan(norm2[0]) - np.arctan(norm1[0])) / 3.141592635

        v_extent = np.argwhere(self.subview_mask[:,hcntr] == subview)

        # Vertical field of view
        norm1 = self.normalise(hcntr,v_extent.min()-0.5)
        norm2 = self.normalise(hcntr,v_extent.max()+0.5)
        fov_v = 180*(np.arctan(norm2[1]) - np.arctan(norm1[1])) / 3.141592635

        return fov_h, fov_v


    # Get line-of-sight directions
    def get_los_direction(self,x=None,y=None,coords='Display',subview=None):

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
                shape = self.image.get_original_shape()
                x,y = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))
                x,y = self.image.transform.original_to_display_coords(x,y)
            else:
                shape = self.image.get_display_shape()
                x,y = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))


        # Output should be the same shape as input + an extra length 3 axis
        output = np.zeros(np.shape(x) + (3,))


        if subview is None:

            # An array the same size as output sepcifying which sub-view calibration to use
            subview_mask = self.subview_lookup(x,y)
            subview_mask = np.tile( subview_mask, [3] + [1]*x.ndim )
            subview_mask = np.moveaxis(subview_mask,0,subview_mask.ndim-1)
            
            for nview in range(self.n_subviews):
                losdir = self.view_models[nview].get_los_direction(x,y)
                output[subview_mask == nview] = losdir[subview_mask == nview]

        else:

            output = self.view_models[subview].get_los_direction(x,y)


        return np.squeeze(output)


    def project_points(self,points_3d,coords='display',check_occlusion_by=None,fill_value=np.nan):

        # This will be the output
        points_2d = []


        for nview in range(self.n_subviews):

            # Do the point projection for this sub-fov
            p2d =  self.view_models[nview].project_points(points_3d) 

            # If we're checking whether the 3D points are actually visible...
            if fill_value is not None:

                # Create a mask indicating where the projected points are outside their correct
                # sub-view (including out of the image completely)
                wrong_subview_mask = self.subview_lookup(p2d[:,0],p2d[:,1]) != nview

                # If given a CAD model to check occlusion against, compare the distances to the 3D points with
                # the distance to a surface in the CAD model
                if check_occlusion_by is not None:

                    point_vectors = points_3d - np.tile( self.view_models[nview].get_pupilpos() , (points_3d.shape[0],1) )
                    point_distances = np.sqrt( np.sum(point_vectors**2),axis= 1)

                    # Ray cast to get the ray lengths
                    ray_lengths = raycast_sightlines(self,check_occlusion_by,p2d[:,0],p2d[:,1],verbose=False,force_subview=nview)

                    # The 3D points are invisible where the distance to the point is larger than the
                    # ray length
                    occluded_mask = ray_lengths < point_distances

                    p2d[ np.tile(occluded_mask,(1,2))  ] = fill_value


                p2d[ np.tile(wrong_subview_mask,(1,2)) ] = fill_value


            # If the user asked for things in original coordinates, convert them.
            if coords.lower() == 'display':
                p2d[:,0],p2d[:,1] = self.geometry.display_to_original_coords(p2d[:,0],p2d[:,1])

            points_2d.append(p2d)

        return points_2d


    def undistort_image(self,image,coords='display'):

        if coords.lower() == 'original':
            image = self.geometry.original_to_display_image(image)

        image_out = np.zeros(image.shape)

        for nview in range(self.n_subviews):

            subview_mask = self.subview_mask == nview

            im_in = im.copy()
            im_in[ np.tile(subview_mask,(1,1,im_in.shape[2])) == 0 ] = 0
            undistorted =  self.view_models[nview].undistort_image(im_in)
            image_out[ np.tile(subview_mask,(1,1,im_in.shape[2])) ] = undistorted[ np.tile(subview_mask,(1,1,im_in.shape[2])) ]

        return image_out


    def get_cam_to_lab_rotation(self,subview=None):

        if subview is None:
            if self.n_subviews > 1:
                raise Exception('This calibration has multiple sub-views; you must specify a pixel location to get_cam_to_lab_rotation!')
            else:
                subview = 0

        return self.view_models[subview].get_cam_to_lab_rotation()


    def normalise(self,x,y,subview=None):

        if subview is None:
            if self.n_subviews > 1:
                raise Exception('This calibration has multiple sub-views; you must specify a pixel location to normalise!')
            else:
                subview = 0

        return self.view_models[subview].normalise(x,y)



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
            free_params = free_params - 2*fixcc

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


    def add_intrinsics_pointpairs(self,pointpairs,subfield=0):
        
        self.pointpairs.append(pointpairs,subfield)


    def clear_intrinsics_pointpairs(self):
        del self.pointpairs[1:]