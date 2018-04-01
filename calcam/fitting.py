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
import csv
from . import paths, pointpairs
import os
import pickle
from .coordtransformer import CoordTransformer
from scipy.ndimage.measurements import center_of_mass as CoM
import copy
from scipy.io.netcdf import netcdf_file
from .image import Image as CalCam_Image
from .pointpairs import PointPairs

opencv_major_version = int(cv2.__version__[0])



# The 'fitter' class 
class Fitter:

    def __init__(self):
        
        self.pointpairs = [None]

        self.nfields = 0
        
        self.transform = None


    def set_model(self,model,field=0):

        if model == 'perspective':
            self.model[field] = model
        elif model == 'fisheye':
            if opencv_major_version < 3:
                raise ValueError('Fisheye model fitting requires OpenCV 3 or newer, you have ' + cv2.__version__)
            else:
                self.model[field] = model
        else:
            raise ValueError('Unknown model type ' + model)


    # Based on the current fit parameters, returns the fit flags
    # in the format required by OpenCV fitting functions.
    # Output: fitflags (long int)
    def get_fitflags(self,field=0):

        if self.model[field] == 'perspective':
            fitflags = cv2.CALIB_USE_INTRINSIC_GUESS

            if self.fixaspectratio[field]:
                    fitflags = fitflags + cv2.CALIB_FIX_ASPECT_RATIO
            if self.fixk1[field]:
                    fitflags = fitflags + cv2.CALIB_FIX_K1
            if self.fixk2[field]:
                    fitflags = fitflags + cv2.CALIB_FIX_K2
            if self.fixk3[field]:
                    fitflags = fitflags + cv2.CALIB_FIX_K3
            if self.disabletangentialdist[field]:
                    fitflags = fitflags + cv2.CALIB_ZERO_TANGENT_DIST
            if self.fixcc[field]:
                fitflags = fitflags + cv2.CALIB_FIX_PRINCIPAL_POINT
        
        elif self.model[field] == 'fisheye':

            fitflags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS + cv2.fisheye.CALIB_FIX_SKEW

            if self.fixk1[field]:
                fitflags = fitflags + cv2.fisheye.CALIB_FIX_K1
            if self.fixk2[field]:
                fitflags = fitflags + cv2.fisheye.CALIB_FIX_K2
            if self.fixk3[field]:
                fitflags = fitflags + cv2.fisheye.CALIB_FIX_K3
            if self.fixk4[field]:
                fitflags = fitflags + cv2.fisheye.CALIB_FIX_K4
        
        return fitflags


    # Get a list of human-readable strings saying what fitting options are anebled. 
    # Output: list of strings, each string describes an enabled fit option.
    def get_fitflags_strings(self,field=0):
        
        Output = []

        if self.model[field] == 'perspective':

            if self.fixaspectratio[field]:
                Output.append('Fix Fx = Fy')
            if self.fixcc[field]:
                Output.append('Fix CC at ({:.1f},{:.1f})'.format(self.initial_matrix[field][0,2],self.initial_matrix[field][1,2]))
            if self.disabletangentialdist[field]:
                Output.append('Disable Tangential Distortion')

        if self.fixk1[field]:    
            Output.append('Disable k1')
        if self.fixk2[field]:    
            Output.append('Disable k2')
        if self.fixk3[field]:    
            Output.append('Disable k3')
        if self.fixk4[field]:
            Output.append('Disable k4')


        return Output

    def set_fitflags_strings(self,fitflags_strings):

        if len(fitflags_strings) != self.nfields:
            raise ValueError('Number of fit flag string lists provided is different to number of fields!')

        # Start off with resetting everything
        self.fixaspectratio = [False] * self.nfields
        self.fixk1 = [False] * self.nfields
        self.fixk2 = [False] * self.nfields
        self.fixk3 = [False] * self.nfields
        self.fixk4 = [False] * self.nfields
        self.fixskew = [True] * self.nfields
        self.disabletangentialdist=[False] * self.nfields
        self.fixcc = [False] * self.nfields

        # Set fit flags based on provided strings
        for field in range(self.nfields):
            for string in fitflags_strings[field]:
                if string == 'Fix Fx = Fy':
                    self.fixaspectratio[field] = True
                elif string.startswith('Fix CC at'):
                    coords = string.split('(')[1].split(')')[0].split(',')
                    self.fix_cc(True,field=field,Cx = float(coords[0]),Cy = float(coords[1]))
                elif string == 'Disable Tangential Distortion':
                    self.disabletangentialdist[field] = True
                elif string == 'Disable k1':
                    self.fixk1[field] = True
                elif string == 'Disable k2':
                    self.fixk2[field] = True
                elif string == 'Disable k3':
                    self.fixk3[field] = True
                elif string == 'Disable k4':
                    self.fixk4[field] = True



    def fix_cc(self,fix,field=0,Cx=None,Cy=None):
        if self.model[field] == 'fisheye':
            raise Exception('This option is not available for the fisheye camera model.')

        self.fixcc[field] = fix
        if Cx is not None:
            self.initial_matrix[field][0,2] = Cx
        if Cy is not None:
            self.initial_matrix[field][1,2] = Cy


    # Do a fit, using the current input data and specified fit options.
    # Output: CalibResults instance containing the fit results.
    def do_fit(self):

        # A new results object to put the results in:
        Results = CalibResults()
        Results.image_display_shape = self.image_display_shape
        Results.fieldmask = self.fieldmask
        Results.nfields = self.nfields
        Results.fitoptions = [self.get_fitflags_strings(i) for i in range(self.nfields)]
        Results.transform = self.transform
        Results.field_names = self.field_names
        Results.type = 'fit'
        Results.fit_params = []

        for pointpairs in self.pointpairs:
            Results.objectpoints.append(pointpairs.objectpoints)
            Results.imagepoints.append(pointpairs.imagepoints)

        Results.image = self.pointpairs[0].image

        # Do a separate fit for each field
        for field in range(self.nfields):
            # Gather up the image and object points for this field
            ObjPoints = []
            ImgPoints = []
            for PointPairs in self.pointpairs:
                ObjPoints.append([])
                ImgPoints.append([])
                for point in range(len(PointPairs.objectpoints)):
                        if PointPairs.imagepoints[point][field] is not None:
                            ObjPoints[-1].append(PointPairs.objectpoints[point])
                            ImgPoints[-1].append(PointPairs.imagepoints[point][field])
                if len(ImgPoints[-1]) == 0:
                        ObjPoints.remove([])
                        ImgPoints.remove([])

                ObjPoints[-1] = np.array(ObjPoints[-1],dtype='float32')
                ImgPoints[-1] = np.array(ImgPoints[-1],dtype='float32')
                
            ObjPoints = np.array(ObjPoints)
            ImgPoints = np.array(ImgPoints)

            # Do the fit!
            if self.model[field] == 'perspective':
                if int(cv2.__version__[0]) == 3:
                    FitOutput = cv2.calibrateCamera(ObjPoints,ImgPoints,self.image_display_shape,self.initial_matrix[field].copy(),None,flags=self.get_fitflags(field))
                else:
                    FitOutput = cv2.calibrateCamera(ObjPoints,ImgPoints,self.image_display_shape,self.initial_matrix[field].copy(),flags=self.get_fitflags(field))
            elif self.model[field] == 'fisheye':

                # Prepare input arrays in the annoying, strange way OpenCV requires them...
                ObjPoints = np.expand_dims(ObjPoints,1)
                ImgPoints = np.expand_dims(ImgPoints,1)
                rvecs = [np.zeros((1,1,3),dtype='float32') for i in range(len(self.pointpairs))]
                tvecs = [np.zeros((1,1,3),dtype='float32') for i in range(len(self.pointpairs))]
                FitOutput = cv2.fisheye.calibrate(ObjPoints,ImgPoints,self.image_display_shape,self.initial_matrix[field].copy(), np.zeros(4),rvecs,tvecs,flags = self.get_fitflags(field))
            
            Results.fit_params.append(FieldFit(self.model[field],FitOutput))

        return Results


    # Set the primary point pairs to be fit to - these are the ones used for the extrinsics fit.
    # Input: PointPairs, a Calcam PointPairs object containing the points you want to fit.
    # In future, I want to add a similar function for adding additional intrinsics constraints 
    # (e.g. test pattern images taken in the lab)
    def set_PointPairs(self,PointPairs,save_options=False):
        # The primary point pairs are always first in the list.
        self.pointpairs[0] = PointPairs
        
        # Set up image properties from the image that belongs to these point pairs
        self.fieldmask = PointPairs.image.transform.original_to_display_image(PointPairs.image.fieldmask)
        self.nfields = PointPairs.image.n_fields
        self.image_display_shape = tuple(PointPairs.image.transform.get_display_shape())
        self.transform = PointPairs.image.transform
        self.field_names = PointPairs.image.field_names


        # Default fit options, changing these affects the fit
        self.model = ['perspective'] * self.nfields
        self.fixaspectratio = [True] * self.nfields
        self.fixk1 = [False] * self.nfields
        self.fixk2 = [False] * self.nfields
        self.fixk3 = [True] * self.nfields
        self.fixk4 = [False] * self.nfields
        self.fixskew = [True] * self.nfields
        self.disabletangentialdist=[False] * self.nfields
        self.fixcc = [False] * self.nfields

        # Initialise initial values for fitting
        initial_matrix = np.zeros((3,3))
        initial_matrix[0,0] = 1200    # Fx
        initial_matrix[1,1] = 1200    # Fy
        initial_matrix[2,2] = 1
        initial_matrix[0,2] = self.image_display_shape[0]/2    # Cx
        initial_matrix[1,2] = self.image_display_shape[1]/2    # Cy
        self.initial_matrix = [initial_matrix] * self.nfields


    def add_intrinsics_pointpairs(self,pointpairs):
        try:
            self.pointpairs = self.pointpairs + pointpairs
        except TypeError:
            self.pointpairs.append(pointpairs)


    def remove_intrinsics_pointpairs(self,pointpairs):
        try:
            for pp in pointpairs:
                self.pointpairs.remove(pointpairs)
        except ValueError:
            self.pointpairs.remove(pointpairs)


    def clear_intrinsics_pointpairs(self):
        del self.pointpairs[1:]


# Class for storing the calibration results.
# Has methods for post-processing the results to give useful information
# and for loading and saving the results
class CalibResults:

    def __init__(self,SaveName=None):

        if SaveName is not None:
            self.fit_params = []
            self.load(SaveName)
        else:
            self.nfields = 1
            self.fit_params = [None]
            self.image = None
            self.objectpoints = []
            self.imagepoints = []
            self.type = 'manual_alignment'
            self.fitoptions = []
            self.transform = None

    # Get the camera position in the lab (i.e. CAD model) coordinate system
    # Optional inputs: x_pixels, y_pixels - array-likes containing the X and Y pixel coordinates (floats) you want to know the pupil position for. 
    #                                       This is only useful for optical systems with split fields-of-view, where not all pixels 
    #                                       necesserily have the same pupil position. MUST be specified for images with split fields of view.
    #                  field - For split field-of-view cameras, the number of the sub-field you want the pupil position for (int).
    #                  Coords - If specifying x_pixels and y_pixels, this specifies whether the provided pixel coodinates are in display
    #                           or original coordinates. (String, either 'Display' or 'Original')
    # Output: 3-element array containing the camera pupil position (X,Y,Z) in metres (floats)
    def get_pupilpos(self,x_pixels=None,y_pixels=None,field=None,Coords='Display'):

        if x_pixels is not None or y_pixels is not None:
            if x_pixels is None or y_pixels is None:
                raise ValueError("X pixels and Y pixels must both be specified!")
            if np.shape(x_pixels) != np.shape(y_pixels):
                raise ValueError("X pixels array and Y pixels array must be the same size!")

            # Flatten everything and create output array        
            oldshape = np.shape(x_pixels)

            x_pixels = np.reshape(x_pixels,np.size(x_pixels),order='F')
            y_pixels = np.reshape(y_pixels,np.size(y_pixels),order='F')
            out = np.zeros(np.shape(x_pixels) + (3,))

            # Convert pixel coords if we need to
            if Coords.lower() == 'original':
                x_pixels,y_pixels = self.transform.original_to_display_coords(x_pixels,y_pixels)

            # Identify which sub-field each pixel is in
            pointfield = self.fieldmask[y_pixels.round().astype(int),x_pixels.round().astype(int)]
            if np.size(pointfield) == 1:
                pointfield = [pointfield]

            for i in range(np.size(x_pixels)):
                out[i,:] = self.get_pupilpos(field=pointfield[i])

            return np.reshape(out,oldshape + (3,),order='F')

        else:
        
            if field is None:
                if self.nfields == 1:
                    field = 0
                else:
                    raise Exception('This calibration has multiple sub-fields; you must specify a pixel location to get_pupilpos!')

            rotation_matrix = np.matrix(cv2.Rodrigues(self.fit_params[int(field)].rvec)[0])
            CamPos = np.matrix(self.fit_params[int(field)].tvec)
            CamPos = - (rotation_matrix.transpose() * CamPos)
            CamPos = np.array(CamPos)

            return np.array([CamPos[0][0],CamPos[1][0],CamPos[2][0]])


    # Get X and Y field of view of the camera (X and Y being horizontal and vertical of the detector)
    # Optional inputs: field - for cameras with split fields-of-view, the sub-field number to get the FOV of (int).
    #                  FullChipWithoutDistortion - ignores distortion and any split field-of-view, just returns the FOV
    #                                              for the full chip as if there was no distortion. Used in Calcam.Render() but probably not useful otherwise (bool).
    # Output: 2-element tuple with field of view in degrees: (horiz, vert) (tuple of floats)
    def get_fov(self,field=None,FullChipWithoutDistortion=False):

        if field is None:
                if self.nfields > 1:
                    raise Exception('This fit has multiple sub-fields; must specify a field to get_fov!')
                else:
                    field = 0

        if not FullChipWithoutDistortion:

                # Calculate FOV by looking at the angle between sight lines at the image extremes
                ycntr,xcntr = CoM( (self.fieldmask + 1) * (self.fieldmask == field) )

                ycntr = int(np.round(ycntr))
                xcntr = int(np.round(xcntr))

                # Horizontal extent of sub-field
                fieldpoints = np.argwhere(self.fieldmask[ycntr,:] == field)

                # X FOV
                norm1 = self.normalise(fieldpoints.min()-0.5,ycntr,field=field)
                norm2 = self.normalise(fieldpoints.max()+0.5,ycntr,field=field)
                fovx = 180*(np.arctan(norm2[0]) - np.arctan(norm1[0])) / 3.141592635

                fieldpoints = np.argwhere(self.fieldmask[:,xcntr] == field)

                # Y FOV
                norm1 = self.normalise(xcntr,fieldpoints.min()-0.5,field=field)
                norm2 = self.normalise(xcntr,fieldpoints.max()+0.5,field=field)
                fovy = 180*(np.arctan(norm2[1]) - np.arctan(norm1[1])) / 3.141592635

        else:
                # Find FOV from focal length and number of pixels
                [fovx,fovy,_,_,_] = cv2.calibrationMatrixValues(self.fit_params[field].cam_matrix,self.image_display_shape,0,0)

        return fovx,fovy


    # Get a rotation matrix which rotates from the camera frame to the lab frame
    # Optional input: field, for split FOV cameras, which field of view to do.
    # Output: 3D rotation matrix (Numpy matrix)
    def get_cam_to_lab_rotation(self,field=None):

        if field is None:
                if self.nfields > 1:
                    raise Exception('This calibration has multiple sub-fields; you must specify a pixel location to get_cam_to_lab_rotation!')
                else:
                    field = 0

        rotation_matrix = np.matrix(cv2.Rodrigues(self.fit_params[field].rvec)[0])

        return rotation_matrix.transpose()


    # Given pixel coordinates x,y, return the NORMALISED
    # coordinates of the corresponding un-distorted points.
    def normalise(self,x,y,field):

        if np.shape(x) != np.shape(y):
            raise ValueError("X and Y input arrays must be the same size!")

        if self.fit_params[field].model == 'fisheye' and opencv_major_version < 3:
            raise Exception('Fisheye model distortion calculation requires OpenCV 3 or newer! Your version is ' + cv2.__version__)

        # Flatten everything and create output array        
        oldshape = np.shape(x)
        x = np.reshape(x,np.size(x),order='F')
        y = np.reshape(y,np.size(y),order='F')

        input_points = np.zeros([x.size,1,2])
        for point in range(len(x)):
            input_points[point,0,0] = x[point]
            input_points[point,0,1] = y[point]

        if self.fit_params[field].model == 'perspective':
            undistorted = cv2.undistortPoints(input_points,self.fit_params[field].cam_matrix,self.fit_params[field].kc)
        elif self.fit_params[field].model == 'fisheye':
            undistorted = cv2.fisheye.undistortPoints(input_points,self.fit_params[field].cam_matrix,self.fit_params[field].kc)

        undistorted = np.swapaxes(undistorted,0,1)[0]

        return np.reshape(undistorted[:,0],oldshape,order='F') , np.reshape(undistorted[:,1],oldshape,order='F')
 

    # Get the sight-line direction(s) for given pixel coordinates, as unit vector(s) in the lab frame.
    # Input: x_pixel and y_pixel - array-like, x and y pixel coordinates (floats or arrays/lists of floats)
    # Optional inputs: ForceField - for split field cameras, get the sight line direction as if the pixel
    #                               was part of the specified subfield, even if it isn't really (int)
    #                  Coords - whether the input x_pixel and y_pixel values are in display or original 
    #                           coordimates (default Display; string either 'Display' or 'Original')
    # Output: Numpy array with 1 more dimension than the input x_pixels and y_pixels, but otherwise
    #         the same size and shape. The extra dimension indexes the 3 vector components, 0 = X, 1 = Y, 2 = Z.
    #         This is a unit vector in the CAD model coordinate system.
    def get_los_direction(self,x_pixels=None,y_pixels=None,ForceField=None,Coords='Display'):

        if x_pixels is not None or y_pixels is not None:
                if x_pixels is None or y_pixels is None:
                    raise ValueError("X pixels and Y pixels must both be specified!")
                if np.shape(x_pixels) != np.shape(y_pixels):
                    raise ValueError("X pixels array and Y pixels array must be the same size!")
                if Coords.lower() == 'original':
                    x_pixels,y_pixels = self.transform.original_to_display_coords(x_pixels,y_pixels)

        else:
                if Coords.lower() == 'display':
                    x_pixels,y_pixels = np.meshgrid(np.linspace(0,self.image_display_shape[0]-1,self.image_display_shape[0]),np.linspace(0,self.image_display_shape[1]-1,self.image_display_shape[1]))
                else:
                    shape = [self.transform.x_pixels,self.transform.y_pixels]
                    x_pixels,y_pixels = np.meshgrid(np.linspace(0,shape[0]-1,shape[0]),np.linspace(0,shape[1]-1,shape[1]))
                    x_pixels,y_pixels = self.transform.original_to_display_coords(x_pixels,y_pixels)


        # Flatten everything and create output array        
        oldshape = np.shape(x_pixels)
        x_pixels = np.reshape(x_pixels,np.size(x_pixels),order='F')
        y_pixels = np.reshape(y_pixels,np.size(y_pixels),order='F')

        out = np.zeros(np.shape(x_pixels) + (3,))


        if ForceField is None:
                # Identify which sub-field each pixel is in
                pointfield = self.fieldmask[y_pixels.round().astype(int),x_pixels.round().astype(int)]
        else:
                pointfield = np.zeros(np.shape(y_pixels),dtype=int) + ForceField

        fields = set(pointfield)

        for field in fields:

            x_pixels_subset = x_pixels[pointfield == field]
            y_pixels_subset = y_pixels[pointfield == field]

            # Get the normalised 2D coordinates including distortion
            x_norm,y_norm = self.normalise(x_pixels_subset,y_pixels_subset,field)

            # Normalise them to 3D unit vectors
            vect_length = np.sqrt(x_norm**2 + y_norm**2 + 1)
            x_norm = x_norm / vect_length
            y_norm = y_norm / vect_length
            z_norm = np.ones(x_norm.shape) / vect_length

            # Finally, rotate in to lab coordinates
            rotationMatrix = self.get_cam_to_lab_rotation(field)

            # x,y and z components of the LOS vectors
            x = rotationMatrix[0,0]*x_norm + rotationMatrix[0,1]*y_norm + rotationMatrix[0,2]*z_norm
            y = rotationMatrix[1,0]*x_norm + rotationMatrix[1,1]*y_norm + rotationMatrix[1,2]*z_norm
            z = rotationMatrix[2,0]*x_norm + rotationMatrix[2,1]*y_norm + rotationMatrix[2,2]*z_norm

            # Return an array the same shape as the input x and y pixel arrays + an extra dimension
            out[pointfield == field,0] = x
            out[pointfield == field,1] = y
            out[pointfield == field,2] = z


        return np.reshape(out,oldshape + (3,),order='F')


    # Export sight line data to a netCDF file, mainly designed for RaySect.
    # IMPORTANT: There is much room for efficiency improvement in the Raysect / calcam interface
    # and this was written quickly to get some work on the go.
    # SO: THIS OUTPUT FORMAT SHOULD NOT BE RELIED UPON TO STAY THE SAME IN FUTURE VERSIONS!
    def export_sightlines(self,filename,x_pixels=None,y_pixels=None,Coords='Display',binning=1):

        if not filename.endswith('.nc'):
            filename = filename + '.nc'
            
        if x_pixels is None or y_pixels is None:
            if Coords.lower() == 'display':
                xl = np.linspace( (binning-1.)/2,float(self.image_display_shape[0]-1)-(binning-1.)/2,(1+float(self.image_display_shape[0]-1))/binning)
                yl = np.linspace( (binning-1.)/2,float(self.image_display_shape[1]-1)-(binning-1.)/2,(1+float(self.image_display_shape[1]-1))/binning)
                x_pixels,y_pixels = np.meshgrid(xl,yl)
            elif Coords.lower() == 'original':
                xl = np.linspace( (binning-1.)/2,float(self.transform.x_pixels-1)-(binning-1.)/2,(1+float(self.transform.x_pixels-1))/binning)
                yl = np.linspace( (binning-1.)/2,float(self.transform.y_pixels-1)-(binning-1.)/2,(1+float(self.transform.y_pixels-1))/binning)
                x_pixels,y_pixels = np.meshgrid(xl,yl)
            del xl, yl
        
        if Coords.lower() == 'original':
            x_pixels,y_pixels = self.transform.original_to_display_coords(x_pixels,y_pixels)
        
        # To make the image come out from RaySect in the intended orientation,
        # we have to transpose the coordinates here.
        x_pixels = x_pixels.transpose()
        y_pixels = y_pixels.transpose()

        origins = self.get_pupilpos(x_pixels,y_pixels)
        endpoints = self.get_los_direction(x_pixels,y_pixels) + origins

        f = netcdf_file(filename,'w')
        pointdim = f.createDimension('pointdim',3)
        setattr(f,'history','Calcam camera sight lines')
        udim = f.createDimension('udim',x_pixels.shape[1])
        vdim = f.createDimension('vdim',x_pixels.shape[0])
        rayhit = f.createVariable('RayEndCoords','f4',('vdim','udim','pointdim'))
        raystart = f.createVariable('RayStartCoords','f4',('vdim','udim','pointdim'))           
        rayhit[:,:,:] = endpoints
        raystart[:,:,:] = origins
        f.close()

    # Make a pretty summary of the results in the console if someone does print() on this object.
    def __str__(self):

        res = ''

        for field,params in enumerate(self.fit_params):

                pupilPos = self.get_pupilpos(field=field)
                fov = self.get_fov(field)

                # Get CoM of this field on the chip
                ypx,xpx = CoM( (self.fieldmask + 1) * (self.fieldmask == field) )

                # Line of sight at the field centre
                los_centre = self.get_los_direction(xpx,ypx)

                res = res + '\n---------------------------------------\n'
                if self.type == 'fit':
                    res = res + '   ' + params.model.capitalize() + ' Model Calibration Fit'
                    if self.nfields > 1:
                        res = res + '\n           Sub-Field #' + str(field)
                elif self.type == 'manual_alignment':
                    res = res + '    Manual Alignment Calibration'
                res = res + '\n---------------------------------------\n\n'
                if self.type == 'fit':
                    res = res + '      RMS re-projection error\n'
                    res = res + "            {:.1f}".format(params.rms_error) + ' pixels.\n\n'
                res = res + ' Pupil X          = ' + "{: .3f}".format(pupilPos[0]) + ' m.\n'
                res = res + ' Pupil Y          = ' + "{: .3f}".format(pupilPos[1]) + ' m.\n'
                res = res + ' Pupil Z          = ' + "{: .3f}".format(pupilPos[2]) + ' m.\n\n'
                
                if self.nfields > 1:
                    res = res + ' View direction at sub-field centre:\n'
                else:
                    res = res + ' View direction at image centre:\n'
                res = res + ' (' + "{:.3f}".format(los_centre[0]) + "," + "{:.3f}".format(los_centre[1]) + "," + "{:.3f}".format(los_centre[2]) + ")\n\n"
                res = res + ' Horizontal FOV   = ' + "{:.1f}".format(fov[0]) + " degrees\n"
                res = res + ' Vertical FOV     = ' + "{:.1f}".format(fov[1]) + " degrees\n\n"

                if params.cam_matrix[0,0] < 1000:
                    res = res + ' Fx = ' + "{: 4.1f}".format(params.cam_matrix[0,0]) + ' px  |  Fy = ' + "{: 4.1f}".format(params.cam_matrix[1,1]) + ' px\n'
                elif params.cam_matrix[0,0] >= 1000 and params.cam_matrix[0,0] < 10000:
                    res = res + ' Fx = ' + "{: 4.0f}".format(params.cam_matrix[0,0]) + ' px   |  Fy = ' + "{: 4.0f}".format(params.cam_matrix[1,1]) + ' px\n'
                else:
                    res = res + ' Fx = ' + "{: 3.1e}".format(params.cam_matrix[0,0]) + ' px |  Fy = ' + "{: 3.1e}".format(params.cam_matrix[1,1]) + ' px\n'

                res = res + ' Cx = ' + "{: .1f}".format(params.cam_matrix[0,2]) + ' px  |  Cy = ' + "{: .1f}".format(params.cam_matrix[1,2]) + ' px\n'
                
                if params.model == 'perspective':
                    res = res + ' k1 = ' + "{: 2.2e}".format(params.k1) + '  |  k2 = ' + "{: 2.2e}".format(params.k2) + '\n'
                    res = res + ' p1 = ' + "{: 2.2e}".format(params.p1) + '  |  p2 = ' + "{: 2.2e}".format(params.p2) + '\n\n'
                elif params.model == 'fisheye':
                    res = res + ' k1 = ' + "{: 2.2e}".format(params.k1) + '  |  k2 = ' + "{: 2.2e}".format(params.k2) + '\n'
                    res = res + ' k3 = ' + "{: 2.2e}".format(params.k3) + '  |  k4 = ' + "{: 2.2e}".format(params.k4) + '\n'

                if self.type == 'fit':
                    res = res + ' Fit Options:\n'
                    res = res + ' ' + ', '.join(self.fitoptions[field]) + '\n\n'
                
                res = res + '---------------------------------------\n'

        return res

    def import_intrinsics(self,fitresults):
        if fitresults.nfields > 1:
            raise ValueError('Only non split-field virtual views are currently supported!')
        fitresults = copy.deepcopy(fitresults)

        if self.fit_params != [None]:
            tvec = self.fit_params[0].tvec
            rvec = self.fit_params[0].rvec
        else:
            tvec = None
            rvec = None

        self.fit_params = fitresults.fit_params
        
        if tvec is not None and rvec is not None:
            self.fit_params[0].rvec = rvec
            self.fit_params[0].tvec = tvec

        self.image_display_shape = fitresults.image_display_shape
        
        self.nfields = fitresults.nfields
        self.field_names = fitresults.field_names
        self.fieldmask = fitresults.fieldmask


    def set_extrinsics(self,campos,upvec,camtar=None,view_dir=None,opt_axis = False):


        if camtar is not None:
            w = np.squeeze(np.array(camtar) - np.array(campos))
        elif view_dir is not None:
            w = view_dir
        else:
            raise ValueError('Either viewing target or view direction must be specified!')

        w = w / np.sqrt(np.sum(w**2))
        v = upvec / np.sqrt(np.sum(upvec**2))


        # opt_axis specifies whether the input campos or camtar are where we should
        # point the optical axis or the view centre. By defauly we assume the view centre.
        # In this case, we need another rotation matrix to rotate from the image centre
        # view direction (given) to the optical axis direction (stored). This applies for
        # intrinsics where the perspective centre is not at the the detector centre.
        if not opt_axis:
            R = np.zeros([3,3])
            # Optical axis direction in the camera coordinate system
            uz = np.array(self.normalise(self.image_display_shape[0]/2.,self.image_display_shape[1]/2.,0) + (1.,))
            uz = uz / np.sqrt(np.sum(uz**2))

            ux = np.array([1,0,0]) - uz[0]*uz
            ux = ux / np.sqrt(np.sum(ux**2))
            uy = np.cross(ux,uz)
            R[:,0] = ux
            R[:,1] = -uy
            R[:,2] = uz

            R = np.matrix(R).T

        else:
        # If we are pointing the optical axis, set this extra
        # rotation to be the identity.
            R = np.matrix([[1,0,0],[0,1,0],[0,0,1]])


        u = np.cross(w,v)

        Rmatrix = np.zeros([3,3])
        Rmatrix[:,0] = u
        Rmatrix[:,1] = -v
        Rmatrix[:,2] = w
        Rmatrix = np.matrix(Rmatrix)


        Rmatrix = Rmatrix * R
        campos = np.matrix(campos)
        if campos.shape[0] < campos.shape[1]:
            campos = campos.T

        self.fit_params[0].tvec = -Rmatrix.T * campos
        self.fit_params[0].rvec = -cv2.Rodrigues(Rmatrix)[0]


    """
    Project given points in real space (CAD model coordinate system) to pixel coordinates in the image
     
    INPUTS:
    ObjPoints - 3D points to project. Can be EITHER an Nx3 (N = number of points) numpy array, OR a list of 3-element array-likes (where each 3 element list is an X,Y,Z point) OR an array of 3-element arrays (where each 3 element list is an X,Y,Z point)
    OPTIONAL INPUTS:
    CheckVisible - whether to check if the points are occluded by features in the CAD model or not. If true, either RayData or RayCaster must also be provided.
                   If set to True, the pixel coordinates of points which are not visible to the camera (i.e. hidden behind something) are returned as NaN.
                   Default False, slows the calculation down a lot.
    OutOfFrame - whether to return coordinates for points outside the edges of the sensor. Only works for single field images.
    RayData - Calcam raydata object, used if CheckVisible is true
    RayCaster - Calcam raycaster object, used if CheckVisible is true. Takes priority over RayData if both are specified.
    VisibilityMargin - this is a margin of error to allow when checking visibilty. I use this somewhere but I can't remember where.
    Coords - whether to project in to display coords (image right-way-up) or original coords (image as it comes out from the camera), if there's a difference. Default display.
    
    OUTPUT:
    A list of Nx2 numpy arrays. Each list contains the image pixel coordinates [x,y; origin in top left of image] for a sub-field of view of the image.
    For simple images without a split field of view, this is a one-element list where output[0] has the pixel coords in it.
    Points not visible to the camera (off the image edge, or occluded and CheckVisible is enabled) get [nan,nan] returned for their coordinates.
    """
    def project_points(self,ObjPoints,CheckVisible=False,OutOfFrame=False,RayData=None,RayCaster=None,VisibilityMargin=0,Coords='Display'):

        models = []
        for field in self.fit_params:
            models.append(field.model)
        if np.any( np.array(models) == 'fisheye') and opencv_major_version < 3:
            raise Exception('Fisheye model point projection requires OpenCV 3 or newer! Your version is ' + cv2.__version__)

        if RayData is None and RayCaster is None and CheckVisible:
                raise Exception('To check point visibility either a RayData or RayCaster object is required!')

        if OutOfFrame and self.nfields > 1:
            raise Exception('Cannot return coordinates for out-of-frame points since this fit has multiple sub-fields!')


        # Check the input points are in a suitable format
        if np.ndim(ObjPoints) < 3:
                ObjPoints = np.array([ObjPoints],dtype='float32')
        else:
                ObjPoints = np.array(ObjPoints,dtype='float32')

        OutPoints = []

        for field in range(self.nfields):
                PointDistance = np.zeros([len(ObjPoints[0])])+1.e5
                OutPoints.append(np.zeros([len(ObjPoints[0]),2])+np.nan)
            
                # Do reprojection
                if self.fit_params[field].model == 'perspective':
                    points,_ = cv2.projectPoints(ObjPoints,self.fit_params[field].rvec,self.fit_params[field].tvec,self.fit_params[field].cam_matrix,self.fit_params[field].kc)
                elif self.fit_params[field].model == 'fisheye':
                    points,_ = cv2.fisheye.projectPoints(ObjPoints,self.fit_params[field].rvec,self.fit_params[field].tvec,self.fit_params[field].cam_matrix,self.fit_params[field].kc)
                    points = np.swapaxes(points,0,1)
                
                PupilPos = self.get_pupilpos(field=field)

                for i in range(len(ObjPoints[0])):

                    # If we only have 1 sub-field and want points even outside the FoV, we have to set the field number
                    # manually instead of looking in fieldmask, which can produce IndexErrors.
                    if OutOfFrame and field == 0:
                        pointfield = 0
                    else:
                        # Check which part of the image the projected point is in
                        try:
                            pointfield = self.fieldmask[points[i][0][1].round().astype(int),points[i][0][0].round().astype(int)]
                        # If it's outside the image, leave it as [nan,nan]
                        except (IndexError):
                            continue
                        if points[i][0][1] < 0 or points[i][0][0] < 0:
                            continue

                    # If it's in another field, leave it as [nan,nan]. Otherwise add it to the output.
                    if pointfield != field:
                        continue
                    else:
                        if Coords.lower() == 'original':
                            OutPoints[field][i] = self.transform.display_to_original_coords(points[i][0][0],points[i][0][1])
                        else:
                            OutPoints[field][i] = points[i][0]
                    
                    # Distance to the object points from the camera pupil
                    PointDistance[i] = (np.sqrt((ObjPoints[0][i][0] - PupilPos[0])**2 + (ObjPoints[0][i][1] - PupilPos[1])**2 + (ObjPoints[0][i][2] - PupilPos[2])**2))

                # If requested, check visibility due to occlusion
                if CheckVisible:

                        # If we have a ray caster, ray cast the pixels in question
                        if RayCaster is not None:
                            RayData = RayCaster.raycast_pixels(OutPoints[field][:,0],OutPoints[field][:,1],Coords='Display')

                        # Check if points are occluded by any CAD features
                        RayLength = RayData.get_ray_lengths(OutPoints[field][:,0],OutPoints[field][:,1],Coords='Display')
                        RayLength[np.isnan(RayLength)] = 0                
                        OutPoints[field][RayLength + VisibilityMargin < PointDistance,:] = np.nan
                   
        return OutPoints


    # Save the fit results for later use. 
    # Input: SaveName, string to identify the saved retults.
    def save(self,SaveName):

        # File we're going to save in
        SaveFile = open(os.path.join(paths.fitresults,SaveName + '.pickle'),'wb')

        # Gather up the fit parameters
        FitParams = []
        model = []
        for field in self.fit_params:
                FitParams.append([field.rms_error,field.cam_matrix,field.kc,[field.rvec],[field.tvec]])
                model.append(field.model)
        # Shove everything in a nested dictionary
        SaveDict = {'model':model,'nfields':self.nfields,'field_mask':self.fieldmask,'image_display_shape':self.image_display_shape,'fitoptions':self.fitoptions,'fitparams':FitParams,'transform_pixels':[self.transform.x_pixels,self.transform.y_pixels],'transform_actions':self.transform.transform_actions,'transform_pixel_aspect':self.transform.pixel_aspectratio,'objectpoints':self.objectpoints,'imagepoints':self.imagepoints,'field_names':self.field_names,'type':self.type}

        # Pickle it!
        pickle.dump(SaveDict,SaveFile,2)


    # Load saved fit results.
    # Input: SaveName, string to identify saved retults.
    def load(self,SaveName):

        # File to load from - prioritise new .pickle format, but still supports ye old CSVs.
        if os.path.isfile(os.path.join(paths.fitresults,SaveName + '.pickle')):
            SaveFile = open(os.path.join(paths.fitresults,SaveName + '.pickle'),'rb')
        elif os.path.isfile(os.path.join(paths.fitresults,SaveName + '.csv')):
            self.load_old(SaveName)
            return
        else:
            nearest_saves = paths.get_nearest_names('FitResults',SaveName)
            if len(nearest_saves) == 0:
                raise Exception('Save "{:s}" not found!'.format(SaveName))
            else:
                raise Exception('Save "{:s}" not found! Did you mean {:s}?'.format(SaveName,'"' + '" or "'.join(nearest_saves) + '"'))
        try:
            save = pickle.load(SaveFile)
        except:
            SaveFile.seek(0)
            save = pickle.load(SaveFile,encoding='latin1')

        self.nfields = save['nfields']
        self.fieldmask = save['field_mask']
        self.image_display_shape = save['image_display_shape']
        self.fitoptions = save['fitoptions']
        if len(self.fitoptions) == 0:
            self.fitoptions = [[]]
        elif type(self.fitoptions[0]) != list:
            self.fitoptions = [self.fitoptions]

        if 'PointPairs' in save:
            for ppname in save['PointPairs']:
                pp = pointpairs.PointPairs(pp)
                self.imagepoints.append(pp.imagepoints)
                self.objectpoints.append(pp.objectpoints)
        elif 'objectpoints' in save:
            self.objectpoints = save['objectpoints']
            self.imagepoints = save['imagepoints']

        if 'field_names' in save:
            self.field_names = save['field_names']
        else:
            if self.nfields == 1:
                self.field_names = ['Image']
            else:
                self.field_names = []
                for field in range(self.nfields):
                    self.field_names.append('Sub FOV # {:d}'.format(field+1))

        if 'type' in save:
            self.type = save['type']
        else:
            self.type = 'fit'

        for field in range(self.nfields):
                if 'model' in save:
                    if save['model'][field] == 'fisheye' and opencv_major_version < 3:
                        raise Exception('This calibration result uses the fisheye camera model and requires OpenCV3 to be loaded (you are using {:s}).'.format(cv2.__version__))
                    self.fit_params.append(FieldFit(save['model'][field],save['fitparams'][field],from_save=True))
                else:
                    self.fit_params.append(FieldFit('perspective',save['fitparams'][field],from_save=True))

        self.transform = CoordTransformer()
        self.transform.x_pixels = save['transform_pixels'][0]
        self.transform.y_pixels = save['transform_pixels'][1]
        self.transform.pixel_aspectratio = save['transform_pixel_aspect']
        self.transform.set_transform_actions(save['transform_actions'])
        SaveFile.close()


    # Load saved fit results.
    # Input: SaveName, string to identify saved retults.
    def load_old(self,SaveName):


        import csv
        # File to load from
        SaveFile = open(os.path.join(paths.fitresults,SaveName + '.csv'),'r')

        # Temporary storage...
        cmat = np.zeros((3,3))
        tvec = np.zeros((3,1))
        rvec = np.zeros((3,1))
        dist_vec = np.zeros((1,5))
        pixels_x = 0
        pixels_y = 0
        maxk = 0

        csvreader = csv.reader(SaveFile)

        # Skip header row
        try:  # python <3.6?
            csvreader.next()
        except AttributeError as e:  # Python 3.6+
            next(csvreader)

        for row in csvreader:
            # Read the RMS error
            if row[0] == "RMS_error":
                rms_error = float(row[1])
            # Read the camera matrix
            if row[0] == "Camera_Matrix[0 0]":
                cmat[0][0] = float(row[1])
            if row[0] == "Camera_Matrix[0 1]":
                cmat[0][1] = float(row[1])
            if row[0] == "Camera_Matrix[0 2]":
                cmat[0][2] = float(row[1])
            if row[0] == "Camera_Matrix[1 0]":
                cmat[1][0] = float(row[1])
            if row[0] == "Camera_Matrix[1 1]":
                cmat[1][1] = float(row[1])
            if row[0] == "Camera_Matrix[1 2]":
                cmat[1][2] = float(row[1])
            if row[0] == "Camera_Matrix[2 0]":
                cmat[2][0] = float(row[1])
            if row[0] == "Camera_Matrix[2 1]":
                cmat[2][1] = float(row[1])
            if row[0] == "Camera_Matrix[2 2]":
                cmat[2][2] = float(row[1])
            # Read the translation vector
            if row[0] == "Translation_vector[0]":
                tvec[0][0] = float(row[1])
            if row[0] == "Translation_vector[1]":
                tvec[1][0] = float(row[1])
            if row[0] == "Translation_vector[2]":
                tvec[2][0] = float(row[1])
            # Read the rotation vector
            if row[0] == "Rotation_vector[0]":
                rvec[0][0] = float(row[1])
            if row[0] == "Rotation_vector[1]":
                rvec[1][0] = float(row[1])
            if row[0] == "Rotation_vector[2]":
                rvec[2][0] = float(row[1])
            # Read the distortion coefficients
            if row[0] == "K1":
                k1 = float(row[1])
            if row[0] == "K2":
                k2 = float(row[1])
            if row[0] == "K3":
                k3 = float(row[1])
                maxk = max([3,maxk])
            if row[0] == "K4":
                k4 = float(row[1])
                maxk = max([4,maxk])
            if row[0] == "K5":
                k5 = float(row[1])
                maxk = max([5,maxk])
            if row[0] == "K6":
                k6 = float(row[1])
                maxk = max([6,maxk])
            if row[0] == "P1":
                p1 = float(row[1])
            if row[0] == "P2":
                p2 = float(row[1])
            if row[0] == "Num_x_pixels":
                pixels_x = int(row[1])
            if row[0] == "Num_y_pixels":
                pixels_y = int(row[1])
        # Close the file
        SaveFile.close()


        # Make distortion vector
        dist_vec = np.zeros((1,maxk+2))
        dist_vec[0][0] = k1
        dist_vec[0][1] = k2
        dist_vec[0][2] = p1
        dist_vec[0][3] = p2
        dist_vec[0][4] = k3
        if maxk == 4:
            dist_vec[0][5] = k4
        if maxk == 5:
            dist_vec[0][5] = k4
            dist_vec[0][6] = k5
        if maxk == 6:
            dist_vec[0][5] = k4
            dist_vec[0][6] = k5
            dist_vec[0][7] = k6


        self.nfields = 1
        self.fieldmask = np.zeros([pixels_y,pixels_x],dtype='int')
        self.image_display_shape = (pixels_x,pixels_y)
        self.fitoptions = [['Unknown']]
        self.field_names = ['Image']
        self.objectpoints = None
        self.imagepoints = None
        self.type = 'fit'
        self.fit_params.append(FieldFit('perspective',[rms_error,cmat,dist_vec,[rvec],[tvec]] ,from_save=True))
        self.transform = CoordTransformer()
        self.transform.x_pixels = pixels_x
        self.transform.y_pixels = pixels_y
        self.transform.pixel_aspectratio = 1.
        

    def undistort_image(self,image,coords=None):

        if self.nfields > 1:
            raise Exception('This feature is not supported for split-FOV images!')

        imobj_out = None
        if isinstance(image,CalCam_Image):
            imobj_out = copy.deepcopy(image)
            image = imobj_out.transform.original_to_display_image(imobj_out.data)
            coords='display'
        
        display_shape = self.transform.get_display_shape()
        if coords is None:
            coords = 'display'
            if image.shape[0] != display_shape[1] or image.shape[1] != display_shape[0]:
                if image.shape[0] == self.transform.y_pixels and image.shape[1] == self.transform.x_pixels:
                    coords = 'original'
                else:
                    raise ValueError('Supplied image is the wrong shape! Expected {:d}x{:d} or {:d}x{:d} pixels.'.format(display_shape[0],display_shape[1],self.transform.x_pixels,self.transform.y_pixels))

        if coords == 'original':
            im = self.transform.original_to_display_image(image)
        else:
            im = image

        im = cv2.undistort(im,self.fit_params[0].cam_matrix,self.fit_params[0].kc)

        if coords == 'original':
            im = self.transform.display_to_original_image(im)

        if imobj_out is not None:
            imobj_out.data = imobj_out.transform.display_to_original_image(im)
            return imobj_out
        else:
            return im


# Class for storing the calibration results.
# Has methods for post-processing the results to give useful information
# and for loading and saving the results
class VirtualCalib(CalibResults):

    def __init__(self,SaveName=None):

        self.nfields = 1
        self.fit_params = [None]

        if SaveName is not None:
                self.load(SaveName)


    # Make a pretty summary of the results in the console if someone does print() on this object.
    def __str__(self):

        res = ''

        for field,params in enumerate(self.fit_params):

                pupilPos = self.get_pupilpos(field=field)
                fov = self.get_fov(field)

                # Get CoM of this field on the chip
                ypx,xpx = CoM( (self.fieldmask + 1) * (self.fieldmask == field) )

                # Line of sight at the field centre
                los_centre = self.get_los_direction(xpx,ypx)

                res = res + '\n---------------------------------------\n'
                res = res + '   ' + params.model.capitalize() + ' Virtual Camera View'
                if self.nfields > 1:
                    res = res + '\n           Sub-Field #' + str(field)
                res = res + '\n---------------------------------------\n\n'
                res = res + ' Pupil X          = ' + "{: .3f}".format(pupilPos[0]) + ' m.\n'
                res = res + ' Pupil Y          = ' + "{: .3f}".format(pupilPos[1]) + ' m.\n'
                res = res + ' Pupil Z          = ' + "{: .3f}".format(pupilPos[2]) + ' m.\n\n'
                
                if self.nfields > 1:
                    res = res + ' View direction at sub-field centre:\n'
                else:
                    res = res + ' View direction at image centre:\n'
                res = res + ' (' + "{:.3f}".format(los_centre[0]) + "," + "{:.3f}".format(los_centre[1]) + "," + "{:.3f}".format(los_centre[2]) + ")\n\n"
                res = res + ' Horizontal FOV   = ' + "{:.1f}".format(fov[0]) + " degrees\n"
                res = res + ' Vertical FOV     = ' + "{:.1f}".format(fov[1]) + " degrees\n\n"

                if params.cam_matrix[0,0] < 1000:
                    res = res + ' Fx = ' + "{: 4.1f}".format(params.cam_matrix[0,0]) + ' px  |  Fy = ' + "{: 4.1f}".format(params.cam_matrix[1,1]) + ' px\n'
                elif params.cam_matrix[0,0] >= 1000 and params.cam_matrix[0,0] < 10000:
                    res = res + ' Fx = ' + "{: 4.0f}".format(params.cam_matrix[0,0]) + ' px   |  Fy = ' + "{: 4.0f}".format(params.cam_matrix[1,1]) + ' px\n'
                else:
                    res = res + ' Fx = ' + "{: 3.1e}".format(params.cam_matrix[0,0]) + ' px |  Fy = ' + "{: 3.1e}".format(params.cam_matrix[1,1]) + ' px\n'

                res = res + ' Cx = ' + "{: .1f}".format(params.cam_matrix[0,2]) + ' px  |  Cy = ' + "{: .1f}".format(params.cam_matrix[1,2]) + ' px\n'
                
                if params.model == 'perspective':
                    res = res + ' k1 = ' + "{: 2.2e}".format(params.k1) + '  |  k2 = ' + "{: 2.2e}".format(params.k2) + '\n'
                    res = res + ' p1 = ' + "{: 2.2e}".format(params.p1) + '  |  p2 = ' + "{: 2.2e}".format(params.p2) + '\n\n'
                elif params.model == 'fisheye':
                    res = res + ' k1 = ' + "{: 2.2e}".format(params.k1) + '  |  k2 = ' + "{: 2.2e}".format(params.k2) + '\n'
                    res = res + ' k3 = ' + "{: 2.2e}".format(params.k3) + '  |  k4 = ' + "{: 2.2e}".format(params.k4) + '\n'
                
                res = res + '---------------------------------------\n'

        return res





    # Save the fit results for later use. 
    # Input: SaveName, string to identify the saved retults.
    def save(self,SaveName):

        # File we're going to save in
        SaveFile = open(os.path.join(paths.virtualcameras,SaveName + '.pickle'),'wb')

        # Gather up the fit parameters
        FitParams = []
        model = []
        for field in self.fit_params:
                FitParams.append([field.rms_error,field.cam_matrix,field.kc,[field.rvec],[field.tvec]])
                model.append(field.model)

        # Shove everything in a nested dictionary
        SaveDict = {'model':model,'nfields':self.nfields,'field_mask':self.fieldmask,'image_display_shape':self.image_display_shape,'fitparams':FitParams,'transform_pixels':[self.transform.x_pixels,self.transform.y_pixels],'transform_actions':self.transform.transform_actions,'transform_pixel_aspect':self.transform.pixel_aspectratio,'field_names':self.field_names}

        # Pickle it!
        pickle.dump(SaveDict,SaveFile,2)


    # Load saved fit results.
    # Input: SaveName, string to identify saved retults.
    def load(self,SaveName):

        # File to load from
        SaveFile = open(os.path.join(paths.virtualcameras,SaveName + '.pickle'),'rb')

        try:
            save = pickle.load(SaveFile)
        except:
            SaveFile.seek(0)
            save = pickle.load(SaveFile,encoding='latin1')

        self.nfields = save['nfields']
        self.fieldmask = save['field_mask']
        self.image_display_shape = save['image_display_shape']


        if 'field_names' in save:
            self.field_names = save['field_names']
        else:
            if self.nfields == 1:
                self.field_names = ['Image']
            else:
                self.field_names = []
                for field in range(self.nfields):
                    self.field_names.append('Sub FOV # {:d}'.format(field+1))
        self.fit_params = []
        for field in range(self.nfields):
                if 'model' in save:
                    if save['model'][field] == 'fisheye' and opencv_major_version < 3:
                        raise Exception('This calibration result uses the fisheye camera model and requires OpenCV3 to be loaded (you are using {:s}).'.format(cv2.__version__))
                    self.fit_params.append(FieldFit(save['model'][field],save['fitparams'][field],from_save=True))
                else:
                    self.fit_params.append(FieldFit('perspective',save['fitparams'][field],from_save=True))

        self.transform = CoordTransformer()
        self.transform.x_pixels = save['transform_pixels'][0]
        self.transform.y_pixels = save['transform_pixels'][1]
        
        # Due to a bug in early versions of the view designer, some virtual calibrations' transformers
        # were left with x_pixels and y_pixels set to None. This is to fix things when loading
        # virtual calibration objects made with the buggy version (and has no effect for newer versions).
        if self.transform.x_pixels is None or self.transform.y_pixels is None:
            self.transform.x_pixels = self.image_display_shape[0]
            self.transform.y_pixels = self.image_display_shape[1]
        	
        self.transform.pixel_aspectratio = save['transform_pixel_aspect']
        self.transform.set_transform_actions(save['transform_actions'])
        SaveFile.close()


# Small class for storing the actual fit output parameters for a sub-field.
# Instances of this are used inside the CalibResults class.
class FieldFit:

    def __init__(self,fit_model,FitParams,from_save=False):

        self.model = fit_model

        # RMS reprojection error
        self.rms_error = FitParams[0]
    
        # Camera matrix
        self.cam_matrix = FitParams[1].copy()
    
        # Distortion coefficients array
        self.kc = FitParams[2].copy()

        # Extrinsics: rotation and translation vectors
        self.rvec = FitParams[3][0].copy()
        self.tvec = FitParams[4][0].copy()

        
        if fit_model == 'perspective':
            # Split distortion coeffs in to individual properties
            self.k1 = self.kc[0][0]
            self.k2 = self.kc[0][1]
            self.k3 = 0
            self.p1 = self.kc[0][2]
            self.p2 = self.kc[0][3]
            if len(FitParams[2]) == 5:
                    self.k3 = self.kc[0][4]
            elif len(FitParams[2]) == 8:
                raise Exceotion('Calcam does not (yet) support rational OpenCV model!')
        
        elif fit_model == 'fisheye':
            # Annoyingly, fisheye calibration returns its extrinsics in a different
            # format than the perspective fitting. *sigh*. Also the format that comes
            # back from OpenCV depends in some interesting way on versions of things,
            # so we need this try...except here. Grumble.
            if not from_save:
                try:
                    self.rvec[0] = self.rvec[0].T
                    self.tvec[0] = self.tvec[0].T
                except ValueError:
                    self.rvec = self.rvec[0].T
                    self.tvec = self.tvec[0].T
                                        
            self.k1 = self.kc[0]
            self.k2 = self.kc[1]
            self.k3 = self.kc[2]
            self.k4 = self.kc[3]


def update_calibration(calibration,new_image,old_image = None,region_size=None,debug=False,display=False):
    
    if calibration.nfields > 1:
        raise Exception('Cannot do this for images with more than one sub-field yet!')

    if old_image is None:
        old_image = calibration.Image

    if old_image is None:
        raise Exception('No old image provided and calibration does not have an image stored with it!')

    if new_image.transform.get_display_shape() != old_image.transform.get_display_shape():
        raise Exception('New image is a different shape to the original one! New and old images must be the same shape.')


    if region_size is None:
        region_size = int(np.round(np.mean(new_image.transform.get_display_shape()) / 20) )

    old_image_data = old_image.transform.original_to_display_image(old_image.data).sum(axis=2).astype('float64')
    new_image_data = new_image.transform.original_to_display_image(new_image.data).sum(axis=2).astype('float64')

    # Create a new point pairs object which will contain the updated points,
    # and copy the CAD coordinates straight over.
    new_pointpairs = PointPairs()
    new_pointpairs.set_image(new_image)
    new_pointpairs.objectpoints = calibration.objectpoints[0]

    if debug or display:
        import matplotlib.pyplot as plt

    # Update points based on the old and new images...
    # Loop over every point pair
    for pointpair in calibration.imagepoints[0]:
        new_pointpairs.imagepoints.append([])
        # Loop over each image sub field
        for point in pointpair:
            if point is None:
                new_pointpairs.imagepoints[-1].append(None)
            else:
                # Take pieces of each image centred at the point and +- region_size
                int_coords = np.round(point).astype('int')

                top = max(0,int_coords[1]-region_size)
                bot = min(new_image_data.shape[0],int_coords[1]+region_size)
                left = max(0,int_coords[0]-region_size)
                right = min(new_image_data.shape[1],int_coords[0]+region_size)

                new_image_sample = np.diff(new_image_data[top:bot,left:right])
                old_image_sample = np.diff(old_image_data[top:bot,left:right])


                shift = cv2.phaseCorrelate(old_image_sample,new_image_sample)

                # In OpenCV2 phaseCorrelate just returns the image shift;
                # for later versions it also returns the value of the correlation maximum.
                if opencv_major_version > 2:
                    shift = shift[0]

                if debug:
                    plt.imshow(new_image_sample,cmap='gray')
                    ox = point[0] - left
                    oy = point[1] - top
                    plt.plot([ox+shift[0]],[oy+shift[1]],'o')
                    plt.title('New')
                    plt.figure()
                    plt.imshow(old_image_sample,cmap='gray')
                    plt.plot([ox],[oy],'o')
                    plt.title('Old')
                    plt.show()

                # Add the shifted point to the new point pairs
                new_pointpairs.imagepoints[-1].append([point[0]+shift[0],point[1]+shift[1]])

    # Now we need to do the fit, so make a new fitter and set it up with options copied from the existing calibration
    fitter = Fitter()
    fitter.set_PointPairs(new_pointpairs)
    for field in range(calibration.nfields):
        fitter.set_model(calibration.fit_params[field].model,field=field)
    fitter.set_fitflags_strings(calibration.fitoptions)

    # Do the new fit and return it!
    new_calibration = fitter.do_fit()


    if display:
        ax = plt.subplot(121)
        plt.imshow(old_image_data,cmap='gray')
        for pointpair in calibration.imagepoints[0]:
            # Loop over each image sub field
            for point in pointpair:
                if point is not None:
                    plt.plot(point[0],point[1],'bo')
                    plt.plot([point[0]-region_size,point[0]-region_size,point[0]+region_size,point[0]+region_size,point[0]-region_size],[point[1]+region_size,point[1]-region_size,point[1]-region_size,point[1]+region_size,point[1]+region_size],'g')
        plt.title('Old image and calib points')
        plt.subplot(122,sharex=ax,sharey=ax)
        plt.imshow(new_image_data,cmap='gray')
        for pointpair in new_calibration.imagepoints[0]:
            # Loop over each image sub field
            for point in pointpair:
                if point is not None:
                    plt.plot(point[0],point[1],'bo')
        for pointpair in calibration.imagepoints[0]:
            # Loop over each image sub field
            for point in pointpair:
                if point is not None:
                    #plt.plot(point[0],point[1],'o',color=(0.5,0.5,0.5),alpha=0.5)
                    plt.plot([point[0]-region_size,point[0]-region_size,point[0]+region_size,point[0]+region_size,point[0]-region_size],[point[1]+region_size,point[1]-region_size,point[1]-region_size,point[1]+region_size,point[1]+region_size],'g')

        plt.title('New image and updated calib points')
        plt.show()

    return new_calibration
