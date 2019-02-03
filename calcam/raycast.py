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
Ray tracing tools for CalCam_py

Written by Scott Silburn
2015-05-17
"""

import time
import sys
import os
import random

try:
    import vtk
except:
    vtk = None

import numpy as np
from scipy.io.netcdf import netcdf_file

from . import coordtransformer
from . import misc
from . import __version__ as calcam_version


def raycast_sightlines(calibration,cadmodel,x=None,y=None,binning=1,coords='Display',verbose=True,force_subview=None,status_callback=None):
    '''
    Ray cast camera sight-lines to determine where they intersect the given CAD model.

    Parameters:

        calibration (calcam.Calibration) : Calibration whose sight-lines to raycast.
        
        cadmodel (calcam.CADModel)       : CAD model to check intersection with.
        
        x, y (array-like)                : x and y image pixel coordinates for which to cast sight-lines. \
                                           If not specified, one ray is cast at the centre of every detector pixel.\
                                           x and y must be the same shape.
                                           
        binning (int)                    : If not explicitly providing x and y image coordinates, pixel binning for ray casting.\
                                           This specifies NxN binning, i.e. for a value of 2, one ray is cast at the centre of \
                                           every 2x2 cluster of pixels.
                                           
        coords (str)                     : Either ``Display`` or ``Original``. If specifying x and y coordinates,\
                                           specifies whether the input x and y are in original or display coords. \
                                           Otherwise, specifies the orientation of the returned data.
        
        force_subview (int)              : If specified, forces use of the camera model from this index of sub-view \
                                           in the calibration. Otherwise, sub-views are chosen according to the \
                                           sub-view mask in the calibration.
                                           
        verbose (bool)                   : Whether to print status updates during ray casting (depreciated in favour of status_callback)

        status_callback (callable)       : Callable which takes a single argument to be called with status updates. The argument will \
                                           either be a string for textual status updates or a float from 0 to 1 specifying the progress \
                                           of the calculation. If set to None, no status updates are issued. For backwards compatibility, \
                                           if set to None but verbose is set to True, status_callback will be set such that status updates \
                                           go to stdout.

    Returns:

        calcam.RayData                   : Object containing the results.

    '''

    if vtk is None:
        raise Exception('VTK is not available, and this Calcam feature requires VTK!')

    if status_callback is None:
        if verbose:
            status_callback = misc.LoopProgPrinter().update
            
    if status_callback is not None:
        original_callback = cadmodel.get_status_callback()
        cadmodel.set_status_callback(status_callback)


    # Work out how big the model is. This is to make sure the rays we cast aren't too short.
    model_extent = cadmodel.get_extent()
    model_size = model_extent[1::2] - model_extent[::2]
    max_ray_length = model_size.max() * 4
    
    if status_callback is not None:
        status_callback('Getting CAD model octree...')

    # Get the CAD model's octree
    cell_locator = cadmodel.get_cell_locator()


    # If no pixels are specified, do the whole chip at the specified binning level.
    fullchip = False
    if x is None and y is None:
        fullchip = True
        if coords.lower() == 'display':
            shape = calibration.geometry.get_display_shape()
            xl = np.linspace( (binning-1.)/2,float(shape[0]-1)-(binning-1.)/2,(1+float(shape[0]-1))/binning)
            yl = np.linspace( (binning-1.)/2,float(shape[1]-1)-(binning-1.)/2,(1+float(shape[1]-1))/binning)
            x,y = np.meshgrid(xl,yl)
        else:
            shape = calibration.geometry.get_original_shape()
            xl = np.linspace( (binning-1.)/2,float(shape[0]-1)-(binning-1.)/2,(1+float(shape[0]-1))/binning)
            yl = np.linspace( (binning-1.)/2,float(shape[1]-1)-(binning-1.)/2,(1+float(shape[1]-1))/binning)
            x,y = np.meshgrid(xl,yl)
            x,y = calibration.geometry.original_to_display_coords(x,y)
        valid_mask = np.ones(x.shape,dtype=bool)
    elif x is None or y is None:
        raise ValueError('Either both or none of x and y pixel coordinates must be given!')
    else:

        if np.array(x).ndim == 0:
            x = np.array([x])
        else:
            x = np.array(x)
   		
        if np.array(y).ndim == 0:
            y = np.array([y])
        else:
            y = np.array(y)
   	
        if x.shape != y.shape:
            raise ValueError('x and y arrays must be the same shape!')
        valid_mask = np.logical_and(np.isnan(x) == 0 , np.isnan(y) == 0 )
        if coords.lower() == 'original':
            x,y = calibration.geometry.original_to_display_coords(x,y)

    results = RayData()
    
    if fullchip:
        results.fullchip = coords
    else:
        results.fullchip = False
        
    results.x = np.copy(x).astype('float')
    results.x[valid_mask == 0] = 0
    results.y = np.copy(y).astype('float')
    results.y[valid_mask == 0] = 0
    results.transform = calibration.geometry
    if calibration.filename is not None:
        splitname = os.path.split(calibration.filename)
        results.history = 'Ray cast of calibration "{:s}" [from: {:s}] by {:s} on {:s} at {:s}'.format(splitname[1].replace('.ccc',''),splitname[0],misc.username,misc.hostname,misc.get_formatted_time())
    else:
        results.history = 'Ray cast by {:s} on {:s} at {:s}'.format(misc.username,misc.hostname,misc.get_formatted_time())

    orig_shape = np.shape(results.x)
    results.x = np.reshape(results.x,np.size(results.x),order='F')
    results.y = np.reshape(results.y,np.size(results.y),order='F')
    valid_mask = np.reshape(valid_mask,np.size(valid_mask),order='F')
    totpx = np.size(results.x)


    # New results object to store results
    if fullchip:
        results.binning = binning
        results.coords = coords
    else:
        results.binning = None
        results.coords = None


    results.ray_end_coords = np.ndarray([np.size(x),3])

    # Line of sight directions
    LOSDir = calibration.get_los_direction(results.x,results.y,coords='Display',subview=force_subview)
    if len(LOSDir.shape) == 1:
        LOSDir = [LOSDir]
    results.ray_start_coords = calibration.get_pupilpos(results.x,results.y,coords='Display',subview=force_subview)

    
    if status_callback is not None:
        oom = np.floor( np.log(np.size(x)) / np.log(10) / 3. )
        status_callback('Casting {:s} rays...'.format( ['{:.0f}','{:.1f}k','{:.2f}M'][int(oom)].format(np.size(x)/10**(3*oom)) ) )


    # Some variables to give to VTK becasue of its annoying C-like interface
    t = vtk.mutable(0)
    pos = np.zeros(3)
    coords_ = np.zeros(3)
    subid = vtk.mutable(0)

    last_status_update = 0.
    n_done = 0
    
    # We will do the ray casting in a random order,
    # purely to get better time remaining estimation.
    inds = list(range(np.size(x)))
    random.shuffle(inds)
    
    for ind in inds:

        if not valid_mask[ind]:
            results.ray_end_coords[ind,:] = np.nan
            results.ray_start_coords[ind,:] = np.nan
            continue

        # Do the raycast and put the result in the output array
        rayend = results.ray_start_coords[ind] + max_ray_length * LOSDir[ind]
        retval = cell_locator.IntersectWithLine(results.ray_start_coords[ind],rayend,1.e-6,t,pos,coords_,subid)

        if abs(retval) > 0:
            results.ray_end_coords[ind,:] = pos[:]
        else:
            results.ray_end_coords[ind,:] = rayend

        n_done = n_done + 1
        
        if time.time() - last_status_update > 1 and status_callback is not None:
            status_callback(n_done / len(inds))
            last_status_update = time.time()

    if status_callback is not None:
        status_callback(1.)

    results.x[valid_mask == 0] = np.nan
    results.y[valid_mask == 0] = np.nan

    results.ray_end_coords = np.reshape(results.ray_end_coords,orig_shape + (3,),order='F')
    results.ray_start_coords = np.reshape(results.ray_start_coords,orig_shape + (3,),order='F')
    results.x = np.reshape(results.x,orig_shape,order='F')
    results.y = np.reshape(results.y,orig_shape,order='F')

    if not verbose:
        cadmodel.set_status_callback(original_callback)

    return results





class RayData:
    '''
    Class representing ray casting results.

    Objects of this class are returned by :func:`calcam.raycast_sightlines`.
    It can also be used to save and load ray cast results to disk.

    Parameters:
        
        filename (str)  : File name of netCDF file containing saved RayData to load. \
                          If not given, an empty RayData object is created.

    '''
    def __init__(self,filename=None):

        self.ray_end_coords = None
        '''
        np.ndarray :An array containing the 3D coordinates, in metres, of the points where \
        the sight lines intersect the CAD model. If x and y coordinates were \
        input to raycast_sightlines, the shape of this array is the same as the \
        input x and y arrays with an additional dimension added which contains the \
        [X,Y,Z] 3D coordinates. Otherwise the shape is (h x w x 3) where w and h are \
        the image width and height.
        '''

        self.ray_start_coords = None
        '''
        np.ndarray :An array containing the 3D coordinates, in metres, of the start of \
        each sight line (i.e. camera pupil position). If x and y coordinates were \
        input to raycast_sightlines, the shape of this array is the same as the \
        input x and y arrays with an additional dimension added which contains the \
        [X,Y,Z] 3D coordinates. Otherwise the shape is (h x w x 3) where w and h are \
        the image width and height.
        '''

        self.binning = None
        self.transform = None
        self.fullchip = None
        self.x = None
        self.y = None
        self.filename = None
        self.history = None
        if filename is not None:
            self._load(filename)


    # Save to a netCDF file
    def save(self,filename):
        '''
        Save the RayData to a netCDF file.

        Parameters:

            filename (str) : File name to save to.
        '''
        if not filename.endswith('.nc'):
            filename = filename + '.nc'
			
        f = netcdf_file(filename,'w')
        f.title = 'Calcam v{:s} RayData (ray cast results) file.'.format(calcam_version)
        f.history = self.history
        f.image_transform_actions = "['" + "','".join(self.transform.transform_actions) + "']"
        f.fullchip = self.fullchip

        pointdim = f.createDimension('pointdim',3)

        if len(self.x.shape) == 2:
            udim = f.createDimension('udim',self.x.shape[1])
            vdim = f.createDimension('vdim',self.x.shape[0])
            rayhit = f.createVariable('RayEndCoords','f4',('vdim','udim','pointdim'))
            raystart = f.createVariable('RayStartCoords','f4',('vdim','udim','pointdim'))
            x = f.createVariable('PixelXLocation','i4',('vdim','udim'))
            y = f.createVariable('PixelYLocation','i4',('vdim','udim'))
            
            rayhit[:,:,:] = self.ray_end_coords
            raystart[:,:,:] = self.ray_start_coords
            x[:,:] = self.x
            y[:,:] = self.y
        elif len(self.x.shape) == 1:
            udim = f.createDimension('udim',self.x.size)
            rayhit = f.createVariable('RayEndCoords','f4',('udim','pointdim'))
            raystart = f.createVariable('RayStartCoords','f4',('udim','pointdim'))
            x = f.createVariable('PixelXLocation','i4',('udim',))
            y = f.createVariable('PixelYLocation','i4',('udim',))

            rayhit[:,:] = self.ray_end_coords
            raystart[:,:] = self.ray_start_coords

            x[:] = self.x
            y[:] = self.y
        else:
            raise Exception('Cannot save RayData with >2D x and y arrays!')

        binning = f.createVariable('Binning','i4',())

        if self.binning is not None:
            binning.assignValue(self.binning)
        else:
            binning.assignValue(0)

        pixelsdim = f.createDimension('pixelsdim',2)

        xpx = f.createVariable('image_original_shape','i4',('pixelsdim',))
        xpx[:] = [self.transform.x_pixels,self.transform.y_pixels]

        pixelaspect = f.createVariable('image_original_pixel_aspect','f4',())
        pixelaspect.assignValue(self.transform.pixel_aspectratio)

        binning.units = 'pixels'
        raystart.units = 'm'
        rayhit.units = 'm'
        x.units = 'pixels'
        y.units = 'pixels'
        f.close()



    def _load(self,filename):
        '''
        Load RayData from a file.

        Parameters:

            filename (str) : File name to load from.
        '''
        f = netcdf_file(filename, 'r',mmap=False)
        self.filename = filename
               
        self.ray_end_coords = f.variables['RayEndCoords'].data
        self.ray_start_coords = f.variables['RayStartCoords'].data
        self.binning = f.variables['Binning'].data[()]

        self.transform = coordtransformer.CoordTransformer()
        self.transform.set_transform_actions(eval(f.image_transform_actions))
        self.transform.x_pixels = f.variables['image_original_shape'][0]
        self.transform.y_pixels = f.variables['image_original_shape'][1]
        self.transform.pixel_aspectratio = f.variables['image_original_pixel_aspect'].data[()]

        try:
            self.history = f.history.decode('utf-8')
            self.fullchip = f.fullchip
            if not self.fullchip:
                self.binning = None
            elif self.fullchip != True:
                self.fullchip = self.fullchip.decode('utf-8')
        except AttributeError:
            self.history = 'Loaded from legacy raydata file "{:s} by {:s} on {:s} at {:s}'.format(os.path.split(filename)[-1],misc.username,misc.hostname,misc.get_formatted_time())
            if self.binning == 0:
                self.binning = None
                self.fullchip = False
            else:
                if len(self.transform.transform_actions) == 0:
                    self.fullchip = 'Original'
                else:                
                    self.fullchip = True

        self.x = f.variables['PixelXLocation'].data
        self.y = f.variables['PixelYLocation'].data

        f.close()


    def get_ray_lengths(self,x=None,y=None,im_position_tol = 1,coords='Display'):
        '''
        Get the sight-line lengths either of all casted sight-lines or at the specified image coordinates.

        Parameters:
        
            x,y (array-like)        : Image pixel coordinates at which to get the sight-line lengths. \
                                      If not specified, the lengths of all casted sight lines will be returned.
            im_position_tol (float) : If x and y are specified but no sight-line was cast at exactly the \
                                      input coordinates, the nearest casted sight-line will be returned \
                                      instead provided the pixel coordinates wre within this many pixels of \
                                      the requested coordinates.
            coords (str)            : Either ``Display`` or ``Coords``, specifies what orientation the input x \
                                      and y correspond to or orientation of the returned array.
        
        Returns:

            np.ndarray              : Array containing the sight-line lengths. If the ray cast was for the \
                                      full detector and x and y are not specified, the array shape will be \
                                      (h x w) where w nd h are the image width and height. Otherwise it will \
                                      be the same shape as the input x and y coordinates.
        '''
        
        # Work out ray lengths for all raytraced pixels
        raylength = np.sqrt(np.sum( (self.ray_end_coords - self.ray_start_coords) **2,axis=-1))
        
        # If no x and y given, return them all
        if x is None and y is None:
            if self.fullchip:
                if coords.lower() == 'display':
                    return raylength
                else:
                    return self.transform.display_to_original_image(raylength)
            else:
                return raylength
        else:
            if self.x is None or self.y is None:
                raise Exception('This ray data does not have x and y pixel indices!')

            # Otherwise, return the ones at given x and y pixel coords.
            if np.shape(x) != np.shape(y):
                raise ValueError('x and y arrays must be the same shape!')
            else:

                if coords.lower() == 'original':
                    x,y = self.transform.original_to_display_coords(x,y)

                orig_shape = np.shape(x)
                x = np.reshape(x,np.size(x),order='F')
                y = np.reshape(y,np.size(y),order='F')
                RL = np.zeros(np.shape(x))

                if not self.fullchip:
                    raylength = raylength.flatten()
                    xflat = self.x.flatten()
                    yflat = self.y.flatten()

                for pointno in range(x.size):
                    if np.isnan(x[pointno]) or np.isnan(y[pointno]):
                        RL[pointno] = np.nan
                        continue

                    if self.fullchip:
                        xind = np.argmin(np.abs(self.x[0,:] - x[pointno]))
                        yind = np.argmin(np.abs(self.y[:,0] - y[pointno]))
                        deltaR = np.sqrt( (self.x[0,xind]-x[pointno])**2 + (self.y[yind,0]-y[pointno])**2)
                        if deltaR < im_position_tol:
                            RL[pointno] = raylength[yind,xind]
                        else:
                            raise Exception('No ray-traced pixel within im_position_tol of requested pixel!')
                    else:
                        # This can be very slow if xflat and yflat are big arrays!
                        deltaR = np.sqrt( (xflat - x[pointno])**2 + (yflat - y[pointno])**2 )
                        if np.nanmin(deltaR) <= im_position_tol:
                            RL[pointno] = raylength[np.nanargmin(deltaR)]
                        else:
                            raise Exception('No ray-traced pixel within im_position_tol of requested pixel!')

                return np.reshape(RL,orig_shape,order='F')



    def get_ray_directions(self,x=None,y=None,im_position_tol=1,coords='Display'):
        '''
        Get unit vectors specifying the sight-line directions. Note that ray casting \
        is not required to get this information: see :func:`calcam.Calibration.get_los_direction` \
        for the same functionality, however this can be useful if you have the RayData \
        but not :class:`calcam.Calibration` object loaded when doing the analysis.

        Parameters:

            x,y (array-like)        : x and y pixel coordinates at which to get the ray directions. \
                                      If not specified, the ray directions of every sight-line are returned.
            im_position_tol (float) : If x and y are specified but no sight-line was cast at exactly the \
                                      input coordinates, the nearest casted sight-line will be returned \
                                      instead provided the pixel coordinates wre within this many pixels of \
                                      the requested coordinates.
            coords (str)            : Either ``Display`` or ``Coords``, specifies what orientation the input x \
                                      and y correspond to or orientation of the returned array.

        Returns:

            np.ndarray              : Array containing the sight-line directions. If the ray cast was for the \
                                      full detector and x and y are not specified, the array shape will be \
                                      (h x w x 3) where w nd h are the image width and height. Otherwise it will \
                                      be the same shape as the input x and y coordinates plus an extra dimension.
        '''
        lengths = self.get_ray_lengths()
        dirs = (self.ray_end_coords - self.ray_start_coords) / np.repeat(lengths.reshape(np.shape(lengths)+(1,)),3,axis=-1)

        if x is None and y is None:
            if self.fullchip:
                if coords.lower() == 'display':
                    return dirs
                else:
                    return self.transform.display_to_original_image(dirs,binning=self.binning)
            else:
                return dirs
        else:
            if self.x is None or self.y is None:
                raise Exception('This ray data does not have x and y pixel indices!')
            if np.shape(x) != np.shape(y):
                raise ValueError('x and y arrays must be the same shape!')
            else:

                if coords.lower() == 'original':
                    x,y = self.transform.original_to_display_coords(x,y)

                oldshape = np.shape(x)
                x = np.reshape(x,np.size(x),order='F')
                y = np.reshape(y,np.size(y),order='F')
                [dirs_X,dirs_Y,dirs_Z] = np.split(dirs,3,-1)
                dirs_X = dirs_X.flatten()
                dirs_Y = dirs_Y.flatten()
                dirs_Z = dirs_Z.flatten()
                xflat = self.x.flatten()
                yflat = self.y.flatten()
                Xout = np.zeros(np.shape(x))
                Yout = np.zeros(np.shape(x))
                Zout = np.zeros(np.shape(x))
                for pointno in range(x.size):
                    deltaX = xflat - x[pointno]
                    deltaY = yflat - y[pointno]
                    deltaR = np.sqrt(deltaX**2 + deltaY**2)
                    if np.min(deltaR) <= im_position_tol:
                        Xout[pointno] = dirs_X[np.argmin(deltaR)]
                        Yout[pointno] = dirs_Y[np.argmin(deltaR)]
                        Zout[pointno] = dirs_Z[np.argmin(deltaR)]
                    else:
                        raise Exception('No ray-traced pixel within im_position_tol of requested pixel!')
                out = np.hstack([Xout,Yout,Zout])

                return np.reshape(out,oldshape + (3,),order='F')
