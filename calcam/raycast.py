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
import os
import random
import copy

try:
    import vtk
except:
    vtk = None

import numpy as np
from scipy.io.netcdf import netcdf_file

from . import coordtransformer
from . import misc
from . import __version__ as calcam_version


def raycast_sightlines(calibration,cadmodel,x=None,y=None,exclusion_radius=0.0,binning=1,coords='Display',verbose=True,intersecting_only=False, force_subview=None,status_callback=None,calc_normals=False):
    '''
    Ray cast camera sight-lines to determine where they intersect the given CAD model.

    Parameters:

        calibration (calcam.Calibration) : Calibration whose sight-lines to raycast.
        
        cadmodel (calcam.CADModel)       : CAD model to check intersection with.
        
        x, y (array-like)                : x and y image pixel coordinates for which to cast sight-lines. \
                                           If not specified, one ray is cast at the centre of every detector pixel.\
                                           x and y must be the same shape.

        exclusion_radius (float)         : Distance from camera pupil (in meters) over which to ignore ray \
                                           intersections with CAD surfaces. \
                                           This is useful for views involving mirrors and/or prisms where unfolding \
                                           the optical path results in the virtual pupil location falling behind \
                                           nearby CAD surfaces. Setting a sufficient exclusion radius will cause the \
                                           rays to be launched from the other side of these surfaces so they intersect \
                                           the correct surfaces present in the image.

        binning (int)                    : If not explicitly providing x and y image coordinates, pixel binning for ray casting.\
                                           This specifies NxN binning, i.e. for a value of 2, one ray is cast at the centre of \
                                           every 2x2 cluster of pixels.
                                           
        coords (str)                     : Either ``Display`` or ``Original``. If specifying x and y coordinates,\
                                           specifies whether the input x and y are in original or display coords. \
                                           Otherwise, specifies the orientation of the returned data.

        intersecting_only (bool)         : If set to True, the ray end coordinates and length for sight-lines which do not intersect \
                                           with the CAD model (i.e. sight lines which "escape" through holes in the model) are set to NaN. \
                                           This is useful if you are only interested in sight-lines which intersect with CAD model surfaces. 
        
        force_subview (int)              : If specified, forces use of the camera model from this index of sub-view \
                                           in the calibration. Otherwise, sub-views are chosen according to the \
                                           sub-view mask in the calibration.
                                           
        verbose (bool)                   : Whether to print status updates during ray casting (depreciated in favour of status_callback)

        status_callback (callable)       : Callable which takes a single argument to be called with status updates. The argument will \
                                           either be a string for textual status updates or a float from 0 to 1 specifying the progress \
                                           of the calculation. If set to None, no status updates are issued. For backwards compatibility, \
                                           if set to None but verbose is set to True, status_callback will be set such that status updates \
                                           go to stdout.

        calc_normals (bool)              : Whether to calculate the normal vectors of the CAD model where the sight-lines intersect it. \
                                           Not turned on by default because it seems to add around 80% extra calculation time, so best used \
                                           only if actyally needed.

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
        x,y = calibration.fullframe_meshgrid(coords=coords,binning=binning)

    elif x is None or y is None:
        raise ValueError('Either both or none of x and y pixel coordinates must be given!')


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


    # New results object to store results
    if fullchip:
        results.binning = binning
        results.coords = coords
    else:
        results.binning = None
        results.coords = None


    results.ray_end_coords = np.ndarray([np.size(x),3])
    results.model_normals = np.ndarray([np.size(x),3]) + np.nan

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
    cellid = vtk.mutable(0)
    cell = vtk.vtkGenericCell()

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
        raystart = results.ray_start_coords[ind] + exclusion_radius * LOSDir[ind]
        rayend = results.ray_start_coords[ind] + max_ray_length * LOSDir[ind]

        retval = cell_locator.IntersectWithLine(raystart,rayend,1.e-6,t,pos,coords_,subid,cellid,cell)

        if abs(retval) > 0:
            results.ray_end_coords[ind,:] = pos[:]
            if calc_normals:
                v0 = np.array(cell.GetPoints().GetPoint(2)) - np.array(cell.GetPoints().GetPoint(0))
                v1 = np.array(cell.GetPoints().GetPoint(2)) - np.array(cell.GetPoints().GetPoint(1))
                n = np.cross(v0,v1)
                n = n / np.sqrt(np.sum(n**2))
                if np.dot(LOSDir[ind],n) > 0:
                    n = -n
                results.model_normals[ind,:] = n

        elif intersecting_only:
            results.ray_end_coords[ind,:] = np.nan
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
    if calc_normals:
        results.model_normals = np.reshape(results.model_normals, orig_shape + (3,), order='F')
    else:
        results.model_normals = None

    results.x = np.reshape(results.x,orig_shape,order='F')
    results.y = np.reshape(results.y,orig_shape,order='F')

    if status_callback is not None:
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
        self.ray_start_coords = None

        self.history = None
        '''
        str: Human readable description of where the raydata came from.
        '''

        self.binning = None
        self.transform = None
        self.fullchip = None
        self.x = None
        self.y = None
        self.filename = None
        self.crop = None
        self.model_normals = None
        
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

        current_crop = self.crop
        self.set_detector_window(None)

        f = netcdf_file(filename,'w')
        f.title = 'Calcam v{:s} RayData (ray cast results) file.'.format(calcam_version)
        f.history = self.history
        f.image_transform_actions = "['" + "','".join(self.transform.transform_actions) + "']"
        f.fullchip = self.fullchip

        f.createDimension('pointdim',3)

        if len(self.x.shape) == 2:
            f.createDimension('udim',self.x.shape[1])
            f.createDimension('vdim',self.x.shape[0])
            rayhit = f.createVariable('RayEndCoords','f4',('vdim','udim','pointdim'))
            raystart = f.createVariable('RayStartCoords','f4',('vdim','udim','pointdim'))
            x = f.createVariable('PixelXLocation','i4',('vdim','udim'))
            y = f.createVariable('PixelYLocation','i4',('vdim','udim'))
            if self.model_normals is not None:
                normals = f.createVariable('ModelNormals', 'f4', ('vdim', 'udim', 'pointdim'))
            
            rayhit[:,:,:] = self.ray_end_coords
            raystart[:,:,:] = self.ray_start_coords
            x[:,:] = self.x
            y[:,:] = self.y
            if self.model_normals is not None:
                normals[:,:,:] = self.model_normals

        elif len(self.x.shape) == 1:
            f.createDimension('udim',self.x.size)
            rayhit = f.createVariable('RayEndCoords','f4',('udim','pointdim'))
            raystart = f.createVariable('RayStartCoords','f4',('udim','pointdim'))
            if self.model_normals is not None:
                normals = f.createVariable('ModelNormals','f4',('udim','pointdim'))

            x = f.createVariable('PixelXLocation','i4',('udim',))
            y = f.createVariable('PixelYLocation','i4',('udim',))

            rayhit[:,:] = self.ray_end_coords
            raystart[:,:] = self.ray_start_coords
            x[:] = self.x
            y[:] = self.y
            if self.model_normals is not None:
                normals[:,:] = self.model_normals
        else:
            raise Exception('Cannot save RayData with >2D x and y arrays!')

        binning = f.createVariable('Binning','i4',())

        if self.binning is not None:
            binning.assignValue(self.binning)
        else:
            binning.assignValue(0)

        f.createDimension('pixelsdim',2)

        xpx = f.createVariable('image_original_shape','i4',('pixelsdim',))
        xpx[:] = [self.transform.x_pixels,self.transform.y_pixels]

        offset = f.createVariable('image_offset','i4',('pixelsdim',))
        offset[:] = self.transform.offset[:]

        pixelaspect = f.createVariable('image_original_pixel_aspect','f4',())
        pixelaspect.assignValue(self.transform.pixel_aspectratio)

        binning.units = 'pixels'
        raystart.units = 'm'
        rayhit.units = 'm'
        x.units = 'pixels'
        y.units = 'pixels'
        f.close()

        self.set_detector_window(current_crop)



    def _load(self,filename):
        '''
        Load RayData from a file.

        Parameters:

            filename (str) : File name to load from.
        '''
        f = netcdf_file(filename, 'r',mmap=False)
        self.filename = filename
               
        self.ray_end_coords = f.variables['RayEndCoords'].data.astype(np.float32)
        self.ray_start_coords = f.variables['RayStartCoords'].data.astype(np.float32)
        self.binning = f.variables['Binning'].data[()]

        self.transform = coordtransformer.CoordTransformer()
        self.transform.set_transform_actions(eval(f.image_transform_actions))
        self.transform.x_pixels = f.variables['image_original_shape'][0]
        self.transform.y_pixels = f.variables['image_original_shape'][1]
        self.transform.pixel_aspectratio = f.variables['image_original_pixel_aspect'].data[()]

        try:
            self.transform.offset = f.variables['image_offset'][:]
        except KeyError:
            pass

        try:
            self.model_normals = f.variables['ModelNormals'].data.astype(np.float32)
        except KeyError:
            self.model_normals = None

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
                if len(self.transform.transform_actions) == 0 and self.transform.pixel_aspectratio == 1:
                    self.fullchip = 'Original'
                else:                
                    self.fullchip = True

        self.x = f.variables['PixelXLocation'].data
        self.y = f.variables['PixelYLocation'].data

        f.close()


    def set_detector_window(self,window):
        '''
        Adjust the raydata to apply to a different detector region for than was used
        to perform the original raycast. Useful for example if a CMOS camera has been calibrated
        over the full frame, but you now want to use this calibration for data which
        has been cropped.

        Calling this function with `None` as the single argument sets the raydata
        back to its "native" state. Otherwise, call with a 4 element tuple specifying the
        left,top,width and height of the detector window.

        Detector window coordinates must always be in original coordinates.

        Parameters:

            window (tuple or list) : A 4-element tuple or list of integers defining the \
                                     detector window coordinates (Left,Top,Width,Height)
        '''
        if window is None:

            if self.crop is not None:

                dx = self.crop[0] - self.native_geometry[2][0]
                dy = self.crop[1] - self.native_geometry[2][1]

                ox, oy = self.transform.display_to_original_coords(self.x, self.y)
                ox = ox + dx
                oy = oy + dy

                self.transform.x_pixels,self.transform.y_pixels,self.transform.offset = self.native_geometry

                self.x,self.y = self.transform.original_to_display_coords(ox,oy)

                del self.native_geometry

                self.crop_inds = None
                self.crop = None

        elif len(window) == 4:

            dx = window[0] - self.transform.offset[0]
            dy = window[1] - self.transform.offset[1]

            ox,oy = self.transform.display_to_original_coords(self.x,self.y)
            ox = ox - dx
            oy = oy - dy

            self.native_geometry = (self.transform.x_pixels,self.transform.y_pixels,self.transform.offset)

            self.transform.x_pixels = window[2]
            self.transform.y_pixels = window[3]
            self.transform.offset = (window[0],window[1])

            nx,ny = self.transform.original_to_display_coords(ox,oy)

            newshape = self.transform.get_display_shape()

            if self.fullchip and (np.all( nx > 0) or np.all(nx + 1 < newshape[0]) or np.all(ny > 0) or np.all(ny + 1 < newshape[1])):
                self.transform.x_pixels,self.transform.y_pixels,self.transform.offset = self.native_geometry

                raise ValueError('Requested crop window ({:d}x{:d} at {:d}x{:d}) is outside the raycasted area ({:d}x{:d} at {:d}x{:d})'.format(window[2],window[3],window[0],window[1],self.transform.x_pixels,self.transform.y_pixels,self.transform.offset[0],self.transform.offset[1]))

            self.x = nx
            self.y = ny

            if self.fullchip:
                rowinds = np.squeeze(np.argwhere(np.logical_and(self.y[:,0] >= 0, self.y[:,0] < newshape[1])))
                colinds = np.squeeze(np.argwhere(np.logical_and(self.x[rowinds,:][0,:] >= 0, self.x[rowinds,:][0,:] < newshape[0])))
                self.crop_inds = (rowinds,colinds)


            elif self.x.ndim == 2:
                xinds = np.argwhere(np.logical_and(self.x >= 0, self.x < newshape[0]))
                yinds = np.argwhere(np.logical_and(self.y[xinds] >= 0, self.y[xinds] < newshape[1]))
                self.crop_inds = (xinds,yinds)
            else:
                raise Exception("Cropping a raydata object with strange dimensions! I don't know how to do that; help!")

            self.crop = copy.deepcopy(window)

        else:
            raise ValueError('Cannot understand detector window; should be None or (Left,Top,Width,Height)')


    def get_ray_start(self,x=None,y=None,im_position_tol = 1,coords='Display'):
        '''
        Get the 3D x,y,z coordinates of the "start" of the casted rays / sightlines.

        Parameters:

            x,y (array-like)        : Image pixel coordinates at which to get the sight-line start coordinates.\
                                      If not specified, the start coordinates of all casted sight lines will be returned.
            im_position_tol (float) : If x and y are specified but no sight-line was cast at exactly the \
                                      input coordinates, the nearest casted sight-line will be returned \
                                      instead provided the pixel coordinates wre within this many pixels of \
                                      the requested coordinates.
            coords (str)            : Either ``Display`` or ``Coords``, specifies what orientation the input x \
                                      and y correspond to or orientation of the returned array.

        Returns:

            np.ndarray              : An array containing the 3D coordinates, in metres, of the start of \
                                      each sight line (i.e. camera pupil position). If x and y coordinates were \
                                      given either to this function or to calcam.raycast_sightlines(), the shape of \
                                      this array is the same as the input x and y arrays with an additional \
                                      dimension added which contains the  [X,Y,Z] 3D coordinates. Otherwise the shape\
                                      is (h x w x 3) where w and h are the image width and height (in display coords).
        '''
        if x is None and y is None:
            if self.fullchip:
                if coords.lower() == 'display':
                    if self.crop is None:
                        return self.ray_start_coords
                    else:
                        return self.ray_start_coords[self.crop_inds[0],:,:][:,self.crop_inds[1],:]
                else:
                    if self.crop is None:
                        return self.transform.display_to_original_image(self.ray_start_coords)
                    else:
                        return self.transform.display_to_original_image(self.ray_start_coords[self.crop_inds[0],:,:][:,self.crop_inds[1],:])
            else:
                if self.crop is None:
                    return self.ray_start_coords
                else:
                    return self.ray_start_coords[self.crop_inds[0],:][self.crop_inds[1],:]
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
                [start_X,start_Y,start_Z] = np.split(self.ray_start_coords,3,-1)
                start_X = start_X.flatten()
                start_Y = start_Y.flatten()
                start_Z = start_Z.flatten()
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
                        Xout[pointno] = start_X[np.argmin(deltaR)]
                        Yout[pointno] = start_Y[np.argmin(deltaR)]
                        Zout[pointno] = start_Z[np.argmin(deltaR)]
                    else:
                        raise Exception('No ray-traced pixel within im_position_tol of requested pixel ({:.1f},{:.1f})!'.format(x[pointno],y[pointno]))
                out = np.hstack([Xout,Yout,Zout])

                return np.reshape(out,oldshape + (3,),order='F')


    def get_ray_end(self,x=None,y=None,im_position_tol = 1,coords='Display'):
        '''
        Get the 3D x,y,z coordinates where the casted rays / sightlines intersect the CAD model.

        Parameters:

            x,y (array-like)        : Image pixel coordinates at which to get the sight-line end coordinates.\
                                      If not specified, the end coordinates of all casted sight lines will be returned.
            im_position_tol (float) : If x and y are specified but no sight-line was cast at exactly the \
                                      input coordinates, the nearest casted sight-line will be returned \
                                      instead provided the pixel coordinates wre within this many pixels of \
                                      the requested coordinates.
            coords (str)            : Either ``Display`` or ``Coords``, specifies what orientation the input x \
                                      and y correspond to or orientation of the returned array.

        Returns:

            np.ndarray              : An array containing the 3D coordinates, in metres, of the points where\
                                      each sight line intersects the CAD model. If x and y coordinates were \
                                      given either to this function or to calcam.raycast_sightlines(), the shape of \
                                      this array is the same as the input x and y arrays with an additional \
                                      dimension added which contains the  [X,Y,Z] 3D coordinates. Otherwise the shape\
                                      is (h x w x 3) where w and h are the image width and height (in display coords).

        '''
        if x is None and y is None:
            if self.fullchip:
                if coords.lower() == 'display':
                    if self.crop is None:
                        return self.ray_end_coords
                    else:
                        return self.ray_end_coords[self.crop_inds[0],:,:][:,self.crop_inds[1],:]
                else:
                    if self.crop is None:
                        return self.transform.display_to_original_image(self.ray_end_coords)
                    else:
                        return self.transform.display_to_original_image(self.ray_end_coords[self.crop_inds[0],:,:][:, self.crop_inds[1],:])
            else:
                if self.crop is None:
                    return self.ray_end_coords
                else:
                    return self.ray_end_coords[self.crop_inds[0],:][self.crop_inds[1],:]
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
                [end_X,end_Y,end_Z] = np.split(self.ray_end_coords,3,-1)
                end_X = end_X.flatten()
                end_Y = end_Y.flatten()
                end_Z = end_Z.flatten()
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
                        Xout[pointno] = end_X[np.argmin(deltaR)]
                        Yout[pointno] = end_Y[np.argmin(deltaR)]
                        Zout[pointno] = end_Z[np.argmin(deltaR)]
                    else:
                        raise Exception('No ray-traced pixel within im_position_tol of requested pixel!')
                out = np.hstack([Xout,Yout,Zout])

                return np.reshape(out,oldshape + (3,),order='F')


    def get_model_normals(self,x=None,y=None,im_position_tol = 1,coords='Display'):
        '''
        Get the 3D unit normal vectors of the CAD model surface where the camera sight-lines hit the model.
        Only available if calc_normals = True was given when running raycast_sightlines().

        Parameters:

            x,y (array-like)        : Image pixel coordinates at which to get the model normals.\
                                      If not specified, the end coordinates of all casted sight lines will be returned.
            im_position_tol (float) : If x and y are specified but no sight-line was cast at exactly the \
                                      input coordinates, the nearest casted sight-line will be returned \
                                      instead provided the pixel coordinates wre within this many pixels of \
                                      the requested coordinates.
            coords (str)            : Either ``Display`` or ``Coords``, specifies what orientation the input x \
                                      and y correspond to or orientation of the returned array.

        Returns:

            np.ndarray              : An array containing the normal vectors of the CAD model surface where\
                                      each sight line intersects the CAD model. If x and y coordinates were \
                                      given either to this function or to calcam.raycast_sightlines(), the shape of \
                                      this array is the same as the input x and y arrays with an additional \
                                      dimension added which contains the  [X,Y,Z] components of the normals. Otherwise the shape\
                                      is (h x w x 3) where w and h are the image width and height (in display coords).

        '''

        if self.model_normals is None:
            return Exception('Model normals were not calculated when doing the ray-cast. To use this function you must use raycast_sightlines() with calc_normals=True.')

        if x is None and y is None:
            if self.fullchip:
                if coords.lower() == 'display':
                    if self.crop is None:
                        return self.model_normals
                    else:
                        return self.model_normals[self.crop_inds[0],:,:][:,self.crop_inds[1],:]
                else:
                    if self.crop is None:
                        return self.transform.display_to_original_image(self.model_normals)
                    else:
                        return self.transform.display_to_original_image(self.model_normals[self.crop_inds[0],:,:][:, self.crop_inds[1],:])
            else:
                if self.crop is None:
                    return self.model_normals
                else:
                    return self.model_normals[self.crop_inds[0],:][self.crop_inds[1],:]
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
                [norm_X,norm_Y,norm_Z] = np.split(self.model_normals,3,-1)
                norm_X = norm_X.flatten()
                norm_Y = norm_Y.flatten()
                norm_Z = norm_Z.flatten()
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
                        Xout[pointno] = norm_X[np.argmin(deltaR)]
                        Yout[pointno] = norm_Y[np.argmin(deltaR)]
                        Zout[pointno] = norm_Z[np.argmin(deltaR)]
                    else:
                        raise Exception('No ray-traced pixel within im_position_tol of requested pixel!')
                out = np.hstack([Xout,Yout,Zout])

                return np.reshape(out,oldshape + (3,),order='F')

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
                    if self.crop is None:
                        return raylength
                    else:
                        return raylength[self.crop_inds[0],:][:,self.crop_inds[1]]
                else:
                    if self.crop is None:
                        return self.transform.display_to_original_image(raylength)
                    else:
                        return self.transform.display_to_original_image(raylength[self.crop_inds[0],:][:,self.crop_inds[1]])
            else:
                if self.crop is None:
                    return raylength
                else:
                    return raylength[self.crop_inds[0]][self.crop_inds[1]]
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
                            raise Exception('No ray-traced pixel within im_position_tol of requested pixel ({:.1f},{:.1f})!'.format(x[pointno], y[pointno]))
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
            coords (str)            : Either ``Display`` or ``Original``, specifies what orientation the input x \
                                      and y correspond to or orientation of the returned array.

        Returns:

            np.ndarray              : Array containing the sight-line directions. If the ray cast was for the \
                                      full detector and x and y are not specified, the array shape will be \
                                      (h x w x 3) where w nd h are the image width and height. Otherwise it will \
                                      be the same shape as the input x and y coordinates plus an extra dimension.
        '''
        vectors = (self.ray_end_coords - self.ray_start_coords)
        lengths = np.sqrt(np.sum(vectors**2,axis=-1))
        dirs =  vectors / np.repeat(lengths.reshape(np.shape(lengths)+(1,)),3,axis=-1)

        if x is None and y is None:
            if self.fullchip:
                if coords.lower() == 'display':
                    if self.crop is None:
                        return dirs
                    else:
                        return dirs[self.crop_inds[0],:,:][:,self.crop_inds[1],:]
                else:
                    if self.crop is None:
                        return self.transform.display_to_original_image(dirs)
                    else:
                        return self.transform.display_to_original_image(dirs[self.crop_inds[0],:,:][:, self.crop_inds[1],:])
            else:
                if self.crop is None:
                    return dirs
                else:
                    return dirs[self.crop_inds[0]][self.crop_inds[1]]
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
