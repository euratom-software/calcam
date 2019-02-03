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


'''
Geometry matrix module for Calcam.

Written by Scott Silburn & James Harrison.
'''

import multiprocessing
import copy
import time
import json
import os
import random

import numpy as np
import scipy.sparse
import scipy.io
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon as PolyPatch
from matplotlib.collections import PatchCollection
import matplotlib.path as mplpath

try:
    import meshpy.triangle
    import meshpy.tet
    meshpy_err = None
except Exception as e:
    meshpy_err = '{:}'.format(e)

from . import config
from . import misc
from .io import ZipSaveFile


class PoloidalVolumeGrid:
    '''
    Class for representing tomographic reconstruction grids with
    polygonal grid cells in the R, Z plane. Quantities are assumed to be
    uniform within the volume of each grid cell. Grid cells can be arbitrary 
    polygons but all cells in the grid must all have the same number of 
    sides.
    
    Grids can be constructed by directly instantiating this class with the 
    following parameters, or alternatively  a number of convenience 
    functions are provided for easily creating various types of grid.

    Parameters:
        
        vertices (np.ndarray)    : (N_verts x 2) array of floats containing the \
                                   (R,Z) coordinates of the grid cell vertices.
                                
        cells (np.ndarray)       : (N_cells x N_verts_per_cell) array of integers \
                                   specifying which vertices (indexes in to the vertices \
                                   array) define each grid cell. For each cell (array row), \
                                   the vertices must be listed in order around the \
                                   cell perimeter (in either direction).
                                
        wall_contour (np.ndaray) : Nx2 array containing the R,Z wall contour of the machine. \
                                   If provided, this is used for plotting purposes.
                                   
        src (str)                 : Human readable string describing where the grid came from.
                                   
    '''
    
    def __init__(self,vertices,cells,wall_contour=None,src=None):

        
        self.vertices = vertices.copy()
        self.cells = cells.copy()
        self.wall_contour = wall_contour.copy()
        
        if src is None:
            self.history = 'Created by {:s} on {:s} at {:s}'.format(misc.username,misc.hostname,misc.get_formatted_time())
        elif 'by' in src and 'on' in src and 'at' in src:
            self.history = history
        else:
            self.history = history + ' by {:s} on {:s} at {:s}'.format(misc.username,misc.hostname,misc.get_formatted_time())

        self._validate_grid()
        self._build_edge_list()
        self._cull_unused_verts()


    def _validate_grid(self):
        # Some code here to check the generation worked OK.
        pass



    def _cull_unused_verts(self):
        '''
        Remove any un-used vertices from the mesh definition.
        '''
        # Check what members of the set of all vertices
        # appear in the list of vertices used by cells
        used_vert_inds = set(self.cells.flatten())
        all_vert_inds = set(range(self.vertices.shape[0]))
        used_verts = np.array( sorted(list(all_vert_inds & used_vert_inds)),dtype=np.uint32)

        # Remove the unused vertices and keep track of vertex indexing
        ind_translation = np.zeros(self.vertices.shape[0],dtype=np.uint32)
        ind_translation[used_verts] = np.arange(used_verts.size,dtype=np.uint32)

        self.vertices = self.vertices[used_verts,:]
        self.cells = ind_translation[self.cells]
        self.segments = ind_translation[self.segments]


    def _build_edge_list(self):
        '''
        Build the list of line segments in the grid
        and which line segments border which grid cells.
        '''
        # Make the list of line segments and which line segments are
        # associated with each cell.
        self.segments = np.zeros((np.prod(self.cells.shape),2),dtype=np.uint32)
        self.cell_sides = np.zeros(self.cells.shape,dtype=np.uint32)

        segment_index = 0
        verts_per_cell = self.verts_per_cell()

        # For each grid cell
        for cell_index in range(self.n_cells()):

            # For each side of the triangle
            for seg_index in range(verts_per_cell):

                seg = sorted( [self.cells[cell_index][seg_index],self.cells[cell_index][(seg_index + 1) % verts_per_cell]] )

                # Store the index if this line segment is already in the segment list,
                # otherwise add it to the segment list and then store its index.
                try:
                    self.cell_sides[cell_index][seg_index] = np.where( (self.segments[:,0] == seg[0]) & (self.segments[:,1] == seg[1]) )[0][0]
                except IndexError:
                    self.segments[segment_index,:] = seg
                    self.cell_sides[cell_index][seg_index] = segment_index
                    segment_index += 1


        self.segments = self.segments[:segment_index,:]



    def n_cells(self):
        '''
        Get the number of grid cells.

        Returns:

            int : Number of grid cells

        '''
        return self.cells.shape[0]


    def n_segments(self):
        '''
        Get the number of line segments in the grid.

        Returns:

            int : Number of grid cells

        '''
        return self.segments.shape[0]


    def verts_per_cell(self):
        '''
        Get the number of vertices per grid cell.

        Returns:

            int : Number of vertices per cell

        '''
        return self.cells.shape[1]

    def n_vertices(self):
        '''
        Get the number of vertices in the grid.
        
        Returns:
            
            int : Number of vertices.
        '''
        return self.vertices.shape[0]


    def get_extent(self):
        '''
        Get the R,Z extent of the grid.
        
        Returns:
            
            tuple : 4-element tuple of floats containing the \
                    grid extent Rmin,Rmax,Zmin,Zmax
        '''
        rmin = self.vertices[:,0].min()
        rmax = self.vertices[:,0].max()
        zmin = self.vertices[:,1].min()
        zmax = self.vertices[:,1].max()

        return rmin,rmax,zmin,zmax




    def get_cell_intersections(self,ray_start,ray_end,plot=False):
        '''
        Get the intersections of a ray, i.e. a straight line
        in 3D space, with the grid cell boundaries.
        
        Parameters:
            
            ray_start (sequence) : 3-element sequence containing the X,Y,Z coordinates \
                                   of the ray's start position.

            ray_end (sequence)   : 3-element sequence containing the X,Y,Z coordinates \
                                   of the ray's end position.
                                   
            plot (bool)          : If set to True, the function will also plot the ray \
                                   in R,Z and red circles at each intersection location. \
                                   Mainly intended for debugging and algorithm demonstration.
                        
        Returns:
            
            tuple   : Intersection information:
                      
                      * NumPy array containing the lengths along the ray where intersections \
                        with grid cell boundaries lines were found (in order from start to end of the ray).
                         
                      * list of lists containing the indices of the grid cell(s) which were involved \
                        in  each intersection. Each intersection will be associated with 1 or more grid cells.

        '''
        # Turn off some NumPy warnings because we will inevitably
        # have some dividing by zero and such in here, but it's harmless.
        with np.errstate(divide='ignore',invalid='ignore'):
            
            ray_start = np.array(ray_start)
            ray_end = np.array(ray_end)
            
            ray_length = np.sqrt( np.sum( (ray_end - ray_start)**2 ) )
            
            # Parametric coefficients for ray
            pax,pay,paz = ray_start
            dpx,dpy,dpz = (ray_end - ray_start) / ray_length
            
            # Parametric coefficients for grid segments
            lar = self.vertices[self.segments[:,0],0]
            laz = self.vertices[self.segments[:,0],1]
            dlr = self.vertices[self.segments[:,1],0] - lar
            dlz = self.vertices[self.segments[:,1],1] - laz
    
            a = -dlz**2*dpx**2 - dlz**2*dpy**2 + dlr**2*dpz**2
            b = 2*dlr*dlz*dpz*lar - 2*dlr**2*dpz*laz - 2*dlz**2*dpx*pax - 2*dlz**2*dpy*pay + 2*dlr**2*dpz*paz
            c = (dlz**2*lar**2 - 2*dlr*dlz*lar*laz + dlr**2*laz**2 - dlz**2*pax**2 - dlz**2*pay**2 + 2*dlr*dlz*lar*paz - 2*dlr**2*laz*paz + dlr**2*paz**2)
            
            # These will be the intersection positions
            n_lines = self.n_segments()
            t_ray0 = np.zeros(n_lines) - 1.
            t_seg0 = np.zeros(n_lines) - 1.
            t_ray1 = np.zeros(n_lines) - 1.
            t_seg1 = np.zeros(n_lines) - 1.
    
            # The magic number!
            d = b**2 - 4*a*c
    
            # d > 0 means two real solutions and hence two intersections
            indx = np.where(d > 0)
            if len(indx) > 0:
                q = -0.5 * (b[indx] + np.sign(b[indx]) * np.sqrt(d[indx]))
                t_ray0[indx] = q/a[indx]
                t_seg0[indx] = (-laz[indx] + paz + dpz * t_ray0[indx])/dlz[indx]
                t_ray1[indx] = c[indx] / q
                t_seg1[indx] = (-laz[indx] + paz + dpz * t_ray1[indx])/dlz[indx]
    
            # d == 0 means one real solution so one intersection
            indx = np.where(d == 0)
            if len(indx) > 0:
                q = -0.5 * (b[indx] + np.sign(b[indx]) * np.sqrt(d[indx]))
                t_ray0[indx] = q / a[indx]
                t_seg0[indx] = (-laz[indx] + paz + dpz * t_ray0[indx])/dlz[indx]
            
            # Special case for exactly horizontal rays.
            indx = np.where(np.abs(dlz) <1e-14)
            if len(indx) > 0:
                t_ray0[indx] = (-paz + laz[indx])/dpz
                hitr = np.sqrt((pax+t_ray0[indx]*dpx)**2+(pay+t_ray0[indx]*dpy)**2)
                t_seg0[indx] = (-lar[indx] + hitr)/dlr[indx]
                
            
            # Valid intersections are ones within the line segment length and 
            # within the ray length
            valid_inds0 = (t_seg0 >= 0.) & (t_seg0 <= 1.) & (t_ray0 >= 0.) & (t_ray0 <= ray_length)
            valid_inds1 = (t_seg1 >= 0.) & (t_seg1 <= 1.) & (t_ray1 >= 0.) & (t_ray1 <= ray_length)
            

            # Full list of intersection distance and segment index
            t_ray = np.concatenate( (t_ray0[valid_inds0],t_ray1[valid_inds1]) )     
            seg_inds = np.arange(self.n_segments(),dtype=np.uint32)
            seg_inds = np.concatenate( (seg_inds[valid_inds0],seg_inds[valid_inds1]) )
    
            
            # Sort the intersections by length along the sight-line.
            # Also round t_ray to 9 figures because we'll want to find unique values
            # of it shortly, so round to something well over machine precision.
            sort_order = np.argsort(t_ray)
            t_ray = t_ray[sort_order].round(decimals=12)
            seg_inds = seg_inds[sort_order]
            
            # This will be the output list of cell indices
            cell_inds = []
            
            # Check which cells the intersected line segments belong to.
            for intersection_pos in np.unique( t_ray ):
                
                segs = seg_inds[t_ray == intersection_pos]
                
                cells = set()
                for seg in seg_inds[ t_ray == intersection_pos]:
                    cells.update( np.where(self.cell_sides == seg)[0] )
                
                cell_inds.append(list(cells))
            
            # We only want to return the unique intersection lengths
            t_ray = np.unique(t_ray)

            # If we're asked to, plot the sight line and intersection points.
            if plot:
                
                # Plot the sight line
                ray_dir = (ray_end - ray_start) / ray_length
                l = np.linspace(0,ray_length,int(ray_length/1e-2))
                R = np.sqrt( (ray_start[0] + l*ray_dir[0])**2 + (ray_start[1] + l*ray_dir[1])**2  )
                Z = ray_start[2] + l*ray_dir[2]
                plt.plot(R,Z)
                
                # Plot the intersections
                points3d = np.tile(ray_start[np.newaxis,:],(t_ray.size,1)) + np.tile(t_ray[:,np.newaxis],(1,3)) * np.tile(ray_dir[np.newaxis,:],(t_ray.size,1))
                R = np.sqrt(np.sum(points3d[:,:2]**2,axis=1))
                Z = points3d[:,2]
                plt.plot(R,Z,'ro')


        return t_ray,cell_inds




    def plot(self,data=None,clim=None,cmap=None,line_colour=(0,0,0),cell_linewidth=None,cblabel='',axes=None):
        '''
        Either plot a given data vector on the grid, or if no data vector is given,
        plot the grid itself.

        Parameters:

            data (array)                    : Data to plot on the grid. Must be a 1D array with as many elements \
                                              as there are grid cells. If not given, only the grid structure is plotted.

            clim (sequence or None)         : 2 element sequence giving the colourmap limits for the data [min,max]. \
                                              If set to None, the data min and max are used.

            cmap (str or None)              : Name of the matplotlib colourmap to use to display the data. If not given, \
                                              matplotlib's default colourmap will be used.

            line_colour (sequence or None)  : 3-element sequence of values between 0 and 1 specifying the R,G,B \
                                              colour with which to draw the wall contour and grid cell boundaries. \
                                              If set to None, the wall and cell boundaries are not drawn.

            cell_linewidth (float)          : Line width to use to show the grid cell boundaries. If set to 0, the grid \
                                              cell boundaries will not be drawn.

            cblabel (str)                   : Label for the data colour bar

            axes (matplotlib.pyplot.Axes)   : Matplotlib axes on which to plot. If not given, a new figure will be created.


        Returns:
            
            tuple : MatPlotLib objects for the plot:
                    
                    * matplotlib.axes.Axes                    : The matplotlib axes containing the plot.
            
                    * matplotlib.collections.PatchCollection  : PatchCollection containing the patches used to show the data and \
                                                                grid cells. This object has useful methods for further adjusting \
                                                                the plot like set_cmap(), set_clim() etc.

                    * list of matplotlib.lines.Line2D         : List of matpltolib line objects making up the wall contour.

        '''
        # If we have some data to plot, validate that we have 1 value per grid cell
        if data is not None:

            data = np.array(data)

            if len(np.squeeze(data).shape) > 1:
                raise ValueError('Data must be a 1D array the same length as the number of grid cells ({:d}).'.format(self.n_cells()))

            if data.size != self.n_cells():
                raise ValueError('Data vector is the wrong length! Data has {:d} values but the grid gas {:d} cells.'.format(data.size,self.n_cells()))

            cmap = cm.get_cmap(cmap)

            # If we have data, grid cell edges default to off
            if cell_linewidth is None:
                cell_linewidth = 0

        else:
            # If we have no data, grid cell edges default to on
            if cell_linewidth is None:
                cell_linewidth = 0.25


        # Create a matplotlib patch collection to show the grid cells,
        # with the data associated with it if necessary
        patches = []
        verts_per_cell = self.cells.shape[1]

        for cell_ind in range(self.n_cells()):
            xy = np.vstack( [ self.vertices[self.cells[cell_ind,i],:] for i in range(verts_per_cell)] )
            patches.append( PolyPatch( xy, closed=True) )
        pcoll = PatchCollection(patches)
        pcoll.set_array(np.squeeze(data))


        # Set up appearance of the grid cells
        if data is not None:
            pcoll.set_cmap(cmap)
            if clim is not None:
                pcoll.set_clim(clim)

        pcoll.set_edgecolor(line_colour)
        pcoll.set_linewidth(cell_linewidth)


        # If not given an existing set of axes to use, make a new figure.
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        
        # Plot the grid cells!
        axes.add_collection(pcoll)

        # Plot the wall
        if line_colour is not None and self.wall_contour is not None:
            wall_lines = axes.plot(self.wall_contour[:,0],self.wall_contour[:,1],color=line_colour,linewidth=2,marker=None)
            wall_lines = wall_lines + axes.plot([self.wall_contour[-1,0],self.wall_contour[0,0]],[self.wall_contour[-1,1],self.wall_contour[0,1]],color=line_colour,linewidth=2,marker=None)
        else:
            wall_lines = []

        # Add a colour bar if we have data
        if data is not None:
            plt.colorbar(pcoll,label=cblabel)

        # Prettification
        rmin,rmax,zmin,zmax = self.get_extent()
        headroom = 0.1
        axes.set_xlim([rmin - headroom,rmax + headroom])
        axes.set_ylim([zmin - headroom, zmax + headroom])
        axes.set_aspect('equal')
        axes.set_xlabel('R [m]')
        axes.set_ylabel('Z [m]')

        return axes,pcoll,wall_lines



    def remove_cells(self,cell_inds):
        '''
        Remove grid cells with the given indices from the grid.
        
        Parameters:
            
            cell_inds (sequence) : Sequence of integers specifying which \
                                   cell indices to remove.
        '''
        cell_inds = np.array(cell_inds).astype(np.uint32)

        self.cells = np.delete(self.cells,cell_inds,axis=0)
        self.cell_sides = np.delete(self.cell_sides,cell_inds,axis=0)

        self._cull_unused_verts()




class GeometryMatrix:
    '''
    Class to represent a scalar geometry matrix and associated metadata.
    
    A new geometry matrix can be created by instantiating this class 
    with the below parameters, or a saved geometry matrix can be loaded
    from disk using the GeometryMatrix.load() class method.
    
    The matrix itself can be accessed in the "data" attribute, where it is
    stored as a sparse matrix using the scipy.sprase.csr_matrix class.

    Parameters:

        grid (calcam.PoloidalVolumeGrid)  : Reconstruction grid to use

        raydata (calcam.RayData)          : Ray data for the camera to be inverted
        
        coords (str)                      : Either 'Original' (default) or 'Display'; whether the image \
                                            being inverted will be in original or display orientation.

        pixel_order (str)                 : What pixel order to use when flattening \
                                            the 2D image array in to the 1D data vector. \
                                            Default 'F' goes verticallytop-to-bottom, column-by-column, \
                                            alternatively 'C' goes horizontally left-to-right,  row-by-row.
        
        calc_status_callback (callable)   : Callable which takes a single argument to be called with status updates. \
                                            The argument will either be a string for textual status updates or a float \
                                            from 0 to 1 specifying the progress of the calculation. If set to None, no \
                                            status updates are issued.

    '''

    def __init__(self,grid,raydata,pixel_order='F',calc_status_callback = misc.LoopProgPrinter().update):

        if grid is not None and raydata is not None:

            if raydata.fullchip:
                if raydata.fullchip == True:
                    raise Exception('Raydata object does not contain information on the image orientation used for ray casting.\n Please set the "fullchip" attribute of the raydata object to either "Display" or "Original".')
                else:
                    self.image_coords = raydata.fullchip
                    self.binning = raydata.binning
                    self.pixel_order = pixel_order
            else:
                
                self.image_coords = None
                self.binning = None
                self.pixel_order = None
                
            self.grid = copy.copy(grid)
            '''
            calcam.PoloidalVolumeGrid : The inversion grid o
            '''    
            
            self.image_geometry = raydata.transform

            self.history = {'los':raydata.history,'grid':grid.history,'matrix':'Created by {:s} on {:s} at {:s}'.format(misc.username,misc.hostname,misc.get_formatted_time())}
            
            # Number of grid cells and sight lines
            n_cells = grid.n_cells()
            n_los = raydata.x.size
    
            # Flatten out the ray start and end coords
            ray_start_coords = raydata.ray_start_coords.reshape(-1,3,order=self.pixel_order)
            ray_end_coords = raydata.ray_end_coords.reshape(-1,3,order=self.pixel_order)


            # Multi-threadedly loop over each sight-line in raydata and calculate its matrix row.
            # Store the results as coords + data then build the matrix after, because that is much faster.
            
            if calc_status_callback is not None:
                calc_status_callback('Calculating geometry matrix elements using {:d} CPUs...'.format(config.n_cpus))
            
            last_status_update = 0.
            
            # We will do the calculation in a random order,
            # purely to get better time remaining estimation.
            inds = list(range(n_los))
            random.shuffle(inds)

            colinds = []
            rowinds = []
            data = []            

            with multiprocessing.Pool( config.n_cpus ) as cpupool:
                calc_status_callback(0.)
                for i , row_data in enumerate( cpupool.imap( self._calc_row_volume, np.hstack((ray_start_coords[inds,:],ray_end_coords[inds,:])) , 10 ) ):          
                    rowinds.append(np.zeros(row_data[0].shape,dtype=np.uint32) + inds[i])                    
                    colinds.append(row_data[0])
                    data.append(row_data[1])
                    
                    if time.time() - last_status_update > 1. and calc_status_callback is not None:
                        calc_status_callback(float(i) / n_los)
                        last_status_update = time.time()

            # Build the matrix!
            self.data = scipy.sparse.csr_matrix((np.concatenate(data),(np.concatenate(rowinds),np.concatenate(colinds))),shape=(n_los,n_cells))            
            
            if calc_status_callback is not None:
                calc_status_callback(1.)
            
            
            # Remove any grid cells + matrix rows which have no sight-line coverage.
            unused_cells = np.where(np.abs(self.data.sum(axis=0)) == 0)[1]
            self.grid.remove_cells(unused_cells)

            used_cols = np.where(self.data.sum(axis=0) > 0)[1]
            self.data = self.data[:,used_cols]


    def get_los_coverage(self):
        '''
        Get the number of lines of sight viewing each inversion element.
        
        Returns:
            
            np.ndarray : Vector with as many elements as there a
        '''
        return np.diff(self.data.tocsc().indptr)



    def get_mean_rcc(self,status_callback=misc.LoopProgPrinter().update):
        '''
        Get the mean reflective correlation coefficient of each matrix column 
        with every other, but with the mean excluding other columns which are
        not correlated at all.
        
        Indicates which grid cells are likely to have correlation & artefact problems
        in inversions.
        
        Note: this is done in a way which maintains the sparse storage structure of the
        matrix to avoid memory issues, but can be extremely slow to calculate for
        large matrices.
        
        Parameters:
            
            status_callback (callable) : Callable which takes a single argument to be called \
                                         with status updates. The argument will either be a \
                                         string for textual status updates or a float from 0 to 1 \
                                         specifying the progress of the calculation. If set to None, \
                                         no status updates are issued. By default, status updates are \
                                         printed to stdout.
                                                                  
        Returns:
            
            np.ndarray : Array with as many elements as there are matrix columns, giving the \
                         mean(ish) correlation coefficients.

        '''
        if status_callback is not None:
            status_callback('Calculating matrix column correlations...'.format(config.n_cpus)) 

        dat_csc = scipy.sparse.csc_matrix(self.data,copy=True)
        datm = 1./np.sqrt((dat_csc.power(2)).sum(axis=0).A.ravel())
        dat_csc = dat_csc * scipy.sparse.diags(datm)       
        
        rcc_out = np.zeros(self.grid.n_cells())
        last_status_update = 0    
        

        for col_ind in range(self.grid.n_cells()):

            # Clever fast way to do dot product and 
            yy = np.squeeze(dat_csc[:,col_ind].toarray())

            rcc = np.add.reduceat(dat_csc.data * yy[dat_csc.indices], dat_csc.indptr[:-1])

            rcc = np.delete(correl,col_ind)
            rcc_out[col_ind] = correl[correl > 1e-3].mean(axis=0)

            
            if time.time() - last_status_update > 1. and status_callback is not None:
                status_callback(float(col_ind) / (self.grid.n_cells()-1))
                last_status_update = time.time()
        
        
        if status_callback is not None:
            status_callback(1.)
        
        return rcc_out
    

    def set_binning(self,binning):
        '''
        Set the level of image binning. Can be used to
        decrease the size of the matrix to reduce memory or
        computation requirements for inversions.
        
        Parameters:
            
            binning (float) : Desired image binning. Must be larger than the \
                              binning already 
                              
        '''
        if binning < self.binning:
            raise ValueError('Specified binning is lower than existing binning! The binning can only be increased.')
        elif binning == self.binning:
            return
        else:
            
            if self.image_coords is not None:
                if self.image_coords.lower() == 'display':
                    image_dims = self.image_geometry.get_display_shape()
                elif self.image_coords.lower() == 'original':
                    image_dims = self.image_geometry.get_original_shape()
            else:
                raise Exception('Nope, no worky.')
                
            bin_factor = int(binning / self.binning)

            init_shape = (np.array(image_dims) / self.binning).astype(np.uint32)
            row_inds = np.arange(np.prod(init_shape),dtype=np.uint32)
            row_inds = np.reshape(row_inds,init_shape,order=self.pixel_order)
            
            ind_arrays = []
            for colshift in range(bin_factor):
                for rowshift in range(bin_factor):
                    ind_arrays.append(row_inds[colshift::bin_factor,rowshift::bin_factor].reshape(int(np.prod(init_shape)/bin_factor**2),order=self.pixel_order))
            
            new_data = self.data[ind_arrays[0],:]
            for indarr in ind_arrays[1:]:
                new_data = new_data + self.data[indarr,:]
                
            norm_factor = scipy.sparse.diags(np.ones(self.grid.n_cells())/bin_factor**2,format='csr')
            
            self.data = new_data * norm_factor
                
            
        



    def save(self,filename):
        '''
        Save the geometry matrix to a file.
        
        Parameters:
            
            filename (str) : File name to save to, including file extension. \
                             The file extension determines the format to be saved: \
                             '.npz' for compressed NumPy binary format, \
                             '.mat' for MATLAB format or 
                             '.zip' for Zipped collection of ASCII files.
        '''
        try:
            fmt = filename.split('.')[1:][-1]
        except IndexError:
            raise ValueError('Given file name does not include file extension; extension .npz, .mat or .zip must be included to determine file type!')

        if fmt == 'npz':
            self._save_npz(filename)
        elif fmt == 'mat':
            self._save_matlab(filename)
        elif fmt == 'zip':
            self._save_txt(filename)
        else:
            raise ValueError('File extension "{:s}" not understood; options are "npz", "mat" or "zip".'.format(fmt))

            
            

    def _calc_row_volume(self,ray_endpoints):
        '''
        Calculate a matrix row given the sight-line start and end in 3D,
        for volume type grids (i.e. quantities constant within grid cell volume).
        
        Parameters:
            
            ray_endpoints (sequence) : 6-element sequence containing the \
                                       ray start and end coordinates: \
                                       (Xstart, Ystart, Zstart, Xend, Yend, Zend)
            
        Returns:
            
            tuple : Calculated row elements: \
            
                    * numpy.ndarray containing row indices of non-zero elements \
                      in the column.

                    * numpy.ndarray containing the values of the matrix elements at \
                      those row indices.

        '''
        
        ray_start_coords = np.array(ray_endpoints[:3])
        ray_end_coords = np.array(ray_endpoints[3:])
        
        # Total length of the ray
        ray_length = np.sqrt( np.sum( (ray_end_coords - ray_start_coords)**2 ) )
        
        # Output will be stored in here
        #row_out = scipy.sparse.lil_matrix( (1,self.grid.n_cells()) )
        col_inds = np.arange(self.grid.n_cells(),dtype=np.uint32)
        data = np.zeros(self.grid.n_cells())
        
        # Get the ray intersections with the grid cells
        positions,intersected_cells = self.grid.get_cell_intersections(ray_start_coords,ray_end_coords)

        # Convert the lists of intersected cells in to sets for later convenience.
        intersected_cells = [set(cells) for cells in intersected_cells]

        # For keeping track of which cell we're currently in
        in_cell = set()
        
        # Loop over each intersection
        for i in range(positions.size):
            
            if len(in_cell) == 0:
                # Entering the grid
                in_cell = intersected_cells[i]
                
            else:
                # Going from one cell to another         
                leaving_cell = list(intersected_cells[i] & in_cell)
                
                if len(leaving_cell) == 1:
                    
                    #row_out[0,leaving_cell[0]] = row_out[0,leaving_cell[0]] + (positions[i] - positions[i-1])
                    data[leaving_cell[0]] = data[leaving_cell[0]] + (positions[i] - positions[i-1])
                    in_cell = intersected_cells[i]
                    in_cell.remove(leaving_cell[0])
                
                else:
                    '''
                    # Make a plot to show where on the grid the error has occured.
                    self.grid.plot()
                    self.grid.get_cell_intersections(ray_start_coords,ray_end_coords,plot=True)
                    problem_point = ray_start_coords + positions[i]*(ray_end_coords - ray_start_coords)/ray_length
                    plt.plot(np.sqrt(problem_point[0]**2 + problem_point[1]**2),problem_point[2],'bo')
                    plt.title('Error for: {:},{:}'.format(ray_start_coords,ray_end_coords))
                    plt.show()
                    '''
                    raise Exception('Error generating geometry matrix row: could not identify which grid cell the LoS left.')
        
        
        # If the sight line ends inside a cell, add the length it was inside that cell.
        if len(in_cell) > 0:
            
            leaving_cell = list(in_cell)
            
            if len(leaving_cell) == 1:
                data[leaving_cell[0]] = data[leaving_cell[0]] + (ray_length - positions[-1])
            else:
                raise Exception('Error generating geometry matrix row: could not identify which grid cell the LoS left.')

        return col_inds[data > 0],data[data > 0]



    def _save_npz(self,filename):
        '''
        Save the geometry matrix in compressed NumPy binary format.
        '''
        coo_data = self.data.tocoo()
        
        np.savez_compressed( filename,
                             mat_row_inds = coo_data.row,
                             mat_col_inds = coo_data.col,
                             mat_data = coo_data.data,
                             mat_shape = self.data.shape,
                             grid_verts = self.grid.vertices,
                             grid_cells = self.grid.cells,
                             grid_wall = self.grid.wall_contour,
                             binning = self.binning,
                             pixel_order = self.pixel_order,
                             history = self.history,
                             grid_type = 'volume',
                            )

    def _load_npz(self,filename):
        '''
        Load a geometry matrix from a compressed NumPy binary file
        '''
        f = np.load(filename)
        
        self.grid = PoloidalVolumeGrid(f['grid_verts'],f['grid_cells'],f['grid_wall'])
        
        self.binning = float(f['binning'])
        self.pixel_order = str(f['pixel_order'])
        self.history = f['history']
        
        self.data = scipy.sparse.csr_matrix((f['mat_data'],(f['mat_row_inds'],f['mat_col_inds'])),shape=f['mat_shape'])



    def _save_matlab(self,filename):
        '''
        Save the geometry matrix in MATLAB format.
        '''
        scipy.io.savemat( filename,
                         { 'geom_mat': self.data,
                           'grid_verts':self.grid.vertices,
                           'grid_cells':self.grid.cells,
                           'grid_wall':self.grid.wall_contour,
                           'binning':self.binning,
                           'pixel_order':self.pixel_order,
                           'sightline_history':self.history['los'],
                           'matrix_history':self.history['matrix'],
                           'grid_history':self.history['grid'],
                           'grid_type':'volume',
                         }
                        )
        
        
    def _load_matlab(self,filename):
        '''
        Load geometry matrix from a MATLAB file.
        '''
        f = scipy.io.loadmat(filename)
        
        self.grid = PoloidalVolumeGrid(f['grid_verts'],f['grid_cells'],f['grid_wall'])
        
        self.binning = float(f['binning'][0,0])
        self.pixel_order = str(f['pixel_order'][0])
        self.history = {'los':str(f['sightline_history'][0]),'grid':str(f['grid_history'][0]),'matrix':str(f['matrix_history'][0])}
        
        self.data = f['geom_mat'].tocsr()
        
         

    def _save_txt(self,filename):
        '''
        Save the geometry matrix as a set of zipped ASCII files.
        '''
        coo_data = self.data.tocoo()
        
        with ZipSaveFile(filename,'w') as zfile:
            
            dest = zfile.get_temp_path()
            np.savetxt(os.path.join(dest,'mat_row_ind.txt'),coo_data.row,fmt='%d')
            np.savetxt(os.path.join(dest,'mat_col_ind.txt'),coo_data.col,fmt='%d')
            np.savetxt(os.path.join(dest,'mat_data.txt'),coo_data.data,fmt='%.5e')
            np.savetxt(os.path.join(dest,'grid_verts.txt'),self.grid.vertices,fmt='%.5e')
            np.savetxt(os.path.join(dest,'grid_cells.txt'),self.grid.cells,fmt='%d')
            np.savetxt(os.path.join(dest,'grid_wall.txt'),self.grid.wall_contour)
            
            meta = {'mat_shape':self.data.shape, 'binning':self.binning, 'pixel_order':self.pixel_order, 'grid_type':'volume','history':self.history}
            
            with zfile.open_file('metadata.json','w') as metafile:
                
                json.dump(meta,metafile,indent=4)
            
    
    def _load_txt(self,filename):
        '''
        Load a geometry matrix from a set of zipped ASCII files.
        '''
        with ZipSaveFile(filename,'r') as zfile:
            
            src = zfile.get_temp_path()

            
            with zfile.open_file('metadata.json','r') as metafile:
                
                meta = json.load(metafile)

            verts = np.loadtxt(os.path.join(src,'grid_verts.txt'))
            cells = np.loadtxt(os.path.join(src,'grid_cells.txt'),dtype=np.uint32)
            wall = np.loadtxt(os.path.join(src,'grid_wall.txt'))
            
            self.grid = PoloidalVolumeGrid(verts,cells,wall,meta['history']['grid'])
                
            self.binning = meta['binning']
            self.pixel_order = meta['pixel_order']
            self.history = meta['history']
            
            row_ind = np.loadtxt(os.path.join(src,'mat_row_ind.txt'),dtype=np.uint32)
            col_ind = np.loadtxt(os.path.join(src,'mat_col_ind.txt'),dtype=np.uint32)
            data = np.loadtxt(os.path.join(src,'mat_data.txt'))
            
            self.data = scipy.sparse.csr_matrix((data,(row_ind,col_ind)),shape=meta['mat_shape'])
                
        

    @classmethod
    def load(cls,filename):
        '''
        Load a Calcam saved geometry matrix from disk.
        
        Parameters:
            
            filename (str)  : File name to load from. Can be a  \
                              NumPy (.npz), MATLAB (.mat) or \
                              zipped ASCII (.zip) file.
                             
        Returns:
            
            calcam.GeometryMatrix : Loaded geometry matrix.
            
        '''
        # Create an empty geometry matrix object
        geommat = cls(None,None)
        
        try:
            fmt = filename.split('.')[1:][-1]
        except IndexError:
            raise ValueError('Given file name does not include file extension; extension must be specified to determine file type!')

        if fmt == 'npz':
            geommat._load_npz(filename)
        elif fmt == 'mat':
            geommat._load_matlab(filename)
        elif fmt == 'zip':
            geommat._load_txt(filename)
        else:
            raise ValueError('File extension "{:s}" not understood; should be an "npz", "mat" or "zip" file.'.format(fmt))

        return geommat
        

def squaregrid(wall_contour,cell_size,rmin=None,rmax=None,zmin=None,zmax=None):
    '''
    Create a reconstruction grid with square grid cells.
    
    Parameters:
        
        wall_contour (str or np.ndarray) : Either the name of a Calcam CAD model \
                                           from which to use the wall contour, or an \
                                           N x 2 array of R,Z points defining the machine wall.
                                           
        cell_size (float)                : Side length of each grid cell in metres.
        
        rmin, rmax, zmin, zmax  (float)  : Optional limits of the grid extent in the R, Z plane. \
                                           Any combination of these may or may not be given; if none \
                                           are given the entire wall contour interior is gridded.
                                           
    Returns:
        
        calcam.PoloidalVolumeGrid    : Generated grid.
        
    '''
    
    # If given a machine name for the wall contour, get the R,Z contour
    if type(wall_contour) is str:
        wall_contour = _get_ccm_wall(wall_contour)

    # If we have a wall contour which joins back up with itself, remove
    # the last point.
    if np.abs(wall_contour[0,:] - wall_contour[-1,:]).max() < 1e-15:
        wall_contour = wall_contour[:-1]


    if rmin is None:
        rmin = wall_contour[:,0].min()
    else:
        rmin = max(rmin,wall_contour[:,0].min())

    if rmax is None:
        rmax = wall_contour[:,0].max()
    else:
        rmax = min(rmax,wall_contour[:,0].max())

    if zmin is None:
        zmin = wall_contour[:,1].min()
    else:
        zmin = max(zmin,wall_contour[:,1].min())

    if zmax is None:
        zmax = wall_contour[:,1].max() 
    else:
        zmax = min(zmax,wall_contour[:,1].max())


    Rmax = cell_size * np.ceil( (rmax - rmin) / cell_size ) + rmin
    Zmax = cell_size * np.ceil( (zmax - zmin) / cell_size ) + zmin

    nr = int( (Rmax - rmin)/cell_size) + 1
    nz = int( (Zmax - zmin)/cell_size) + 1

    Rpts = np.linspace(rmin,Rmax,nr)
    Zpts = np.linspace(zmin,Zmax,nz)

    wall_path = mplpath.Path(np.vstack((wall_contour,wall_contour[-1,:])),closed=True)

    vertices = np.zeros((nr * nz,2))
    cells = np.zeros( ( (nr-1)*(nz-1) , 4 ),dtype=np.uint32 )

    cell_ind = 0
    vert_ind = 0

    _verts = np.zeros((4,2))

    for iz in range(Zpts.size-1):
        for ir in range(Rpts.size-1):

            # 4 corners of the square grid cell
            _verts[0,:] = [Rpts[ir],Zpts[iz]]
            _verts[1,:] = [Rpts[ir+1],Zpts[iz]]
            _verts[2,:] = [Rpts[ir+1],Zpts[iz+1]]
            _verts[3,:] = [Rpts[ir],Zpts[iz+1]]

            # If the cell is completely outside the wall, don't bother.
            if not np.any(wall_path.contains_points(_verts)):
                continue

            for vert in range(4):
                try:
                    cells[cell_ind][vert] = np.where( (vertices[:,0] == _verts[vert,0]) & (vertices[:,1] == _verts[vert,1]) )[0][0]
                except IndexError:
                    vertices[vert_ind,:] = _verts[vert,:]
                    cells[cell_ind][vert] = vert_ind
                    vert_ind += 1
            cell_ind += 1

    
    return PoloidalVolumeGrid(vertices[:vert_ind,:],cells[:cell_ind,:],wall_contour)




def trigrid(wall_contour,max_cell_area,rmin=None,rmax=None,zmin=None,zmax=None,**kwargs):
    '''
    Create a reconstruction grid with triangular grid cells conforming to the 
    wall contour. Generated using J. Shewchuk's "triangle" via the MeshPy module.
    
    Parameters:
        
        wall_contour (str or np.ndarray) : Either the name of a Calcam CAD model \
                                           from which to use the wall contour, or an \
                                           N x 2 array of R,Z points defining the machine wall.
                                           
        max_cell_area (float)            : Maximum area of a grid cell area in square metres.
        
        rmin, rmax, zmin, zmax  (float)  : Optional limits of the grid extent in the R, Z plane. \
                                           Any combination of these may or may not be given; if none \
                                           are given the entire wall contour interior is gridded.
                                           
    Any additional keyword arguments will be passed directly to the triangle mesher, meshpy.triangle.build().
    This can be used to further control the meshing; see the MeshPy documentation for available arguments.
    
                                           
    Returns:
        
        calcam.PoloidalVolumeGrid    : Generated grid.
        
    '''
    # Before trying to generate a triangular grid, check we have the meshpy package available.
    if meshpy_err is not None:
        raise Exception('Cannot initialise MeshPy based mesh: MeshPy module failed to import with error: {:s}'.format(meshpy_err))

    # If given a machine name for the wall contour, get the R,Z contour
    if type(wall_contour) is str:
        wall_contour = _get_ccm_wall(wall_contour)

    # If we have a wall contour which joins back up with itself, remove
    # the last point.
    if np.abs(wall_contour[0,:] - wall_contour[-1,:]).max() < 1e-15:
        wall_contour = wall_contour[:-1]


    # Whatever happens, the wall will define some vertices to pass to the mesher.
    verts = wall_contour

    # Connectivity between wall contour vertices is simply each one
    # connected to the next, then the last connected to the first
    segments = np.zeros((verts.shape[0],2),dtype=np.uint32)
    segments[:,0] = np.arange(verts.shape[0])
    segments[:,1] = np.arange(1,verts.shape[0]+1)
    segments[-1][-1] = 0


    if not all( [limit is None for limit in [rmin,rmax,zmin,zmax] ] ):

        epsilon = 2*np.sqrt(max_cell_area)
        if rmin is None:
            rmin = wall_contour[:,0].min() - epsilon
        else:
            rmin = max(rmin,wall_contour[:,0].min() - epsilon)

        if rmax is None:
            rmax = wall_contour[:,0].max() + epsilon
        else:
            rmax = min(rmax,wall_contour[:,0].max() + epsilon)

        if zmin is None:
            zmin = wall_contour[:,1].min() - epsilon
        else:
            zmin = max(zmin,wall_contour[:,1].min() - epsilon)

        if zmax is None:
            zmax = wall_contour[:,1].max() + epsilon
        else:
            zmax = min(zmax,wall_contour[:,1].max() + epsilon)

        bbox_verts = np.zeros((4,2))
        bbox_verts[0,:] = [rmin,zmin]
        bbox_verts[1,:] = [rmax,zmin]
        bbox_verts[2,:] = [rmax,zmax]
        bbox_verts[3,:] = [rmin,zmax]

        bbox_segments = np.array( [ [0,1] , [1,2] , [2,3] , [3,0] ] ) + verts.shape[0] 

        epsilon = epsilon / 2.

        holeR = np.linspace(rmin - epsilon,rmax + epsilon, int((2*epsilon+rmax-rmin)/epsilon))
        holeZ = np.linspace(zmin - epsilon,zmax + epsilon, int((zmax-zmin)/epsilon))
        
        holeR,holeZ = np.meshgrid(holeR,holeZ)
        holecoords = np.hstack( (holeR.reshape(holeR.size,1), holeZ.reshape(holeZ.size,1)) )

        wall_path = mplpath.Path(np.vstack((verts,verts[-1,:])),closed=True)
        bbox_path = mplpath.Path(np.vstack((bbox_verts,bbox_verts[-1,:])),closed=True)

        inds =  np.logical_xor( bbox_path.contains_points(holecoords) , wall_path.contains_points(holecoords) )
        holecoords = holecoords[inds,:]

        geometry = {'vertices' : np.vstack( (verts,bbox_verts) ) , 'segments' : np.vstack( (segments,bbox_segments) ), 'holes': holecoords}

    else:

        geometry = {'vertices': verts, 'segments' : segments}


    mesher_kwargs = dict(kwargs)
    mesher_kwargs['max_volume'] = max_cell_area

    # Run triangle to make the grid!
    # Do this in a separate process so that if the triangle library has a fail,
    # it doesn't drag this whole python process down with it.
    q = multiprocessing.Queue()
    triprocess = multiprocessing.Process(target=_run_triangle_python,args=(geometry,q),kwargs=mesher_kwargs)
    triprocess.start()
    while triprocess.is_alive() and q.empty():
        time.sleep(0.05)
    
    if not q.empty():
        mesh = q.get()
        if isinstance(mesh,Exception):
            raise mesh
    else:
        raise Exception('Call to meshpy.triangle to generate the grid failed silently. The triangle library can be a bit flaky; maybe try again and/or change settings.')

    return PoloidalVolumeGrid(mesh['vertices'],mesh['cells'],wall_contour)



def _get_ccm_wall(model_name):
    '''
    Get the wall contour points for a named machine, if Calcam has a 
    CAD model for the machine with a wall contour in it.
    
    Parameters:
        
        model_name (str) : Name of the Calcam CAD model to get the wall for.
        
    Returns:
        
        np.ndarray : Nx2 array of (R,Z) coordinates around the wall contour
        
    '''
    cadmodels = config.CalcamConfig().get_cadmodels()
    if model_name not in cadmodels.keys():
        raise ValueError('Unknown name "{:s}" for wall contour; available machine names are: {:s}'.format(model_name,', '.join(cadmodels.keys())))
    else:
        with ZipSaveFile(cadmodels[model_name][0],'rs') as deffile:
            # Load the wall contour, if present
            if 'wall_contour.txt' in deffile.list_contents():
                with deffile.open_file('wall_contour.txt','r') as cf:
                    wall_contour = np.loadtxt(cf)
            else:
                raise Exception('CAD model "{:s}" does not define a wall contour!'.format(model_name))

    return wall_contour


def _run_triangle_python(geometry,queue,**kwargs):
    '''
    Run triangular mesh generation with the triangle library via MeshPy module.
    See MeshPy documentation.

    This function is designed to be used from a multiprocessing call, to avoid
    errors in the underlying triangle code from dragging down the main Python process
    (seen on Windows 10 / Py3.5.)

    Parameters:

        geometry (dict)               : Dictionary defining the bounding geometry, see triangle \
                                        manual.

        queue (multiprocessing.Queue) : Multiprocessing queue on which to place the result.
        
    Any keyword arguments are all passed to meshpy.triangle.build()

    '''
    # Re-arrange the input geometry and mesher arguments. This
    # is all done here because MeshPy crashes silently otherwise,
    # it appears to be highly not process-safe.

    # Put the input geometry in to a meshpy object
    boundary_geom = meshpy.triangle.MeshInfo()
    boundary_geom.set_points(geometry['vertices'])
    boundary_geom.set_facets(geometry['segments'])
    if 'holes' in geometry:
        boundary_geom.set_holes(geometry['holes'])

    # Don't just pass the kwargs straight to MeshPy or it crashes.
    mesher_args = dict(kwargs)

    try:
        # Run triangle and send the results back to the calling thread.
        mesh = meshpy.triangle.build(boundary_geom,**mesher_args)
        queue.put( {'vertices':np.array(mesh.points), 'cells':np.array(mesh.elements)}  )
    except Exception as e:
        queue.put(e)






