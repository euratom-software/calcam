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

Written by James Harrison, Mark Smithies & Scott Silburn.
'''

import multiprocessing
import copy
import time

import numpy as np
import scipy.sparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon as PolyPatch
from matplotlib.collections import PatchCollection
import matplotlib.path as mplpath

from .config import CalcamConfig
from .io import ZipSaveFile

try:
    import meshpy.triangle
    import meshpy.tet
    meshpy_err = None
except Exception as e:
    meshpy_err = '{:}'.format(e)



class PoloidalPolygonGrid():
    '''
    Base class for representing tomographic reconstruction grids
    in the R, Z plane. Grid cells can be arbitrary polygons but all 
    cells in the grid must all have the same number of sides, for now. 
    Generating the grid geometry must be implemented in a subclass; this 
    parent class provides generic methods for plotting, sight-line intersection etc.

    Paremeters needed to construct a grid are defined by the subclass.
    '''
    
    def __init__(self,*args,**kwargs):

        self.wall_contour = None
        self.vertices = None
        self.cells = None

        self._generate_grid(*args,**kwargs)

        self._validate_grid()
        self._cull_unused_verts()
        self._build_edge_list()


    def _validate_grid(self):
        # Some code here to check the generation worked OK.
        pass


    def _get_calcam_wall_contour(self,model_name):
        '''
        Get wall contour points for a named machine, if Calcam has a 
        CAD model for the machine with a wall contour in it.
        '''
        cadmodels = CalcamConfig().get_cadmodels()
        if model_name not in cadmodels.keys():
            raise ValueError('Unknown name "{:s}" for wall contour; available machine names are: {:s}'.format(model_name,', '.join(calcam_config.get_cadmodels.keys())))
        else:
            with ZipSaveFile(cadmodels[model_name][0],'rs') as deffile:
                # Load the wall contour, if present
                if 'wall_contour.txt' in deffile.list_contents():
                    with deffile.open_file('wall_contour.txt','r') as cf:
                        wall_contour = np.loadtxt(cf)
                else:
                    raise Exception('CAD model "{:s}" does not define a wall contour!'.format(model_name))

        return wall_contour



    def _generate_grid(self,*args,**kwargs):
        '''
        Abstract method to create the grid geometry. This must be overloaded
        by a derived class. The requirement is that this method sets the following
        attributes of the grid object:

           vertices (np.ndarray) :   N_verts x 2 NumPy array of floats containing the  \
                                     R,Z coordinates of grid vertices.

           cells  (np.ndarray)   :   N_cells x N_poly_corners NumPy array of integers specifying which \
                                     vertices (indexes in to self.vertices) define each grid cell.  \
                                     On each row, the vertex indices for a single polygonal grid cell must be \
                                     listed in order around the polygon perimeter (doesn't matter in what direction).
                                     
        It is also recommended but not mandatory to set the following attribute:
            
           wall_contour (np.ndarray) : Nx2 NumPy array containing the R,Z wall contour.

        '''
        raise NotImplementedError('Mesh generation must be handled by a derived class; PoloidalPolygonGrid cannot be instantiated directly.')



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

        Reurns:

            int : Number of grid cells

        '''
        return self.cells.shape[0]

    def n_segments(self):
        '''
        Get the number of line segments in the grid.

        Reurns:

            int : Number of grid cells

        '''
        return self.segments.shape[0]

    def verts_per_cell(self):
        '''
        Get the number of vertices per grid cell.

        Reurns:

            int : Number of vertices per cell

        '''
        return self.cells.shape[1]



    def get_extent(self):
        '''
        Get the R,Z extent of the grid.
        
        Returns:
            
            tuple : 4-element tuple of floats containing the \
                    grid extent R_min,R_max,Z_min,Z_max
        '''
        rmin = self.vertices[:,0].min()
        rmax = self.vertices[:,0].max()
        zmin = self.vertices[:,1].min()
        zmax = self.vertices[:,1].max()

        return rmin,rmax,zmin,zmax




    def get_cell_intersections(self,ray_start,ray_end):
        '''
        Get the intersections of a ray, i.e. a straight line
        in 3D space, with the grid.
        
        Parameters:
            
            ray_start (sequence) : 3-element sequence containing the X,Y,Z coordinates \
                                   of the ray's start position.

            ray_end (sequence)   : 3-element sequence containing the X,Y,Z coordinates \
                                   of the ray's end position.
                        
        Returns:
            
            np.ndarray    : Lengths along the ray where intersections with grid \
                            lines were found (in order from start to end of the ray).
                         
            list of lists : Indices of the grid cell(s) which were involved in \
                            each intersection. There may be 1 or more cells per \
                            intersection.
            
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
            indx = np.where(dlz == 0)
            if len(indx) > 0:
                t_ray0[indx] = (-paz + laz[indx])/dpz
                hitr      = np.sqrt((pax+t_ray0[indx]*dpx)**2+(pay+t_ray0[indx]*dpy)**2)
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
            t_ray = t_ray[sort_order].round(decimals=9)
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
        
        return t_ray,cell_inds




    def plot(self,data=None,clim=None,cmap=None,line_colour=(0,0,0),cell_linewidth=None,cblabel=None,axes=None):
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
            
            matplotlib.axes.Axes                    : The matplotlib axes containing the plot.
            
            matplotlib.collections.PatchCollection  : PatchCollection containing the patches used to show the data and \
                                                      grid cells. This object has useful methods for further adjusting \
                                                      the plot like set_cmap(), set_clim() etc.

            list of matplotlib.lines.Line2D         : List of matpltolib line objects making up the wall contour.

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
        '''
        cell_inds = np.array(cell_inds).astype(np.uint32)

        self.cells = np.delete(self.cells,cell_inds,axis=0)
        self.cell_sides = np.delete(self.cell_sides,cell_inds,axis=0)

        self._cull_unused_verts()



class SquareGrid(PoloidalPolygonGrid):
    '''
    A reconstruction grid with square grid cells.
    
    Parameters:
        
        wall_contour (str or np.ndarray) : Either the name of a Calcam CAD model \
                                           from which to use the wall contour, or an \
                                           N x 2 array of R,Z points defining the machine wall.
                                           
        cell_size (float)                : Side length of each grid cell in metres.
        
        rmin, rmax, zmin, zmax  (float)  : Optional limits of the grid extent in the R, Z plane. \
                                           Any combination of these may or may not be given; if none \
                                           are given the entire wall contour interior is gridded.
    '''
    def _generate_grid(self,wall_contour,cell_size,rmin=None,rmax=None,zmin=None,zmax=None):

        # If given a machine name for the wall contour, get the R,Z contour
        if type(wall_contour) is str:
            wall_contour = self._get_calcam_wall_contour(wall_contour)

        # If we have a wall contour which joins back up with itself, remove
        # the last point.
        if np.abs(wall_contour[0,:] - wall_contour[-1,:]).max() < 1e-15:
            wall_contour = wall_contour[:-1]

        self.wall_contour = wall_contour

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

        self.vertices = np.zeros((nr * nz,2))
        self.cells = np.zeros( ( (nr-1)*(nz-1) , 4 ),dtype=np.uint32 )

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
                        self.cells[cell_ind][vert] = np.where( (self.vertices[:,0] == _verts[vert,0]) & (self.vertices[:,1] == _verts[vert,1]) )[0][0]
                    except IndexError:
                        self.vertices[vert_ind,:] = _verts[vert,:]
                        self.cells[cell_ind][vert] = vert_ind
                        vert_ind += 1
                cell_ind += 1


        self.vertices = self.vertices[:vert_ind,:]
        self.cells = self.cells[:cell_ind,:]



class TriGrid(PoloidalPolygonGrid):
    '''
    A reconstruction grid with triangular grid cells conforming to the 
    wall contour. Generated using triangle via the MeshPy module.
    
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
    '''
    def _generate_grid(self,wall_contour,max_cell_area,rmin=None,rmax=None,zmin=None,zmax=None,**kwargs):

        # Before trying to generate a triangular grid, check we have the meshpy package available.
        if meshpy_err is not None:
            raise Exception('Cannot initialise MeshPy based mesh: MeshPy module failed to import with error: {:s}'.format(triangle_err))

        # If given a machine name for the wall contour, get the R,Z contour
        if type(wall_contour) is str:
            wall_contour = self._get_calcam_wall_contour(wall_contour)

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

        self.wall_contour = wall_contour

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
        triprocess = multiprocessing.Process(target=self._run_triangle_python,args=(geometry,q),kwargs=mesher_kwargs)
        triprocess.start()
        while triprocess.is_alive() and q.empty():
            time.sleep(0.05)
        
        if not q.empty():
            mesh = q.get()
            if isinstance(mesh,Exception):
                raise mesh
        else:
            raise Exception('Call to meshpy.triangle to generate the grid failed silently. The triangle library can be a bit flaky; maybe try again and/or change settings.')

        # Store the results in the appropriate attributes
        self.vertices = mesh['vertices']
        self.cells = mesh['cells']




    def _run_triangle_python(self,geometry,queue,**kwargs):
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



class GeometryMatrix():
    '''
    Class to represent a scalar geometry matrix.

    Parameters:

        grid (calcam.geometry_matrix.PoloidalPolygonGrid)  : Reconstruction grid to use

        raydata (calcam.Raydata)                           : Ray data for the camera to be inverted

        los_order (str)                                    : What pixel order to use when flattening \
                                                             the 2D image array in to the 1D data vector. \
                                                             Default 'F' goes verticallytop-to-bottom, column-by-column, \
                                                             alternatively 'C' goes horizontally left-to-right,  row-by-row.

        cull_grid (bool)                                   : Whether to remove grid cells without any sight-line coverage from the grid.

        n_processes (int)                                  : For multi-CPU computers, how many execution processes \
                                                             to use for calculations: speeds up processing ~linearly with \
                                                             number of processes. Default is 1 fewer than the number of CPUs present.
    '''

    def __init__(self,grid,raydata,los_order='F',cull_grid = True,n_processes=multiprocessing.cpu_count()-1):

        # We'll take a copy of the grid because we might mess with it.
        self.grid = copy.copy(grid)

        self.los_order = los_order

        # Number of grid cells and sight lines
        n_cells = grid.n_cells()
        n_los = raydata.x.size

        # Flatten out the ray start and end coords
        ray_start_coords = raydata.ray_start_coords.reshape(-1,3,order=self.los_order)
        ray_end_coords = raydata.ray_end_coords.reshape(-1,3,order=self.los_order)
        
        # Initialise the geometry matrix.
        # Start off with a row-based-linked-list representation
        # because it's quick and easy to construct.
        self.data = scipy.sparse.lil_matrix((n_los,n_cells))

        # Multi-threadedly loop over each sight-line in raydata and calculate its matrix row
        with multiprocessing.Pool(n_processes) as cpupool:
            for row_ind , row_data in enumerate( cpupool.imap( self._calc_matrix_row, np.hstack((ray_start_coords,ray_end_coords)) , 10 ) ):          
                self.data[row_ind,:] = row_data

        
        # If enabled, remove any grid cells + matrix rows which have no sight-line coverage.
        if cull_grid:
            unused_cells = np.where(np.abs(self.data.sum(axis=0)) == 0)[1]
            self.grid.remove_cells(unused_cells)

            used_cols = np.where(self.data.sum(axis=0) > 0)[1]
            self.data = self.data[:,used_cols]
            
        
        # Convert to a Compressed Sparse Row Matrix representation,
        # which should be more convenient to actually use.
        self.data = scipy.sparse.csr_matrix(self.data)
        

    def _calc_matrix_row(self,ray_endpoints,plot=False):
        '''
        Calculate a matrix row given the sight-line start and end in 3D.
        
        Parameters:
            
            ray_endpoints (sequence) : 6-element sequence containing the \
                                       ray start and end coordinates: \
                                       (Xstart, Ystart, Zstart, Xend, Yend, Zend)/
            
            plot (bool)              : Whether to make a pretty plot of the \
                                       ray, ray-grid intersections and grid cell coverage.\
                                       Intended for debugging and algorithm demonstration purposes.   
            
        Returns:
            
            np.ndarray : Calculated geometry matrix row.

        '''
        
        ray_start_coords = np.array(ray_endpoints[:3])
        ray_end_coords = np.array(ray_endpoints[3:])
        
        # Total length of the ray
        ray_length = np.sqrt( np.sum( (ray_end_coords - ray_start_coords)**2 ) )
        
        # Output will be stored in here
        row_out = np.zeros( (1,self.grid.n_cells()) )       
        
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
                    
                    row_out[0,leaving_cell[0]] = row_out[0,leaving_cell[0]] + (positions[i] - positions[i-1])
                
                    in_cell = intersected_cells[i]
                    in_cell.remove(leaving_cell[0])
                
                else:
                    raise Exception('Something is wrong with this algorithm...could not identify which cell I have left.')
        
        
        # If the sight line ends inside a cell, add the length it was inside that cell.
        if len(in_cell) > 0:
            
            leaving_cell = list(in_cell)
            
            if len(leaving_cell) == 1:
                row_out[0,leaving_cell[0]] = row_out[0,leaving_cell[0]] + (ray_length - positions[-1])
            else:
                raise Exception('Something is wrong with this algorithm...could not identify which cell I have left.')

        
        # If we're plotting, make a plot!
        if plot:
            
            # Plot the grid and the calculated cell weights
            plot_data = np.squeeze(row_out.toarray())
            self.grid.plot(plot_data,cell_linewidth=0.25,cblabel='Sight-line length in cell [m]')
            
            # Plot the sight line
            ray_dir = (ray_end_coords - ray_start_coords) / ray_length
            l = np.linspace(0,ray_length,int(ray_length/1e-3))
            R = np.sqrt( (ray_start_coords[0] + l*ray_dir[0])**2 + (ray_start_coords[1] + l*ray_dir[1])**2  )
            Z = ray_start_coords[2] + l*ray_dir[2]
            plt.plot(R,Z)
            
            # Plot the intersections
            for i in range(positions.size):
                pos = ray_start_coords + positions[i]*ray_dir
                R = np.sqrt(np.sum(pos[:2]**2))
                Z = pos[2]
                plt.plot(R,Z,'ro')
                
            plt.show()
                
        return row_out
        


    def plot_coverage(self,metric='n_los'):
        '''
        Plot various metrics related to the coverage of the inversion grid
        by the camera sight-lines. Possible choices are:

        * 'l_tot' : Total length of all sight-lines passing through each cell.

        * n_los   : Number of different lines-of-sight covering each cell.


        Parameters:

            metric (str)    : What metric to plot.

        '''
        if metric == 'l_tot':

            coverage = self.data.sum(axis=0)
            self.grid.plot(coverage,cblabel='Total sight-line coverage [m]')

        elif metric == 'n_los':
            coverage = np.diff(self.data.tocsc().indptr)
            self.grid.plot(coverage,cblabel='Number of sight-lines seeing grid cell')
