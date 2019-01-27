import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon as PolyPatch
from matplotlib.collections import PatchCollection
import matplotlib.path as mplpath
import scipy.sparse
import multiprocessing
import copy
from .config import CalcamConfig
from .io import ZipSaveFile

import time

calcam_config = CalcamConfig()

try:
    import triangle
    triangle_err = None
except Exception as e:
    triangle_err = '{:s}'.format(e)



class PoloidalPolygonGrid():
    '''
    Parent class for representing a tomographic reconstruction grid
    in the R, Z plane. Grid cells can be arbitrary polygons but all 
    cells in the grid must all have the same number of sides, for now. 
    Generating the polygons  must be implemented in a subclass; this 
    parent class provides generic methods for plotting, sight-line intersection etc.

    Paremeters:

        wall_contour (str or np.ndarray) : Either the name of a Calcam CAD model whose wall contour to use, \
                                           or an N x 2 array containing the bounding wall contour in R,Z.

        scale(float)                     : Characteristic scale of a grid cell in metres.

        rmin, rmax, zmin, zmax (float)   : Optional bounds to limit the gridded area within the R,Z plane. \
                                           If not given, the entire area of the wall contour is meshed.
    '''
    def __init__(self,wall_contour,scale,rmin = None, rmax = None, zmin = None, zmax = None):

        cell_area = scale**2

        # If given a machine name for the wall contour, get the R,Z contour
        if type(wall_contour) is str:
            cadmodels = calcam_config.get_cadmodels()
            if wall_contour not in cadmodels.keys():
                raise ValueError('Unknown name "{:s}" for wall contour; available machine names are: {:s}'.format(wall_contour,', '.join(calcam_config.get_cadmodels.keys())))
            else:
                with ZipSaveFile(cadmodels[wall_contour][0],'rs') as deffile:
                    # Load the wall contour, if present
                    if 'wall_contour.txt' in deffile.list_contents():
                        with deffile.open_file('wall_contour.txt','r') as cf:
                            wall_contour = np.loadtxt(cf)
                    else:
                        raise Exception('CAD model "{:s}" does not define a wall contour!'.format(wall_contour))


        # If we have a wall contour which joins back up with itself, remove
        # the last point.
        if np.abs(wall_contour[0,:] - wall_contour[-1,:]).max() < 1e-15:
            wall_contour = wall_contour[:-1]

        self.wall_contour = wall_contour
        epsilon = scale*2
        if any([limit is not None for limit in [rmin, rmax, zmin, zmax]]):

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

            bbox = [rmin,rmax,zmin,zmax]

        else:

            bbox = None

        self._generate_grid(wall_contour,cell_area,bbox)
        self._cull_unused_verts()



    def _generate_grid(self,wall_contour,cell_area,bbox):
        '''
        Function to generate the grid geometry; this must be handled by
        a subclass. This method must generate the following two attributes 
        for the object:

            * self.vertices :   N_verts x 2 NumPy array of floats containing the  \
                                R,Z coordinates of grid vertices.

            * self.cells    :   N_cells x N_poly_corners NumPy array of integers specifying which \
                                vertices (indexes in to self.vertices) belong to each \
                                grid cell. On each row, the vertex indices for a single grid cell must be \
                                listed in order around the polygon perimiter (doesn't matter in what direction).

        Parameters:

            wall_contour (np.ndarray)         : Nx2 NumPy array containing the wall contour in R,Z

            cell_area (float)                 : Desired characteristic cell area in square metres

            bbox (4 element sequence or None) : Rmin, Rmax, Zmin, Zmax limits of the desired grid extent.
        '''

        raise NotImplementedError('Mesh generation must be handled by a subclass; PoloidalPolygonGrid cannot be used directly.')


    def get_n_cells(self):
        '''
        Get the number of grid cells.

        Reurns:

            int : Number of grid cells

        '''
        return self.cells.shape[0]


    def get_extent(self):

        rmin = self.vertices[:,0].min()
        rmax = self.vertices[:,0].max()
        zmin = self.vertices[:,1].min()
        zmax = self.vertices[:,1].max()

        return rmin,rmax,zmin,zmax


    def get_cell_los_lengths(self,los_ends,step_length=2e-2,plot=False):
        '''
        Given line-of-sight start and end coordinates in 3D cartesian coordinates,
        get the length for which that sight-line is in each grid cell. Approximates the
        line-of-sight in R,Z as a series of straight line segments and finds intersections
        with the grid cell boundaries using this technique: https://stackoverflow.com/a/565282

        A current limitation is that every cell has to have the same number of vertices.

        This will 

        Parameters:

            los_ends (np.array)    : 6 element array containing X,Y,Z sight-line start and end coordinates

            step_length (float)  : Step length along the sight-line

            plot (bool)          : Whether to plot the sight-line and intersection points.

        '''

        # Array to store the results
        arr_out = np.zeros((1,self.get_n_cells()))

        los_start = los_ends[:3]
        los_end = los_ends[3:]

        # Generate evenly spaced R,Z points along the sight line according to step_length
        los_len = np.sqrt(np.sum((los_end-los_start)**2 ))
        los_dir = (los_end - los_start) / los_len
        npts = int( los_len / step_length ) + 1
        step_length = los_len / (npts - 1)
        xyz = np.tile(los_start[np.newaxis,:],(npts,1)) + np.tile(los_dir[np.newaxis,:],(npts,1)) * np.tile(np.linspace( 0., los_len, npts )[:,np.newaxis],(1,3))
        rz = np.hstack( ( np.sqrt(np.sum(xyz[:,:2]**2,axis=1))[:,np.newaxis] , xyz[:,2][:,np.newaxis] ) )
        del xyz

        # Start and end points for every LoS segment
        q = rz[:-1,:]
        s = rz[1:,:] - q

        # Start and end points for every line segment in the grid
        p = self.vertices[self.segments[:,0],:]
        r = self.vertices[self.segments[:,1],:] - p
        
        if plot:

            # Plot the sight-line
            plt.plot(rz[:,0],rz[:,1],'-')


        # Turn off numpy divide by zero warnings for the main calculation, 
        # because it often does divide by zero but this is harmless.
        with np.errstate(divide='ignore'):

            # Step along the sight-line and check intersection
            in_cell = set()
            l_entry = None
            left_grid = False
            for los_segment in range(npts - 1):

                s_ = np.tile(s[los_segment,:][np.newaxis,:],(self.segments.shape[0],1))

                crs = np.cross(r, s_ )

                diff = np.tile(q[los_segment,:][np.newaxis,:],(self.segments.shape[0],1)) - p

                u = np.cross( diff , r) / crs
                t = np.cross( diff , s_) / crs

                intersections = np.logical_and(u >= 0, u < 1)
                intersections = np.logical_and(intersections, t >= 0)
                intersections = np.logical_and(intersections, t <= 1)
                intersections = np.argwhere(intersections == True)

                # If we have intersections, keep track of which cells we're in
                if intersections.size > 0:

                    # Distances along the LoS where we found intersections
                    intersect_l = np.squeeze( (u[intersections] + los_segment) * step_length )

                    # If we have multiplt intersections, sort them so we step 
                    # along the intersections in the right order
                    if intersect_l.size > 1:
                        sort_order = np.argsort(intersect_l)
                        intersect_l = intersect_l[sort_order]
                        intersections = intersections[sort_order]

                    # Now step along each intersection to get cell lengths.
                    # Since we can have multiple segment intersections at the same
                    # location, we step by distance along sight line and not by index.
                    for los_pos in np.unique( intersect_l.round(decimals=10)):

                        # Check which cells we've hit the border of
                        cells = set()
                        segments = intersections[np.abs(intersect_l - los_pos) < 1e-10]
                        for segment in segments:
                            cells.update(np.where(self.cell_sides == segment)[0])                    

                        if len(in_cell) == 0:
                            # Initially entering the grid
                            in_cell = cells
                            l_entry = los_pos

                        else:
                            # Leaving a cell
                            leaving_cell = list(cells & in_cell)
                            if len(leaving_cell) == 1:

                                leaving_cell = leaving_cell[0]

                                # The actual important bit: add the length to the output for the cell we're leaving
                                arr_out[0,leaving_cell] = arr_out[0,leaving_cell] + los_pos - l_entry

                                in_cell = cells
                                in_cell.remove(leaving_cell)
                                l_entry = los_pos

                            # This isn't really needed, it's just reassuring for me while debugging the algorithm.
                            elif len(leaving_cell) > 1 :
                                raise Exception('Something is wrong with the algorithm; this number should always be 1!')


                        if plot:
                            point_coords_xyz = los_start + los_pos * los_dir
                            point_R = np.sqrt(np.sum(point_coords_xyz[:2]**2))
                            point_Z = point_coords_xyz[2]
                            plt.plot( point_R, point_Z, 'ro')

                # This is invoked if we hit the end of the LoS inside the body of a grid cell.
                elif los_segment == npts - 2:
                    if len(in_cell) == 1:
                        arr_out[0,list(in_cell)[0]] = arr_out[0,list(in_cell)[0]] + los_len - l_entry

        return arr_out


    def plot(self,data=None,clim=None,cmap=None,line_colour=(0,0,0),cell_linewidth=None,cblabel=None,axes=None):
        '''
        Either plot a given data vector on the grid, or if no data vector is given,
        plot the grid itself. Note: this does not call plt.show() itself when done.

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
            
            matplotlib.collections.PatchCollection  : PatchCollection containing the patches used to show the data and \
                                                      grid cells. This object has useful methods for further adjusting \
                                                      the plot like set_cmap(), set_clim() etc.

            list of matplotlib.lines.Line2D         : List of matpltolib line objects making up the wall contour.

        '''

        # If we have some data to plot, validate that we have 1 value per grid cell
        if data is not None:

            data = np.array(data)

            if len(np.squeeze(data).shape) > 1:
                raise ValueError('Data must be a 1D array the same length as the number of grid cells ({:d}).'.format(self.get_n_cells()))

            if data.size != self.get_n_cells():
                raise ValueError('Data vector is the wrong length! Data has {:d} values but the grid gas {:d} cells.'.format(data.size,self.get_n_cells()))

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

        for cell_ind in range(self.get_n_cells()):
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
        if line_colour is not None:
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

        return pcoll,wall_lines



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
    '''

    def _generate_grid(self,wall_contour,cell_area,bbox):

        cell_size = np.sqrt(cell_area)

        # If the bounding box is not given, set it to the extent of the wall outline
        if bbox is None:
            bbox = [wall_contour[:,0].min(),wall_contour[:,0].max(),wall_contour[:,1].min(),wall_contour[:,1].max()]

        Rmax = cell_size * np.ceil( (bbox[1] - bbox[0]) / cell_size ) + bbox[0]
        Zmax = cell_size * np.ceil( (bbox[3] - bbox[2]) / cell_size ) + bbox[2]

        Rpts = np.linspace(bbox[0],Rmax,int( (Rmax - bbox[0])/cell_size) + 1)
        Zpts = np.linspace(bbox[2],Zmax,int( (Zmax - bbox[2])/cell_size) + 1)

        wall_path = mplpath.Path(np.vstack((wall_contour,wall_contour[-1,:])),closed=True)

        self.vertices = []
        self.cells = []
        self.segments = []
        self.cell_sides = []

        for iz in range(Zpts.size-1):
            for ir in range(Rpts.size-1):

                # 4 corners of the square grid cell
                verts = []
                verts.append([Rpts[ir],Zpts[iz]])
                verts.append([Rpts[ir+1],Zpts[iz]])
                verts.append([Rpts[ir+1],Zpts[iz+1]])
                verts.append([Rpts[ir],Zpts[iz+1]])

                # If the cell is completely outside the wall, don't bother.
                if not np.any(wall_path.contains_points(verts)):
                    continue

                self.cells.append([])
                for vert_ind in range(4):

                    try:
                        self.cells[-1].append(self.vertices.index(verts[vert_ind]))
                    except ValueError:
                        self.vertices.append(verts[vert_ind])
                        self.cells[-1].append(len(self.vertices)-1)

                self.cell_sides.append([])
                for seg_ind in range(4):
                    seg = sorted( [self.cells[-1][seg_ind], self.cells[-1][(seg_ind + 1) % 4]] )

                    try:
                        self.cell_sides[-1].append(self.segments.index(seg))
                    except ValueError:
                        self.segments.append(seg)
                        self.cell_sides[-1].append(len(self.segments)-1)

        self.vertices = np.array(self.vertices)
        self.cells = np.array(self.cells)
        self.segments = np.array(self.segments)
        self.cell_sides = np.array(self.cell_sides)



class TriangularGrid(PoloidalPolygonGrid):
    '''
    A reconstruction grid with triangular grid cells,
    allowing it to conform nicely to the wall contour.
    '''

    def __init__(*args,**kwargs):

        # When instantiating a TriangularGrid, make sure the triangle library is available.
        if triangle_err is not None:
            raise Exception('Cannot initialise triangular geometry matrix: triangle module failed to import with error: {:s}'.format(triangle_err))

        PoloidalPolygonGrid.__init__(*args,**kwargs)


    def _generate_grid(self,wall_contour,triangle_area,bbox):

        # Whatever happens, the wall will define some vertices to pass to the mesher.
        verts = wall_contour

        # Connectivity between wall contour vertices is simply each one
        # connected to the next, then the last connected to the first
        segments = np.zeros((verts.shape[0],2),dtype=np.uint32)
        segments[:,0] = np.arange(verts.shape[0])
        segments[:,1] = np.arange(1,verts.shape[0]+1)
        segments[-1][-1] = 0


        # If we have a bounding box, add segments for it to the input geometry and 
        # put an array of hole markers around it so we only grid inside the bounding box.
        if bbox is not None:

            bbox_verts = np.zeros((4,2))
            bbox_verts[0,:] = [bbox[0],bbox[2]]
            bbox_verts[1,:] = [bbox[1],bbox[2]]
            bbox_verts[2,:] = [bbox[1],bbox[3]]
            bbox_verts[3,:] = [bbox[0],bbox[3]]

            bbox_segments = np.array( [ [0,1] , [1,2] , [2,3] , [3,0] ] ) + verts.shape[0] 

            epsilon = np.sqrt(triangle_area)

            holeR = np.linspace(bbox[0] - epsilon,bbox[1] + epsilon, int((2*epsilon+bbox[1]-bbox[0])/epsilon))
            holeZ = np.linspace(bbox[2] - epsilon,bbox[3] + epsilon, int((bbox[3]-bbox[2])/epsilon))
            
            holeR,holeZ = np.meshgrid(holeR,holeZ)
            holecoords = np.hstack( (holeR.reshape(holeR.size,1), holeZ.reshape(holeZ.size,1)) )

            wall_path = mplpath.Path(np.vstack((verts,verts[-1,:])),closed=True)
            bbox_path = mplpath.Path(np.vstack((bbox_verts,bbox_verts[-1,:])),closed=True)

            inds =  np.logical_xor( bbox_path.contains_points(holecoords) , wall_path.contains_points(holecoords) )
            holecoords = holecoords[inds,:]

            geometry = {'vertices' : np.vstack( (verts,bbox_verts) ) , 'segments' : np.vstack( (segments,bbox_segments) ), 'holes': holecoords}

        else:

            geometry = {'vertices': verts, 'segments' : segments}


        # Run triangle to make the grid!
        # Do this in a separate process so that if the triangle library has a fail,
        # it doesn't drag this whole python process down with it.
        q = multiprocessing.Queue()
        triprocess = multiprocessing.Process(target=self._run_triangle_python,args=(geometry,'pqa{:.4f}'.format(triangle_area),q))
        triprocess.start()
        while triprocess.is_alive() and q.empty():
            time.sleep(0.1)
        
        if not q.empty():
            mesh = q.get()
        else:
            raise Exception('Gridding using triangle module failed; most likely a precision issue caused by the boundary definition.')


        # The grid vertices and vertex connectivity to form cells come
        # straight out from triangle. 
        self.vertices = mesh['vertices']
        self.cells = mesh['triangles']

        # Make the list of line segments and which line segments are
        # associated with each cell.
        self.segments = []
        self.cell_sides = np.zeros(self.cells.shape,dtype=np.uint64)

        # For each grid cell
        for cell_index in range(self.cells.shape[0]):

            # For each side of the triangle
            for seg_index in range(3):

                seg = sorted( [self.cells[cell_index][seg_index],self.cells[cell_index][(seg_index + 1) % 3]] )

                # Store the index if this line segment is already in the segment list,
                # otherwise add it to the segment list and then store its index.
                try:
                    self.cell_sides[cell_index][seg_index] = self.segments.index(seg)
                except ValueError:
                    self.segments.append(seg)
                    self.cell_sides[cell_index][seg_index] = len(self.segments)-1


        self.segments = np.array(self.segments)


    def _run_triangle_python(self,geometry,options,queue):
        '''
        Run triangular mesh generation with the triangle library.
        See: https://www.cs.cmu.edu/~quake/triangle.html
    
        This function is designed to be used from a multiprocessing call.

        Parameters:

            geometry (dict)               : Dictionary defining the bounding geometry, see triangle \
                                            manual.

            options (str)                 : Options to give to triangle, see triangle manual.

            queue (multiprocessing.Queue) : Multiprocessing queue on which to place the result.

        '''
        queue.put( triangle.triangulate( geometry , options ) )




class GeometryMatrix():
    '''
    Class to represent a geometry matrix.

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
        n_cells = grid.get_n_cells()
        n_los = raydata.x.size

        # Flatten out the ray start and end coords
        ray_start_coords = raydata.ray_start_coords.reshape(-1,3,order=self.los_order)
        ray_end_coords = raydata.ray_end_coords.reshape(-1,3,order=self.los_order)
        
        # Create the (so far empty) geometry matrix!
        self.data = scipy.sparse.dok_matrix((n_los,n_cells))

        # Multi-threadedly loop over each sight-line in raydata and calculate its matrix row
        with multiprocessing.Pool(n_processes) as cpupool:
            for los_ind , mat_row in enumerate( cpupool.imap( self.grid.get_cell_los_lengths , np.hstack((ray_start_coords,ray_end_coords)) , 10 ) ):
                self.data[los_ind,:] = mat_row

        # If enabled, remove any grid cells + matrix rows which have no sight-line coverage.
        if cull_grid:
            unused_cells = np.where(np.abs(np.sum(self.data,axis=0)) == 0)[1]
            self.grid.remove_cells(unused_cells)

            used_cols = np.where(np.abs(np.sum(self.data,axis=0)) > 0)[1]
            self.data = self.data[:,used_cols]


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

            coverage = np.abs(np.sum(self.data,axis=0))
            self.grid.plot(coverage,cblabel='Total sight-line coverage [m]')

        elif metric == 'n_los':
            coverage = np.sum(self.data > 0,axis=0)
            self.grid.plot(coverage,cblabel='Number of sight-lines seeing grid cell')
