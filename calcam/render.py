'''
* Copyright 2015-2021 European Atomic Energy Community (EURATOM)
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

import vtk
import cv2
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import time
from .raycast import raycast_sightlines, RayData
import copy
from .misc import bin_image, get_contour_intersection, LoopProgPrinter, ColourCycle
from .calibration import Calibration
from matplotlib.cm import get_cmap

# This is the maximum image dimension which we expect VTK can succeed at rendering in a single RenderWindow.
# Hopefully 5120 is conservative enough to be safe on most systems but also not restrict render quality too much.
# If getting blank images when trying to render at high resolutions, try reducing this.
# TODO: Add a function to this module to determine the best value for this automatically
max_render_dimension = 5120


class CoordsActor(vtk.vtkAssembly):

    def __init__(self,coords,lines=True,markers=False,markersize=1e-2,marker_flat_shading=True,linewidth=2):
        super().__init__()
        self.coords = coords
        self.markersize = markersize
        self.linewidth = linewidth
        self.marker_flat_shading = marker_flat_shading
        self.line_actors = []
        self.marker_actors = []
        self.colour = (1,1,1)

        self.lines = False
        self.markers = False

        self.set_lines(lines)
        self.set_markers(markers)


    def set_colour(self,colour):
        for actor in self.line_actors + self.marker_actors:
            actor.GetProperty().SetColor(colour)
        self.colour = colour

    def set_lines(self,enable):

        if enable and not self.lines:
            # Create an actor for the lines
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            point_ind = -1
            if self.coords.shape[1] == 6:

                for lineseg in range(self.coords.shape[0]):
                    points.InsertNextPoint(self.coords[lineseg, :3])
                    points.InsertNextPoint(self.coords[lineseg, 3:])
                    point_ind = point_ind + 2
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, point_ind - 1)
                    line.GetPointIds().SetId(1, point_ind)
                    lines.InsertNextCell(line)

            elif self.coords.shape[1] == 3:

                for pointind in range(self.coords.shape[0]):

                    points.InsertNextPoint(self.coords[pointind, :])
                    point_ind = point_ind + 1

                    if point_ind > 0:
                        line = vtk.vtkLine()
                        line.GetPointIds().SetId(0, point_ind - 1)
                        line.GetPointIds().SetId(1, point_ind)
                        lines.InsertNextCell(line)

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetLines(lines)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            actor.GetProperty().SetLineWidth(self.linewidth)
            actor.GetProperty().SetColor(self.colour)

            self.line_actors.append(actor)
            self.AddPart(actor)

        elif not enable and self.lines:
            # Remove line actors
            for actor in self.line_actors:
                self.RemovePart(actor)
            self.line_actors = []

        self.lines = enable


    def set_markers(self,enable):

        if enable and not self.markers:
            if self.coords.shape[1] == 6:
                x = np.concatenate((self.coords[:, 0], self.coords[:, 3]))
                y = np.concatenate((self.coords[:, 1], self.coords[:, 4]))
                z = np.concatenate((self.coords[:, 2], self.coords[:, 5]))
            elif self.coords.shape[1] == 3:
                x = self.coords[:, 0]
                y = self.coords[:, 1]
                z = self.coords[:, 2]

            for x_, y_, z_ in zip(x, y, z):
                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(x_, y_, z_)
                sphere.SetRadius(self.markersize / 2)
                sphere.SetPhiResolution(12)
                sphere.SetThetaResolution(12)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(sphere.GetOutputPort())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(self.colour)
                if self.marker_flat_shading:
                    actor.GetProperty().LightingOff()
                self.marker_actors.append(actor)
                self.AddPart(actor)

        elif not enable and self.markers:

            for actor in self.marker_actors:
                self.RemovePart(actor)
            self.marker_actors = []

        self.markers = enable




def render_cam_view(cadmodel,calibration,extra_actors=[],filename=None,oversampling=1,aa=1,transparency=False,verbose=True,coords = 'display',interpolation='cubic'):
    '''
    Render an image of a given CAD model from the point of view of a given calibration.

    .. note::
        This function uses off-screen OpenGL rendering which fails above some image dimension which depends on the system.
        The workaround for this is that above a render dimension set by ``calcam.render.max_render_dimension``, the image is rendered
        at lower resolution and then scaled up using nearest-neighbour scaling. For this reason, when rendering very high
        resolution images, the rendered image quality may be lower than expected.

    Parameters:

        cadmodel (calcam.CADModel)          : CAD model of scene
        calibration (calcam.Calibration)    : Calibration whose point-of-view to render from.
        extra_actors (list of vtk.vtkActor) : List containing any additional vtkActors to add to the scene \
                                              in addition to the CAD model.
        filename (str)                      : Filename to which to save the resulting image. If not given, no file is saved.
        oversampling (float)                : Used to render the image at higher (if > 1) or lower (if < 1) resolution than the \
                                              calibrated camera. Must be an integer if > 1 or if <1, 1/oversampling must be a \
                                              factor of both image width and height.
        aa (int)                            : Anti-aliasing factor, 1 = no anti-aliasing.
        transparency (bool)                 : If true, empty areas of the image are set transparent. Otherwise they are black.
        verbose (bool)                      : Whether to print status updates while rendering.
        coords (str)                        : Either ``Display`` or ``Original``, the image orientation in which to return the image.
        interpolation(str)                  : Either ``nearest`` or ``cubic``, inerpolation used when applying lens distortion.

    Returns:
        
        np.ndarray                          : Array containing the rendered 8-bit per channel RGB (h x w x 3) or RGBA (h x w x 4) image.\
                                              Also saves the result to disk if the filename parameter is set.
    '''
    if np.any(calibration.view_models) is None:
        raise ValueError('This calibration object does not contain any fit results! Cannot render an image without a calibration fit.')

    if interpolation.lower() == 'nearest':
        interp_method = cv2.INTER_NEAREST
    elif interpolation.lower() == 'cubic':
        interp_method = cv2.INTER_CUBIC
    else:
        raise ValueError('Invalid interpolation method "{:s}": must be "nearest" or "cubic".'.format(interpolation))

    aa = int(max(aa,1))

    if oversampling > 1:
        if int(oversampling) - oversampling > 1e-5:
            raise ValueError('If using oversampling > 1, oversampling must be an integer!')

    elif oversampling < 1:
        shape = calibration.geometry.get_display_shape()
        undersample_x = oversampling * shape[0]
        undersample_y = oversampling * shape[1]

        if abs(int(undersample_x) - undersample_x) > 1e-5 or abs(int(undersample_y) - undersample_y) > 1e-5:
            raise ValueError('If using oversampling < 1, 1/oversampling must be a common factor of the display image width and height ({:d}x{:d})'.format(shape[0],shape[1]))

    if verbose:
        tstart = time.time()
        print('[Calcam Renderer] Preparing...')

    # This will be our result. To start with we always render in display coords.
    orig_display_shape = calibration.geometry.get_display_shape()
    output = np.zeros([int(orig_display_shape[1]*oversampling),int(orig_display_shape[0]*oversampling),3+transparency],dtype='uint8')

    # The un-distorted FOV is over-rendered to allow for distortion.
    # FOV_factor is how much to do this by; too small and image edges might be cut off.
    models = []
    for view_model in calibration.view_models:
        try:
            models.append(view_model.model)
        except AttributeError:
            pass
    if np.any( np.array(models) == 'fisheye'):
        fov_factor = 3.
    else:
        fov_factor = 1.5

    x_pixels = orig_display_shape[0]
    y_pixels = orig_display_shape[1]

    renwin = vtk.vtkRenderWindow()
    renwin.OffScreenRenderingOn()
    renwin.SetBorders(0)

    # Set up render window for initial, un-distorted window
    renderer = vtk.vtkRenderer()
    renwin.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()

    cad_linewidths = np.array(cadmodel.get_linewidth())
    cadmodel.set_linewidth(list(cad_linewidths*aa))
    cadmodel.add_to_renderer(renderer)

    for actor in extra_actors:
        if isinstance(actor,vtk.vtkAssembly):
            actors = actor.GetParts()
            while True:
                part = actors.GetNextProp3D()
                if part is not None:
                    part.GetProperty().SetLineWidth( actor.GetProperty().GetLineWidth() * aa)
                else:
                    break
        else:
            actor.GetProperty().SetLineWidth( actor.GetProperty().GetLineWidth() * aa)

        renderer.AddActor(actor)

    # We need a field mask the same size as the output
    fieldmask = cv2.resize(calibration.get_subview_mask(coords='Display'),(int(x_pixels*oversampling),int(y_pixels*oversampling)),interpolation=cv2.INTER_NEAREST)

    for field in range(calibration.n_subviews):

        render_shrink_factor = 1

        if calibration.view_models[field] is None:
            continue

        cx = calibration.view_models[field].cam_matrix[0,2]
        cy = calibration.view_models[field].cam_matrix[1,2]
        fy = calibration.view_models[field].cam_matrix[1,1]

        vtk_win_im = vtk.vtkWindowToImageFilter()
        vtk_win_im.SetInput(renwin)

        # Width and height - initial render will be put optical centre in the window centre
        width = int(2 * fov_factor * max(cx, x_pixels - cx))
        height = int(2 * fov_factor * max(cy, y_pixels - cy))

        # To avoid trying to render an image larger than the available OpenGL texture buffer,
        # check if the image will be too big and if it will, first try reducing the AA, and if
        # that isn't enough we will render a smaller image and then resize it afterwards.
        # Not ideal but avoids ending up with black images. What would be nicer, is if
        # VTK could fix their large image rendering code to preserve the damn field of view properly!
        longest_side = max(width,height)*aa*oversampling

        while longest_side > max_render_dimension and aa > 1:
            aa = aa - 1
            longest_side = max(width, height) * aa * oversampling

        while longest_side > max_render_dimension:
            render_shrink_factor = render_shrink_factor + 1
            longest_side = max(width,height)*aa*oversampling/render_shrink_factor

        renwin.SetSize(int(width*aa*oversampling/render_shrink_factor),int(height*aa*oversampling/render_shrink_factor))

        # Set up CAD camera
        fov_y = 360 * np.arctan( height / (2*fy) ) / 3.14159
        cam_pos = calibration.get_pupilpos(subview=field)
        cam_tar = calibration.get_los_direction(cx,cy,subview=field) + cam_pos
        upvec = -1.*calibration.get_cam_to_lab_rotation(subview=field)[:,1]
        camera.SetPosition(cam_pos)
        camera.SetViewAngle(fov_y)
        camera.SetFocalPoint(cam_tar)
        camera.SetViewUp(upvec)

        if verbose:
            print('[Calcam Renderer] Rendering (Sub-view {:d}/{:d})...'.format(field + 1,calibration.n_subviews))

        # Do the render and grab an image
        renwin.Render()

        # Make sure the light lights up the whole model without annoying shadows or falloff.
        light = renderer.GetLights().GetItemAsObject(0)
        light.PositionalOn()
        light.SetConeAngle(180)

        # Do the render and grab an image
        renwin.Render()

        vtk_win_im.Update()

        vtk_image = vtk_win_im.GetOutput()
        vtk_array = vtk_image.GetPointData().GetScalars()
        dims = vtk_image.GetDimensions()

        im = np.flipud(vtk_to_numpy(vtk_array).reshape(dims[1], dims[0] , 3))

        # If we have had to do a smaller render for graphics driver reasons, scale the render up to the resolution
        # we really wanted.
        if render_shrink_factor > 1:
            im = cv2.resize(im,(width*aa*oversampling,height*aa*oversampling),interpolation=cv2.INTER_NEAREST)

        if transparency:
            alpha = 255 * np.ones([np.shape(im)[0],np.shape(im)[1]],dtype='uint8')
            alpha[np.sum(im,axis=2) == 0] = 0
            im = np.dstack((im,alpha))

        if verbose:
            print('[Calcam Renderer] Applying lens distortion (Sub-view {:d}/{:d})...'.format(field + 1,calibration.n_subviews))

        # Pixel locations we want on the final image
        [xn,yn] = np.meshgrid(np.linspace(0,x_pixels-1,int(x_pixels*oversampling*aa)),np.linspace(0,y_pixels-1,int(y_pixels*oversampling*aa)))

        xn,yn = calibration.normalise(xn,yn,field)

        # Transform back to pixel coords where we want to sample the un-distorted render.
        # Both x and y are divided by Fy because the initial render always has Fx = Fy.
        xmap = (xn * fy * oversampling * aa) + (width * oversampling * aa - 1)/2
        ymap = (yn * fy * oversampling * aa) + (height * oversampling * aa - 1)/2
        xmap = xmap.astype('float32')
        ymap = ymap.astype('float32')

        # Actually apply distortion
        im  = cv2.remap(im,xmap,ymap,interp_method)

        # Anti-aliasing by binning
        if aa > 1:
            im = bin_image(im,aa,np.mean)

        output[fieldmask == field,:] = im[fieldmask == field,:]


    if coords.lower() == 'original':
        output = calibration.geometry.display_to_original_image(output,interpolation=interpolation)
    
    if verbose:
        print('[Calcam Renderer] Completed in {:.1f} s.'.format(time.time() - tstart))

    # Save the image if given a filename
    if filename is not None:

        # If we have transparency, we can only save as PNG.
        if transparency and filename[-3:].lower() != 'png':
            print('[Calcam Renderer] Images with transparency can only be saved as PNG! Overriding output file type to PNG.')
            filename = filename[:-3] + 'png'

        # Re-shuffle the colour channels for saving (openCV needs BGR / BGRA)
        save_im = copy.copy(output)
        save_im[:,:,:3] = save_im[:,:,2::-1]
        cv2.imwrite(filename,save_im)
        if verbose:
            print('[Calcam Renderer] Result saved as {:s}'.format(filename))


    # Tidy up after ourselves!
    cadmodel.set_linewidth(list(cad_linewidths))
    cadmodel.remove_from_renderer(renderer)

    for actor in extra_actors:
        if isinstance(actor,vtk.vtkAssembly):
            actors = actor.GetParts()
            while True:
                part = actors.GetNextProp3D()
                if part is not None:
                    part.GetProperty().SetLineWidth( actor.GetProperty().GetLineWidth() / aa )
                else:
                    break
        else:
            actor.GetProperty().SetLineWidth( actor.GetProperty().GetLineWidth() / aa )
        renderer.RemoveActor(actor)

    renwin.Finalize()

    return output



def render_hires(renderer,oversampling=1,aa=1,transparency=False,legendactor=None):
    """
    Render the contents of an existing vtkRenderer to an image array, if requested at higher resolution
    than the existing renderer's current size. N.B. if calling with oversampling > 1 or aa > 1 and using
    VTK versions > 8.2, the returned images will have a slightly smaller field of view than requested. This
    is a known problem caused by the behaviour of VTK; I don't currently have a way to do anything about it.

    Parameters:

        renderer (vtk.vtkRenderer)  : Renderer from which to create the image
        oversampling (int)          : Factor by which to enlarge the resolution
        aa (int)                    : Factor by which to anti-alias by rendering at higher resolution then re-sizing down
        transparency (bool)         : Whether to make black image areas transparent

    """

    # Thicken up all the lines according to AA setting to make sure
    # they don't end upp invisibly thin.
    actorcollection = renderer.GetActors()
    actorcollection.InitTraversal()
    actor = actorcollection.GetNextItemAsObject()

    if legendactor is not None:
        renderer.RemoveActor(legendactor)

    while actor is not None:
        actor.GetProperty().SetLineWidth( actor.GetProperty().GetLineWidth() * aa )
        actor = actorcollection.GetNextItemAsObject()

    hires_renderer = vtk.vtkRenderLargeImage()
    hires_renderer.SetInput(renderer)
    hires_renderer.SetMagnification( oversampling * aa )

    renderer.Render()
    hires_renderer.Update()
    vtk_im = hires_renderer.GetOutput()
    dims = vtk_im.GetDimensions()

    vtk_im_array = vtk_im.GetPointData().GetScalars()
    im = np.flipud(vtk_to_numpy(vtk_im_array).reshape(dims[1], dims[0] , 3))

    # Put all the line widths back to normal
    actorcollection.InitTraversal()
    actor = actorcollection.GetNextItemAsObject()
    while actor is not None:
        actor.GetProperty().SetLineWidth( actor.GetProperty().GetLineWidth() / aa )
        actor = actorcollection.GetNextItemAsObject()

    # If we have a legend, we have to render that separately.
    if legendactor is not None:

        downscale = int(np.ceil(im.shape[1]/max_render_dimension))
        renwin = vtk.vtkRenderWindow()
        renwin.OffScreenRenderingOn()
        renwin.SetBorders(0)
        renwin.SetSize(im.shape[1]//downscale,im.shape[0]//downscale)
        legendrenderer = vtk.vtkRenderer()
        renwin.AddRenderer(legendrenderer)
        legendrenderer.AddActor(legendactor)

        renwin.Render()
        vtk_win_im = vtk.vtkWindowToImageFilter()
        vtk_win_im.SetInput(renwin)
        vtk_win_im.Update()
        vtk_image = vtk_win_im.GetOutput()
        vtk_array = vtk_image.GetPointData().GetScalars()
        dims = vtk_image.GetDimensions()
        del vtk_win_im
        legendrenderer.RemoveActor(legendactor)
        renwin.Finalize()
        legendim = np.flipud(vtk_to_numpy(vtk_array).reshape(dims[1], dims[0], 3))
        legendim = cv2.resize(legendim,(im.shape[1],im.shape[0]))
        legendmask = np.tile( (legendim.sum(axis=2) > 0)[:,:,np.newaxis],[1,1,3])
        im[legendmask] = legendim[legendmask]

        renderer.AddActor(legendactor)

    if transparency:
        alpha = 255 * np.ones([np.shape(im)[0],np.shape(im)[1]],dtype='uint8')
        alpha[np.sum(im,axis=2) == 0] = 0
        im = np.dstack((im,alpha))

    # Anti-aliasing by binning
    im = bin_image(im,aa,np.mean)

    return im


def get_fov_actor(cadmodel,calib,actor_type='volume',resolution=None,subview=None):

    if actor_type.lower() == 'volume':

        if resolution is None:
            resolution = 64

    elif actor_type.lower() == 'lines':

        if resolution is None:
            resolution = 32

    else:
        raise ValueError('"actor_type" argument must be "volume" or "lines"')

    raydata = raycast_sightlines(calib,cadmodel,binning=max(calib.geometry.get_display_shape())/resolution,verbose=False)

    # Before we do anything, we need to arrange our triangle corners
    points = vtk.vtkPoints()

    if actor_type.lower() == 'volume': 

        x_horiz,y_horiz = np.meshgrid( np.arange(raydata.ray_start_coords.shape[1]-1), np.arange(raydata.ray_start_coords.shape[0]))
        x_horiz = x_horiz.flatten()
        y_horiz = y_horiz.flatten()

        x_vert,y_vert = np.meshgrid( np.arange(raydata.ray_start_coords.shape[1]), np.arange(raydata.ray_start_coords.shape[0]-1))
        x_vert = x_vert.flatten()
        y_vert = y_vert.flatten()

        for n in range(len(x_horiz)):
            if np.abs(raydata.ray_start_coords[y_horiz[n],x_horiz[n],:] - raydata.ray_start_coords[y_horiz[n],x_horiz[n]+1,:]).max() < 1e-3:
                if subview is not None:
                    if calib.subview_lookup(raydata.x[y_horiz[n],x_horiz[n]],raydata.y[y_horiz[n],x_horiz[n]]) != subview:
                        continue
                points.InsertNextPoint(raydata.ray_start_coords[y_horiz[n],x_horiz[n],:])
                points.InsertNextPoint(raydata.ray_end_coords[y_horiz[n],x_horiz[n],:])
                points.InsertNextPoint(raydata.ray_end_coords[y_horiz[n],x_horiz[n]+1,:])

        for n in range(len(x_vert)):
            if np.abs( raydata.ray_start_coords[y_vert[n],x_vert[n],:] - raydata.ray_start_coords[y_vert[n]+1,x_vert[n],:] ).max() < 1e-3:
                if subview is not None:
                    if calib.subview_lookup(raydata.x[y_vert[n],x_vert[n]],raydata.y[y_vert[n],x_vert[n]]) != subview:
                        continue
                points.InsertNextPoint(raydata.ray_start_coords[y_vert[n],x_vert[n],:])
                points.InsertNextPoint(raydata.ray_end_coords[y_vert[n],x_vert[n],:])
                points.InsertNextPoint(raydata.ray_end_coords[y_vert[n]+1,x_vert[n],:])


        # Go through and make polygons!
        polygons = vtk.vtkCellArray()
        for n in range(0,points.GetNumberOfPoints(),3):
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(3)
            for i in range(3):
                polygon.GetPointIds().SetId(i,i+n)
            polygons.InsertNextCell(polygon)


        # Make Polydata!
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polygons)


    elif actor_type.lower() == 'lines':

        lines = vtk.vtkCellArray()
        point_ind = -1
        for i in range(raydata.ray_end_coords.shape[0]):
            for j in range(raydata.ray_end_coords.shape[1]):
                if subview is not None:
                    if calib.subview_lookup(raydata.x[i,j],raydata.y[i,j]) != subview:
                        continue

                points.InsertNextPoint(raydata.ray_end_coords[i,j,:])
                points.InsertNextPoint(raydata.ray_start_coords[i,j,:])
                point_ind = point_ind + 2

                line = vtk.vtkLine()
                line.GetPointIds().SetId(0,point_ind - 1)
                line.GetPointIds().SetId(1,point_ind)
                lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()

    if actor_type == 'lines':
        actor.GetProperty().SetLineWidth(2)

    return actor


def get_wall_coverage_actor(cal,cadmodel=None,image=None,imagecoords='Original',clim=None,lower_transparent=True,cmap='jet',clearance=5e-3,resolution=None,subview=None,verbose=False):
    """
    Get a VTK actor representing the wall area coverage of a given camera.
    Optionally, pass an image to colour the actor according to the image (e.g. to map data to the wall for visualisation)

    Parameters:
        cal                        : Calcam calibration to use - can be an instance of calcam.Calibration or calcam.RayData
        cadmodel (calcam.CADModel) : Calcam CAD model to use for wall. Must be given if 'cal' is calcam.Calibration
        image (np.ndarray)         : Data image to use for colouring. Can be EITHER a h*w size array of any type \
                                     which will be colour mapped, or an 8-bit RGB image for colour data.
        imagecoords (str)          : If passing an image, whether that image is in display or original orientation
        clim (2 element sequence)  : If passing data toe colour mapped, the colour limits. Default is full range of data.
        lower_transparent (bool)   : For colour mapped images, whether to make pixels below the lower colour limit \
                                     transparent (don't show them) or give them the lowest colour in the colour map.
        cmap                       : For colour mapped data, the name of the matplotlib colour map to use.
        clearance (float)          : A distance in metres, the returned actor will be slightly in front of the wall surface by \
                                     this much. Used to ensure this actor does not get buried inside the CAD geometry.
        resolution (int)           : Maximum side length in pixels for the ray casting. Smaller values result in faster calculation \
                                     but uglier result. Default will calculate the full resolution of the camera. Ignored if an image is provided.
        subview (int)              : If given, the actor will only include the given subview index.
        verbose (bool)             : Whether to print information about progress.

    Returns:

        if image is None, returns a vtkActor containing a wall-hugging surface showing the camera coverage.
        if image is not None, returns a calcam.render.MappedImageActor containing a wall-hugging mapped image.

        if return_raydata is True, returns the calcam.RayData object containing the geometry info.

    """

    # If an image is supplied, check & organise it
    if image is not None:

        fr = np.array(image).copy().astype(np.float32)

        if len(image.shape) < 3:
            # Single channel image: will be colour mapped so normalise to CLIM and load colour map
            if clim is None:
                clim = [np.nanmin(image), np.nanmax(image)]

            if lower_transparent:
                fr[fr < clim[0]] = np.nan
            else:
                fr[fr < clim[0]] = clim[0]

            fr[fr > clim[1]] = clim[1]

            fr = (fr - clim[0]) / (clim[1] - clim[0])

            cmap = get_cmap(cmap)

        else:
            # > single channel, assume 8-bit RGB: ensure correct datatype and throw away any extra channels
            fr = fr[:,:,:3].astype(np.uint8)

        if isinstance(cal,Calibration):
            expected_shape = cal.geometry.get_image_shape(imagecoords)
        else:
            expected_shape = np.array(cal.x.shape[::-1]) - 1

        factor = np.array(image.shape[1::-1]) / expected_shape

        if factor[0] != factor[1]:
            raise ValueError('Image size ({:d}x{:d} px) does not match expected ({:s} image shape for this calibration ({:d}x{:d} px)'.format(image.shape[1],image.shape[0],imagecoords,expected_shape[0],expected_shape[1]))
        binning = factor[0]

        if binning != 1 and isinstance(cal,RayData):
            image = cv2.resize(image,tuple(expected_shape))

    else:
        if resolution is None:
            binning = 1
        elif isinstance(cal, Calibration):
            binning = max(cal.geometry.get_display_shape()) / resolution

    if isinstance(cal,Calibration):
        # If we're given a calcam calibration, ray cast to get the wall coords
        if cadmodel is None:
            raise Exception('If passing a Calcam calibration object, a CAD model must also be given!')

        w, h = cal.geometry.get_image_shape(imagecoords)
        x = np.linspace(- 0.5, w - 0.5, int(w/binning) + 1)
        y = np.linspace(- 0.5, h - 0.5, int(h/binning) + 1)
        x,y = np.meshgrid(x,y)
        rd = raycast_sightlines(cal,cadmodel,x=x,y=y,coords=imagecoords,calc_normals=True,verbose=verbose)
        
    elif isinstance(cal,RayData):
        # If already given a raydata object, just use that!
        rd = cal

    ray_start = rd.get_ray_start()
    ray_dir = rd.get_ray_directions()
    ray_len = rd.get_ray_lengths()
    normals = rd.get_model_normals()


    # Ray end coordinates at the wall: put them 2mm in front of the wall to make sure the resulting actor
    # is not part buried in the CAD model
    ray_end = ray_start + np.tile(ray_len[:,:,np.newaxis],(1,1,3)) * ray_dir + normals * clearance

    # VTK objects with coordinates of each pixel corner.
    # Which index in the VTK array corresponds to which pixel is tracked in pointinds
    verts = vtk.vtkPoints()
    pointinds = np.zeros((ray_end.shape[0],ray_end.shape[1]),dtype=int)
    ci = 0
    for i in range(ray_end.shape[0]):
        for j in range(ray_end.shape[1]):
            verts.InsertNextPoint(*ray_end[i,j,:])
            pointinds[i,j] = ci
            ci += 1

    ray_end = ray_end - normals * clearance

    # Create VTK polygon array
    polys  = vtk.vtkCellArray()

    # If we're mapping an image, create the array of colours for each polygon
    if image is not None:
        colours = vtk.vtkUnsignedCharArray()
        colours.SetNumberOfComponents(3)
        im_inds = []

    # If we have an image, map of where the NaNs are
    if image is not None:
        if len(image.shape) < 3:
            isnan = np.isnan(image)
        else:
            isnan = np.isnan(image.sum(axis=2))

    if verbose:
        lp = LoopProgPrinter()
        lp.update('Constructing 3D mesh...')
        i = 0
        tot_px = (ray_end.shape[1]-1) * (ray_end.shape[0] - 1)

    # Go through the image
    for xi in range(ray_end.shape[1]-1):
        for yi in range(ray_end.shape[0] - 1):
            i += 1
            # Don't map any pixels which are NaN
            if image is not None:
                if isnan[yi,xi]:
                    continue

            if subview is not None:
                if np.any(cal.subview_lookup(rd.x[yi:yi+2,xi:xi+2],rd.y[yi:yi+2,xi:xi+2]) != subview):
                    continue
            elif np.unique( cal.subview_lookup(rd.x[yi:yi+2,xi:xi+2],rd.y[yi:yi+2,xi:xi+2])).size > 1:
                # Also don't do any cells which are split across subviews.
                continue

            # Coordinates and indices of corners for this pixel.
            # Any coordinates == NaN indicates sight lines which did not hit the model so we skip those polys.
            polycoords = ray_end[yi:yi+2,xi:xi+2,:]
            if np.any(np.isnan(polycoords)):
                continue

            losdirs = ray_dir[yi:yi+2,xi:xi+2,:]
            pixel_dir = np.array( [losdirs[:,:,0].mean(),losdirs[:,:,1].mean(),losdirs[:,:,2].mean()])
            pixel_dir = pixel_dir / np.sqrt(np.sum(pixel_dir**2))

            inds = pointinds[yi:yi+2,xi:xi+2]


            # Check if a pixel is "torn" in real space by checking if any of its sides
            # are very close to parallel with the camera sight lines. If so, skip it.
            sides = np.zeros((6,3))
            sides[0,:] = polycoords[0,1,:] - polycoords[0,0,:]
            sides[1,:] = polycoords[1,1,:] - polycoords[0,1,:]
            sides[2,:] = polycoords[1,0,:] - polycoords[1,1,:]
            sides[3,:] = polycoords[0,0,:] - polycoords[1,0,:]
            sides[4,:] = polycoords[0, 0,:] - polycoords[1, 1,:]
            sides[5,:] = polycoords[0, 1,:] - polycoords[1, 0,:]

            side_lengths = np.sqrt(np.sum(sides**2,axis=1))
            sides = sides / np.tile(side_lengths[:,np.newaxis],(1,3))

            dot_prods = [ np.dot(pixel_dir,sides[i,:]) for i in range(4) ]

            if max(dot_prods) > 0.999 or side_lengths.max() > 8*side_lengths.min():
                continue

            # Create a quad representing the pixel
            poly = vtk.vtkPolygon()
            poly.GetPointIds().SetNumberOfIds(4)
            poly.GetPointIds().SetId(0,inds[0,0])
            poly.GetPointIds().SetId(1, inds[1, 0])
            poly.GetPointIds().SetId(2, inds[1, 1])
            poly.GetPointIds().SetId(3, inds[0, 1])
            polys.InsertNextCell(poly)

            # If we're mapping an image, colour the quad according to the image data
            if image is not None:
                im_inds.append((yi,xi))
                if len(image.shape) < 3:
                    rgb = (np.array(cmap(fr[yi,xi]))[:-1] * 255).astype(np.uint8)
                    colours.InsertNextTypedTuple(rgb)
                else:
                    colours.InsertNextTypedTuple(fr[yi,xi,:])

            if verbose:
                lp.update(i/tot_px)

    if verbose:
        lp.update(1.)
        del lp

    # Put it all togetehr in a vtkPolyDataActor
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(verts)
    polydata.SetPolys(polys)

    if image is not None:
        polydata.GetCellData().SetScalars(colours)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    if image is not None:
        mapper.SetColorModeToDirectScalars()
        actor = MappedImageActor()
        actor.image_inds = im_inds
        actor.nanmap = isnan
        actor.celldata = polydata.GetCellData()
    else:
        actor = vtk.vtkActor()

    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()

    return actor


class MappedImageActor(vtk.vtkActor):

    def update_image(self,new_image,clim=None,lower_transparent=True,cmap='jet'):

        if new_image.shape[:2] != self.nanmap.shape:
            raise ValueError('New image is not the same shape as the original one!')

        if len(new_image.shape) < 3:
            isnan = np.isnan(new_image)
        else:
            isnan = np.isnan(new_image.sum(axis=2))

        if np.not_equal(isnan,self.nanmap).sum() > 0:
            raise ValueError('New image conatins NaNs at different positions to the original! You must build a new actor with get_wall_coverage_actor() instead.')

        fr = np.array(new_image).copy().astype(np.float32)

        if len(fr.shape) < 3:
            # Single channel image: will be colour mapped so normalise to CLIM and load colour map
            if clim is None:
                clim = [np.nanmin(fr), np.nanmax(fr)]

            if lower_transparent:
                fr[fr < clim[0]] = np.nan
            else:
                fr[fr < clim[0]] = clim[0]

            fr[fr > clim[1]] = clim[1]

            fr = (fr - clim[0]) / (clim[1] - clim[0])

            cmap = get_cmap(cmap)

        else:
            # > single channel, assume 8-bit RGB: ensure correct datatype and throw away any extra channels
            fr = fr[:,:,:3].astype(np.uint8)

        colours = vtk.vtkUnsignedCharArray()
        colours.SetNumberOfComponents(3)

        for yi,xi in self.image_inds:
            if len(fr.shape) < 3:
                rgb = (np.array(cmap(fr[yi, xi]))[:-1] * 255).astype(np.uint8)
                colours.InsertNextTypedTuple(rgb)
            else:
                colours.InsertNextTypedTuple(fr[yi, xi, :])

        self.celldata.SetScalars(colours)


def get_wall_contour_actor(wall_contour,actor_type='contour',phi=None,toroidal_res=128):

    if actor_type == 'contour' and phi is None:
        raise ValueError('Toroidal angle must be specified if type==contour!')

    points = vtk.vtkPoints()

    if actor_type == 'contour':

        lines = vtk.vtkCellArray()
        x = wall_contour[-1,0]*np.cos(phi)
        y = wall_contour[-1,0]*np.sin(phi)
        points.InsertNextPoint(x,y,wall_contour[-1,1])

        for i in range(wall_contour.shape[0]-1):
            x = wall_contour[i,0]*np.cos(phi)
            y = wall_contour[i,0]*np.sin(phi)
            points.InsertNextPoint(x,y,wall_contour[i,1])

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0,i)
            line.GetPointIds().SetId(1,i+1)
            lines.InsertNextCell(line)

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0,i+1)
        line.GetPointIds().SetId(1,0)
        lines.InsertNextCell(line)

        polydata = polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)

    elif actor_type == 'surface':

        polygons = vtk.vtkCellArray()

        npoints = 0
        for i in range(wall_contour.shape[0]):
            points.InsertNextPoint(wall_contour[i,0],0,wall_contour[i,1])
            npoints = npoints + 1

        tor_step = 3.14159*(360./toroidal_res)/180
        for phi in np.linspace(tor_step,2*3.14159-tor_step,toroidal_res-1):

            x = wall_contour[0,0]*np.cos(phi)
            y = wall_contour[0,0]*np.sin(phi)
            points.InsertNextPoint(x,y,wall_contour[0,1])
            npoints = npoints + 1

            for i in range(1,wall_contour.shape[0]):

                x = wall_contour[i,0]*np.cos(phi)
                y = wall_contour[i,0]*np.sin(phi)
                points.InsertNextPoint(x,y,wall_contour[i,1])
                npoints = npoints + 1
                lasttor = npoints - wall_contour.shape[0] - 1

                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(3)
                polygon.GetPointIds().SetId(0,lasttor-1)
                polygon.GetPointIds().SetId(1,lasttor)
                polygon.GetPointIds().SetId(2,npoints-1)
                polygons.InsertNextCell(polygon)
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(3)
                polygon.GetPointIds().SetId(0,npoints-2)
                polygon.GetPointIds().SetId(1,npoints-1)
                polygon.GetPointIds().SetId(2,lasttor-1)
                polygons.InsertNextCell(polygon)

            # Close the end (poloidally)
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(3)
            polygon.GetPointIds().SetId(0,npoints-2*wall_contour.shape[0])
            polygon.GetPointIds().SetId(1,npoints-wall_contour.shape[0]-1)
            polygon.GetPointIds().SetId(2,npoints-1)
            polygons.InsertNextCell(polygon)
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(3)
            polygon.GetPointIds().SetId(0,npoints-wall_contour.shape[0])
            polygon.GetPointIds().SetId(1,npoints-1)
            polygon.GetPointIds().SetId(2,npoints-2*wall_contour.shape[0])
            polygons.InsertNextCell(polygon)
        
        # Close the end (toroidally)
        startpoint = npoints - wall_contour.shape[0]
        for i in range(1,wall_contour.shape[0]):
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(3)
            polygon.GetPointIds().SetId(0,startpoint+i-1)
            polygon.GetPointIds().SetId(1,startpoint+i)
            polygon.GetPointIds().SetId(2,i)
            polygons.InsertNextCell(polygon)
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(3)
            polygon.GetPointIds().SetId(0,i-1)
            polygon.GetPointIds().SetId(1,i)
            polygon.GetPointIds().SetId(2,startpoint+i-1)
            polygons.InsertNextCell(polygon)
        

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polygons)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor



# Return VTK image actor and VTK image reiszer objects for this image.
def get_image_actor(image_array,clim=None,actortype='vtkImageActor',scaling=1):

    # VTK image arrays work up-down flipped compared to usual.
    image = np.flipud(image_array.copy())

    if len(image.shape) > 2 and image.dtype == np.uint8:
        scale_image = False
    else:
        scale_image = True


    if scale_image:

        if clim is None:
            if image.min() != image.max():
                clim = [image.min(),image.max()]
            else:
                clim = [0,255]

        clim = np.array(clim)

        # If the array isn't already 8-bit int, make it 8-bit int...
        if image.dtype != np.uint8:
            # If we're given a higher bit-depth integer, it's easy to downcast it.
            if image.dtype == np.uint16 or image.dtype == np.int16:
                image = np.uint8(image/2**8)
                clim = np.uint8(clim/2**8)
            elif image.dtype == np.uint32 or image.dtype == np.int32:
                image = np.uint8(image/2**24)
                clim = np.uint8(clim/2**24)
            elif image.dtype == np.uint64 or image.dtype == np.int64:
                image = np.uint8(image/2**56)
                clim = np.uint8(clim/2**24)
            # Otherwise, scale it in a floating point way to its own max & min
            # and strip out any transparency info (since we can't be sure of the scale used for transparency)
            else:

                if image.min() < 0:
                    image = image - image.min()
                    clim = clim - image.min()

                if len(image.shape) == 3:
                    if image.shape[2] == 4:
                        image = image[:,:,:-1]

                image = np.uint8(255.*(image - image.min())/(image.max() - image.min()))


    vtk_im_importer = vtk.vtkImageImport()

    if scale_image:
        # Create a temporary floating point copy of the data, for scaling.
        # Also flip the data (vtk addresses y from bottom right) and put in display coords.
        image = np.float32(image)
        clim = np.float32(clim)

        # Scale to colour limits and convert to uint8 for displaying.
        image -= clim[0]
        image /= (clim[1]-clim[0])
        image *= 255.
        image = np.uint8(image)

    im_data_string = image.tostring()
    vtk_im_importer.CopyImportVoidPointer(im_data_string,len(im_data_string))
    vtk_im_importer.SetDataScalarTypeToUnsignedChar()

    if len(image.shape) == 2:
        vtk_im_importer.SetNumberOfScalarComponents(1)
    else:
        vtk_im_importer.SetNumberOfScalarComponents(image.shape[2])

    vtk_im_importer.SetDataExtent(0,image.shape[1]-1,0,image.shape[0]-1,0,0)
    vtk_im_importer.SetDataSpacing(scaling,scaling,scaling)
    vtk_im_importer.SetWholeExtent(0,image.shape[1]-1,0,image.shape[0]-1,0,0)

    if actortype == 'vtkImageActor':
        actor = vtk.vtkImageActor()
        actor.InterpolateOff()
        mapper = actor.GetMapper()
        mapper.SetInputConnection(vtk_im_importer.GetOutputPort())
        actor.image = np.flipud(image)

        return actor

    elif actortype == 'vtkActor2D':
        resizer = vtk.vtkImageResize()
        resizer.SetInputConnection(vtk_im_importer.GetOutputPort())
        resizer.SetResizeMethodToOutputDimensions()
        resizer.SetOutputDimensions((image_array.shape[1],image_array.shape[0],1))
        resizer.InterpolateOff()

        mapper = vtk.vtkImageMapper()
        mapper.SetInputConnection(resizer.GetOutputPort())
        mapper.SetColorWindow(255)
        mapper.SetColorLevel(127.5)

        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetDisplayLocationToForeground()

        actor.image = np.flipud(image)

        return actor,resizer





def render_unfolded_wall(cadmodel,calibrations=[],labels = [],colours=None,cal_opacity=0.7,w=None,theta_start=90,phi_start=0,progress_callback=LoopProgPrinter().update,cancel=lambda : False,theta_steps=18,phi_steps=360,r_equiscale=None,extra_actors=[],filename=None):
    """
    Render an image of the tokamak wall "flattened" out. Creates an image where the horizontal direction is the toroidal direction and
    vertical direction is poloidal.

    Parameters:

        cadmodel (calcam.CADModel)                  : CAD Model to render. The CAD model must have an R, Z wall contour embedded in it (this is \
                                                      ued in the wall flattening calculations), which can be added in the CAD model editor.
        calibrations (list of calcam.Calibration)   : List of camera calibrations to visualise on the wall. If provided, each camera calibration \
                                                      will be shown on the image as a colour shaded area indicating which parts of the wall \
                                                      the camera can see.
        labels (list of strings)                    : List of strings containing legend text for the calibrations. If not provided, no legend will be added \
                                                      to the image. If provided, must be the same length as the list of calibrations.
        colours (list of tuple)                     : List of 3-element tuples specifying the colours to use for the displayed calibrations. Each element of the \
                                                      list must have the format (R, G, B) where 0 <= R, G and B <= 1.
        cal_opcity (float)                          : How opaque to make the wall shading when showing calibrations. 0 = completely invisible, 1 = complete opaque. \
                                                      Default is 0.7.
        w (int)                                     : Desired approximate width of the rendered image in pixels. If not given, the image width will be chosen \
                                                      to give a scale of about 2mm/pixel.
        theta_start (float)                         : Poloidal angle in degrees to "split" the image i.e. this angle will be at the top and bottom of the image. \
                                                      0 corresponds to the outboard midplane. Default is 90 degrees i.e. the top of the machine.
        phi_start (float)                           : Toroidal angle in degrees to "split" the image i.e. this angle will be at the left and right of the image.
        progress_callback (callable)                : Used for GUI integration - a callable which will be called with the fraction of the render completed. \
                                                      Default is to print an estimate of how long the render will take.
        cancel (ref to bool)                        : Used for GUI integration - a booleam which starts False, and if set to True during the calculation, the function \
                                                      will stop and return.
        theta_steps (int)                           : Number of tiles to use in the poloidal direction. The default is optimised for image quality so it is advised not \
                                                      to change this. Effects the calculation time linearly.
        phi_steps (int)                             : Number of tiles to use in the toroidal direction. The default is optimised for image quality so it is advised not \
                                                      to change this. Effects the calculation time linearly.
        r_equiscale (float)                         : Due to the unwrapping of the torus to a rectangle, objects will appear stretched or compressed depending on their major \
                                                      radius. This parameter sets at what major radius objects appear at their correct shape. If not specified, the \
                                                      centre of the wall contour is used so objects on the inboard side appear "fatter" than in real life and objects on the \
                                                      outboard side will be "skinnier".
        filename (string)                           : If provided, the result will be saved to an image file with this name in addition to being returned as an array. \
                                                      Must include file extension.
    Returns:

        A NumPy array of size ( h * w * 3 ) and dtype uint8 containing the RGB image result.
    """
    if cadmodel.wall_contour is None:
        raise Exception('[render_unfolded_wall] This CAD model does not have a wall contour included. This function can only be used with CAD models which have wall contours.')

    cal_actors = []
    legend_items = []
    if len(calibrations) > 0:
        if len(labels) > 0:
            if len(labels) != len(calibrations):
                raise ValueError('[render_unfolded_wall] Length of labels list different from number of calibrations given!')
        if colours is not None:
            if len(colours) != len(calibrations):
                raise ValueError('[render_unfolded_wall] Length of colours list different from number of calibrations given!')
        else:
            ccycle = ColourCycle()

        for i,calib in enumerate(calibrations):
            try:
                progress_callback('Calculating high-res wall coverage for calibration {:s}'.format(labels[i] if len(labels) > 0 else '...{:s}'.format(calib.filename[-16:])))
            except Exception:
                print('Calculating high-res wall coverage for calibration {:s}'.format(labels[i] if len(labels) > 0 else '...{:s}'.format(calib.filename[-16:])))
            actor = get_wall_coverage_actor(calib,cadmodel,verbose=True)
            actor.GetProperty().SetOpacity(cal_opacity)
            if colours is not None:
                col = colours[i]
            else:
                col = next(ccycle)
            actor.GetProperty().SetColor(col)
            cal_actors.append(actor)
            if len(labels) > 0:
                legend_items.append((labels[i], col))


    # The camera will be placed in the centre of the R, Z wall contour
    r_min = cadmodel.wall_contour[:,0].min()
    r_max = cadmodel.wall_contour[:,0].max()
    z_min = cadmodel.wall_contour[:,1].min()
    z_max = cadmodel.wall_contour[:,1].max()
    r_mid = (r_max + r_min) / 2
    z_mid = (z_max + z_min) / 2

    # Calculate image width for ~2mm / pixel if none is given
    if w is None:
        tor_len = 2*np.pi*r_mid
        w = int(tor_len/2e-3)

    if r_equiscale is None:
        r_equiscale = r_mid

    # Maximum line length which will be used for wall contour intersection tests.
    linelength = np.sqrt((r_max - r_min)**2 + (z_max - z_min)**2)

    # Theta and phi steps in radians
    theta_step = 2*np.pi/theta_steps
    phi_step = 2*np.pi/phi_steps

    # Convert starting angles to radians
    theta_start = theta_start / 180 * np.pi
    phi_start = phi_start / 180 * np.pi

    # Set up a VTK off screen window to do the image rendering
    renwin = vtk.vtkRenderWindow()
    renwin.OffScreenRenderingOn()
    renwin.SetBorders(0)
    renderer = vtk.vtkRenderer()
    renwin.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    camera.SetViewAngle(theta_step/np.pi * 180)
    renderer.Render()
    light = renderer.GetLights().GetItemAsObject(0)
    light.SetPositional(True)

    # Width of each tile in the horizontal direction
    xsz = int(w/phi_steps)
    w = phi_steps * xsz

    cadmodel.add_to_renderer(renderer)

    for actor in extra_actors:
        renderer.AddActor(actor)

    for actor in cal_actors:
        renderer.AddActor(actor)

    # Initialise an array for the output image.
    out_im = np.empty((0,w,3),dtype=np.uint8)

    # Update status callback if present
    if callable(progress_callback):
        try:
            progress_callback('Rendering un-folded wall image ({:d} pixels width)...'.format(w))
        except Exception:
            print('Rendering un-folded wall image ({:d} pixels width)...'.format(w))
        progress_callback(0.)
        n = 0

    # Outer loop over poloidal angle
    for theta in np.linspace(theta_start + theta_step/2,theta_start + 2*np.pi - theta_step/2,theta_steps):

        # See where a line from the camera position at this poloidal angle hits the wall contour
        line_end = [linelength*np.cos(theta) + r_mid,linelength*np.sin(theta)+z_mid]
        rtarg,ztarg = get_contour_intersection(cadmodel.wall_contour,[r_mid,z_mid],line_end)

        # What aspect ratio the render window needs to be to cover the correct poloidal and toroidal angles
        r = np.sqrt((rtarg - r_mid)**2 + (ztarg - z_mid)**2)
        xangle = phi_step * rtarg / r
        aspect = theta_step / xangle
        ysz = int(xsz * aspect)

        # Camera upvec
        view_up_rz = [np.cos(theta + np.pi/2), np.sin(theta + np.pi/2)]

        # Figure out what height this strip of image should be by normalising it with distance
        # from the camera to the wall in that direction.
        dr = theta_step*r

        height = int(dr/(r_equiscale*2*np.pi/w))

        # This will be our horizontal strip of image corresponding to this poloidal angle
        row = np.empty((height, 0, 3), dtype=np.uint8)

        # Set the render window size
        renwin.SetSize(xsz,ysz)

        # Inner loop is over toroidal angle
        for phi in np.linspace(phi_start + phi_step/2,phi_start + 2*np.pi - phi_step/2,phi_steps):

            # Shuffle the camera and light around toroidally and point them at the right place on the wall
            camera.SetPosition(r_mid*np.cos(phi),r_mid*np.sin(phi),z_mid)
            light.SetPosition(r_mid*np.cos(phi),r_mid*np.sin(phi),z_mid)
            upvec = [np.cos(phi)*view_up_rz[0],np.sin(phi)*view_up_rz[0],view_up_rz[1]]
            camera.SetViewUp(upvec)
            camera.SetFocalPoint(rtarg*np.cos(phi),rtarg*np.sin(phi),ztarg)

            # Render the image tile
            renwin.Render()
            vtk_win_im = vtk.vtkWindowToImageFilter()
            vtk_win_im.SetInput(renwin)
            vtk_win_im.Update()
            vtk_image = vtk_win_im.GetOutput()
            vtk_array = vtk_image.GetPointData().GetScalars()
            dims = vtk_image.GetDimensions()
            im = np.flipud(vtk_to_numpy(vtk_array).reshape(dims[1], dims[0] , 3))

            # Squash or stretch it vertically to get the poloidal scale right, as calculated earlier.
            im = cv2.resize(im,(xsz,height))

            # Glue it on to the output
            row = np.hstack((im,row))

            # Update the status callback, if present
            if callable(progress_callback):
                n += 1
                progress_callback(n/(theta_steps*phi_steps))
            # Stop if cancellation has been requested
            if cancel():
                break

        if cancel():
            break

        # Stick this image strip on to the output.
        out_im = np.vstack((row,out_im))


    if callable(progress_callback):
        progress_callback(1.)

    cadmodel.remove_from_renderer(renderer)
    for cal_actor in cal_actors:
        renderer.RemoveActor(cal_actor)

    for actor in extra_actors:
        renderer.RemoveActor(actor)

    if len(legend_items) > 0 and not cancel():

        longest_name = max([len(item[0]) for item in legend_items])

        legend = vtk.vtkLegendBoxActor()
        legend.SetNumberOfEntries(len(legend_items))

        for i, entry in enumerate(legend_items):
            legend.SetEntryString(i, entry[0])
            legend.SetEntryColor(i, entry[1])

        legend.UseBackgroundOn()
        legend.SetBackgroundColor((0.1, 0.1, 0.1))
        legend.SetPadding(9)

        legend_scale = 0.03

        abs_height = int(legend_scale * len(legend_items) * out_im.shape[0])
        width_per_char = legend_scale * out_im.shape[0] * 0.5
        legend_width =  int(width_per_char * longest_name)

        legend.GetPosition2Coordinate().SetCoordinateSystemToDisplay()
        legend.GetPositionCoordinate().SetCoordinateSystemToDisplay()

        legend.GetPosition2Coordinate().SetValue(legend_width, abs_height)

        legend.GetPositionCoordinate().SetValue(0,0)

        renderer.AddActor(legend)
        renwin.SetSize(legend_width,abs_height)
        renwin.Render()
        vtk_win_im = vtk.vtkWindowToImageFilter()
        vtk_win_im.SetInput(renwin)
        vtk_win_im.Update()
        vtk_image = vtk_win_im.GetOutput()
        vtk_array = vtk_image.GetPointData().GetScalars()
        dims = vtk_image.GetDimensions()
        legend_im = np.flipud(vtk_to_numpy(vtk_array).reshape(dims[1], dims[0], 3))
        alpha = np.tile( (0.8 * (legend_im.sum(axis=2) > 0))[:,:,np.newaxis],(1,1,3))

        x_offs = out_im.shape[1] - legend_im.shape[1] - 20
        y_offs = out_im.shape[0] - legend_im.shape[0] - 20

        out_im[y_offs:y_offs + legend_im.shape[0],x_offs:x_offs + legend_im.shape[1],:] = out_im[y_offs:y_offs + legend_im.shape[0],x_offs:x_offs + legend_im.shape[1],:] * (1-alpha) + alpha * legend_im

    renwin.Finalize()



    # Save the image if given a filename
    if filename is not None and not cancel():

        # Re-shuffle the colour channels for saving (openCV needs BGR / BGRA)
        save_im = copy.copy(out_im)
        save_im[:,:,:3] = save_im[:,:,2::-1]
        cv2.imwrite(filename,save_im)
        try:
            progress_callback('Result saved as {:s}'.format(filename))
        except Exception:
            print('Result saved as {:s}'.format(filename))

    return out_im