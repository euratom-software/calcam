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
from .raycast import raycast_sightlines
import copy
from .misc import bin_image

# This is the maximum image dimension which we expect VTK can succeed at rendering in a single RenderWindow.
# Hopefully 5120 is conservative enough to be safe on most systems but also not restrict render quality too much.
# If getting blank images when trying to render at high resolutions, try reducing this.
# TODO: Add a function to this module to determine the best value for this automatically
max_render_dimension = 5120

def render_cam_view(cadmodel,calibration,extra_actors=[],filename=None,oversampling=1,aa=1,transparency=False,verbose=True,coords = 'display',interpolation='cubic'):
    '''
    Render an image of a given CAD model from the point of view of a given calibration.

    NOTE: This function uses off-screen OpenGL rendering which fails above some image dimension which depends on the system.
    The workaround for this is that above a render dimension set by calcam.render.max_render_dimension, the image is rendered
    at lower resolution and then scaled up using nearest-neighbour scaling. For this reason, when rendering high resolution
    images the rendered image quality may be lower than expected.

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
        actor.GetProperty().SetLineWidth( actor.GetProperty().GetLineWidth() / aa ) 
        renderer.RemoveActor(actor)

    renwin.Finalize()

    return output



def render_hires(renderer,oversampling=1,aa=1,transparency=False):
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


    if transparency:
        alpha = 255 * np.ones([np.shape(im)[0],np.shape(im)[1]],dtype='uint8')
        alpha[np.sum(im,axis=2) == 0] = 0
        im = np.dstack((im,alpha))

    # Anti-aliasing by binning
    im = bin_image(im,aa,np.mean)

    return im


def get_fov_actor(cadmodel,calib,actor_type='volume',resolution=None):


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
        
        polygons = vtk.vtkCellArray()

        for n in range(len(x_horiz)):
            if np.abs(raydata.ray_start_coords[y_horiz[n],x_horiz[n],:] - raydata.ray_start_coords[y_horiz[n],x_horiz[n]+1,:]).max() < 1e-3:
                points.InsertNextPoint(raydata.ray_start_coords[y_horiz[n],x_horiz[n],:])
                points.InsertNextPoint(raydata.ray_end_coords[y_horiz[n],x_horiz[n],:])
                points.InsertNextPoint(raydata.ray_end_coords[y_horiz[n],x_horiz[n]+1,:])

        for n in range(len(x_vert)):
            if np.abs( raydata.ray_start_coords[y_vert[n],x_vert[n],:] - raydata.ray_start_coords[y_vert[n]+1,x_vert[n],:] ).max() < 1e-3:
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
def get_image_actor(image_array,clim=None,actortype='vtkImageActor'):

    image = image_array.copy()

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
    
    # Create a temporary floating point copy of the data, for scaling.
    # Also flip the data (vtk addresses y from bottom right) and put in display coords.
    image = np.float32(np.flipud(image))
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
    vtk_im_importer.SetWholeExtent(0,image.shape[1]-1,0,image.shape[0]-1,0,0)

    if actortype == 'vtkImageActor':
        actor = vtk.vtkImageActor()
        actor.InterpolateOff()
        mapper = actor.GetMapper()
        mapper.SetInputConnection(vtk_im_importer.GetOutputPort())

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

        return actor,resizer





# Get a VTK actor of 3D lines based on a set of 3D point coordinates.
def get_lines_actor(coords):

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    point_ind = -1
    if coords.shape[1] == 6:

        for lineseg in range(coords.shape[0]):
            points.InsertNextPoint(coords[lineseg,:3])
            points.InsertNextPoint(coords[lineseg,3:])
            point_ind = point_ind + 2
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0,point_ind - 1)
            line.GetPointIds().SetId(1,point_ind)
            lines.InsertNextCell(line)

    elif coords.shape[1] == 3:

        for pointind in range(coords.shape[0]):

            points.InsertNextPoint(coords[pointind,:])
            point_ind = point_ind + 1

            if point_ind > 0:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0,point_ind - 1)
                line.GetPointIds().SetId(1,point_ind)
                lines.InsertNextCell(line)

    polydata = polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetLineWidth(2)

    return actor