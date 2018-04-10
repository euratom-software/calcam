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
Functions for making high quality CAD model renders
from a camera's point of view, and related stuff.

Written by Scott Silburn
"""

import vtk
import cv2
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import os
import sys
import time
from .raytrace import raycast_sightlines

vtk_major_version = vtk.vtkVersion().GetVTKMajorVersion()


def render_cam_view(cadmodel,calibration,extra_actors=[],filename=None,oversampling=1,aa=1,transparency=False,remap_nearest_neighbour=False,verbose=True,coords = 'Display',screensize=(800,600)):

    """
    Make CAD model renders from the camera's point of view, including all distortion effects etc.

    INPUTS:

    Required:
    CADModel        - CAD model object
    FitResults      - Fit results for the camera from whose viewpoint to do the render
    
    Optional (keyword):
    filename        - filename, including file extension, to save resulting image (if not specified, the image is not saved).
    oversampling    - the size of the rendered image is  <real camera image size> * oversampling
    AA              - Anti-aliasing factor. Larger numbers look nicer but take more memory to render
    Edges           - Bool, if set to True renders the CAD model in wireframe (but still with occlusion)
    EdgeColour      - 3-element tuple of floats 0-1, If Edges=True, what colour to make the edges in the wireframe (R,G,B)
    EdgeWidth       - If Edges=True, line width for edges
    Transparency    - whether to make black areas (i.e. background) transparent.
    ROI             - An ROI definition object: if supplied, the ROI will be rendered on the image.
    ROIColour       - (R,G,B) colour to render the ROI
    ROIOpacity      - Opacity to render ROI
    qt_avilable_geometry - If being called from a Qt app, this must be used. 

    OUTPUTS:

    im - numpy array with RGB[A] image

    If filename is provided, saves the render with the specified filename
    """

    if coords.lower() == 'original' and oversampling != 1:
        raise Exception('Cannot render in original coordinates with oversampling!')


    logbase2 = np.log(oversampling) / np.log(2)
    if abs(int(logbase2) - logbase2) > 1e-5:
        raise ValueError('Oversampling must be a power of two!')


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
        models.append(view_model.model)
    if np.any( np.array(models) == 'fisheye'):
        fov_factor = 3.
    else:
        fov_factor = 1.5

    x_pixels = orig_display_shape[0]
    y_pixels = orig_display_shape[1]

    renwin = vtk.vtkRenderWindow()
    renwin.OffScreenRenderingOn()


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
        

    # This whole thing is in a try() except() because it is prone to memory errors, and I need to put the CAD model
    # colour back the way it started if we have a problem.
    try:

        # We need a field mask the same size as the output
        fieldmask = cv2.resize(calibration.subview_mask,(int(x_pixels*oversampling),int(y_pixels*oversampling)),interpolation=cv2.INTER_NEAREST)

        for field in range(calibration.n_subviews):

            cx = calibration.view_models[field].cam_matrix[0,2]
            cy = calibration.view_models[field].cam_matrix[1,2]
            fy = calibration.view_models[field].cam_matrix[1,1]

            vtk_win_im = vtk.vtkRenderLargeImage()
            vtk_win_im.SetInput(renderer)

            # Width and height - initial render will be put optical centre in the window centre
            wt = int(2*fov_factor*max(cx,x_pixels-cx))
            ht = int(2*fov_factor*max(cy,y_pixels-cy))

            # Make sure the intended render window will fit on the screen
            window_factor = 1
            if wt > screensize[0] or ht > screensize[1]:
                window_factor = int( max( np.ceil(float(wt)/float(screensize[0])) , np.ceil(float(ht)/float(screensize[1])) ) )
        
            vtk_win_im.SetMagnification(int(window_factor*aa*max(oversampling,1)))

            width = int(wt/window_factor)
            height = int(ht/window_factor)

            renwin.SetSize(width,height)

            # Set up CAD camera
            fov_y = 360 * np.arctan( ht / (2*fy) ) / 3.14159
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
            #renwin.Render()

            # Make sure the light lights up the whole model without annoying shadows or falloff.
            #light = renderer.GetLights().GetItemAsObject(0)
            #light.PositionalOn()
            #light.SetConeAngle(180)

            # Do the render and grab an image
            renwin.Render()

            vtk_win_im.Update()

            vtk_image = vtk_win_im.GetOutput()
            vtk_array = vtk_image.GetPointData().GetScalars()
            dims = vtk_image.GetDimensions()

            im = np.flipud(vtk_to_numpy(vtk_array).reshape(dims[1], dims[0] , 3))
            
            if transparency:
                alpha = 255 * np.ones([np.shape(im)[0],np.shape(im)[1]],dtype='uint8')
                alpha[np.sum(im,axis=2) == 0] = 0
                im = np.dstack((im,alpha))

            im = cv2.resize(im,(int(dims[0]/aa*min(oversampling,1)),int(dims[1]/aa*min(oversampling,1))),interpolation=cv2.INTER_AREA)

            if verbose:
                print('[Calcam Renderer] Applying lens distortion (Sub-view {:d}/{:d})...'.format(field + 1,calibration.n_subviews))

            # Pixel locations we want on the final image
            [xn,yn] = np.meshgrid(np.linspace(0,x_pixels-1,x_pixels*oversampling),np.linspace(0,y_pixels-1,y_pixels*oversampling))

            xn,yn = calibration.normalise(xn,yn,field)

            # Transform back to pixel coords where we want to sample the un-distorted render.
            # Both x and y are divided by Fy because the initial render always has Fx = Fy.
            xmap = ((xn * fy) + (width*window_factor)/2.) * oversampling
            ymap = ((yn * fy) + (height*window_factor)/2.) * oversampling
            xmap = xmap.astype('float32')
            ymap = ymap.astype('float32')


            # Actually apply distortion
            if remap_nearest_neighbour:
                interp_method = cv2.INTER_NEAREST
            else:
                interp_method = cv2.INTER_CUBIC
        
            im  = cv2.remap(im,xmap,ymap,interp_method)

            output[fieldmask == field,:] = im[fieldmask == field,:]


        if coords.lower() == 'original':
            output = FitResults.geometry.display_to_original_image(output)
        
        if verbose:
            print('[Calcam Renderer] Completed in {:.1f} s.'.format(time.time() - tstart))

        # Save the image if given a filename
        if filename is not None:

            # If we have transparency, we can only save as PNG.
            if transparency and filename[-3:].lower() != 'png':
                print('[Calcam Renderer] Images with transparency can only be saved as PNG! Overriding output file type to PNG.')
                filename = filename[:-3] + 'png'

            # Re-shuffle the colour channels for saving (openCV needs BGR / BGRA)
            save_im = output
            save_im[:,:,:3] = output[:,:,2::-1]
            cv2.imwrite(filename,save_im)
            if verbose:
                print('[Calcam Renderer] Result saved as {:s}'.format(filename))
    
        raise_exception = False
    except:
        raise_exception = True


    # Tidy up after ourselves!
    cadmodel.set_linewidth(list(cad_linewidths))
    cadmodel.remove_from_renderer(renderer)

    for actor in extra_actors:
        actor.GetProperty().SetLineWidth( actor.GetProperty().GetLineWidth() / aa ) 
        renderer.RemoveActor(actor)

    renwin.Finalize()

    if raise_exception:
        raise
    else:
        return output



def render_hires(renderer,oversampling=1,aa=1,transparency=False):

        # Thicken up all the lines according to AA setting to make sure
        # they dnon't end upp invisibly thin.
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

        # Un-mess with the line widths
        actorcollection.InitTraversal()
        actor = actorcollection.GetNextItemAsObject()
        while actor is not None:
            actor.GetProperty().SetLineWidth( actor.GetProperty().GetLineWidth() / aa )
            actor = actorcollection.GetNextItemAsObject()


        if transparency:
            alpha = 255 * np.ones([np.shape(im)[0],np.shape(im)[1]],dtype='uint8')
            alpha[np.sum(im,axis=2) == 0] = 0
            im = np.dstack((im,alpha))

        im = cv2.resize(im,(int(dims[0]/aa),int(dims[1]/aa)),interpolation=cv2.INTER_AREA)

        return im


def render_material_mask(CADModel,FitResults,Coords='Display'):

    if Coords.lower() == 'display':
        _material_mask = np.zeros([FitResults.image_display_shape[1],FitResults.image_display_shape[0]],dtype=int) - 1
    elif Coords.lower() == 'original':
        _material_mask = np.zeros([FitResults.transform.y_pixels,FitResults.transform.x_pixels],dtype=int) - 1

    material_mask = _material_mask.copy()

    CADModel.colour_by_material(True)
    CADModel.flat_shading(True)

    render = render_cam_view(CADModel,FitResults,AA=1,NearestNeighbourRemap=True,Verbose=False,Transparency=False,Coords=Coords)
    render = render.astype(float) / 255.

    render2 = render[:,:,0] + 2.*render[:,:,1] + 4.*render[:,:,2]

    colours = []
    for material in CADModel.materials:
        colours.append(material[1][0] + 2.*material[1][1] + 4.*material[1][2])

    for i,colour in enumerate(colours):
        delta = abs(render2 - colour)
        _material_mask[delta < 1e-2] = i

    materials = set(_material_mask.flatten())

    material_list = []
    
    for material in materials:
        material_list.append(CADModel.materials[material][0])
        material_mask[_material_mask == material] = len(material_list) - 1
    
    return material_list,material_mask



def get_fov_actor(cadmodel,calib,actor_type='volume',resolution=None):


    if actor_type.lower() == 'volume':

        if resolution is None:
            resolution = 64

    elif actor_type.lower() == 'lines':

        if resolution is None:
            resolution = 8

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

        polydata = polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)


    mapper = vtk.vtkPolyDataMapper()

    if vtk_major_version < 6:
        mapper.SetInput(polydata)
    else:
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
        

        polydata = polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polygons)

    mapper = vtk.vtkPolyDataMapper()

    if vtk_major_version < 6:
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor