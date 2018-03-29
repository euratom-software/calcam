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
import image as CalCamImage
import time
import qt_wrapper as qt
import raytrace

def render_cam_view(CADModel,FitResults,filename=None,oversampling=1,AA=1,Edges=False,EdgeColour=(1,0,0),EdgeWidth=2,Transparency=False,ROI=None,ROIColour=(0.8,0,0),ROIOpacity=0.3,roi_oversize=0,NearestNeighbourRemap=False,Verbose=True,Coords = 'Display',ScreenSize=None):

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

    if Coords.lower() == 'original' and oversampling != 1:
        raise Exception('Cannot render in original coordinates with oversampling!')


    logbase2 = np.log(oversampling) / np.log(2)
    if abs(int(logbase2) - logbase2) > 1e-5:
        raise ValueError('Oversampling must be a power of two!')


    if Verbose:
        tstart = time.time()
        print('[Calcam Renderer] Preparing...')

    # This will be our result. To start with we always render in display coords.
    OutputImage = np.zeros([int(FitResults.image_display_shape[1]*oversampling),int(FitResults.image_display_shape[0]*oversampling),3+Transparency],dtype='uint8')

    # The un-distorted FOV is over-rendered to allow for distortion.
    # FOV_factor is how much to do this by; too small and image edges might be cut off.
    models = []
    for field in FitResults.fit_params:
        models.append(field.model)
    if np.any( np.array(models) == 'fisheye'):
        FOV_factor = 3.
    else:
        FOV_factor = 1.5

    x_pixels = FitResults.image_display_shape[0]
    y_pixels = FitResults.image_display_shape[1]

    renWin = vtk.vtkRenderWindow()
    renWin.OffScreenRenderingOn()

    # Get the screen resolution - even using off screen rendering, if the VTK render window
    # doesn't fit on the screen it causes problems.
    # Maybe slightly hacky way of finding this out? But I don't know a better one.
    if ScreenSize is None:
        try:
            app = qt.QApplication(sys.argv)
            desktopinfo = app.desktop()
            dummydialog = qt.QDialog()
            qt_available_geometry = desktopinfo.availableGeometry(dummydialog)
            ScreenSize = (qt_available_geometry.width(),available_geometry.height())
            del qt_available_geometry,dummydialog,desktopinfo,app
        except:
            ScreenSize = (800,600)

    # Set up render window for initial, un-distorted window
    renderer = vtk.vtkRenderer()
    renWin.AddRenderer(renderer)

    Camera = renderer.GetActiveCamera()

    # Set the model face colour to black if we only want to see edges
    if Edges:
        CADModel.edges = True
        CADModel.edge_width = EdgeWidth
        # If rendering wireframe, set the CAD model colour to the desired edge colour. Before we do that, save the colours we started with.
        oldColours = []
        for Feature in CADModel.features:
            oldColours.append((Feature[4],Feature[0]))
        CADModel.set_colour(EdgeColour)
        

    # This whole thing is in a try() except() because it is prone to memory errors, and I need to put the CAD model
    # colour back the way it started if we have a problem.
    try:
        # Add all the bits of the machine
        for Actor in CADModel.get_vtkActors():
            renderer.AddActor(Actor)

            
        # Add the ROI if provided
        if ROI is not None:
            try:
                n_rois = len(ROI.rois)
                for roi in ROI.rois:
                    ROIActor = roi.get_vtkActor(FitResults.get_pupilpos())
                    ROIActor.GetProperty().SetColor(ROIColour)
                    ROIActor.GetProperty().SetOpacity(ROIOpacity)
                    if roi_oversize > 0:
                        ROIActor.GetProperty().EdgeVisibilityOn()
                        ROIActor.GetProperty().SetLineWidth(roi_oversize*2.)
                        ROIActor.GetProperty().SetEdgeColor(ROIColour)
                    renderer.AddActor(ROIActor)
            except AttributeError:
                ROIActor = ROI.get_vtkActor(FitResults.get_pupilpos(field=0))
                ROIActor.GetProperty().SetColor(ROIColour)
                ROIActor.GetProperty().SetOpacity(ROIOpacity)
                if roi_oversize > 0:
                    ROIActor.GetProperty().EdgeVisibilityOn()
                    ROIActor.GetProperty().SetLineWidth(roi_oversize*2.)
                    ROIActor.GetProperty().SetEdgeColor(ROIColour)
                renderer.AddActor(ROIActor)

        # We need a field mask the same size as the output
        FieldMask = cv2.resize(FitResults.fieldmask,(int(x_pixels*oversampling),int(y_pixels*oversampling)),interpolation=cv2.INTER_NEAREST)


        for field in range(FitResults.nfields):

            Cx = FitResults.fit_params[field].cam_matrix[0,2]
            Cy = FitResults.fit_params[field].cam_matrix[1,2]
            Fy = FitResults.fit_params[field].cam_matrix[1,1]

            vtk_win_im = vtk.vtkRenderLargeImage()
            vtk_win_im.SetInput(renderer)

            # Width and height - initial render will be put optical centre in the window centre
            wt = int(2*FOV_factor*max(Cx,x_pixels-Cx))
            ht = int(2*FOV_factor*max(Cy,y_pixels-Cy))

            # Make sure the intended render window will fit on the screen
            window_factor = 1
            if wt > ScreenSize[0] or ht > ScreenSize[1]:
                window_factor = int( max( np.ceil(float(wt)/float(ScreenSize[0])) , np.ceil(float(ht)/float(ScreenSize[1])) ) )
        
            vtk_win_im.SetMagnification(int(window_factor*AA*max(oversampling,1)))

            width = int(wt/window_factor)
            height = int(ht/window_factor)

            renWin.SetSize(width,height)

            # Set up CAD camera
            FOV_y = 360 * np.arctan( ht / (2*Fy) ) / 3.14159
            CamPos = FitResults.get_pupilpos(field=field)
            CamTar = FitResults.get_los_direction(Cx,Cy,ForceField=field) + CamPos
            UpVec = -1.*FitResults.get_cam_to_lab_rotation(field=field)[:,1]
            Camera.SetPosition(CamPos)
            Camera.SetViewAngle(FOV_y)
            Camera.SetFocalPoint(CamTar)
            Camera.SetViewUp(UpVec)

            if Verbose:
                print('[Calcam Renderer] Rendering (Field {:d}/{:d})...'.format(field + 1,FitResults.nfields))

            # Do the render and grab an image
            renWin.Render()

            vtk_win_im.Update()

            vtk_image = vtk_win_im.GetOutput()

            if field == FitResults.nfields - 1:
                renWin.Finalize()
                if Edges:
                    # Put the colour scheme back to how it was
                    for Feature in oldColours:
                        CADModel.set_colour(Feature[0],Feature[1])
                    Actor.GetProperty().EdgeVisibilityOff()

            vtk_array = vtk_image.GetPointData().GetScalars()
            dims = vtk_image.GetDimensions()
            im = np.flipud(vtk.util.numpy_support.vtk_to_numpy(vtk_array).reshape(dims[1], dims[0] , 3))
            
            if Transparency:
                alpha = 255 * np.ones([np.shape(im)[0],np.shape(im)[1]],dtype='uint8')
                alpha[np.sum(im,axis=2) == 0] = 0
                im = np.dstack((im,alpha))

            im = cv2.resize(im,(int(dims[0]/AA*min(oversampling,1)),int(dims[1]/AA*min(oversampling,1))),interpolation=cv2.INTER_AREA)

            if Verbose:
                print('[Calcam Renderer] Applying lens distortion (Field {:d}/{:d})...'.format(field + 1,FitResults.nfields))

            # Pixel locations we want on the final image
            [xn,yn] = np.meshgrid(np.linspace(0,x_pixels-1,x_pixels*oversampling),np.linspace(0,y_pixels-1,y_pixels*oversampling))

            xn,yn = FitResults.normalise(xn,yn,field)

            # Transform back to pixel coords where we want to sample the un-distorted render.
            # Both x and y are divided by Fy because the initial render always has Fx = Fy.
            xmap = ((xn * Fy) + (width*window_factor)/2.) * oversampling
            ymap = ((yn * Fy) + (height*window_factor)/2.) * oversampling
            xmap = xmap.astype('float32')
            ymap = ymap.astype('float32')


            # Actually apply distortion
            if NearestNeighbourRemap:
                interp_method = cv2.INTER_NEAREST
            else:
                interp_method = cv2.INTER_CUBIC
        
            im  = cv2.remap(im,xmap,ymap,interp_method)

            OutputImage[FieldMask == field,:] = im[FieldMask == field,:]

        CADModel.edges = False

        if Coords.lower() == 'original':
            OutputImage = FitResults.transform.display_to_original_image(OutputImage)
        
        if Verbose:
            print('[Calcam Renderer] Completed in {:.1f} s.'.format(time.time() - tstart))

        # Save the image if given a filename
        if filename is not None:

            # If we have transparency, we can only save as PNG.
            if Transparency and filename[-3:].lower() != 'png':
                print('[Calcam Renderer] Images with transparency can only be saved as PNG! Overriding output file type to PNG.')
                filename = filename[:-3] + 'png'

            # Re-shuffle the colour channels for saving (openCV needs BGR / BGRA)
            SaveIm = OutputImage
            SaveIm[:,:,:3] = OutputImage[:,:,2::-1]
            cv2.imwrite(filename,SaveIm)
            if Verbose:
                print('[Calcam Renderer] Result saved as {:s}'.format(filename))
    except:
        if Edges:
            CADModel.edges = False
            # Put the colour scheme back to how it was
            for Feature in oldColours:
                CADModel.set_colour(Feature[0],Feature[1])
            Actor.GetProperty().EdgeVisibilityOff()
            try:
                renWin.Finalize()
            except:
                pass
        raise

    return OutputImage




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



def get_fov_actor(cadmodel,calib,resolution=64):

    rc = raytrace.RayCaster(calib,cadmodel,verbose=False)

    raydata = rc.raycast_pixels(binning=max(calib.image_display_shape)/resolution)



    # Before we do anything, we need to arrange our triangle corners
    points = vtk.vtkPoints()

    x_horiz,y_horiz = np.meshgrid( np.arange(raydata.ray_start_coords.shape[1]-1), np.arange(raydata.ray_start_coords.shape[0]))
    x_horiz = x_horiz.flatten()
    y_horiz = y_horiz.flatten()

    x_vert,y_vert = np.meshgrid( np.arange(raydata.ray_start_coords.shape[1]), np.arange(raydata.ray_start_coords.shape[0]-1))
    x_vert = x_vert.flatten()
    y_vert = y_vert.flatten()
    
    field = 0

    pupilpos = calib.get_pupilpos(field=field)
    polygons = vtk.vtkCellArray()

    for n in range(len(x_horiz)):
        points.InsertNextPoint(pupilpos)
        points.InsertNextPoint(raydata.ray_end_coords[y_horiz[n],x_horiz[n],:])
        points.InsertNextPoint(raydata.ray_end_coords[y_horiz[n],x_horiz[n]+1,:])

    for n in range(len(x_vert)):
        points.InsertNextPoint(pupilpos)
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

    mapper = vtk.vtkPolyDataMapper()

    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()

    return actor
