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
Region of interest tools for calcam.

While the ROI class does work, more or less, the editing is not ideal and needs a proper GUI
adding, at the moment it's not really very easy or convenient to use. Also there are
things in the ROI / ROISet classes that could use improvement. Work in progress.
"""

import vtk
import vtkinteractorstyles
import numpy as np
import fitting
import matplotlib.path as path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import render
import paths
import os
import csv
import raytrace
import sys
import shutil
from scipy.ndimage.morphology import binary_fill_holes
import subprocess
import re
import os

def edit_ROI(ROI_to_edit,CADmodel,CamFitResults=None,window_size_percent = 80):

    if type(ROI_to_edit) == str:
        ROI_to_edit = ROI(ROI_to_edit)

    # Check we're using the right CAD model
    if ROI_to_edit.machine_name is not None:
        if ROI_to_edit.machine_name.lower() != CADmodel.machine_name.lower():
            raise Exception('Machine name ' + CADmodel.machine_name + ' does not match ROI definition machine name ' + ROI_to_edit.machine_name + '.')
    else:
        ROI_to_edit.machine_name = CADmodel.machine_name

    # Load the CAD model before opening the window, so we don't present the user with a black window for ages.
    print('-> Loading CAD data...')
    CADmodel.load()
    print('   Done.')

    renWin = vtk.vtkRenderWindow()
    renWin.SetWindowName('CalCam - ROI Editor')

    if sys.platform.startswith('linux'):
        # Use a special method on linux because the vtk one doesn't seem to work
        Screensize = get_screensize_linux()
    else:
        # Otherwise try to get screen size using vtk
        try:
            Screensize = renWin.GetScreenSize()
        except:
            # Use a 'safe' low resolution if all else fails:
            Screensize = (1024,768)

    if CamFitResults is not None:
        aspect_ratio = float(CamFitResults.image_display_shape[0])/CamFitResults.image_display_shape[1]
        RayCaster = raytrace.RayCaster(CADModel=CADmodel,FitResults=CamFitResults)
    else:
        aspect_ratio = float(Screensize[0])/Screensize[1]
        RayCaster = None

    # Work out what size to make the window so that the image nicely fills half of it,
    # while conforming to window_size_percent.
    height = int(Screensize[1]*window_size_percent/100)
    width = int(height*aspect_ratio)

    if width > Screensize[0]*window_size_percent/100:
        width = int(Screensize[0]*window_size_percent/100)
        height = int(width/aspect_ratio)

    renWin.SetSize(width,height)

    # Create a RenderWindowInteractor for point picking & navigation
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renWin)
    ROIEditor = vtkinteractorstyles.ROIEditor()
    interactor.SetInteractorStyle(ROIEditor)

    # -----------------------------------------
    # Set up CAD model renderer
    # -----------------------------------------

    renderer = vtk.vtkRenderer()

    # Black background
    renderer.SetBackground(0, 0, 0)

    #Add the renderer to the window
    renWin.AddRenderer(renderer)

    # -----------------------------------------

    # Get things started
    renWin.Render()
    interactor.Initialize()
    ROIEditor.DoInit(renderer,CADmodel,ROI_to_edit,CamFitResults,RayCaster)
    interactor.Start()

    # When the user clicks the X, close the window.
    renWin.Finalize()
    ROIEditor.free_references()
    del renWin


def edit_ROISet(ROISet_camera,ROISet_name,CADmodel,CamFitResults=None,window_size_percent = 80):

    ROISet_to_edit = ROISet(ROISet_camera,ROISet_name)


    # Check we're using the right CAD model
    if ROISet_to_edit.rois[0].machine_name is not None:
        if ROISet_to_edit.rois[0].machine_name.lower() != CADmodel.machine_name.lower():
            raise Exception('Machine name ' + CADmodel.machine_name + ' does not match ROI definition machine name ' + ROISet_to_edit.rois[0].machine_name + '.')

    # Load the CAD model before opening the window, so we don't present the user with a black window for ages.
    print('-> Loading CAD data...')
    CADmodel.load()
    print('   Done.')


    renWin = vtk.vtkRenderWindow()
    renWin.SetWindowName('CalCam - ROI Set Editor')

    if sys.platform.startswith('linux'):
        # Use a special method on linux because the vtk one doesn't seem to work
        Screensize = get_screensize_linux()
    else:
        # Otherwise try to get screen size using vtk
        try:
            Screensize = renWin.GetScreenSize()
        except:
            # Use a 'safe' low resolution if all else fails:
            Screensize = (1024,768)

    if CamFitResults is not None:
        aspect_ratio = float(CamFitResults.image_display_shape[0])/CamFitResults.image_display_shape[1]
        RayCaster = raytrace.RayCaster(CADModel=CADmodel,FitResults=CamFitResults)
    else:
        aspect_ratio = float(Screensize[0])/Screensize[1]
        RayCaster = None

    # Work out what size to make the window so that the image nicely fills half of it,
    # while conforming to window_size_percent.
    height = int(Screensize[1]*window_size_percent/100)
    width = int(height*aspect_ratio)

    if width > Screensize[0]*window_size_percent/100:
        width = int(Screensize[0]*window_size_percent/100)
        height = int(width/aspect_ratio)

    renWin.SetSize(width,height)

    # Create a RenderWindowInteractor for point picking & navigation
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renWin)
    ROISetEditor = vtkinteractorstyles.ROISetEditor()
    interactor.SetInteractorStyle(ROISetEditor)

    # -----------------------------------------
    # Set up CAD model renderer
    # -----------------------------------------

    renderer = vtk.vtkRenderer()

    # Black background
    renderer.SetBackground(0, 0, 0)

    #Add the renderer to the window
    renWin.AddRenderer(renderer)

    # -----------------------------------------

    # Get things started
    renWin.Render()
    interactor.Initialize()
    ROISetEditor.DoInit(renderer,CADmodel,ROISet_to_edit,CamFitResults,RayCaster)
    interactor.Start()

    # When the user clicks the X, close the window.
    renWin.Finalize()
    ROISetEditor.free_references()
    del renWin


# ROI class
class ROI():
    
    def __init__(self,LoadName=None):
        self.vertex_coords = []
        self.machine_name = None
        self.name = ""

        if LoadName is not None:
            self.load(LoadName)

    # Get ROI mask based on ROI definition in 3D (on CAD model)
    # and given camera fit results
    def get_pixel_mask(self,CADModel,CamFitResults,Coords='Original',oversize=0):

        if len(self.vertex_coords) < 3:
            raise Exception('Cannot create pixel mask for an ROI with fewer than 3 vertices!')     

        # We need the CAD model to be temporarily set to all black. Before we do that, save the colours we started with.
        oldColours = []
        for Feature in CADModel.features:
            oldColours.append((Feature[4],Feature[0]))
        CADModel.set_colour((0,0,0))

        # Now, use Render to correctly account for any occlusion of the ROI
        ROIRender = render.render_cam_view(CADModel,CamFitResults,AA=1,ROI=self,Transparency=True,roi_oversize=oversize,NearestNeighbourRemap=True,Verbose=False,Coords=Coords)

        # Put the colour scheme back to how it was
        for Feature in oldColours:
            CADModel.set_colour(Feature[0],Feature[1])

        # The only non-transparent part of this render is the ROI
        ROIRender = ROIRender[:,:,3]

        Mask = ROIRender > 0
        
        Mask = binary_fill_holes(Mask)

        nPixels_final = np.count_nonzero(Mask)

        if nPixels_final == 0:
            raise Exception('This ROI is not visible to the camera!')

        return Mask


    # Set up the ROI definition based on a pixel mask
    # The pixel mask is assumed to be a single filled polygon
    # detect_corners = True attempts to detect the ROI corners and makes an ROI with minimal vertices.
    # This is best for human editing but is not entirely accurate.
    # detect_corners = False just puts a vertex every 7 pixels around the ROI edge.
    def define_from_mask(self,ROIMask,Raycaster,detect_corners=True):

        # Check we're using the right CAD model
        if self.machine_name is not None:
            if self.machine_name.lower() != Raycaster.machine_name.lower():
                raise Exception('Ray caster CAD model name ' + Raycaster.machine_name + ' does not match ROI defition CAD model name ' + self.machine_name + '.')
        else:
            self.machine_name = Raycaster.machine_name

        if ROIMask.shape[1] != Raycaster.fitresults.transform.x_pixels or ROIMask.shape[0] != Raycaster.fitresults.transform.y_pixels:
            if ROIMask.shape[1] != Raycaster.fitresults.image_display_shape[0] or ROIMask.shape[0] != Raycaster.fitresults.image_display_shape[1]:
                raise Exception('Provided ROI mask is the wrong shape!')
            else:
                print('Resizing mask!')
                ROIMask = Raycaster.fitresults.transform.display_to_original_image(ROIMask)


        # First we need to find the corners of the ROI polygon in pixels

        # Subtract copy of mask shifted up by 1 pixel and subtract them to find vertical edges
        mask = np.uint8(ROIMask)
        mask = cv2.medianBlur(mask,3)
        mask2 = np.vstack([ mask[1:,:] , np.zeros([1,np.shape(mask)[1]]) ] )
        delta = mask - mask2

        # The edges are where there is a non-zero value
        # EdgeMask is a mask image containing the ROI outline
        EdgeMask = np.zeros(np.shape(mask))
        Verts2D = list(np.where(delta == -1))
        for i in range(np.size(Verts2D[0])):
            EdgeMask[Verts2D[0][i]+1,Verts2D[1][i]] = 1
        Verts2D = list(np.where(delta == 1))
        for i in range(np.size(Verts2D[0])):
            EdgeMask[Verts2D[0][i],Verts2D[1][i]] = 1

        # Now do the same thing for horizontal edges
        mask2 = np.hstack([ mask[:,1:] , np.zeros([np.shape(mask)[0],1]) ] )
        delta = mask - mask2

        Verts2D = np.where(delta == -1)
        for i in range(np.size(Verts2D[0])):
            EdgeMask[Verts2D[0][i],Verts2D[1][i]+1] = 1
        Verts2D = np.where(delta == 1)
        for  i in range(np.size(Verts2D[0])):
            EdgeMask[Verts2D[0][i],Verts2D[1][i]] = 1
        
        if detect_corners:
            # Find approx corner locations using Harris algorithm
            # verts is a sort of 'corner mask' same size as the image
            # The last 3 parameters in here can be fiddled to make the corner detection behave differently
            verts = cv2.cornerHarris(np.float32(EdgeMask),7,7,0.022)
            verts[verts < np.max(verts)/50.] = 0

        # Now we need to refine the corner locations and sort them in to order around the polygon
        # Start at whatever point on the ROI edge is at the start of Verts2D
        EdgePoints = [ [Verts2D[0][0],Verts2D[1][0]] ]

        # See if it's in the region of a corner according to the corner mask
        if detect_corners:
            if verts[EdgePoints[-1][0],EdgePoints[-1][1]] > 0:
                nearCorner = True
                FirstNearCorner = 0
            else:
                nearCorner = False

        # These will store the X and Y coords of the final corner locations
        CornersX = []
        CornersY = []


        # Shuffle around the ROI edge and check when we enter and leave the vicinity of corners according to the corner mask.
        # The centre of the edge line within each corner mask blob is taken as the corner location
        breaknext = False
        nsteps = 0
        while breaknext == False:
            
            if nsteps % 7 == 0 and not detect_corners:
                CornersX.append(EdgePoints[-1][1])
                CornersY.append(EdgePoints[-1][0])
                
            # Up & right
            if EdgeMask[EdgePoints[-1][0]-1,EdgePoints[-1][1]+1] == 1:
                EdgePoints.append([EdgePoints[-1][0]-1,EdgePoints[-1][1]+1])
                EdgeMask[EdgePoints[-1][0],EdgePoints[-1][1]] = 0
            # Right
            elif EdgeMask[EdgePoints[-1][0],EdgePoints[-1][1]+1] == 1:
                EdgePoints.append([EdgePoints[-1][0],EdgePoints[-1][1]+1])
                EdgeMask[EdgePoints[-1][0],EdgePoints[-1][1]] = 0
            # Down & right
            elif EdgeMask[EdgePoints[-1][0]+1,EdgePoints[-1][1]+1] == 1:
                EdgePoints.append([EdgePoints[-1][0]+1,EdgePoints[-1][1]+1])
                EdgeMask[EdgePoints[-1][0],EdgePoints[-1][1]] = 0
            # Down
            elif EdgeMask[EdgePoints[-1][0]+1,EdgePoints[-1][1]] == 1:
                EdgePoints.append([EdgePoints[-1][0]+1,EdgePoints[-1][1]])
                EdgeMask[EdgePoints[-1][0],EdgePoints[-1][1]] = 0
            # Down & Left
            elif EdgeMask[EdgePoints[-1][0]+1,EdgePoints[-1][1]-1] == 1:
                EdgePoints.append([EdgePoints[-1][0]+1,EdgePoints[-1][1]-1])
                EdgeMask[EdgePoints[-1][0],EdgePoints[-1][1]] = 0
            # Left
            elif EdgeMask[EdgePoints[-1][0],EdgePoints[-1][1]-1] == 1:
                EdgePoints.append([EdgePoints[-1][0],EdgePoints[-1][1]-1])
                EdgeMask[EdgePoints[-1][0],EdgePoints[-1][1]] = 0
            # Left & up
            elif EdgeMask[EdgePoints[-1][0]-1,EdgePoints[-1][1]-1] == 1:
                EdgePoints.append([EdgePoints[-1][0]-1,EdgePoints[-1][1]-1])
                EdgeMask[EdgePoints[-1][0],EdgePoints[-1][1]] = 0
            # Up
            elif EdgeMask[EdgePoints[-1][0]-1,EdgePoints[-1][1]] == 1:
                EdgePoints.append([EdgePoints[-1][0]-1,EdgePoints[-1][1]])
                EdgeMask[EdgePoints[-1][0],EdgePoints[-1][1]] = 0
            else:
                # We've got to the end!
                breaknext = True
                
            nsteps = nsteps + 1
            
            if detect_corners:
                # Check if we've arrived at a corner
                if nearCorner == False and verts[EdgePoints[-1][0],EdgePoints[-1][1]] > 0:
                    nearCorner = True
                    FirstNearCorner = len(EdgePoints) - 1

                # Check if we've left a corner
                if nearCorner == True and verts[EdgePoints[-1][0],EdgePoints[-1][1]] == 0:
                    nearCorner = False
                    CornerInd = np.int(np.mean([len(EdgePoints)-1,FirstNearCorner]))
                    CornersX.append(EdgePoints[CornerInd][1])
                    CornersY.append(EdgePoints[CornerInd][0])

        # Project the ROI corners on to the CAD model!
        PointsList = Raycaster.raycast_pixels(CornersX,CornersY,Coords='Original').ray_end_coords
        self.vertex_coords = []

        for point in PointsList:
            self.vertex_coords.append(tuple(point))

        # This is good for debugging - returns the Harris corner detection mask.
        #return verts

    def get_vtkActor(self,campos,dist_from_surf = 0.08,return_mapper=False):

        points = vtk.vtkPoints()
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(self.vertex_coords))

        dist_to_points = []
        direction_to_points = []

        for point in self.vertex_coords:
            dist_to_points.append(np.sqrt((point[0] - campos[0])**2 + (point[1] - campos[1])**2 + (point[2] - campos[2])**2))
            direction_to_points.append( [ (point[0] - campos[0])/dist_to_points[-1], (point[1] - campos[1])/dist_to_points[-1], (point[2] - campos[2])/dist_to_points[-1] ] )

        dist_to_poly = min(dist_to_points) - dist_from_surf

        for n,direction in enumerate(direction_to_points):

            points.InsertNextPoint( ( campos[0] + direction[0]*(dist_to_points[n] - dist_from_surf), campos[1] + direction[1]*(dist_to_points[n] - dist_from_surf), campos[2] + direction[2]*(dist_to_points[n] - dist_from_surf) ) )
            #points.InsertNextPoint( ( campos[0] + direction[0]*dist_to_poly, campos[1] + direction[1]*dist_to_poly, campos[2] + direction[2]*dist_to_poly ) )
            polygon.GetPointIds().SetId(n,n)

        polygons = vtk.vtkCellArray()
        polygons.InsertNextCell(polygon)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polygons)

        triangulator = vtk.vtkTriangleFilter()
        triangulator.SetInputData(polydata)
        triangulator.Update()

        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(triangulator.GetOutput())
        else:
            mapper.SetInputData(triangulator.GetOutput())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().LightingOff()

        if return_mapper:
            return mapper,actor
        else:
            return actor

    def save(self,SaveName):

        savefile = open(os.path.join(paths.rois,SaveName + '.csv'),'wb')
        csvwriter = csv.writer(savefile)
        
        if self.name == "":
            self.name = SaveName

        csvwriter.writerow(["Machine:",self.machine_name])
        csvwriter.writerow(["ROI Name:",self.name])
        csvwriter.writerow(["Verices:"])
        csvwriter.writerow(["X (m)","Y (m)","Z (m)"])

        for point in self.vertex_coords:
            csvwriter.writerow(point)

        savefile.close()

    def load(self,SaveName):

        savefile = open(os.path.join(paths.rois,SaveName + '.csv'),'r')
        csvreader = csv.reader(savefile)

        for line in csvreader:
            if 'Machine:' in line[0]:
                self.machine_name = line[1]
            elif 'ROI Name:' in line[0]:
                self.name = line[1]
            elif 'X (m)' in line[0]:
                break

        # Loop over remaining rows, adding them to the CAD points
        self.vertex_coords = []
        for point in csvreader:
            self.vertex_coords.append(( float(point[0]), float(point[1]), float(point[2])))

        savefile.close()



class ROISet():

    def __init__(self,camera=None,load_name=None):
        self.name = None
        self.camera = None
        self.rois = []
        if load_name is not None and camera is not None:
            self.load(camera,load_name)

    def save(self,name=None):
        if self.camera is None:
            raise Exception('You must define the camera name (ROISet.camera) before saving!')
        if self.name is None:
            if name is None:
                raise Exception('You must provide a name for this ROI set to save!')
            else:
                self.name = name
        else:
            if name is not None:
                if name != self.name:
                    print('-> Saving as new ROI set with name "' + name + '"')
                    self.name = name

        if os.path.isdir(os.path.join(paths.rois,self.camera,self.name)):
            replace = True
            os.rename(os.path.join(paths.rois,self.camera,self.name),os.path.join(paths.rois,self.camera,self.name) + '_backup')
        else:
            replace = False
            
        os.makedirs(os.path.join(paths.rois,self.camera,self.name))
        
        try:
            for index,roi in enumerate(self.rois):
               roi.save(os.path.join(self.camera,self.name,str(index) + '_' + roi.name))
            if replace:
                shutil.rmtree(os.path.join(paths.rois,self.camera,self.name) + '_backup')
        except:
            shutil.rmtree(os.path.join(paths.rois,self.camera,self.name))
            if replace:
                os.rename(os.path.join(paths.rois,self.camera,self.name) + '_backup',os.path.join(paths.rois,self.camera,self.name))



    def load(self,camera,load_name):

        self.camera = camera
        self.name = load_name

        if not os.path.isdir(os.path.join(paths.rois,camera,self.name)):
            raise Exception('No ROI set with specified name found!')

        next_roi_exists = True
        filelist = os.listdir(os.path.join(paths.rois,camera,self.name))
        while next_roi_exists:
            next_roi_exists = False
            for filename in filelist:
                if int(filename.split('_')[0]) == len(self.rois):
                    self.rois.append(ROI(os.path.join(paths.rois,camera,self.name,filename[0:-4])))
                    next_roi_exists = True

    def get_pixel_mask(self,CADModel,CamFitResults,Coords='Original',oversize=0):

        mask = self.rois[0].get_pixel_mask(CADModel,CamFitResults,Coords,oversize)
        for roi in self.rois[1:]:
            mask = mask | roi.get_pixel_mask(CADModel,CamFitResults,Coords,oversize)

        return mask

# Try to get the screen resolution using xrandr. Needs proper error handling!
def get_screensize_linux():
    output = subprocess.check_output(['xrandr','-q'],stderr=open(os.devnull,'w'))
    strings = output.split('\n')
    for st in strings:
        if '*' in st:
            numbers = re.findall('\d+',st)
            return [int(numbers[0]),int(numbers[1])]