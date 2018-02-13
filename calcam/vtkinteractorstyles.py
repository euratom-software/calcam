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
vtk Interactor style classes for Calcam.

There's a lot more code in here than there needs to be / should be,
a lot of what's done in here should probably be moved to the GUI module.
However, because a lot of this code predates the Qt GUI, it was never moved out of here,
which is why the GUI window object and vtkinteractor object interact with
each other in a sadly messy way.

Written by Scott Silburn 
"""

import vtk
import csv
import fitting
import numpy as np
import os
import pointpairs
import render
import image
import gc
import roi
import sys
import time
import raytrace
import roi
import matplotlib.path as mplPath
import matplotlib.cm
import paths

class PointPairPicker(vtk.vtkInteractorStyleTerrain):
 
    def __init__(self,parent=None):
        # Set callbacks for all the controls
        self.AddObserver("LeftButtonPressEvent",self.OnLeftClick)
        self.AddObserver("RightButtonPressEvent",self.rightButtonPress)
        self.AddObserver("RightButtonReleaseEvent",self.rightButtonRelease)
        self.AddObserver("MiddleButtonPressEvent",self.middleButtonPress)
        self.AddObserver("MiddleButtonReleaseEvent",self.middleButtonRelease)
        self.AddObserver("MouseWheelForwardEvent",self.ZoomIn)
        self.AddObserver("MouseWheelBackwardEvent",self.ZoomOut)
        self.AddObserver("MouseMoveEvent",self.mouse_move)


    # Do various initial setup things, most of which can't be done at the time of __init__
    def DoInit(self,ren_2D,ren_3D,gui_window):
       

        # Get the interactor object
        self.Interactor = self.GetInteractor()

        # Some other objects from higher up which I need access to
        self.Window = self.Interactor.GetRenderWindow()
        self.Renderer = ren_3D
        self.Renderer_2D = ren_2D
        self.Camera3D = self.Renderer.GetActiveCamera()

        self.im_dragging = False

        self.Camera2D = self.Renderer_2D.GetActiveCamera()
        self.Camera2D.ParallelProjectionOn()
        self.Image = None
        self.gui_window = gui_window
        self.field_names = ['Image']
        self.nFields = None


        self.SetAutoAdjustCameraClippingRange(0)

        # Turn off any VTK responses to keyboard input (all necessary keyboard shortcuts etc are done in Q)
        self.Interactor.RemoveObservers('KeyPressEvent')
        self.Interactor.RemoveObservers('CharEvent')

        # Add observer for catching window resizing
        self.Window.AddObserver("ModifiedEvent",self.OnWindowSizeAdjust)

        # Create a picker
        self.Picker = vtk.vtkCellPicker()
        self.Interactor.SetPicker(self.Picker)

        # Variables
        self.ObjectPoints = []
        self.ImagePoints = []
        self.ReProjectedPoints = None
        self.SelectedPoint = None
        self.FitResults = None
        self.overlay_on = False
        self.fit_overlay_actor = None
    

        # We will use this for converting from 3D to screen coords.
        self.CoordTransformer = vtk.vtkCoordinate()
        self.CoordTransformer.SetCoordinateSystemToWorld()

        self.PointPairs = pointpairs.PointPairs()

        # We're going to need a fitter object...
        self.Fitter = fitting.Fitter()

        
        
    def free_references(self):
        del self.Interactor
        del self.Window
        del self.Renderer
        del self.Renderer_2D
        del self.Camera3D
        del self.Camera2D
        del self.Picker


    # Use this image object
    def init_image(self,image,hold_position=False):
        
        # Remove current image, if any
        if self.Image is not None:
            if hold_position:
                # Check if hold position is really justified - override hold position if the image aspect ratio has changed!
                imshape = self.Image.transform.get_display_shape()
                if abs( imshape[1]/imshape[0] - image.transform.get_display_shape()[1] / image.transform.get_display_shape()[0]) > 1e-6:
                    hold_position = False

            self.Renderer_2D.RemoveActor(self.ImageActor)
            self.ImageActor = None
            self.ImageResizer = None
            self.Image = None
        else:
            hold_position = False

        try:

            self.Image = image
            self.ImageOriginalSize = self.Image.transform.get_display_shape()

            self.nFields = np.max(self.Image.fieldmask) + 1

            self.fieldmask = np.flipud(self.Image.transform.original_to_display_image(self.Image.fieldmask))

            self.field_names = self.Image.field_names

            self.ImageActor = self.Image.get_vtkActor()


            self.WinSize = self.Window.GetSize()
            winaspect =  (float(self.WinSize[0])/2)/float(self.WinSize[1])


            ImAspect = float(self.ImageOriginalSize[0])/self.ImageOriginalSize[1]


            #self.ImageResizer.SetOutputDimensions(int(self.ZoomRefSize[0]*self.ZoomLevel),int(self.ZoomRefSize[1]*self.ZoomLevel),1)
            self.Renderer_2D.AddActor2D(self.ImageActor)


            bounds = self.ImageActor.GetBounds()
            xc = bounds[0] + bounds[1] / 2
            yc = bounds[2] + bounds[3] / 2
            ye = bounds[3] - bounds[2]
            xe = bounds[1] - bounds[0]

            self.zoom_ref_cc = (xc,yc)

            im_aspect = xe / ye

            if winaspect >= im_aspect:
                # Base new zero size on y dimension
                self.zoom_ref_scale = 0.5*ye
            else:
                self.zoom_ref_scale = 0.5*xe/winaspect


            if not hold_position:
                # Reset view to fit the whole image on screen.
                self.ZoomLevel = 1.                
                self.Camera2D.SetParallelScale(self.zoom_ref_scale)
                self.Camera2D.SetPosition(xc,yc,1.)
                self.Camera2D.SetFocalPoint(xc,yc,0.)
            
            self.PointPairs.set_image(self.Image)
            self.Fitter.set_PointPairs(self.PointPairs)


            self.gui_window.refresh_vtk()
        except:

            self.Image = None
            raise


    # On the CAD view, middle click + drag to pan
    def middleButtonPress(self,obj,event):
        self.orig_dist = self.Camera3D.GetDistance()
        self.Camera3D.SetDistance(0.5)
        if self.ChooseRenderer() == '3D':
                self.OnMiddleButtonDown()
        elif self.ChooseRenderer() == '2D':
            self.im_dragging = True

    def middleButtonRelease(self,obj,event):
        if self.ChooseRenderer() == '3D':
            self.OnMiddleButtonUp()
            self.Camera3D.SetDistance(self.orig_dist)
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewAngle())
        elif self.ChooseRenderer() == '2D':
            self.im_dragging = False

    # On the CAD view, right click+drag to rotate (usually on left button in this interactorstyle)
    def rightButtonPress(self,obj,event):
        self.orig_dist = self.Camera3D.GetDistance()
        self.Camera3D.SetDistance(0.01)
        if self.ChooseRenderer() == '3D':
            self.OnLeftButtonDown()
        return

    def rightButtonRelease(self,obj,event):
        if self.ChooseRenderer() == '3D':
                self.OnLeftButtonUp()
                self.Camera3D.SetDistance(self.orig_dist)
                self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewAngle())    
        return


    # Left click to move a point or add a new point
    def OnLeftClick(self,obj,event):

        # If the user clicked on the CAD model...
        if self.ChooseRenderer() == '3D':

                # Check if they selected an existing point or a position on the model
                picktype,pickdata = self.GetSelected_3D()

                # If they clicked a model position...
                if picktype == 'Position':

                    # If the user was holding Ctrl, create a new point pair and select it.
                    if self.Interactor.GetControlKey():
                        if self.Image is not None:
                            self.AddPointPair()
                        else:
                            raise UserWarning('Cannot add point pairs without an image loaded. Please load an image first.')
            
                    if self.SelectedPoint is not None:

                        # If there isn't already a CAD point in this pair, add one
                        if self.ObjectPoints[self.SelectedPoint] is None:
                                self.AddPoint3D(pickdata)


                        else:
                                # If there is already a CAD point in this pair, move it to wherever the user clicked.
                                self.ObjectPoints[self.SelectedPoint][0].SetFocalPoint(pickdata[0],pickdata[1],pickdata[2])
                                self.Set3DCursorStyle(self.SelectedPoint,True)

                        self.gui_window.pointpairs_changed = True

                # If the user clicked on an existing point, select that point pair.
                if picktype == 'Cursor':
                    self.CursorDefocus(self.SelectedPoint)
                    self.SelectedPoint = pickdata
                    self.CursorFocus(pickdata)
        else:

                # This proceeds almost exactly as above, but for the image view.
                picktype,pickdata = self.GetSelected_2D()

                if picktype == 'Position':

                    if self.Interactor.GetControlKey():
                        self.AddPointPair()  

                    if self.SelectedPoint is not None:

                        field = self.fieldmask[int(pickdata[1]),int(pickdata[0])]

                        if self.ImagePoints[self.SelectedPoint][field] is None:

                                self.AddPoint2D(pickdata)

                        else:
                                # Move the 2D point: first store the image coords
                                self.ImagePoints[self.SelectedPoint][field][3] = pickdata
                                # Because the image view is still a 3D renderer, we have to convert our 2D image coordinate
                                # in to a 3D location to place the cursor there
                                self.ImagePoints[self.SelectedPoint][field][0].SetFocalPoint(pickdata[0],pickdata[1],0.1)
                   
                self.gui_window.pointpairs_changed = True
 
                if picktype == 'Cursor':
                    self.CursorDefocus(self.SelectedPoint)
                    self.SelectedPoint = pickdata
                    self.CursorFocus(pickdata)

        # If there have been any changes to the point pairs as a result of this click,
        # make sure the structure containing the results to be returned is up to date.
        if picktype == 'Position':
                self.UpdateResults()

        self.update_current_point()
        self.gui_window.refresh_vtk()



    def update_current_point(self):

        if self.SelectedPoint is not None:
            # Update status info for GUI
            impoints = []

            for j in range(self.nFields):
                if self.ImagePoints[self.SelectedPoint][j] is not None:

                        x = self.ImagePoints[self.SelectedPoint][j][3][0]
                        y = self.ImageOriginalSize[1] - self.ImagePoints[self.SelectedPoint][j][3][1]
                        impoints.append((x,y))

            if self.ObjectPoints[self.SelectedPoint] is not None:
                object_coords = self.ObjectPoints[self.SelectedPoint][0].GetFocalPoint()
            else:
                object_coords = None

            self.gui_window.update_current_points(object_coords,impoints)
        else:
            self.gui_window.update_current_points(None,None)




    
    def AddPointPair(self):

        if self.SelectedPoint is not None:
                self.CursorDefocus(self.SelectedPoint)

        self.ObjectPoints.append(None)
        self.ImagePoints.append([])
        for i in range(self.nFields):
                self.ImagePoints[-1].append(None)
        self.SelectedPoint = len(self.ObjectPoints) - 1



    def ZoomIn(self,obj,event):

        if self.ChooseRenderer() == '3D':

                # If ctrl + scroll, change the camera FOV
                if self.Interactor.GetControlKey():
                    self.Camera3D.SetViewAngle(max(self.Camera3D.GetViewAngle()*0.9,1))

                # Otherwise, move the camera forward.
                else:
                    orig_dist = self.Camera3D.GetDistance()
                    self.Camera3D.SetDistance(0.3)
                    self.Camera3D.Dolly(1.5)
                    self.Camera3D.SetDistance(orig_dist)

                # Update cursor sizes depending on their distance from the camera,
                # so they're all comfortably visible and clickable.
                for i in range(len(self.ObjectPoints)):
                    if self.ObjectPoints[i] is not None:
                        self.Set3DCursorStyle(i,self.SelectedPoint == i,self.ImagePoints[i].count(None) < self.nFields)
                self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewAngle())
        else:

                # Re-position the image camera to keep the image point under the mouse pointer fixed when zooming


                zoom_ratio = 1 + 0.2/self.ZoomLevel                 # The zoom ratio we will have

                # Current camera position and scale
                campos = self.Camera2D.GetPosition()
                camscale = self.Camera2D.GetParallelScale() * 2.


                # Where is the zoom target in world coordinates?

                # Zoom coordinates in window pixels
                zoomcoords = list(self.Interactor.GetEventPosition())
                zoomcoords[0] = zoomcoords[0] - self.WinSize[0]/2.

                # Position of current centre from to where we're zooming
                zoomcoords = ( (zoomcoords[0] - (self.WinSize[0]/4.))/(self.WinSize[0]/2.) * camscale * (float(self.WinSize[0])/2.)/float(self.WinSize[1]) + campos[0],
                               (zoomcoords[1] - self.WinSize[1]/2.)/self.WinSize[1] * camscale + campos[1] )

                # Vector from zoom point to current camera centre
                zoomvec = ( campos[0] - zoomcoords[0] , campos[1] - zoomcoords[1] )

                # Now we move the camera along the line bwteen the current camera centre and zoom position
                newxc = zoomvec[0]/zoom_ratio + zoomcoords[0]
                newyc = zoomvec[1]/zoom_ratio + zoomcoords[1]

                # Actually move the camera
                self.Camera2D.SetPosition((newxc,newyc,1.))
                self.Camera2D.SetFocalPoint((newxc,newyc,0.))

                # Actually zoom in.
                self.ZoomLevel = self.ZoomLevel + 0.2
                self.Camera2D.SetParallelScale(self.zoom_ref_scale / self.ZoomLevel)
                self.update_im_cursorstyles()

        self.gui_window.refresh_vtk()



    def ZoomOut(self,obj,event):

        if self.ChooseRenderer() == '3D':

                # If ctrl + scroll, change the camera FOV
                if self.Interactor.GetControlKey():
                    self.Camera3D.SetViewAngle(min(self.Camera3D.GetViewAngle()*1.1,110.))

                # Otherwise, move the camera backward.
                else:
                    orig_dist = self.Camera3D.GetDistance()
                    self.Camera3D.SetDistance(0.3)
                    self.Camera3D.Dolly(0.75)
                    self.Camera3D.SetDistance(orig_dist)

                # Update cursor sizes so they're all well visible:
                for i in range(len(self.ObjectPoints)):
                    if self.ObjectPoints[i] is not None:
                        self.Set3DCursorStyle(i,self.SelectedPoint == i,self.ImagePoints[i].count(None) < self.nFields)
                self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewAngle())
        else:
                # Zoom out smoothly until the whole image is visible
                if self.ZoomLevel > 1.:

                    zoom_ratio = 0.2/(self.ZoomLevel**2 - 1.2*self.ZoomLevel + 0.2)

                    campos = self.Camera2D.GetPosition()

                    zoomvec = ( self.zoom_ref_cc[0] - campos[0] , self.zoom_ref_cc[1] - campos[1] )

                    self.Camera2D.SetPosition((campos[0] + zoomvec[0] * zoom_ratio, campos[1] +  zoomvec[1] * zoom_ratio, 1.))
                    self.Camera2D.SetFocalPoint((campos[0] + zoomvec[0] * zoom_ratio, campos[1] +  zoomvec[1] * zoom_ratio, 0.))

                    self.ZoomLevel = self.ZoomLevel - 0.2
                    self.Camera2D.SetParallelScale(self.zoom_ref_scale / self.ZoomLevel)
                    self.update_im_cursorstyles()

        self.gui_window.refresh_vtk()



    def update_im_cursorstyles(self):
        
        for n in range(len(self.ImagePoints)):
            self.Set2DCursorStyle(n,self.SelectedPoint == n,self.ObjectPoints[n] is not None)


    def clear_all(self):

        for i in range(len(self.ObjectPoints)):
            self.DeletePoint(len(self.ObjectPoints)-1)

        self.HideReprojectedPoints()
        if self.overlay_on:
            self.Renderer_2D.RemoveActor(self.fit_overlay_actor)

        self.gui_window.refresh_vtk()



    # Function to check if the user is interacting with the CAD or image view
    def ChooseRenderer(self):
        coords = self.Interactor.GetEventPosition()
        poked_renderer = self.Interactor.FindPokedRenderer(coords[0],coords[1])
        if poked_renderer == self.Renderer_2D:
                ren_name = '2D'
        else:
                ren_name = '3D'
        return ren_name

    # Defocus cursors for a given point pair
    def CursorDefocus(self,CursorNumber):
        if self.ObjectPoints[CursorNumber] is not None:
                self.Set3DCursorStyle(CursorNumber,False,self.ImagePoints[CursorNumber].count(None) < self.nFields)

        self.Set2DCursorStyle(CursorNumber,False,self.ObjectPoints[CursorNumber] is not None)

    # Bring cursors to focus for a given point pair
    def CursorFocus(self,CursorNumber):

        if self.ObjectPoints[CursorNumber] is not None:
                self.Set3DCursorStyle(CursorNumber,True,self.ImagePoints[CursorNumber].count(None) < self.nFields)
        
        self.Set2DCursorStyle(CursorNumber,True,self.ObjectPoints[CursorNumber] is not None)


    # Check whether the user clicked an existing point or position
    # on the CAD model
    def GetSelected_3D(self):

        # These will be the variables we return. If the user clicked in free space they will stay None.
        picktype = None
        pickdata = None

        # Do a pick with our picker object
        clickcoords = self.Interactor.GetEventPosition()
        retval = self.Picker.Pick(clickcoords[0],clickcoords[1],0,self.Renderer)

        # If something was successfully picked, find out what it was...
        if retval != 0:

                pickedpoints = self.Picker.GetPickedPositions()

                # If more than 1 point is within the picker's tolerance,
                # use the one closest to the camera (this is most intuitive)
                npoints = pickedpoints.GetNumberOfPoints()
                dist_fromcam = []
                campos = self.Camera3D.GetPosition()

                for i in range(npoints):
                    point = pickedpoints.GetPoint(i)
                    dist_fromcam.append(np.sqrt( (campos[0] - point[0])**2 + (campos[1] - point[1])**2 + (campos[2] - point[2])**2 ))

                _, idx = min((val, idx) for (idx, val) in enumerate(dist_fromcam))

                pickcoords = pickedpoints.GetPoint(idx)

                # If the picked point is within 1.5x the cursor radius of any existing point,
                # say that the user clicked on that point
                for i in range(len(self.ObjectPoints)):
                    if self.ObjectPoints[i] is not None:
                        # Get the on-screen position of this cursor
                        self.CoordTransformer.SetValue(self.ObjectPoints[i][0].GetFocalPoint())
                        cursorpos = self.CoordTransformer.GetComputedDisplayValue(self.Renderer)
                        cursorpos_3D = self.ObjectPoints[i][0].GetFocalPoint()
                        # See if it's close enough to the click to be considered 'clicked'
                        cursortol = 7 # Number of pixels the click has to be within
                        cursortol_3D = 0.1 # Distance (in metres) the 3D picked point has to be within.
                        dist_from_cursor = np.sqrt( (cursorpos[0] - clickcoords[0])**2 + (cursorpos[1] - clickcoords[1])**2 )
                        dist_from_cursor_3D =  np.sqrt( (cursorpos_3D[0] - pickcoords[0])**2 + (cursorpos_3D[1] - pickcoords[1])**2 + (cursorpos_3D[2] - pickcoords[2])**2)
                        if dist_from_cursor < cursortol and dist_from_cursor_3D < cursortol_3D and i != self.SelectedPoint:
                                picktype = 'Cursor'
                                pickdata = i

                # If they didn't click on an existing point, they clicked a model position.
                if picktype is None:
                    picktype = 'Position'
                    pickdata = pickcoords


        return picktype,pickdata


    # Check whether the user clicked an existing cursor on the image
    # or just somewhere else
    # Very similar to GetSelected_3D
    def GetSelected_2D(self):

        picktype = None
        pickdata = None

        clickcoords = self.Interactor.GetEventPosition()

        # Check if the click was near enough an existing cursor to be considered as clicking it
        for i in range(len(self.ImagePoints)):
                for j in range(self.nFields):
                    if self.ImagePoints[i][j] is not None:
                        cursortol = 7
                        screencoords = self.ImageToDisplayCoords(self.ImagePoints[i][j][3])
                        dist_from_cursor = np.sqrt( (screencoords[0] - clickcoords[0])**2 + (screencoords[1] - clickcoords[1])**2 )
                        if dist_from_cursor < cursortol and i != self.SelectedPoint:
                                picktype = 'Cursor'
                                pickdata = i

        # Otherwise, if the click was within the bounds of the image,
        # return the pixel coordinates on the image
        if picktype is None and self.Image is not None:
                imcoords = self.DisplayToImageCoords(clickcoords)
                imshape = self.Image.transform.get_display_shape()
                if np.all(np.array(imcoords) > 0.) and np.all(np.array(imcoords) < np.array(imshape)):
                    picktype = 'Position'
                    pickdata = imcoords

        return picktype,pickdata



    # Set the visual style of cursors on the CAD view
    def Set3DCursorStyle(self,CursorNumber,Focus,Paired=True):

        # Cursor appearance settings - mess with this as per your tastes.
        # The actual size & line width numbers correspond to 3D cursor side length in CAD units for a 75 degree FOV.
        focus_size = 0.025
        nofocus_size = 0.0125

        focus_linewidth = 3
        nofocus_linewidth = 2

        focus_colour = (0,0.8,0)
        nofocus_colour = (0.8,0,0)
        unpaired_colour = (0.8,0.8,0) 


        # Cursor size scales with camera FOV to maintain size on screen.
        focus_size = focus_size * (self.Camera3D.GetViewAngle()/75)
        nofocus_size = nofocus_size * (self.Camera3D.GetViewAngle()/75)

        point = self.ObjectPoints[CursorNumber][0].GetFocalPoint()
        campos = self.Camera3D.GetPosition()
        dist_to_cam = np.sqrt( (campos[0] - point[0])**2 + (campos[1] - point[1])**2 + (campos[2] - point[2])**2 )

        if Focus:
                self.ObjectPoints[CursorNumber][0].SetModelBounds([point[0]-focus_size*dist_to_cam,point[0]+focus_size*dist_to_cam,point[1]-focus_size*dist_to_cam,point[1]+focus_size*dist_to_cam,point[2]-focus_size*dist_to_cam,point[2]+focus_size*dist_to_cam])
                self.ObjectPoints[CursorNumber][2].GetProperty().SetColor(focus_colour)
                self.ObjectPoints[CursorNumber][2].GetProperty().SetLineWidth(focus_linewidth)
        else:
                self.ObjectPoints[CursorNumber][0].SetModelBounds([point[0]-nofocus_size*dist_to_cam,point[0]+nofocus_size*dist_to_cam,point[1]-nofocus_size*dist_to_cam,point[1]+nofocus_size*dist_to_cam,point[2]-nofocus_size*dist_to_cam,point[2]+nofocus_size*dist_to_cam])
                self.ObjectPoints[CursorNumber][2].GetProperty().SetLineWidth(nofocus_linewidth)
                if Paired:
                    self.ObjectPoints[CursorNumber][2].GetProperty().SetColor(nofocus_colour)
                else:
                    self.ObjectPoints[CursorNumber][2].GetProperty().SetColor(unpaired_colour)



    # Similar to Set3DCursorStyle but for image points
    def Set2DCursorStyle(self,CursorNumber,Focus,Paired=True):
        
        camscale = self.Camera2D.GetParallelScale()
        focus_size = 0.03 * camscale
        nofocus_size = 0.015 * camscale

        focus_linewidth = 3
        nofocus_linewidth = 2

        focus_colour = (0,0.8,0)
        nofocus_colour = (0.8,0,0)
        unpaired_colour = (0.8,0.8,0)

        for i in range(self.nFields):
                if self.ImagePoints[CursorNumber][i] is not None:
                    pos = self.ImagePoints[CursorNumber][i][0].GetFocalPoint()

                    if Focus:
                        self.ImagePoints[CursorNumber][i][0].SetModelBounds(pos[0]-focus_size,pos[0]+focus_size,pos[1]-focus_size,pos[1]+focus_size,0.0,0.0)
                        self.ImagePoints[CursorNumber][i][2].GetProperty().SetColor(focus_colour)
                        self.ImagePoints[CursorNumber][i][2].GetProperty().SetLineWidth(focus_linewidth)
                    else:
                        self.ImagePoints[CursorNumber][i][0].SetModelBounds([pos[0]-nofocus_size,pos[0]+nofocus_size,pos[1]-nofocus_size,pos[1]+nofocus_size,0.,0.])
                        self.ImagePoints[CursorNumber][i][2].GetProperty().SetLineWidth(nofocus_linewidth)
                        if Paired:
                                self.ImagePoints[CursorNumber][i][2].GetProperty().SetColor(nofocus_colour)
                        else:
                                self.ImagePoints[CursorNumber][i][2].GetProperty().SetColor(unpaired_colour)

        if self.ReProjectedPoints is not None:
            for point in self.ReProjectedPoints:
                point[0].SetModelBounds([point[3][0]-nofocus_size,point[3][0]+nofocus_size,point[3][1]-nofocus_size,point[3][1]+nofocus_size,0.,0.])


    # Adjust 2D image size and cursor positions if the window is resized
    def OnWindowSizeAdjust(self,obg=None,event=None):
        # This is to stop this function erroneously running before
        # the interactor starts (apparently that was a thing??)
        if self.Interactor is not None:

                if self.Image is not None:

                    newWinSize = self.Window.GetSize()
                    winaspect = (float(newWinSize[0])/2)/float(newWinSize[1])

                    bounds = self.ImageActor.GetBounds()
                    xc = bounds[0] + bounds[1] / 2
                    yc = bounds[2] + bounds[3] / 2
                    ye = bounds[3] - bounds[2]
                    xe = bounds[1] - bounds[0]

                    im_aspect = xe / ye

                    if winaspect >= im_aspect:
                        # Base new zero size on y dimension
                        self.zoom_ref_scale = 0.5*ye
                    else:
                        self.zoom_ref_scale = 0.5*xe/winaspect


                    self.Camera2D.SetParallelScale(self.zoom_ref_scale / self.ZoomLevel)
                    self.WinSize = newWinSize



    
    # Function to convert display coordinates to pixel coordinates on the camera image
    def DisplayToImageCoords(self,DisplayCoords):

        camyscale = self.Camera2D.GetParallelScale() * 2.
        camxscale = camyscale * (self.WinSize[0]/2.)/self.WinSize[1]
        cc = self.Camera2D.GetFocalPoint()
        ImCoords = ( (( DisplayCoords[0] - 3.*self.WinSize[0]/4 ) / (self.WinSize[0]/2.)) * camxscale + cc[0], (DisplayCoords[1] - self.WinSize[1]/2.)/self.WinSize[1] * camyscale + cc[1] )

        return ImCoords



    # Function to convert image pixel coordinates to display coordinates
    def ImageToDisplayCoords(self,ImCoords):

        camyscale = self.Camera2D.GetParallelScale() * 2.
        camxscale = camyscale * (self.WinSize[0]/2.)/self.WinSize[1]
        cc = self.Camera2D.GetFocalPoint()

        DisplayCoords = ( float(ImCoords[0] - cc[0]) / camxscale * self.WinSize[0]/2. + 3.*self.WinSize[0]/4. , float(ImCoords[1] - cc[1]) / camyscale*self.WinSize[1] + self.WinSize[1]/2.)


        return DisplayCoords


    # Update the current point pairs FROM the current point pairs object.
    def UpdateFromPPObject(self,append):

        if not append:
                for i in range(len(self.ObjectPoints)):
                    self.DeletePoint(len(self.ObjectPoints)-1,False)
            
        for i in range(len(self.PointPairs.objectpoints)):

                self.AddPointPair()
                self.AddPoint3D(self.PointPairs.objectpoints[i])
                for field in range(self.nFields):
                    if self.PointPairs.imagepoints[i][field] is not None:
                        self.AddPoint2D([self.PointPairs.imagepoints[i][field][0],self.ImageOriginalSize[1] - self.PointPairs.imagepoints[i][field][1]])

        self.UpdateResults()
        self.update_current_point()
        self.update_n_points()
        self.gui_window.refresh_vtk()


    # Delete current point pair
    def DeletePoint(self,pointNumber,update_PPObj = True):

        if self.ObjectPoints[pointNumber] is not None:
                self.Renderer.RemoveActor(self.ObjectPoints[pointNumber][2])

        for i in range(self.nFields):
                if self.ImagePoints[pointNumber][i] is not None:
                    self.Renderer_2D.RemoveActor(self.ImagePoints[pointNumber][i][2])

        self.ImagePoints.remove(self.ImagePoints[pointNumber])
        self.ObjectPoints.remove(self.ObjectPoints[pointNumber])
            
        if len(self.ObjectPoints) > 0:
                if pointNumber == self.SelectedPoint:
                    self.SelectedPoint = (self.SelectedPoint - 1) % len(self.ObjectPoints)
                    self.CursorFocus(self.SelectedPoint)
        else:
                self.SelectedPoint = None

        if update_PPObj:
            self.UpdateResults()

        self.update_n_points()
        self.update_current_point()


    # Add a new point on the image
    def AddPoint2D(self,Imcoords):

        field = self.fieldmask[int(Imcoords[1]),int(Imcoords[0])]

        # Create new cursor, mapper and actor
        self.ImagePoints[self.SelectedPoint][field] = [vtk.vtkCursor3D(),vtk.vtkPolyDataMapper(),vtk.vtkActor(),Imcoords]
        
        # Some setup of the cursor
        self.ImagePoints[self.SelectedPoint][field][0].OutlineOff()
        self.ImagePoints[self.SelectedPoint][field][0].XShadowsOff()
        self.ImagePoints[self.SelectedPoint][field][0].YShadowsOff()
        self.ImagePoints[self.SelectedPoint][field][0].ZShadowsOff()
        self.ImagePoints[self.SelectedPoint][field][0].AxesOn()
        self.ImagePoints[self.SelectedPoint][field][0].TranslationModeOn()
        
        im_dims = self.Image.transform.get_display_shape()

        # Work out where to place the cursor
        worldpos = [Imcoords[0],Imcoords[1],0.1]

        self.ImagePoints[self.SelectedPoint][field][0].SetFocalPoint(worldpos)

        # Mapper setup
        self.ImagePoints[self.SelectedPoint][field][1].SetInputConnection(self.ImagePoints[self.SelectedPoint][field][0].GetOutputPort())

        # Actor setup
        self.ImagePoints[self.SelectedPoint][field][2].SetMapper(self.ImagePoints[self.SelectedPoint][field][1])

        # Add new cursor to screen
        self.Renderer_2D.AddActor(self.ImagePoints[self.SelectedPoint][field][2])

        self.Set2DCursorStyle(self.SelectedPoint,True)

        self.update_current_point()
        self.update_n_points()


    def AddPoint3D(self,Objcoords):

        # Create new cursor, mapper and actor
        self.ObjectPoints[self.SelectedPoint] = (vtk.vtkCursor3D(),vtk.vtkPolyDataMapper(),vtk.vtkActor())

        # Some setup of the cursor
        self.ObjectPoints[self.SelectedPoint][0].XShadowsOff()
        self.ObjectPoints[self.SelectedPoint][0].YShadowsOff()
        self.ObjectPoints[self.SelectedPoint][0].ZShadowsOff()
        self.ObjectPoints[self.SelectedPoint][0].OutlineOff()
        self.ObjectPoints[self.SelectedPoint][0].SetTranslationMode(1)
        self.ObjectPoints[self.SelectedPoint][0].SetFocalPoint(Objcoords[0],Objcoords[1],Objcoords[2])


        # Mapper setup
        self.ObjectPoints[self.SelectedPoint][1].SetInputConnection(self.ObjectPoints[self.SelectedPoint][0].GetOutputPort())
    
        # Actor setup
        self.ObjectPoints[self.SelectedPoint][2].SetMapper(self.ObjectPoints[self.SelectedPoint][1])

        # Add new cursor to screen
        self.Renderer.AddActor(self.ObjectPoints[self.SelectedPoint][2])

        self.Set3DCursorStyle(self.SelectedPoint,True)

        self.update_current_point()
        self.update_n_points()

    # Show the current CAD points re-projected on to the image
    # using the current fit.
    def ShowReprojectedPoints(self):

        camscale = self.Camera2D.GetParallelScale()
        cursorcolour = (0,0,1)
        cursorsize = 0.015 * camscale
        cursorlinewidth = 2

        points = self.FitResults.project_points(self.PointPairs.objectpoints,Coords = 'Display')

        self.ReProjectedPoints = []

        for field in range(self.FitResults.nfields):

                for point in points[field]:

                    if np.isnan(point[0]):
                        continue

                    # Flip y coordinate to match VTK
                    point = (point[0],self.ImageOriginalSize[1]-point[1])

                    # Create new cursor, mapper and actor
                    self.ReProjectedPoints.append((vtk.vtkCursor2D(),vtk.vtkPolyDataMapper(),vtk.vtkActor(),point))
                
                    # Some setup of the cursor
                    self.ReProjectedPoints[-1][0].OutlineOff()
                    self.ReProjectedPoints[-1][0].AxesOn()
                    self.ReProjectedPoints[-1][0].SetTranslationMode(1)
                
                    self.ReProjectedPoints[-1][0].SetFocalPoint((point[0],point[1],0.2))

                    # Mapper setup
                    self.ReProjectedPoints[-1][1].SetInputConnection(self.ReProjectedPoints[-1][0].GetOutputPort())

                    # Actor setup
                    self.ReProjectedPoints[-1][2].SetMapper(self.ReProjectedPoints[-1][1])

                    # Add new cursor to screen
                    self.Renderer_2D.AddActor(self.ReProjectedPoints[-1][2])

                    self.ReProjectedPoints[-1][0].SetModelBounds([point[0]-cursorsize,point[0]+cursorsize,point[1]-cursorsize,point[1]+cursorsize,0.,0.])
                    self.ReProjectedPoints[-1][2].GetProperty().SetColor(cursorcolour)
                    self.ReProjectedPoints[-1][2].GetProperty().SetLineWidth(cursorlinewidth)
                    self.ReProjectedPoints[-1][0].SetRadius(0.)

        self.gui_window.refresh_vtk()


    # Hide any re-projected points
    def HideReprojectedPoints(self):
                if self.ReProjectedPoints is not None:
                    for point in self.ReProjectedPoints:
                        self.Renderer_2D.RemoveActor(point[2])

                    self.ReProjectedPoints = None    
                    self.gui_window.refresh_vtk()


    # Put the current point pairs in to the results object, so they're not lost
    # if the user closes the window.
    def UpdateResults(self):

        self.PointPairs.imagepoints = []
        self.PointPairs.objectpoints = []

        for i in range(len(self.ObjectPoints)):
                if self.ObjectPoints[i] is not None and self.ImagePoints[i].count(None) < self.nFields:
                    self.PointPairs.objectpoints.append(self.ObjectPoints[i][0].GetFocalPoint())
                    self.PointPairs.imagepoints.append([])
                    for j in range(self.nFields):
                        if self.ImagePoints[i][j] is None:
                                self.PointPairs.imagepoints[-1].append(None)
                        else:
                                x = self.ImagePoints[i][j][3][0]
                                y = self.ImageOriginalSize[1] - self.ImagePoints[i][j][3][1]

                                self.PointPairs.imagepoints[-1].append([x,y])


    def set_view_to_fit(self,field):

        orig_dist = self.Camera3D.GetDistance()
        self.Camera3D.SetPosition(self.FitResults.get_pupilpos(field=field))
        self.Camera3D.SetFocalPoint(self.FitResults.get_pupilpos(field=field) + self.FitResults.get_los_direction(self.ImageOriginalSize[0]/2,self.ImageOriginalSize[1]/2,ForceField=field))
        self.Camera3D.SetDistance(orig_dist)
        self.Camera3D.SetViewAngle(self.FitResults.get_fov(field=field)[1])
        self.Camera3D.SetViewUp(-1.*self.FitResults.get_cam_to_lab_rotation(field)[:,1])
        # Update cursor sizes depending on their distance from the camera,
        # so they're all comfortably visible and clickable.
        for i in range(len(self.ObjectPoints)):
                if self.ObjectPoints[i] is not None:
                    self.Set3DCursorStyle(i,self.SelectedPoint == i,self.ImagePoints[i] is not None)
        
        self.gui_window.refresh_vtk()


    def get_view_target(self,zoom=False):


        campos = self.Camera3D.GetPosition()
        view_dir =  self.Camera3D.GetDirectionOfProjection()

        self.view_target = ( campos[0] + view_dir[0] , campos[1] + view_dir[1] , campos[2] + view_dir[2] )

        return self.view_target


    def remove_current_pointpair(self):
            self.DeletePoint(self.SelectedPoint)
            self.gui_window.refresh_vtk()


    def update_n_points(self):

        n_pairs = 0
        n_unpaired = 0

        for i in range(len(self.ObjectPoints)):
            if self.ObjectPoints[i] is not None or self.ImagePoints[i] is not None:
                if self.ObjectPoints[i] is not None and self.ImagePoints[i] is not None:
                    n_pairs = n_pairs + 1
                else:
                    n_unpaired = n_unpaired + 1

        self.gui_window.update_n_points(n_pairs,n_unpaired)


    # Custom mouse move event to enable middle click panning on both
    # CAD and image views.
    def mouse_move(self,obj,event):

        # If the mouse is over the 3D renderer, invoke its normal mouse move callback
        # and make sure the 2D image stops panning.
        if self.ChooseRenderer() == '3D':
            self.im_dragging = False
            self.OnMouseMove()
        else:
            self.OnLeftButtonUp()
            self.OnMiddleButtonUp()
            self.OnRightButtonUp()
            if self.im_dragging and self.ZoomLevel > 1:

                lastXYpos = self.Interactor.GetLastEventPosition() 
                xypos = self.Interactor.GetEventPosition()
                camscale = self.Camera2D.GetParallelScale() * 2
                oldpos = self.Camera2D.GetPosition()
                deltaX = (xypos[0] - lastXYpos[0])/(self.WinSize[0]/2.) * camscale * (self.WinSize[0]/2.)/self.WinSize[1]
                deltaY = (xypos[1] - lastXYpos[1])/self.WinSize[1] * camscale

                newY = oldpos[1] - deltaY
                newX = oldpos[0] - deltaX

                
                # Make sure we don't pan outside the image.
                im_bounds = self.ImageActor.GetBounds()
                xcamscale = (camscale * (self.WinSize[0]/2.)/self.WinSize[1])
                if newX + xcamscale/2. > im_bounds[1]:
                    newX = im_bounds[1] - xcamscale/2.
                elif newX - xcamscale/2. < im_bounds[0]:
                    newX = im_bounds[0] + xcamscale/2.

                if newY + camscale/2. > im_bounds[3]:
                    newY = im_bounds[3] - camscale/2.
                elif newY - camscale/2. < im_bounds[2]:
                    newY = im_bounds[2] + camscale/2.
                
                # Move image camera
                self.Camera2D.SetPosition((newX, newY,1.))
                self.Camera2D.SetFocalPoint((newX,newY,0.))

                self.gui_window.refresh_vtk(im_only=True)


'''
Note: ROIEditor is sort of legacy, but it's still here in this form
because I haven't managed to finish something better yet, so this
is all there is for ROI editing at the moment.
'''
class ROIEditor(vtk.vtkInteractorStyleTrackballCamera):
 
    def __init__(self,parent=None):
        # Set callbacks for all the controls
        self.AddObserver("LeftButtonPressEvent",self.OnLeftClick)
        self.AddObserver("RightButtonPressEvent",self.rightButtonPress)
        self.AddObserver("RightButtonReleaseEvent",self.rightButtonRelease)
        self.AddObserver("MiddleButtonPressEvent",self.middleButtonPress)
        self.AddObserver("MiddleButtonReleaseEvent",self.middleButtonRelease)
        self.AddObserver("MouseWheelForwardEvent",self.ZoomIn)
        self.AddObserver("MouseWheelBackwardEvent",self.ZoomOut)       
        self.AddObserver("KeyPressEvent",self.OnKey)

    # Do various initial setup things, most of which can't be done at the time of __init__
    def DoInit(self,renderer,CADmodel,ROIObject,FitResults=None,RayCaster=None):
       
        # Get the interactor object
        self.Interactor = self.GetInteractor()

        # Some other objects from higher up which I need access to
        self.CADmodel = CADmodel
        self.Window = self.Interactor.GetRenderWindow()
        self.Renderer = renderer
        self.Camera = self.Renderer.GetActiveCamera()
        self.Camera.SetPosition(0,0,0)
        self.Camera.SetFocalPoint(0,0,1)

        # Remove some interactor observers which interfere with my controls
        # NOTE: This only works in VTK 6! This will break some
        # user interaction in VTK 7! Really need to bring the ROI editing
        # up to date!
        self.Interactor.RemoveObserver(40)
        self.Interactor.RemoveObserver(41)
        self.Interactor.RemoveObserver(42)

        # Create a picker
        self.Picker = vtk.vtkCellPicker()
        self.Interactor.SetPicker(self.Picker)

        # We will use this for converting from 3D to screen coords.
        self.CoordTransformer = vtk.vtkCoordinate()
        self.CoordTransformer.SetCoordinateSystemToWorld()


        # Variables
        self.ZoomLevel = 1.
        self.Points = []
        self.ROIActor = None
        self.SelectedPoint = None

        # Adjust the camera target distance to make the CAD camera pivot on itself
        self.Camera.SetDistance(self.Camera.GetDistance()/10)

        self.CamFitResults = FitResults
        self.RayCaster = RayCaster


        # Add all the bits of the machine
        for Actor in self.CADmodel.get_vtkActors():
                renderer.AddActor(Actor)

        if self.CamFitResults is not None:
                # Camera setup based on real camera
                self.Camera.SetPosition(self.CamFitResults.get_pupilpos())
                self.Camera.SetFocalPoint(self.CamFitResults.get_pupilpos() + self.CamFitResults.get_los_direction(self.CamFitResults.image_display_shape[0]/2,self.CamFitResults.image_display_shape[1]/2))
                self.Camera.SetViewAngle(self.CamFitResults.get_fov()[1])
                self.Camera.SetViewUp(-1.*self.CamFitResults.get_cam_to_lab_rotation()[:,1])
        else:
                # Camera setup based on real camera
                self.Camera.SetPosition(CADmodel.cam_pos_default)
                self.Camera.SetFocalPoint(CADmodel.cam_target_default)
                self.Camera.SetViewAngle(CADmodel.cam_fov_default)
                self.Camera.SetViewUp((0,0,1))
        self.Camera.SetDistance(0.4)

        # Store the 'home' view
        self.CameraHome = [self.Camera.GetPosition(),self.Camera.GetFocalPoint(),self.Camera.GetViewUp(),self.Camera.GetViewAngle()]

        # ROI Object
        self.ROI = ROIObject
        for point in self.ROI.vertex_coords:
                self.AddPoint([point,len(self.Points)])
        
        for i in range(len(self.Points)):
                self.SetCursorStyle(i,i==self.SelectedPoint)

        self.UpdateShading()
        self.Window.Render()


    def free_references(self):
        del self.Interactor
        del self.Window
        del self.Renderer
        del self.Camera
        del self.Picker
        
        
    # On the CAD view, middle click + drag to pan
    def middleButtonPress(self,obj,event):
        self.Renderer.RemoveActor(self.ROIActor)
        self.OnMiddleButtonDown()


    def middleButtonRelease(self,obj,event):
        self.UpdateShading()
        self.OnMiddleButtonUp()
        


    # On the CAD view, right click+drag to rotate (usually on left button in this interactorstyle)
    def rightButtonPress(self,obj,event):
        self.Renderer.RemoveActor(self.ROIActor)
        self.OnLeftButtonDown()

    def rightButtonRelease(self,obj,event):
        # Since the user probably doesn't intend to roll the camera,
        # un-roll it automatically after any rotation action.
        self.Camera.SetViewUp(0,0,1)
        self.UpdateShading()
        self.OnLeftButtonUp()



    # Left click to move a point or add a new point
    def OnLeftClick(self,obj,event):

        # Check if they selected an existing point or a position on the model
        picktype,pickdata = self.GetSelected()

        # If they clicked a model position...
        if picktype == 'Position':

                # If the user was holding Ctrl, create a new point and select it.
                if self.Interactor.GetControlKey():
                    if self.SelectedPoint is not None:
                        self.CursorDefocus(self.SelectedPoint)        
                    self.AddPoint(pickdata)

                else:
                    if self.SelectedPoint is not None:
                        # Move the currently selected point to the new location
                        if self.Points[self.SelectedPoint] is not None:
                                self.Points[self.SelectedPoint][0].SetFocalPoint(pickdata[0][0],pickdata[0][1],pickdata[0][2])
                                # Check if the camera can see this point
                                if self.RayCaster is not None and self.CamFitResults is not None:
                                    CamPxCoords = self.CamFitResults.project_points([self.Points[self.SelectedPoint][0].GetFocalPoint()],CheckVisible=True,RayCaster=self.RayCaster,VisibilityMargin=0.07)
                                    self.Points[self.SelectedPoint][3] = np.isfinite(CamPxCoords[0][0][0])
                                self.SetCursorStyle(self.SelectedPoint,True)

                self.UpdateResults()
                self.UpdateShading()

        # If the user clicked on an existing point, select that point.
        if picktype == 'Cursor':
                self.CursorDefocus(self.SelectedPoint)
                self.SelectedPoint = pickdata
                self.CursorFocus(pickdata)


        self.Window.Render()

 


    def ZoomIn(self,obj,event):

        # If ctrl + scroll, change the camera FOV
        if self.Interactor.GetControlKey():
                self.Camera.SetViewAngle(self.Camera.GetViewAngle()*0.9)

        # Otherwise, move the camera forward.
        else:
                orig_dist = self.Camera.GetDistance()
                self.Camera.Dolly(1.5)
                self.Camera.SetDistance(orig_dist)

        # Update cursor sizes depending on their distance from the camera,
        # so they're all comfortably visible and clickable.
        for i in range(len(self.Points)):
                if self.Points[i] is not None:
                    self.SetCursorStyle(i,self.SelectedPoint == i)

        self.UpdateShading()
        self.Window.Render()

        return

    def ZoomOut(self,obj,event):

        # If ctrl + scroll, change the camera FOV
        if self.Interactor.GetControlKey():
                self.Camera.SetViewAngle(min(self.Camera.GetViewAngle()*1.1,100.))

        # Otherwise, move the camera backward.
        else:
                orig_dist = self.Camera.GetDistance()
                self.Camera.Dolly(0.75)
                self.Camera.SetDistance(orig_dist)

        # Update cursor sizes so they're all well visible:
        for i in range(len(self.Points)):
                if self.Points[i] is not None:
                    self.SetCursorStyle(i,self.SelectedPoint == i)
        
        self.UpdateShading()
        self.Window.Render()

        return

    # Key press actions handled in here!
    def OnKey(self,obj,event):

        keypressed = self.Interactor.GetKeySym().lower()

        # 'Delete' key deletes the selected point pair
        if keypressed == 'delete':
                if self.SelectedPoint is not None:
                    self.DeletePoint(self.SelectedPoint)
                    self.Window.Render()

        # 'Home' key resets camera to home position
        if keypressed == 'home':
                self.Camera.SetPosition(self.CameraHome[0])
                self.Camera.SetFocalPoint(self.CameraHome[1])
                self.Camera.SetViewUp(self.CameraHome[2])
                self.Camera.SetViewAngle(self.CameraHome[3])
                self.UpdateShading()
                self.Window.Render()

            # 's' Saves the ROI definition
        if keypressed == 's':
                print('')
                print('--- Save ROI Definition ---')
                print('')
                filename = raw_input('Enter save name for ROI: ')
                if filename != "":
                    self.ROI.save(filename)
                print('Done.')    
                print('')
                print('---------------------------')


    # Defocus cursors for a given point
    def CursorDefocus(self,CursorNumber):
        if self.Points[CursorNumber] is not None:
                self.SetCursorStyle(CursorNumber,False)


    # Bring cursors to focus for a given point
    def CursorFocus(self,CursorNumber):

        if self.Points[CursorNumber] is not None:
                self.SetCursorStyle(CursorNumber,True)



    # Check whether the user clicked an existing point or position
    # on the CAD model
    def GetSelected(self):

        # These will be the variables we return. If the user clicked in free space they will stay None.
        picktype = None
        pickdata = None

        # Hide ROI Actor so we don't accidentally pick it
        self.Renderer.RemoveActor(self.ROIActor)

        # Do a pick with our picker object
        clickcoords = self.Interactor.GetEventPosition()
        retval = self.Picker.Pick(clickcoords[0],clickcoords[1],0,self.Renderer)

        # Put the ROI actor back
        self.Renderer.AddActor(self.ROIActor)

        # If something was successfully picked, find out what it was...
        if retval != 0:

                pickedpoints = self.Picker.GetPickedPositions()

                # If more than 1 point is within the picker's tolerance,
                # use the one closest to the camera (this is most intuitive)
                npoints = pickedpoints.GetNumberOfPoints()
                dist_fromcam = []
                campos = self.Camera.GetPosition()

                for i in range(npoints):
                    point = pickedpoints.GetPoint(i)
                    dist_fromcam.append(np.sqrt( (campos[0] - point[0])**2 + (campos[1] - point[1])**2 + (campos[2] - point[2])**2 ))

                _, idx = min((val, idx) for (idx, val) in enumerate(dist_fromcam))

                pickcoords = pickedpoints.GetPoint(idx)
            

                # If the picked point is within 1.5x the cursor radius of any existing point,
                # say that the user clicked on that point

                for i in range(len(self.Points)):

                    # Get the on-screen position of this cursor
                    self.CoordTransformer.SetValue(self.Points[i][0].GetFocalPoint())
                    cursorpos = self.CoordTransformer.GetComputedDisplayValue(self.Renderer)
                    cursorpos_3D = self.Points[i][0].GetFocalPoint()
                
                    cursortol = 7
                    cursortol_3D = 0.1
                    dist_from_cursor = np.sqrt( (cursorpos[0] - clickcoords[0])**2 + (cursorpos[1] - clickcoords[1])**2 )
                    dist_from_cursor_3D = np.sqrt( (cursorpos_3D[0] - pickcoords[0])**2 + (cursorpos_3D[1] - pickcoords[1])**2 + (cursorpos_3D[2] - pickcoords[2])**2 )

                    if dist_from_cursor < cursortol and dist_from_cursor_3D < cursortol_3D and i != self.SelectedPoint:
                        picktype = 'Cursor'
                        pickdata = i


                # If this is going to be a new point, choose between which two existing points
                # to insert it.
                insertion_point = len(self.Points)
                if len(self.Points) > 2:
                    mindist = 10000
                    for lineno in range(len(self.Points)-1):
                        p0 = self.Points[lineno][0].GetFocalPoint()
                        p1 = self.Points[lineno + 1][0].GetFocalPoint()
                        linevec = np.array([p1[0] - p0[0],p1[1] - p0[1],p1[2] - p0[2]])
                        posvec = np.array([pickcoords[0] - p0[0],pickcoords[1]-p0[1],pickcoords[2]-p0[2]])
                        reldist = np.dot(linevec,posvec)/np.dot(linevec,linevec)
                        if reldist > 0 and reldist < 1:
                                closestpoint = p0 + linevec * reldist
                                dist = np.sqrt(np.sum((pickcoords - closestpoint)**2))
                                if dist < mindist:
                                    insertion_point = lineno + 1
                                    mindist = dist

                    p0 = self.Points[-1][0].GetFocalPoint()
                    p1 = self.Points[0][0].GetFocalPoint()
                    linevec = np.array([p1[0] - p0[0],p1[1] - p0[1],p1[2] - p0[2]])
                    posvec = np.array([pickcoords[0] - p0[0],pickcoords[1]-p0[1],pickcoords[2]-p0[2]])
                    reldist = np.dot(linevec,posvec)/np.dot(linevec,linevec)
                    if reldist > 0 and reldist < 1:
                        closestpoint = p0 + linevec * reldist
                        dist = np.sqrt(np.sum((pickcoords - closestpoint)**2))
                        if dist < mindist:
                                insertion_point = len(self.Points)
                

                # If they didn't click on an existing point, they clicked a model position.
                if picktype is None:
                    picktype = 'Position'
                    pickdata = [pickcoords,insertion_point]


        return picktype,pickdata


    # Set the visual style of cursors on the CAD view
    def SetCursorStyle(self,CursorNumber,Focus):

        # Cursor appearance settings - mess with this as per your tastes.
        # The actual size & line width numbers correspond to 3D cursor side length in CAD units for a 75 degree FOV.
        focus_size = 0.025
        nofocus_size = 0.0125

        focus_linewidth = 3
        nofocus_linewidth = 2

        focus_colour = (0,0.8,0)
        nofocus_colour = (0.8,0,0)
        hidden_colour = (0.8,0.8,0) 


        # Cursor size scales with camera FOV to maintain size on screen.
        focus_size = focus_size * (self.Camera.GetViewAngle()/75)
        nofocus_size = nofocus_size * (self.Camera.GetViewAngle()/75)

        point = self.Points[CursorNumber][0].GetFocalPoint()
        campos = self.Camera.GetPosition()
        dist_to_cam = np.sqrt( (campos[0] - point[0])**2 + (campos[1] - point[1])**2 + (campos[2] - point[2])**2 )

        if Focus:
                self.Points[CursorNumber][0].SetModelBounds([point[0]-focus_size*dist_to_cam,point[0]+focus_size*dist_to_cam,point[1]-focus_size*dist_to_cam,point[1]+focus_size*dist_to_cam,point[2]-focus_size*dist_to_cam,point[2]+focus_size*dist_to_cam])
                self.Points[CursorNumber][2].GetProperty().SetColor(focus_colour)
                self.Points[CursorNumber][2].GetProperty().SetLineWidth(focus_linewidth)
        else:
                self.Points[CursorNumber][0].SetModelBounds([point[0]-nofocus_size*dist_to_cam,point[0]+nofocus_size*dist_to_cam,point[1]-nofocus_size*dist_to_cam,point[1]+nofocus_size*dist_to_cam,point[2]-nofocus_size*dist_to_cam,point[2]+nofocus_size*dist_to_cam])
                self.Points[CursorNumber][2].GetProperty().SetLineWidth(nofocus_linewidth)
                self.Points[CursorNumber][2].GetProperty().SetColor(nofocus_colour)
            
        if not self.Points[CursorNumber][3]:
                self.Points[CursorNumber][2].GetProperty().SetColor(hidden_colour)



    # Delete current point
    def DeletePoint(self,pointNumber):

        self.Renderer.RemoveActor(self.Points[pointNumber][2])

        self.Points.remove(self.Points[pointNumber])
            
        if len(self.Points) > 0:
                if pointNumber == self.SelectedPoint:
                    self.SelectedPoint = (self.SelectedPoint - 1) % len(self.Points)
                    self.CursorFocus(self.SelectedPoint)
        else:
                self.Points = []
                self.SelectedPoint = None

        self.UpdateResults()
        self.UpdateShading()


    def AddPoint(self,SelectionData):

        # Create new cursor, mapper and actor
        self.Points.insert(SelectionData[1],[vtk.vtkCursor3D(),vtk.vtkPolyDataMapper(),vtk.vtkActor(),True])
        self.SelectedPoint = SelectionData[1]

        # Some setup of the cursor
        self.Points[self.SelectedPoint][0].XShadowsOff()
        self.Points[self.SelectedPoint][0].YShadowsOff()
        self.Points[self.SelectedPoint][0].ZShadowsOff()
        self.Points[self.SelectedPoint][0].OutlineOff()
        self.Points[self.SelectedPoint][0].SetTranslationMode(1)
        self.Points[self.SelectedPoint][0].SetFocalPoint(SelectionData[0][0],SelectionData[0][1],SelectionData[0][2])

        # Mapper setup
        self.Points[self.SelectedPoint][1].SetInputConnection(self.Points[SelectionData[1]][0].GetOutputPort())
    
        # Actor setup
        self.Points[self.SelectedPoint][2].SetMapper(self.Points[self.SelectedPoint][1])

        # Add new cursor to screen
        self.Renderer.AddActor(self.Points[self.SelectedPoint][2])

        # Check if the camera can see this point
        if self.RayCaster is not None and self.CamFitResults is not None:
                CamPxCoords = self.CamFitResults.project_points([self.Points[self.SelectedPoint][0].GetFocalPoint()],CheckVisible=True,RayCaster=self.RayCaster,VisibilityMargin=0.07)
                self.Points[self.SelectedPoint][3] = np.isfinite(CamPxCoords[0][0][0])

        self.SetCursorStyle(self.SelectedPoint,True)
        self.UpdateResults()

    def UpdateShading(self):

        Shading_Colour = (0,0.8,0)
        Shading_Opacity = 0.3

        self.Renderer.RemoveActor(self.ROIActor)

        if self.Points is not None:
                if len(self.Points) > 2:
                    self.ROIActor = self.ROI.get_vtkActor(self.Camera.GetPosition())
                    self.ROIActor.GetProperty().SetColor(Shading_Colour)
                    self.ROIActor.GetProperty().SetOpacity(Shading_Opacity)
                    self.Renderer.AddActor(self.ROIActor)


    def UpdateResults(self):
        self.ROI.vertex_coords = []
        for point in self.Points:
                    self.ROI.vertex_coords.append(point[0].GetFocalPoint())


class SplitFieldEditor(vtk.vtkInteractorStyleTrackballCamera):


    def __init__(self,parent=None):
        # Set callbacks for all the controls
        self.AddObserver("LeftButtonPressEvent",self.OnLeftClick)
        self.AddObserver("MiddleButtonPressEvent",self.middleButtonPress)
        self.AddObserver("MiddleButtonReleaseEvent",self.middleButtonRelease)
        self.AddObserver("MouseWheelForwardEvent",self.ZoomIn)
        self.AddObserver("MouseWheelBackwardEvent",self.ZoomOut)
        self.AddObserver("MouseMoveEvent",self.mouse_move)     

    # Do various initial setup things, most of which can't be done at the time of __init__
    def DoInit(self,renderer,Image,parent_window,overlay_opacity=0.25):
       
        # Get the interactor object
        self.Interactor = self.GetInteractor()

        self.im_dragging = False

        # Some other objects from higher up which I need access to
        self.Window = self.Interactor.GetRenderWindow()
        self.Renderer = renderer
        self.camera = self.Renderer.GetActiveCamera()
        self.camera.ParallelProjectionOn()
        self.Image = Image
        self.ImageActor = self.Image.get_vtkActor()
        self.overlay_actor = None
        self.overlay_image = None
        self.set_mask_opacity(overlay_opacity)
        self.window = parent_window


        # Turn off any VTK responses to keyboard input (all necessary keyboard shortcuts etc are done in Qt)
        self.Interactor.RemoveObservers('KeyPressEvent')
        self.Interactor.RemoveObservers('CharEvent')
        self.Interactor.RemoveObservers('RightButtonReleaseEvent')
        self.Interactor.RemoveObservers('RightButtonPressEvent')

        # Add observer for catching window resizing
        self.Window.AddObserver("ModifiedEvent",self.OnWindowSizeAdjust)

        # Variables
        self.ZoomLevel = 1.
        self.SelectedPoint = None
        self.Points = []
        self.WinSize = self.Window.GetSize()

        bounds = self.ImageActor.GetBounds()
        xc = bounds[0] + bounds[1] / 2
        yc = bounds[2] + bounds[3] / 2
        ye = bounds[3] - bounds[2]
        xe = bounds[1] - bounds[0]

        self.zoom_ref_cc = (xc,yc)

        im_aspect = xe / ye
        winaspect = float(self.WinSize[0])/float(self.WinSize[1])

        if winaspect >= im_aspect:
            # Base new zero size on y dimension
            self.zoom_ref_scale = 0.5*ye
        else:
            self.zoom_ref_scale = 0.5*xe/winaspect
    
        self.camera.SetParallelScale(self.zoom_ref_scale)
        self.camera.SetPosition(xc,yc,1.)
        self.camera.SetFocalPoint(xc,yc,0.)
        

    def free_references(self):
        del self.Interactor
        del self.Window
        del self.Renderer

    # Left click to move a point or add a new point
    def OnLeftClick(self,obj,event):

        if not self.click_enabled:
            return
            
        pickdata,closest_cursor = self.GetSelected()
        if pickdata is None:
            return

        if len(self.Points) < 2:
            self.AddPoint(pickdata)

        else:

            if closest_cursor != self.SelectedPoint:
                self.CursorDefocus(self.SelectedPoint)
                self.SelectedPoint = closest_cursor
                self.CursorFocus(closest_cursor)

            # Move the 2D point: first store the image coords
            self.Points[self.SelectedPoint][3] = pickdata
            # Because the image view is still a 3D renderer, we have to convert our 2D image coordinate
            # in to a 3D location to place the cursor there
            imshape = self.Image.transform.get_display_shape()
            self.Points[self.SelectedPoint][0].SetFocalPoint(pickdata[0],imshape[1] -pickdata[1],0.01)
            self.update_fieldmask()


        self.Window.Render()

        return
 

    def ZoomIn(self,obj,event):


        # Re-position the image camera to keep the image point under the mouse pointer fixed when zooming
        zoom_ratio = 1 + 0.2/self.ZoomLevel                 # The zoom ratio we will have

        # Current camera position and scale
        campos = self.camera.GetPosition()
        camscale = self.camera.GetParallelScale() * 2.

        # Where is the zoom target in world coordinates?

        # Zoom coordinates in window pixels
        zoomcoords = list(self.Interactor.GetEventPosition())
        zoomcoords[0] = zoomcoords[0] - self.WinSize[0]/2.

        # Position of current centre from to where we're zooming
        zoomcoords = ( (zoomcoords[0] - (self.WinSize[0]/4.))/(self.WinSize[0]/2.) * camscale * (float(self.WinSize[0])/2.)/float(self.WinSize[1]) + campos[0],
                       (zoomcoords[1] - self.WinSize[1]/2.)/self.WinSize[1] * camscale + campos[1] )

        # Vector from zoom point to current camera centre
        zoomvec = ( campos[0] - zoomcoords[0] , campos[1] - zoomcoords[1] )

        # Now we move the camera along the line bwteen the current camera centre and zoom position
        newxc = zoomvec[0]/zoom_ratio + zoomcoords[0]
        newyc = zoomvec[1]/zoom_ratio + zoomcoords[1]

        # Actually move the camera
        self.camera.SetPosition((newxc,newyc,1.))
        self.camera.SetFocalPoint((newxc,newyc,0.))

        # Actually zoom in.
        self.ZoomLevel = self.ZoomLevel + 0.2
        self.camera.SetParallelScale(self.zoom_ref_scale / self.ZoomLevel)

        for point in range(len(self.Points)):
            self.SetCursorStyle(point,self.SelectedPoint == point)

        self.window.qvtkWidget.update()



    def ZoomOut(self,obj,event):

        # Zoom out smoothly until the whole image is visible
        if self.ZoomLevel > 1.:

            zoom_ratio = 0.2/(self.ZoomLevel**2 - 1.2*self.ZoomLevel + 0.2)

            campos = self.camera.GetPosition()

            zoomvec = ( self.zoom_ref_cc[0] - campos[0] , self.zoom_ref_cc[1] - campos[1] )

            self.camera.SetPosition((campos[0] + zoomvec[0] * zoom_ratio, campos[1] +  zoomvec[1] * zoom_ratio, 1.))
            self.camera.SetFocalPoint((campos[0] + zoomvec[0] * zoom_ratio, campos[1] +  zoomvec[1] * zoom_ratio, 0.))

            self.ZoomLevel = self.ZoomLevel - 0.2
            self.camera.SetParallelScale(self.zoom_ref_scale / self.ZoomLevel)

            for point in range(len(self.Points)):
                self.SetCursorStyle(point,self.SelectedPoint == point)

            self.window.qvtkWidget.update()





    def mouse_move(self,obj,event):


        if self.im_dragging and self.ZoomLevel > 1:

            lastXYpos = self.Interactor.GetLastEventPosition() 
            xypos = self.Interactor.GetEventPosition()
            camscale = self.camera.GetParallelScale() * 2
            oldpos = self.camera.GetPosition()
            deltaX = (xypos[0] - lastXYpos[0])/(self.WinSize[0]/2.) * camscale * (self.WinSize[0]/2.)/self.WinSize[1]
            deltaY = (xypos[1] - lastXYpos[1])/self.WinSize[1] * camscale

            newY = oldpos[1] - deltaY
            newX = oldpos[0] - deltaX

            
            # Make sure we don't pan outside the image.
            im_bounds = self.ImageActor.GetBounds()
            xcamscale = (camscale * (self.WinSize[0]/2.)/self.WinSize[1])
            if newX + xcamscale/2. > im_bounds[1]:
                newX = im_bounds[1] - xcamscale/2.
            elif newX - xcamscale/2. < im_bounds[0]:
                newX = im_bounds[0] + xcamscale/2.

            if newY + camscale/2. > im_bounds[3]:
                newY = im_bounds[3] - camscale/2.
            elif newY - camscale/2. < im_bounds[2]:
                newY = im_bounds[2] + camscale/2.
            
            # Move image camera
            self.camera.SetPosition((newX, newY,1.))
            self.camera.SetFocalPoint((newX,newY,0.))

            self.Window.Render()




    # Defocus cursors for a given point pair
    def CursorDefocus(self,CursorNumber):

        if self.Points[CursorNumber][0] is not None:
                self.SetCursorStyle(CursorNumber,False)

    # Bring cursors to focus for a given point pair
    def CursorFocus(self,CursorNumber):

        if self.Points[CursorNumber][0] is not None:
                self.SetCursorStyle(CursorNumber,True)


    def middleButtonPress(self,obj,event):
        self.im_dragging = True

    def middleButtonRelease(self,obj,event):
        self.im_dragging = False


    # Check whether the user clicked an existing cursor on the image
    # or just somewhere else
    # Very similar to GetSelected_3D
    def GetSelected(self):

        pickdata = None
        closest_cursor = None

        clickcoords = self.Interactor.GetEventPosition()
        imcoords = self.DisplayToImageCoords(clickcoords)

        # Check if the click was near enough an existing cursor to be considered as clicking it
        min_dist = 1e5
        for i in range(len(self.Points)):
            dist_from_cursor = np.sqrt( (self.Points[i][3][0] - imcoords[0])**2 + (self.Points[i][3][1] - imcoords[1])**2 )
            if dist_from_cursor < min_dist:
                closest_cursor = i
                min_dist = dist_from_cursor


        
        imshape = self.Image.transform.get_display_shape()
        if np.all(np.array(imcoords) > 0.) and np.all(np.array(imcoords) < np.array(imshape)):
            pickdata = imcoords

        return pickdata,closest_cursor





    def SetCursorStyle(self,CursorNumber,Focus):
        
        camscale = self.camera.GetParallelScale()
        focus_size = 0.03 * camscale 
        nofocus_size = 0.015 * camscale

        focus_linewidth = 3
        nofocus_linewidth = 2

        focus_colour = (0,0.8,0)
        nofocus_colour = (0.8,0,0)

        pos = self.Points[CursorNumber][0].GetFocalPoint()

        if Focus:
            self.Points[CursorNumber][0].SetModelBounds([pos[0]-focus_size,pos[0]+focus_size,pos[1]-focus_size,pos[1]+focus_size,0.,0.])
            self.Points[CursorNumber][2].GetProperty().SetColor(focus_colour)
            self.Points[CursorNumber][2].GetProperty().SetLineWidth(focus_linewidth)
        else:
            self.Points[CursorNumber][0].SetModelBounds([pos[0]-nofocus_size,pos[0]+nofocus_size,pos[1]-nofocus_size,pos[1]+nofocus_size,0.,0.])
            self.Points[CursorNumber][2].GetProperty().SetLineWidth(nofocus_linewidth)
            self.Points[CursorNumber][2].GetProperty().SetColor(nofocus_colour)






    # Adjust 2D image size and cursor positions if the window is resized
    def OnWindowSizeAdjust(self,obg=None,event=None):
        # This is to stop this function erroneously running before
        # the interactor starts (apparently that was a thing??)
        if self.Interactor is not None:

            newWinSize = self.Window.GetSize()
            winaspect = float(newWinSize[0])/float(newWinSize[1])

            bounds = self.ImageActor.GetBounds()
            xc = bounds[0] + bounds[1] / 2
            yc = bounds[2] + bounds[3] / 2
            ye = bounds[3] - bounds[2]
            xe = bounds[1] - bounds[0]

            im_aspect = xe / ye

            if winaspect >= im_aspect:
                # Base new zero size on y dimension
                self.zoom_ref_scale = 0.5*ye
            else:
                self.zoom_ref_scale = 0.5*xe/winaspect


            self.camera.SetParallelScale(self.zoom_ref_scale / self.ZoomLevel)
            self.WinSize = newWinSize
                      

    def DoNothing(self):
        pass

    
    # Function to convert display coordinates to pixel coordinates on the camera image
    def DisplayToImageCoords(self,DisplayCoords):

        camyscale = self.camera.GetParallelScale() * 2.
        camxscale = camyscale * float(self.WinSize[0])/self.WinSize[1]
        cc = self.camera.GetFocalPoint()
        ImCoords = [ (( DisplayCoords[0] - self.WinSize[0]/2. ) / (self.WinSize[0])) * camxscale + cc[0], (DisplayCoords[1] - self.WinSize[1]/2.)/self.WinSize[1] * camyscale + cc[1] ]
        imshape = self.Image.transform.get_display_shape()
        ImCoords[1]  = imshape[1] - ImCoords[1]

        return tuple(ImCoords)



    # Function to convert image pixel coordinates to display coordinates
    def ImageToDisplayCoords(self,ImCoords):

        camyscale = self.camera.GetParallelScale() * 2.
        camxscale = camyscale * float(self.WinSize[0])/self.WinSize[1]
        cc = self.camera.GetFocalPoint()

        DisplayCoords = ( float(ImCoords[0] - cc[0]) / camxscale * self.WinSize[0] + self.WinSize[0]/2. , cc[1] - float(ImCoords[1]) / camyscale*self.WinSize[1] + self.WinSize[1]/2.)

        return DisplayCoords



    # Delete current point pair
    def DeletePoint(self,pointNumber):

        if self.Points[pointNumber][0] is not None:
                self.Renderer.RemoveActor(self.Points[pointNumber][2])

        self.Points.remove(self.Points[pointNumber])
            
        if len(self.Points) > 0:
                if pointNumber == self.SelectedPoint:
                    self.SelectedPoint = (self.SelectedPoint - 1) % len(self.Points)
                    self.CursorFocus(self.SelectedPoint)
        else:
                self.Points = []
                self.SelectedPoint = None

        self.update_fieldmask()


    # Add a new point on the image
    def AddPoint(self,Imcoords):

        if self.SelectedPoint is not None:
                self.CursorDefocus(self.SelectedPoint)

        self.SelectedPoint = len(self.Points)

        self.Points.append([vtk.vtkCursor3D(),vtk.vtkPolyDataMapper(),vtk.vtkActor(),Imcoords])
        
        # Some setup of the cursor
        self.Points[self.SelectedPoint][0].AxesOn()
        self.Points[self.SelectedPoint][0].OutlineOff()
        self.Points[self.SelectedPoint][0].XShadowsOff()
        self.Points[self.SelectedPoint][0].YShadowsOff()
        self.Points[self.SelectedPoint][0].ZShadowsOff()
        self.Points[self.SelectedPoint][0].SetTranslationMode(1)
        
        # Place it in the right place.
        imshape = self.Image.transform.get_display_shape()
        self.Points[self.SelectedPoint][0].SetFocalPoint(Imcoords[0],imshape[1] - Imcoords[1],0.02)

        # Mapper setup
        self.Points[self.SelectedPoint][1].SetInputConnection(self.Points[self.SelectedPoint][0].GetOutputPort())

        # Actor setup
        self.Points[self.SelectedPoint][2].SetMapper(self.Points[self.SelectedPoint][1])

        # Add new cursor to screen
        self.Renderer.AddActor(self.Points[self.SelectedPoint][2])

        self.SetCursorStyle(self.SelectedPoint,True)
        self.update_fieldmask()


    def get_points(self):
        points = []
        for point in self.Points:
                x,y = self.Image.transform.display_to_original_coords(point[3][0],point[3][1])
                points.append([x,y])
        return points



    def update_fieldmask(self,mask=None,names=[]):

        # If we're doing this from points...
        if mask is None:
            points = self.get_points()

            if len(points) == 2:

                m = (points[1][1] - points[0][1])/(points[1][0] - points[0][0])
                c = points[1][1] - m*points[1][0]

                if c < 0:
                    side1 = 0
                elif c > self.Image.data.shape[0]-1:
                    side1 = 2
                else:
                    side1 = 1

                c2  = c + m * self.Image.data.shape[1]-1

                if c2 < 0:
                    side2 = 0
                elif c2 > self.Image.data.shape[0]-1:
                    side2 = 2
                else:
                    side2 = 3


                if side1 == 1 and side2 == 0:
                    pathpoints = np.zeros([3,2]) - 1
                    pathpoints[0,1] = c
                    pathpoints[1,0] = -c/m

                elif side1 == 1 and side2 == 3:
                    pathpoints = np.zeros([5,2]) - 1
                    pathpoints[0,1] = c
                    pathpoints[1,1] = self.Image.data.shape[0]
                    pathpoints[2,0] = self.Image.data.shape[1]
                    pathpoints[2,1] = self.Image.data.shape[0]
                    pathpoints[3,0] = self.Image.data.shape[1]
                    pathpoints[3,1] = c2

                elif side1 == 1 and side2 == 2:
                    pathpoints = np.zeros([4,2]) - 1
                    pathpoints[0,1] = c
                    pathpoints[1,0] = (self.Image.data.shape[0]-c)/m
                    pathpoints[1,1] = self.Image.data.shape[0]
                    pathpoints[2,0] = self.Image.data.shape[1]
                    pathpoints[2,1] = self.Image.data.shape[0]

                elif side1 == 2 and side2 == 3:
                    pathpoints = np.zeros([4,2]) - 1
                    pathpoints[0,1] = self.Image.data.shape[0]
                    pathpoints[0,0] = (self.Image.data.shape[0]-c)/m
                    pathpoints[1,1] = self.Image.data.shape[0]
                    pathpoints[1,0] = self.Image.data.shape[1]
                    pathpoints[2,0] = self.Image.data.shape[1]
                    pathpoints[2,1] = c2

                elif side1 == 0 and side2 == 3:
                    pathpoints = np.zeros([4,2]) - 1
                    pathpoints[0,0] = -c/m
                    pathpoints[1,0] = self.Image.data.shape[1]
                    pathpoints[2,0] = self.Image.data.shape[1]
                    pathpoints[2,1] = c2

                else:
                    pathpoints = np.zeros([5,2]) - 1
                    pathpoints[0,1] = self.Image.data.shape[0]
                    pathpoints[0,0] = (self.Image.data.shape[0]-c)/m
                    pathpoints[1,0] = self.Image.data.shape[1]
                    pathpoints[1,1] = self.Image.data.shape[0]
                    pathpoints[2,0] =self.Image.data.shape[1]
                    pathpoints[3,0] = -c/m

                path = mplPath.Path(pathpoints,closed=True)

                x,y = np.meshgrid(np.linspace(0,self.Image.data.shape[1]-1,self.Image.data.shape[1]),np.linspace(0,self.Image.data.shape[0]-1,self.Image.data.shape[0]))
                x = np.reshape(x,[np.size(x),1])
                y = np.reshape(y,[np.size(y),1])
                impoints = np.hstack((x,y))

                mask = path.contains_points(impoints)

                self.fieldmask = np.uint8(np.reshape(mask,self.Image.data.shape[0:2]))
                n_fields = np.max(self.fieldmask) + 1
            else:
                return

        # or if we're doing it from a mask..
        else:

            if mask.shape[0:2] != self.Image.data.shape[0:2]:
                if mask.shape[0] == self.Image.transform.get_display_shape()[1] and mask.shape[1] == self.Image.transform.get_display_shape()[0]:
                    mask = self.Image.transform.display_to_original_image(mask)
                else:
                    raise ValueError('Provided field mask is the wrong shape for this image!')

            if len(mask.shape) > 2:
                for channel in range(mask.shape[2]):
                    mask[:,:,channel] = mask[:,:,channel] * 2**channel
                mask = np.sum(mask,axis=2)

            lookup = list(np.unique(mask))
            n_fields = len(lookup)

            for value in lookup:
                mask[mask == value] = lookup.index(value)
            self.fieldmask = np.uint8(mask)

        if self.overlay_actor is not None:
            self.Renderer.RemoveActor(self.overlay_actor)
            self.overlay_actor = None
            self.overlay_resizer = None
        
        self.field_colours = []

        fieldmask = self.Image.transform.original_to_display_image(self.fieldmask)

        if n_fields > 1:
    
            colours = [matplotlib.cm.get_cmap('nipy_spectral')(i) for i in np.linspace(0.1,0.9,n_fields)]
            
            for colour in colours:
                self.field_colours.append( np.uint8(np.array(colour) * 255) )
            shape = self.Image.transform.get_display_shape()
            x = shape[0]
            y = shape[1]

            mask_image = np.zeros([y,x,4],dtype=np.uint8)

            for field in range(n_fields):
                inds = np.where(fieldmask == field)
                mask_image[inds[0],inds[1],0] = self.field_colours[field][0]
                mask_image[inds[0],inds[1],1] = self.field_colours[field][1]
                mask_image[inds[0],inds[1],2] = self.field_colours[field][2]

            mask_image[:,:,3] = self.overlay_opacity

            self.overlay_image = mask_image
            self.overlay_actor = image.from_array(mask_image).get_vtkActor()
            self.overlay_actor.SetPosition(0.,0.,0.01)
            self.Renderer.AddActor(self.overlay_actor)        
        self.Renderer.Render()


        if len(names) != n_fields:
            names = []
            if n_fields == 2:

                    xpx = np.zeros(2)
                    ypx = np.zeros(2)

                    for field in range(n_fields):

                        # Get CoM of this field on the chip
                        x,y = np.meshgrid(np.linspace(0,self.Image.transform.get_display_shape()[0]-1,self.Image.transform.get_display_shape()[0]),np.linspace(0,self.Image.transform.get_display_shape()[1]-1,self.Image.transform.get_display_shape()[1]))
                        x[fieldmask != field] = 0
                        y[fieldmask != field] = 0
                        xpx[field] = np.sum(x)/np.count_nonzero(x)
                        ypx[field] = np.sum(y)/np.count_nonzero(y)

                    names = ['','']

                    if ypx.max() - ypx.min() > 20:
                        names[np.argmin(ypx)] = 'Upper '
                        names[np.argmax(ypx)] = 'Lower '

                    if xpx.max() - xpx.min() > 20:
                        names[np.argmax(xpx)] = names[np.argmax(xpx)] + 'Right '
                        names[np.argmin(xpx)] = names[np.argmin(xpx)] + 'Left '

                    if names == ['','']:
                        names.append('Sub FOV # 1', 'Sub FOV # 2')
                    else:
                        names[0] = names[0] + 'View'
                        names[1] = names[1] + 'View'

            elif n_fields > 2:
                names = []
                for field in range(n_fields):
                    names.append('Sub FOV # ' + str(field + 1))

            elif n_fields == 1:
                names = ['Image']

        self.window.update_fieldnames_gui(n_fields,self.field_colours,names)


    def set_mask_opacity(self,opacity):
        self.overlay_opacity = np.uint8(255 * opacity)

        if self.overlay_image is not None:
            self.overlay_image[:,:,3] = self.overlay_opacity
            self.Renderer.RemoveActor(self.overlay_actor)
            self.overlay_actor = image.from_array(self.overlay_image).get_vtkActor()
            self.Renderer.AddActor(self.overlay_actor)    
            self.Renderer.Render()              




class ROISetEditor(vtk.vtkInteractorStyleTrackballCamera):
 
    def __init__(self,parent=None):
        # Set callbacks for all the controls
        self.AddObserver("LeftButtonPressEvent",self.OnLeftClick)
        self.AddObserver("RightButtonPressEvent",self.rightButtonPress)
        self.AddObserver("RightButtonReleaseEvent",self.rightButtonRelease)
        self.AddObserver("MiddleButtonPressEvent",self.middleButtonPress)
        self.AddObserver("MiddleButtonReleaseEvent",self.middleButtonRelease)
        self.AddObserver("MouseWheelForwardEvent",self.ZoomIn)
        self.AddObserver("MouseWheelBackwardEvent",self.ZoomOut)       
        self.AddObserver("KeyPressEvent",self.OnKey)

    # Do various initial setup things, most of which can't be done at the time of __init__
    def DoInit(self,renderer,CADmodel,ROISetObject,FitResults=None,RayCaster=None):
       
        # Get the interactor object
        self.Interactor = self.GetInteractor()

        # Some other objects from higher up which I need access to
        self.CADmodel = CADmodel
        self.Window = self.Interactor.GetRenderWindow()
        self.Renderer = renderer
        self.Camera = self.Renderer.GetActiveCamera()


        # Create a picker
        self.Picker = vtk.vtkCellPicker()
        self.Interactor.SetPicker(self.Picker)

        # We will use this for converting from 3D to screen coords.
        self.CoordTransformer = vtk.vtkCoordinate()
        self.CoordTransformer.SetCoordinateSystemToWorld()

        self.select_roi_mode = False

        # Variables
        self.ZoomLevel = 1.
        self.Points = []
        self.ROIActor = None
        self.BackgroundROIActors = []
        self.SelectedPoint = None
        self.selected_roi = 0
        self.roi_set = ROISetObject

        self.CamFitResults = FitResults
        self.RayCaster = RayCaster


        # Add all the bits of the machine
        for Actor in self.CADmodel.get_vtkActors():
                renderer.AddActor(Actor)

        if self.CamFitResults is not None:
                # Camera setup based on real camera
                self.Camera.SetPosition(self.CamFitResults.get_pupilpos())
                self.Camera.SetFocalPoint(self.CamFitResults.get_pupilpos() + self.CamFitResults.get_los_direction(self.CamFitResults.image_display_shape[0]/2,self.CamFitResults.image_display_shape[1]/2))
                self.Camera.SetViewAngle(self.CamFitResults.get_fov()[1])
                self.Camera.SetViewUp(-1.*self.CamFitResults.get_cam_to_lab_rotation()[:,1])
        else:
                # Camera setup based on model defaults
                self.Camera.SetPosition(CADmodel.cam_pos_default)
                self.Camera.SetFocalPoint(CADmodel.cam_target_default)
                self.Camera.SetViewAngle(CADmodel.cam_fov_default)
                self.Camera.SetViewUp((0,0,1))
        self.Camera.SetDistance(0.4)

        # Store the 'home' view
        self.CameraHome = [self.Camera.GetPosition(),self.Camera.GetFocalPoint(),self.Camera.GetViewUp(),self.Camera.GetViewAngle()]

        # ROI Object
        self.ROI = self.roi_set.rois[self.selected_roi]

        for point in self.ROI.vertex_coords:
                self.AddPoint([point,len(self.Points)])
        
        for i in range(len(self.Points)):
                self.SetCursorStyle(i,i==self.SelectedPoint)

        self.UpdateShading()
        self.Window.Render()

    def free_references(self):
        del self.Interactor
        del self.Window
        del self.Renderer
        del self.Camera
        del self.Picker
        
    # On the CAD view, middle click + drag to pan
    def middleButtonPress(self,obj,event):
        self.Renderer.RemoveActor(self.ROIActor)
        for actor in self.BackgroundROIActors:
                self.Renderer.RemoveActor(actor)
        self.OnMiddleButtonDown()


    def middleButtonRelease(self,obj,event):
        self.UpdateShading()
        self.OnMiddleButtonUp()
        


    # On the CAD view, right click+drag to rotate (usually on left button in this interactorstyle)
    def rightButtonPress(self,obj,event):
        self.Renderer.RemoveActor(self.ROIActor)
        for actor in self.BackgroundROIActors:
                self.Renderer.RemoveActor(actor)
        self.OnLeftButtonDown()

    def rightButtonRelease(self,obj,event):
        # Since the user probably doesn't intend to roll the camera,
        # un-roll it automatically after any rotation action.
        self.Camera.SetViewUp(0,0,1)
        self.UpdateShading()
        self.OnLeftButtonUp()



    # Left click to move a point or add a new point
    def OnLeftClick(self,obj,event):

        if self.select_roi_mode:

                self.select_clicked_roi()

        else:

                # Check if they selected an existing point or a position on the model
                picktype,pickdata = self.GetSelected()

                # If they clicked a model position...
                if picktype == 'Position':

                    # If the user was holding Ctrl, create a new point and select it.
                    if self.Interactor.GetControlKey():
                        if self.SelectedPoint is not None:
                                self.CursorDefocus(self.SelectedPoint)        
                        self.AddPoint(pickdata)

                    else:
                        if self.SelectedPoint is not None:
                                # Move the currently selected point to the new location
                                if self.Points[self.SelectedPoint] is not None:
                                    self.Points[self.SelectedPoint][0].SetFocalPoint(pickdata[0][0],pickdata[0][1],pickdata[0][2])
                                    # Check if the camera can see this point
                                    if self.RayCaster is not None and self.CamFitResults is not None:
                                        CamPxCoords = self.CamFitResults.project_points([self.Points[self.SelectedPoint][0].GetFocalPoint()],CheckVisible=True,RayCaster=self.RayCaster,VisibilityMargin=0.07)
                                        self.Points[self.SelectedPoint][3] = np.isfinite(CamPxCoords[0][0][0])
                                    self.SetCursorStyle(self.SelectedPoint,True)

                    self.UpdateResults()
                    self.UpdateShading()

                # If the user clicked on an existing point, select that point.
                if picktype == 'Cursor':
                    self.CursorDefocus(self.SelectedPoint)
                    self.SelectedPoint = pickdata
                    self.CursorFocus(pickdata)


        self.Window.Render()

 

    def select_clicked_roi(self):

        clicked_roi = None

        # Do a pick with our picker object
        clickcoords = self.Interactor.GetEventPosition()
        retval = self.Picker.Pick(clickcoords[0],clickcoords[1],0,self.Renderer)

        # If something was successfully picked, find out what it was...
        if retval != 0:
                pickedmapper = self.Picker.GetMapper()
                for i,mapper in enumerate(self.roi_mappers):
                    if pickedmapper == mapper:
                        clicked_roi = i


        if clicked_roi is not None:

                self.selected_roi = clicked_roi

                # ROI Object
                self.ROI = self.roi_set.rois[clicked_roi]

                self.Points = []

                for point in self.ROI.vertex_coords:
                    self.AddPoint([point,len(self.Points)])
            
                for i in range(len(self.Points)):
                    self.SetCursorStyle(i,i==self.SelectedPoint)

                self.select_roi_mode = False

                self.UpdateShading()
                self.Window.Render()

                print('=> Now editing ROI index ' + str(self.selected_roi) +  ' : ' + self.roi_set.rois[self.selected_roi].name)


    def ZoomIn(self,obj,event):

        # If ctrl + scroll, change the camera FOV
        if self.Interactor.GetControlKey():
                self.Camera.SetViewAngle(self.Camera.GetViewAngle()*0.9)

        # Otherwise, move the camera forward.
        else:
                orig_dist = self.Camera.GetDistance()
                self.Camera.Dolly(1.5)
                self.Camera.SetDistance(orig_dist)

        # Update cursor sizes depending on their distance from the camera,
        # so they're all comfortably visible and clickable.
        for i in range(len(self.Points)):
                if self.Points[i] is not None:
                    self.SetCursorStyle(i,self.SelectedPoint == i)

        self.UpdateShading()
        self.Window.Render()

        return

    def ZoomOut(self,obj,event):

        # If ctrl + scroll, change the camera FOV
        if self.Interactor.GetControlKey():
                self.Camera.SetViewAngle(min(self.Camera.GetViewAngle()*1.1,100.))

        # Otherwise, move the camera backward.
        else:
                orig_dist = self.Camera.GetDistance()
                self.Camera.Dolly(0.75)
                self.Camera.SetDistance(orig_dist)

        # Update cursor sizes so they're all well visible:
        for i in range(len(self.Points)):
                if self.Points[i] is not None:
                    self.SetCursorStyle(i,self.SelectedPoint == i)
        
        self.UpdateShading()
        self.Window.Render()

        return

    # Key press actions handled in here!
    def OnKey(self,obj,event):

        keypressed = self.Interactor.GetKeySym().lower()

        # 'Delete' key deletes the selected point pair
        if keypressed == 'delete':
            if self.SelectedPoint is not None:
                self.DeletePoint(self.SelectedPoint)
                self.Window.Render()

        # 'Home' key resets camera to home position
        if keypressed == 'home':
            self.Camera.SetPosition(self.CameraHome[0])
            self.Camera.SetFocalPoint(self.CameraHome[1])
            self.Camera.SetViewUp(self.CameraHome[2])
            self.Camera.SetViewAngle(self.CameraHome[3])
            self.UpdateShading()
            self.Window.Render()

        # 's' Saves the ROI set definition
        if keypressed == 's':
            print('')
            print('------ Save ROI Set -------')
            print('')
            filename = raw_input('Enter save name for ROI set (blank=overwrite existing): ')
            if filename == "" and self.roi_set.name is not None:
                filename=None
            self.roi_set.save(filename)
            print('Done.')    
            print('')
            print('---------------------------')

        # 'Escape' enters ROI selection mode
        if keypressed == 'escape':

            userissure = None

            if len(self.Points) < 2:
                check_sure = raw_input('=> This will delete ROI index ' + str(self.selected_roi) + '(' + self.ROI.name + '), are you sure? (y/n) ')
                if check_sure.lower() == 'y':
                    self.roi_set.rois.remove(self.ROI)
                    userissure = True

            if len(self.Points) > 2 or (len(self.Points) < 2 and userissure):
                self.select_roi_mode = True
                
                for point in self.Points:
                    self.Renderer.RemoveActor(point[2])

                self.UpdateShading()
                self.Window.Render()


        # 'n' adds a new ROI
        if keypressed == 'n':
            self.ROI = roi.ROI()
            self.roi_set.rois.append(self.ROI)

            self.selected_roi = len(self.roi_set.rois) - 1

            # ROI Object
            
            self.ROI.machine_name = self.CADmodel.machine_name

            print('-> Adding new ROI with index ' + str(len(self.roi_set.rois) - 1))
            self.ROI.name = raw_input('-> Please enter a name for this ROI: ')

            for point in self.Points:
                self.Renderer.RemoveActor(point[2])

            self.Points = []
            self.SelectedPoint = None

            self.select_roi_mode = False

            self.UpdateShading()
            self.Window.Render()


        # 'r' renames current ROI
        if keypressed == 'r':
            newname = raw_input('-> Enter a new name for ROI ' + str(self.selected_roi) + '(' + self.ROI.name + '): ')
            if newname != '':
                self.ROI.name = newname


    # Defocus cursors for a given point
    def CursorDefocus(self,CursorNumber):
        if self.Points[CursorNumber] is not None:
                self.SetCursorStyle(CursorNumber,False)


    # Bring cursors to focus for a given point
    def CursorFocus(self,CursorNumber):

        if self.Points[CursorNumber] is not None:
                self.SetCursorStyle(CursorNumber,True)



    # Check whether the user clicked an existing point or position
    # on the CAD model
    def GetSelected(self):

        # These will be the variables we return. If the user clicked in free space they will stay None.
        picktype = None
        pickdata = None

        # Hide ROI Actors so we don't accidentally pick them
        self.Renderer.RemoveActor(self.ROIActor)
        for actor in self.BackgroundROIActors:
                self.Renderer.RemoveActor(actor)

        # Do a pick with our picker object
        clickcoords = self.Interactor.GetEventPosition()
        retval = self.Picker.Pick(clickcoords[0],clickcoords[1],0,self.Renderer)

        # Put the ROI actor back
        self.Renderer.AddActor(self.ROIActor)
        for actor in self.BackgroundROIActors:
                self.Renderer.AddActor(actor)

        # If something was successfully picked, find out what it was...
        if retval != 0:

                pickedpoints = self.Picker.GetPickedPositions()

                # If more than 1 point is within the picker's tolerance,
                # use the one closest to the camera (this is most intuitive)
                npoints = pickedpoints.GetNumberOfPoints()
                dist_fromcam = []
                campos = self.Camera.GetPosition()

                for i in range(npoints):
                    point = pickedpoints.GetPoint(i)
                    dist_fromcam.append(np.sqrt( (campos[0] - point[0])**2 + (campos[1] - point[1])**2 + (campos[2] - point[2])**2 ))

                _, idx = min((val, idx) for (idx, val) in enumerate(dist_fromcam))

                pickcoords = pickedpoints.GetPoint(idx)
            

                # If the picked point is within 1.5x the cursor radius of any existing point,
                # say that the user clicked on that point
                for i in range(len(self.Points)):

                    # Get the on-screen position of this cursor
                    self.CoordTransformer.SetValue(self.Points[i][0].GetFocalPoint())
                    cursorpos = self.CoordTransformer.GetComputedDisplayValue(self.Renderer)
                    cursorpos_3D = self.Points[i][0].GetFocalPoint()
                
                    cursortol = 7
                    cursortol_3D = 0.1
                    dist_from_cursor = np.sqrt( (cursorpos[0] - clickcoords[0])**2 + (cursorpos[1] - clickcoords[1])**2 )
                    dist_from_cursor_3D = np.sqrt( (cursorpos_3D[0] - pickcoords[0])**2 + (cursorpos_3D[1] - pickcoords[1])**2 + (cursorpos_3D[2] - pickcoords[2])**2 )

                    if dist_from_cursor < cursortol and dist_from_cursor_3D < cursortol_3D and i != self.SelectedPoint:
                        picktype = 'Cursor'
                        pickdata = i


                # If this is going to be a new point, choose between which two existing points
                # to insert it.
                insertion_point = len(self.Points)
                if len(self.Points) > 2:
                    mindist = 10000
                    for lineno in range(len(self.Points)-1):
                        p0 = self.Points[lineno][0].GetFocalPoint()
                        p1 = self.Points[lineno + 1][0].GetFocalPoint()
                        linevec = np.array([p1[0] - p0[0],p1[1] - p0[1],p1[2] - p0[2]])
                        posvec = np.array([pickcoords[0] - p0[0],pickcoords[1]-p0[1],pickcoords[2]-p0[2]])
                        reldist = np.dot(linevec,posvec)/np.dot(linevec,linevec)
                        if reldist > 0 and reldist < 1:
                                closestpoint = p0 + linevec * reldist
                                dist = np.sqrt(np.sum((pickcoords - closestpoint)**2))
                                if dist < mindist:
                                    insertion_point = lineno + 1
                                    mindist = dist

                    p0 = self.Points[-1][0].GetFocalPoint()
                    p1 = self.Points[0][0].GetFocalPoint()
                    linevec = np.array([p1[0] - p0[0],p1[1] - p0[1],p1[2] - p0[2]])
                    posvec = np.array([pickcoords[0] - p0[0],pickcoords[1]-p0[1],pickcoords[2]-p0[2]])
                    reldist = np.dot(linevec,posvec)/np.dot(linevec,linevec)
                    if reldist > 0 and reldist < 1:
                        closestpoint = p0 + linevec * reldist
                        dist = np.sqrt(np.sum((pickcoords - closestpoint)**2))
                        if dist < mindist:
                                insertion_point = len(self.Points)
                

                # If they didn't click on an existing point, they clicked a model position.
                if picktype is None:
                    picktype = 'Position'
                    pickdata = [pickcoords,insertion_point]


        return picktype,pickdata


    # Set the visual style of cursors on the CAD view
    def SetCursorStyle(self,CursorNumber,Focus):

        # Cursor appearance settings - mess with this as per your tastes.
        # The actual size & line width numbers correspond to 3D cursor side length in CAD units for a 75 degree FOV.
        focus_size = 0.025
        nofocus_size = 0.0125

        focus_linewidth = 3
        nofocus_linewidth = 2

        focus_colour = (0,0.8,0)
        nofocus_colour = (0.8,0,0)
        hidden_colour = (0.8,0.8,0) 


        # Cursor size scales with camera FOV to maintain size on screen.
        focus_size = focus_size * (self.Camera.GetViewAngle()/75)
        nofocus_size = nofocus_size * (self.Camera.GetViewAngle()/75)

        point = self.Points[CursorNumber][0].GetFocalPoint()
        campos = self.Camera.GetPosition()
        dist_to_cam = np.sqrt( (campos[0] - point[0])**2 + (campos[1] - point[1])**2 + (campos[2] - point[2])**2 )

        if Focus:
                self.Points[CursorNumber][0].SetModelBounds([point[0]-focus_size*dist_to_cam,point[0]+focus_size*dist_to_cam,point[1]-focus_size*dist_to_cam,point[1]+focus_size*dist_to_cam,point[2]-focus_size*dist_to_cam,point[2]+focus_size*dist_to_cam])
                self.Points[CursorNumber][2].GetProperty().SetColor(focus_colour)
                self.Points[CursorNumber][2].GetProperty().SetLineWidth(focus_linewidth)
        else:
                self.Points[CursorNumber][0].SetModelBounds([point[0]-nofocus_size*dist_to_cam,point[0]+nofocus_size*dist_to_cam,point[1]-nofocus_size*dist_to_cam,point[1]+nofocus_size*dist_to_cam,point[2]-nofocus_size*dist_to_cam,point[2]+nofocus_size*dist_to_cam])
                self.Points[CursorNumber][2].GetProperty().SetLineWidth(nofocus_linewidth)
                self.Points[CursorNumber][2].GetProperty().SetColor(nofocus_colour)
            
        if not self.Points[CursorNumber][3]:
                self.Points[CursorNumber][2].GetProperty().SetColor(hidden_colour)



    # Delete current point
    def DeletePoint(self,pointNumber):

        self.Renderer.RemoveActor(self.Points[pointNumber][2])

        self.Points.remove(self.Points[pointNumber])
            
        if len(self.Points) > 0:
                if pointNumber == self.SelectedPoint:
                    self.SelectedPoint = (self.SelectedPoint - 1) % len(self.Points)
                    self.CursorFocus(self.SelectedPoint)
        else:
                self.Points = []
                self.SelectedPoint = None

        self.UpdateResults()
        self.UpdateShading()


    def AddPoint(self,SelectionData):

        # Create new cursor, mapper and actor
        self.Points.insert(SelectionData[1],[vtk.vtkCursor3D(),vtk.vtkPolyDataMapper(),vtk.vtkActor(),True])
        self.SelectedPoint = SelectionData[1]

        # Some setup of the cursor
        self.Points[self.SelectedPoint][0].XShadowsOff()
        self.Points[self.SelectedPoint][0].YShadowsOff()
        self.Points[self.SelectedPoint][0].ZShadowsOff()
        self.Points[self.SelectedPoint][0].OutlineOff()
        self.Points[self.SelectedPoint][0].SetTranslationMode(1)
        self.Points[self.SelectedPoint][0].SetFocalPoint(SelectionData[0][0],SelectionData[0][1],SelectionData[0][2])

        # Mapper setup
        self.Points[self.SelectedPoint][1].SetInputConnection(self.Points[SelectionData[1]][0].GetOutputPort())
    
        # Actor setup
        self.Points[self.SelectedPoint][2].SetMapper(self.Points[self.SelectedPoint][1])

        # Add new cursor to screen
        self.Renderer.AddActor(self.Points[self.SelectedPoint][2])

        # Check if the camera can see this point
        if self.RayCaster is not None and self.CamFitResults is not None:
                CamPxCoords = self.CamFitResults.project_points([self.Points[self.SelectedPoint][0].GetFocalPoint()],CheckVisible=True,RayCaster=self.RayCaster,VisibilityMargin=0.07)
                self.Points[self.SelectedPoint][3] = np.isfinite(CamPxCoords[0][0][0])

        self.SetCursorStyle(self.SelectedPoint,True)
        self.UpdateResults()


    def UpdateShading(self):

        Shading_Colour_selected = (0,0.8,0)
        Shading_Opacity_selected = 0.3

        Shading_Colour_background = (0.8,0,0)
        Shading_Opacity_background = 0.3

        self.Renderer.RemoveActor(self.ROIActor)

        if self.Points is not None and not self.select_roi_mode:
                if len(self.Points) > 2:
                    self.ROIActor = self.ROI.get_vtkActor(self.Camera.GetPosition())
                    self.ROIActor.GetProperty().SetColor(Shading_Colour_selected)
                    self.ROIActor.GetProperty().SetOpacity(Shading_Opacity_selected)
                    self.Renderer.AddActor(self.ROIActor)

        for actor in self.BackgroundROIActors:
                self.Renderer.RemoveActor(actor)

        self.BackgroundROIActors = []
        self.roi_mappers = []

        for ind in range(len(self.roi_set.rois)):
                if ind != self.selected_roi or self.select_roi_mode:
                    mapper,actor = self.roi_set.rois[ind].get_vtkActor(self.Camera.GetPosition(),return_mapper=True)
                    self.BackgroundROIActors.append(actor)
                    self.roi_mappers.append(mapper)
                    self.BackgroundROIActors[-1].GetProperty().SetColor(Shading_Colour_background)
                    self.BackgroundROIActors[-1].GetProperty().SetOpacity(Shading_Opacity_background)
                    self.Renderer.AddActor(self.BackgroundROIActors[-1])


    def UpdateResults(self):
        self.ROI.vertex_coords = []
        for point in self.Points:
                    self.ROI.vertex_coords.append(point[0].GetFocalPoint())


class CADExplorer(vtk.vtkInteractorStyleTerrain):
 
    def __init__(self,parent=None):
        # Set callbacks for all the controls
        self.AddObserver("LeftButtonPressEvent",self.OnLeftClick)
        self.AddObserver("RightButtonPressEvent",self.rightButtonPress)
        self.AddObserver("RightButtonReleaseEvent",self.rightButtonRelease)
        self.AddObserver("MiddleButtonPressEvent",self.middleButtonPress)
        self.AddObserver("MiddleButtonReleaseEvent",self.middleButtonRelease)
        self.AddObserver("MouseWheelForwardEvent",self.ZoomIn)
        self.AddObserver("MouseWheelBackwardEvent",self.ZoomOut)
        
             

    # Do various initial setup things, most of which can't be done at the time of __init__
    def DoInit(self,ren_3D,gui_window):

        # Get the interactor object
        self.Interactor = self.GetInteractor()

        # Turn off any VTK responses to keyboard input (all necessary keyboard shortcuts etc are done in Q)
        self.Interactor.RemoveObservers('KeyPressEvent')
        self.Interactor.RemoveObservers('CharEvent')

        # Some other objects from higher up which I need access to
        self.Window = self.Interactor.GetRenderWindow()
        self.Renderer = ren_3D
        self.Camera3D = self.Renderer.GetActiveCamera()
        self.gui_window = gui_window


        # Create a picker
        self.Picker = vtk.vtkCellPicker()
        self.Interactor.SetPicker(self.Picker)
        
        self.point = None 
    
        self.rays = {}
        self.rois = {}

        self.view_target = None
        self.view_target_dummy = True

        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewAngle())

        
    def free_references(self):
        del self.Interactor
        del self.Window
        del self.Renderer
        del self.Camera3D
        del self.Picker

    # On the CAD view, middle click + drag to pan
    def middleButtonPress(self,obj,event):
        self.Camera3D.SetDistance(0.5)
        for roi in self.rois.values():
            roi[1].GetProperty().SetOpacity(0)
        self.OnMiddleButtonDown()

    def middleButtonRelease(self,obj,event):
        self.OnMiddleButtonUp()
        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewAngle())
        self.update_rois()
        self.gui_window.refresh_vtk()

    # On the CAD view, right click+drag to rotate (usually on left button in this interactorstyle)
    def rightButtonPress(self,obj,event):
        self.Camera3D.SetDistance(0.01)
        for roi in self.rois.values():
            roi[1].GetProperty().SetOpacity(0)
        self.OnLeftButtonDown()


    def rightButtonRelease(self,obj,event):  

        # Since the user probably doesn't intend to roll the camera,
        # un-roll it automatically after any rotation action.
        self.Camera3D.SetViewUp(0,0,1)

        self.OnLeftButtonUp()

        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewAngle())

        self.update_rois()
        self.gui_window.refresh_vtk()


    # Left click to move a point or add a new point
    def OnLeftClick(self,obj,event):
        
        # Do a pick with our picker object
        clickcoords = self.Interactor.GetEventPosition()
        retval = self.Picker.Pick(clickcoords[0],clickcoords[1],0,self.Renderer)

        # If something was successfully picked, find out what it was...
        if retval != 0:

                pickedpoints = self.Picker.GetPickedPositions()

                # If more than 1 point is within the picker's tolerance,
                # use the one closest to the camera (this is most intuitive)
                npoints = pickedpoints.GetNumberOfPoints()
                dist_fromcam = []
                campos = self.Camera3D.GetPosition()

                for i in range(npoints):
                    point = pickedpoints.GetPoint(i)
                    dist_fromcam.append(np.sqrt( (campos[0] - point[0])**2 + (campos[1] - point[1])**2 + (campos[2] - point[2])**2 ))

                _, idx = min((val, idx) for (idx, val) in enumerate(dist_fromcam))

                pickcoords = pickedpoints.GetPoint(idx)
                
                if self.point is None:
                    # Create new cursor, mapper and actor
                    self.point = [vtk.vtkCursor3D(),vtk.vtkPolyDataMapper(),vtk.vtkActor()]

                    # Some setup of the cursor
                    self.point[0].XShadowsOff()
                    self.point[0].YShadowsOff()
                    self.point[0].ZShadowsOff()
                    self.point[0].OutlineOff()
                    self.point[0].SetTranslationMode(1)
                    self.point[0].SetFocalPoint(pickcoords[0],pickcoords[1],pickcoords[2])

                    # Mapper setup
                    self.point[1].SetInputConnection(self.point[0].GetOutputPort())
    
                    # Actor setup
                    self.point[2].SetMapper(self.point[1])
                    self.point[2].GetProperty().SetColor((0,0.8,0))
                    self.point[2].GetProperty().SetLineWidth(3)
                    
                    # Add new cursor to screen
                    self.Renderer.AddActor(self.point[2])
                
                else:
                    self.point[0].SetFocalPoint(pickcoords[0],pickcoords[1],pickcoords[2])
                    
                self.Update3DCursorSize()
                
                self.gui_window.update_cursor_position(self.point[0].GetFocalPoint())

                self.gui_window.refresh_vtk()
                
                
    # Set the visual style of cursors on the CAD view
    def Update3DCursorSize(self):

        # Cursor appearance settings - mess with this as per your tastes.
        # The actual size & line width numbers correspond to 3D cursor side length in CAD units for a 75 degree FOV.
        focus_size = 0.025

        # Cursor size scales with camera FOV to maintain size on screen.
        focus_size = focus_size * (self.Camera3D.GetViewAngle()/75)

        point = self.point[0].GetFocalPoint()
        campos = self.Camera3D.GetPosition()
        dist_to_cam = np.sqrt( (campos[0] - point[0])**2 + (campos[1] - point[1])**2 + (campos[2] - point[2])**2 )

        self.point[0].SetModelBounds([point[0]-focus_size*dist_to_cam,point[0]+focus_size*dist_to_cam,point[1]-focus_size*dist_to_cam,point[1]+focus_size*dist_to_cam,point[2]-focus_size*dist_to_cam,point[2]+focus_size*dist_to_cam])



    def ZoomIn(self,obj,event):


        # If ctrl + scroll, change the camera FOV
        if self.Interactor.GetControlKey():
            self.Camera3D.SetViewAngle(self.Camera3D.GetViewAngle()*0.9)

        # Otherwise, move the camera forward.
        else:
            orig_dist = self.Camera3D.GetDistance()
            self.Camera3D.SetDistance(0.3)
            self.Camera3D.Dolly(1.5)
            self.Camera3D.SetDistance(orig_dist)

        if self.point is not None:
            self.Update3DCursorSize()

        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(zoom=True),self.Camera3D.GetViewAngle())

        self.update_rois()
        self.gui_window.refresh_vtk()



    def ZoomOut(self,obj,event):


        # If ctrl + scroll, change the camera FOV
        if self.Interactor.GetControlKey():
            self.Camera3D.SetViewAngle(min(self.Camera3D.GetViewAngle()*1.1,100.))

        # Otherwise, move the camera backward.
        else:
            orig_dist = self.Camera3D.GetDistance()
            self.Camera3D.SetDistance(0.3)
            self.Camera3D.Dolly(0.75)
            self.Camera3D.SetDistance(orig_dist)

        if self.point is not None:
            self.Update3DCursorSize()

        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(zoom=True),self.Camera3D.GetViewAngle())

        self.update_rois()
        self.gui_window.refresh_vtk()


    def get_view_target(self,zoom=False):


        campos = self.Camera3D.GetPosition()
        view_dir =  self.Camera3D.GetDirectionOfProjection()

        self.view_target = ( campos[0] + view_dir[0] , campos[1] + view_dir[1] , campos[2] + view_dir[2] )

        return self.view_target


    def add_raydata(self,raydata,n_rays):

        name = raydata
        raydata = raytrace.RayData(raydata)

        if len(raydata.ray_start_coords.shape) == 3:

            startX = raydata.ray_start_coords[:,:,0]
            startY = raydata.ray_start_coords[:,:,1]
            startZ = raydata.ray_start_coords[:,:,2]
            endX = raydata.ray_end_coords[:,:,0]
            endY = raydata.ray_end_coords[:,:,1]
            endZ = raydata.ray_end_coords[:,:,2]


            if raydata.ray_start_coords.size > n_rays:
                n = int(np.sqrt(raydata.ray_start_coords.shape[0]*float(raydata.ray_start_coords.shape[1])/n_rays))

                x,y = np.meshgrid(range(0,raydata.ray_start_coords.shape[1],n),range(0,raydata.ray_start_coords.shape[0],n))
                x = x.flatten()
                y = y.flatten()
            else:
                x = np.array(range(raydata.ray_start_coords.shape[1])).flatten()
                y = np.array(range(raydata.ray_start_coords.shape[0])).flatten()

            inds = np.ravel_multi_index((y,x),startX.shape)


            startX = startX.flatten()
            startY = startY.flatten()
            startZ = startZ.flatten()
            endX = endX.flatten()
            endY = endY.flatten()
            endZ = endZ.flatten()


        elif len(raydata.ray_start_coords.shape) == 2:

            startX = raydata.ray_start_coords[:,0]
            startY = raydata.ray_start_coords[:,1]
            startZ = raydata.ray_start_coords[:,2]
            endX = raydata.ray_end_coords[:,0]
            endY = raydata.ray_end_coords[:,1]
            endZ = raydata.ray_end_coords[:,2]

            #raydir = raydata.ray_end_coords - (raydata.ray_end_coords - raydata.ray_start_coords) * 0.002
            #startX = raydir[:,0]
            #startY = raydir[:,1]
            #startZ = raydir[:,2]

            if startX.size > n_rays:
                n = int(startX.size/n_rays)
                inds = list(range(0,startX.size,n))
            else:
                inds = list(range(0,startX.size))

        self.rays[name] = []
        for ind in inds:
            ls = vtk.vtkLineSource()
            ls.SetPoint1(startX[ind],startY[ind],startZ[ind])
            ls.SetPoint2(endX[ind],endY[ind],endZ[ind])
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(ls.GetOutputPort())
            self.rays[name].append(vtk.vtkActor())
            self.rays[name][-1].SetMapper(mapper)
            self.Renderer.AddActor(self.rays[name][-1])

        self.gui_window.refresh_vtk()

    def remove_raydata(self,raydata=None):
        try:
            if raydata is None:
                raydata = self.rays.keys()
            else:
                raydata = [raydata]

            for rdname in raydata:
                for ray in self.rays[rdname]:
                    self.Renderer.RemoveActor(ray)
                del self.rays[rdname]

            self.gui_window.refresh_vtk()
        except:
            pass


    def add_roi(self,roi_to_add):

        Shading_Colour = (0.8,0,0)
        Shading_Opacity = 0.3

        name = roi_to_add
        if name in self.rois:
            self.Renderer.RemoveActor(self.rois[name][1])
            
        self.rois[name] = [roi.ROI(name),None]
        self.rois[name][1] = self.rois[name][0].get_vtkActor(self.Camera3D.GetPosition())

        self.rois[name][1].GetProperty().SetColor(Shading_Colour)
        self.rois[name][1].GetProperty().SetOpacity(Shading_Opacity)
        self.Renderer.AddActor(self.rois[name][1])
        self.gui_window.refresh_vtk()


    def remove_roi(self,roi):

        try:
            self.Renderer.RemoveActor(self.rois[roi][1])
            del self.rois[roi]
            self.gui_window.refresh_vtk()
        except:
            pass

    def update_rois(self):

        Shading_Colour = (0.8,0,0)
        Shading_Opacity = 0.3

        for roi in self.rois.values():
            self.Renderer.RemoveActor(roi[1])
            roi[1] = roi[0].get_vtkActor(self.Camera3D.GetPosition())
            roi[1].GetProperty().SetColor(Shading_Colour)
            roi[1].GetProperty().SetOpacity(Shading_Opacity)
            self.Renderer.AddActor(roi[1])

class ViewDesigner(vtk.vtkInteractorStyleTerrain):
 
    def __init__(self,parent=None):
        # Set callbacks for all the controls
        self.AddObserver("RightButtonPressEvent",self.rightButtonPress)
        self.AddObserver("RightButtonReleaseEvent",self.rightButtonRelease)
        self.AddObserver("MiddleButtonPressEvent",self.middleButtonPress)
        self.AddObserver("MiddleButtonReleaseEvent",self.middleButtonRelease)
        self.AddObserver("MouseWheelForwardEvent",self.ZoomIn)
        self.AddObserver("MouseWheelBackwardEvent",self.ZoomOut)
             

    # Do various initial setup things, most of which can't be done at the time of __init__
    def DoInit(self,ren_3D,gui_window):

        # Get the interactor object
        self.Interactor = self.GetInteractor()

        # Some other objects from higher up which I need access to
        self.Window = self.Interactor.GetRenderWindow()
        self.Renderer = ren_3D
        self.Camera3D = self.Renderer.GetActiveCamera()
        self.gui_window = gui_window

        # Remove some interactor observers which interfere with my controls
        self.Interactor.RemoveObservers('LeftButtonPressEvent')
        self.Interactor.RemoveObservers('LeftButtonReleaseEvent')

        # Turn off any VTK responses to keyboard input (all necessary keyboard shortcuts etc are done in Q)
        self.Interactor.RemoveObservers('KeyPressEvent')
        self.Interactor.RemoveObservers('CharEvent')

        self.view_target = None
        self.view_target_dummy = True

        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target())

        
    def free_references(self):
        del self.Interactor
        del self.Window
        del self.Renderer
        del self.Camera3D

    # On the CAD view, middle click + drag to pan
    def middleButtonPress(self,obj,event):
        self.Camera3D.SetDistance(0.5)
        self.OnMiddleButtonDown()

    def middleButtonRelease(self,obj,event):
        self.OnMiddleButtonUp()
        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target())
        self.gui_window.refresh_vtk()

    # On the CAD view, right click+drag to rotate (usually on left button in this interactorstyle)
    def rightButtonPress(self,obj,event):
        self.Camera3D.SetDistance(0.01)
        self.OnLeftButtonDown()


    def rightButtonRelease(self,obj,event):  

        # Since the user probably doesn't intend to roll the camera,
        # un-roll it automatically after any rotation action.

        self.OnLeftButtonUp()

        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target())

        self.gui_window.refresh_vtk()


    def ZoomIn(self,obj,event):


        # Move the camera forward.
        orig_dist = self.Camera3D.GetDistance()
        self.Camera3D.SetDistance(0.3)
        self.Camera3D.Dolly(1.5)
        self.Camera3D.SetDistance(orig_dist)


        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(zoom=True))

        self.gui_window.refresh_vtk()



    def ZoomOut(self,obj,event):

        # Move the camera backward.
        orig_dist = self.Camera3D.GetDistance()
        self.Camera3D.SetDistance(0.3)
        self.Camera3D.Dolly(0.75)
        self.Camera3D.SetDistance(orig_dist)


        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(zoom=True))

        self.gui_window.refresh_vtk()


    def get_view_target(self,zoom=False):

        '''
        #This is very nice, but far too slow for most situations.

        full_pick = True

        if self.view_target is not None and zoom:
            targetvect = [self.view_target[0] - self.Camera3D.GetPosition()[0], self.view_target[1] - self.Camera3D.GetPosition()[1], self.view_target[2] - self.Camera3D.GetPosition()[2]]
            camviewdir = self.Camera3D.GetDirectionOfProjection()
            if (targetvect[0]*camviewdir[0] + targetvect[1]*camviewdir[1] + targetvect[2]*camviewdir[2]) < 0 and not self.view_target_dummy:
                full_pick = True
            else:
                full_pick = False

        if full_pick:
            WinSize = self.Window.GetSize()

            retval = self.Picker.Pick(WinSize[0]/2,WinSize[1]/2,0,self.Renderer)

            # If something was successfully picked, find out what it was...
            if retval != 0:

                self.view_target_dummy = False
                pickedpoints = self.Picker.GetPickedPositions()

                # If more than 1 point is within the picker's tolerance,
                # use the one closest to the camera (this is most intuitive)
                npoints = pickedpoints.GetNumberOfPoints()
                dist_fromcam = []
                campos = self.Camera3D.GetPosition()

                for i in range(npoints):
                    point = pickedpoints.GetPoint(i)
                    dist_fromcam.append(np.sqrt( (campos[0] - point[0])**2 + (campos[1] - point[1])**2 + (campos[2] - point[2])**2 ))

                _, idx = min((val, idx) for (idx, val) in enumerate(dist_fromcam))

                self.view_target = pickedpoints.GetPoint(idx)

            else:
                self.view_target = self.Camera3D.GetPosition() + self.Camera3D.GetDirectionOfProjection()
                self.view_target_dummy = True

            '''

        campos = self.Camera3D.GetPosition()
        view_dir =  self.Camera3D.GetDirectionOfProjection()

        self.view_target = ( campos[0] + view_dir[0] , campos[1] + view_dir[1] , campos[2] + view_dir[2] )

        return self.view_target


class ViewAligner(vtk.vtkInteractorStyleTerrain):
 
    def __init__(self,parent=None):
        # Set callbacks for all the controls
        self.AddObserver("RightButtonPressEvent",self.rightButtonPress)
        self.AddObserver("RightButtonReleaseEvent",self.rightButtonRelease)
        self.AddObserver("MiddleButtonPressEvent",self.middleButtonPress)
        self.AddObserver("MiddleButtonReleaseEvent",self.middleButtonRelease)
        self.AddObserver("MouseWheelForwardEvent",self.ZoomIn)
        self.AddObserver("MouseWheelBackwardEvent",self.ZoomOut)
        self.Image = None             

    # Do various initial setup things, most of which can't be done at the time of __init__
    def DoInit(self,ren_3D,gui_window):

        # Get the interactor object
        self.Interactor = self.GetInteractor()

        # Some other objects from higher up which I need access to
        self.Window = self.Interactor.GetRenderWindow()
        self.Renderer = ren_3D
        self.Camera3D = self.Renderer.GetActiveCamera()
        self.gui_window = gui_window

        # Add observer for catching window resizing
        self.Window.AddObserver("ModifiedEvent",self.OnWindowSizeAdjust)

        # Remove some interactor observers which interfere with my controls
        self.Interactor.RemoveObservers('LeftButtonPressEvent')
        self.Interactor.RemoveObservers('LeftButtonReleaseEvent')

        # Turn off any VTK responses to keyboard input (all necessary keyboard shortcuts etc are done in Q)
        self.Interactor.RemoveObservers('KeyPressEvent')
        self.Interactor.RemoveObservers('CharEvent')

        self.view_target = None
        self.view_target_dummy = True

        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewUp())

        
    def free_references(self):
        del self.Interactor
        del self.Window
        del self.Renderer
        del self.Camera3D

    # On the CAD view, middle click + drag to pan
    def middleButtonPress(self,obj,event):
        self.Camera3D.SetDistance(0.05)
        self.OnMiddleButtonDown()

    def middleButtonRelease(self,obj,event):
        self.OnMiddleButtonUp()
        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewUp())
        self.gui_window.refresh_vtk()

    # On the CAD view, right click+drag to rotate (usually on left button in this interactorstyle)
    def rightButtonPress(self,obj,event):
        self.Camera3D.SetDistance(0.01)
        self.OnLeftButtonDown()


    def rightButtonRelease(self,obj,event):  

        # Since the user probably doesn't intend to roll the camera,
        # un-roll it automatically after any rotation action.

        self.OnLeftButtonUp()
        #self.Camera3D.SetRoll(45)
        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewUp())

        self.gui_window.refresh_vtk()


    def ZoomIn(self,obj,event):


        # Move the camera forward.
        orig_dist = self.Camera3D.GetDistance()
        self.Camera3D.SetDistance(0.1)
        self.Camera3D.Dolly(1.5)
        self.Camera3D.SetDistance(orig_dist)


        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(zoom=True),self.Camera3D.GetViewUp())

        self.gui_window.refresh_vtk()



    def ZoomOut(self,obj,event):

        # Move the camera backward.
        orig_dist = self.Camera3D.GetDistance()
        self.Camera3D.SetDistance(0.1)
        self.Camera3D.Dolly(0.75)
        self.Camera3D.SetDistance(orig_dist)


        if self.gui_window is not None:
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(zoom=True),self.Camera3D.GetViewUp())

        self.gui_window.refresh_vtk()


    def get_view_target(self,zoom=False):

        campos = self.Camera3D.GetPosition()
        view_dir =  self.Camera3D.GetDirectionOfProjection()

        self.view_target = ( campos[0] + view_dir[0] , campos[1] + view_dir[1] , campos[2] + view_dir[2] )

        return self.view_target


    def init_image(self,image,opacity,cx=None,cy=None):
        
        # Remove current image, if any
        if self.Image is not None:

            self.Renderer.RemoveActor(self.ImageActor)
            self.ImageActor = None
            self.ImageResizer = None
            self.Image = None


        try:

            self.Image = image
            self.ImageOriginalSize = self.Image.transform.get_display_shape()

            [xpx,ypx] = self.Image.transform.get_display_shape()

            if cx is None or cy is None:
                cx = xpx/2.
                cy = ypx/2.

            #self.nFields = np.max(self.Image.fieldmask) + 1

            #self.fieldmask = np.flipud(self.Image.transform.original_to_display_image(self.Image.fieldmask))

            #self.field_names = self.Image.field_names

            self.ImageActor,self.ImageResizer = self.Image.get_vtkobjects(opacity=opacity)


            self.WinSize = self.Window.GetSize()
            winaspect =  float(self.WinSize[0])/float(self.WinSize[1])


            ImAspect = float(self.ImageOriginalSize[0])/self.ImageOriginalSize[1]

            newRefSize = [0,0,1]

            # Base new zero size on y dimension
            newRefSize[0] = self.WinSize[1]*ImAspect
            newRefSize[1] = self.WinSize[1]
            self.ZoomRefPos = [((self.WinSize[0] - self.WinSize[1]*ImAspect))/2,0.]
            self.ZoomRefPos[0] = self.ZoomRefPos[0] - (cx - xpx/2.)*(self.WinSize[1]/float(ypx))
            self.ZoomRefPos[1] = self.ZoomRefPos[1] + (cy - ypx/2.)*(self.WinSize[1]/float(ypx))            

            self.cx = cx
            self.cy = cy

            self.ZoomRefSize = tuple(newRefSize)
            self.ZoomLevel = 1.

            # Set the initial size of the image to fit the window size
            self.ImageActor.SetPosition(self.ZoomRefPos)

            self.ImageResizer.SetOutputDimensions(int(self.ZoomRefSize[0]*self.ZoomLevel),int(self.ZoomRefSize[1]*self.ZoomLevel),1)
            self.ImageActor.GetProperty().SetDisplayLocationToForeground()

            self.Renderer.AddActor2D(self.ImageActor)
            
            self.gui_window.refresh_vtk()
        except:

            self.Image = None
            raise

    # Adjust 2D image size and cursor positions if the window is resized
    def OnWindowSizeAdjust(self,obg=None,event=None):
        # This is to stop this function erroneously running before
        # the interactor starts (apparently that was a thing??)
        if self.Interactor is not None:
            if self.Image is not None:
                self.WinSize = self.Window.GetSize()
                winaspect =  float(self.WinSize[0])/float(self.WinSize[1])

                ImAspect = float(self.ImageOriginalSize[0])/self.ImageOriginalSize[1]

                newRefSize = [0,0,1]

                [xpx,ypx] = self.Image.transform.get_display_shape()

                # Base new zero size on y dimension
                newRefSize[0] = self.WinSize[1]*ImAspect
                newRefSize[1] = self.WinSize[1]
                self.ZoomRefPos = [((self.WinSize[0] - self.WinSize[1]*ImAspect))/2,0.]

                self.ZoomRefPos[0] = self.ZoomRefPos[0] - (self.cx - xpx/2.)*(self.WinSize[1]/float(ypx))
                self.ZoomRefPos[1] = self.ZoomRefPos[1] + (self.cy - ypx/2.)*(self.WinSize[1]/float(ypx))  

                self.ZoomRefSize = tuple(newRefSize)
                self.ZoomLevel = 1.

                # Set the initial size of the image to fit the window size
                self.ImageActor.SetPosition(self.ZoomRefPos)

                self.ImageResizer.SetOutputDimensions(int(self.ZoomRefSize[0]*self.ZoomLevel),int(self.ZoomRefSize[1]*self.ZoomLevel),1)
           
                self.gui_window.refresh_vtk()


class ImageAnalyser(vtk.vtkInteractorStyleTrackballCamera):
 
    def __init__(self,parent=None):
        # Set callbacks for all the controls
        self.AddObserver("LeftButtonPressEvent",self.OnLeftClick)
        self.AddObserver("RightButtonPressEvent",self.rightButtonPress)
        self.AddObserver("RightButtonReleaseEvent",self.rightButtonRelease)
        self.AddObserver("MiddleButtonPressEvent",self.middleButtonPress)
        self.AddObserver("MiddleButtonReleaseEvent",self.middleButtonRelease)
        self.AddObserver("MouseWheelForwardEvent",self.ZoomIn)
        self.AddObserver("MouseWheelBackwardEvent",self.ZoomOut)
        self.AddObserver("MouseMoveEvent",self.mouse_move)


    # Do various initial setup things, most of which can't be done at the time of __init__
    def DoInit(self,ren_2D,ren_3D,gui_window,raycaster):
       

        # Get the interactor object
        self.Interactor = self.GetInteractor()

        # Some other objects from higher up which I need access to
        self.Window = self.Interactor.GetRenderWindow()
        self.Renderer = ren_3D
        self.Renderer_2D = ren_2D
        self.Camera3D = self.Renderer.GetActiveCamera()
        self.flyto3d = False
        self.cursor2D = []
        self.cursor3D = None
        self.sightlines = []

        self.im_dragging = False

        self.Camera2D = self.Renderer_2D.GetActiveCamera()
        self.Camera2D.ParallelProjectionOn()
        self.Image = None
        self.gui_window = gui_window


        self.SetAutoAdjustCameraClippingRange(0)

        # Turn off any VTK responses to keyboard input (all necessary keyboard shortcuts etc are done in Q)
        self.Interactor.RemoveObservers('KeyPressEvent')
        self.Interactor.RemoveObservers('CharEvent')

        # Add observer for catching window resizing
        self.Window.AddObserver("ModifiedEvent",self.OnWindowSizeAdjust)

        # Create a picker
        self.Picker = vtk.vtkCellPicker()
        self.Interactor.SetPicker(self.Picker)

        # Variables
        self.overlay_on = False
        self.fit_overlay_actor = None
    


        # We will use this for converting from 3D to screen coords.
        self.CoordTransformer = vtk.vtkCoordinate()
        self.CoordTransformer.SetCoordinateSystemToWorld()

        
        
    def free_references(self):
        del self.Interactor
        del self.Window
        del self.Renderer
        del self.Renderer_2D
        del self.Camera3D
        del self.Camera2D
        del self.Picker


    # Use this image object
    def init_image(self,image,hold_position=False):
        
        # Remove current image, if any
        if self.Image is not None:

            # Don't try to hold position if the image aspect ratio has changed!
            imshape = self.Image.transform.get_display_shape()
            if abs( imshape[1]/imshape[0] - image.transform.get_display_shape()[1] / image.transform.get_display_shape()[0]) > 1e-6:
                hold_position = False            

            self.Renderer_2D.RemoveActor(self.ImageActor)
            self.ImageActor = None
            self.Image = None
        else:
            hold_position = False

        try:

            self.Image = image
            self.ImageOriginalSize = self.Image.transform.get_display_shape()

            self.ImageActor = self.Image.get_vtkActor()

            self.WinSize = self.Window.GetSize()
            winaspect =  (float(self.WinSize[0])/2)/float(self.WinSize[1])



            bounds = self.ImageActor.GetBounds()
            xc = bounds[0] + bounds[1] / 2
            yc = bounds[2] + bounds[3] / 2
            ye = bounds[3] - bounds[2]
            xe = bounds[1] - bounds[0]

            self.zoom_ref_cc = (xc,yc)

            im_aspect = xe / ye

            if winaspect >= im_aspect:
                # Base new zero size on y dimension
                self.zoom_ref_scale = 0.5*ye
            else:
                self.zoom_ref_scale = 0.5*xe/winaspect


            if not hold_position:
                # Reset view to fit the whole image on screen.
                self.ZoomLevel = 1.                
                self.Camera2D.SetParallelScale(self.zoom_ref_scale)
                self.Camera2D.SetPosition(xc,yc,1.)
                self.Camera2D.SetFocalPoint(xc,yc,0.)


            self.Renderer_2D.AddActor2D(self.ImageActor)
            

            self.gui_window.refresh_vtk()
        except:

            self.Image = None
            raise


    # On the CAD view, middle click + drag to pan
    def middleButtonPress(self,obj,event):
        self.orig_dist = self.Camera3D.GetDistance()
        self.Camera3D.SetDistance(0.5)
        if self.ChooseRenderer() == '3D':
                self.OnMiddleButtonDown()
        elif self.ChooseRenderer() == '2D':
            self.im_dragging = True

    def middleButtonRelease(self,obj,event):
        if self.ChooseRenderer() == '3D':
            self.OnMiddleButtonUp()
            self.Camera3D.SetDistance(self.orig_dist)
            self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewAngle())
        elif self.ChooseRenderer() == '2D':
            self.im_dragging = False

    # On the CAD view, right click+drag to rotate (usually on left button in this interactorstyle)
    def rightButtonPress(self,obj,event):
        self.orig_dist = self.Camera3D.GetDistance()
        self.Camera3D.SetDistance(0.01)
        if self.ChooseRenderer() == '3D':
            self.OnLeftButtonDown()
        return

    def rightButtonRelease(self,obj,event):
        if self.ChooseRenderer() == '3D':
                self.OnLeftButtonUp()
                self.Camera3D.SetDistance(self.orig_dist)
                self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewAngle())    
        return


    # Left click to move the cursor
    def OnLeftClick(self,obj,event):

        if self.gui_window.raycaster.vtkCellLocator is None or self.gui_window.image is None or self.gui_window.raycaster.fitresults is None:
            return

        clickcoords = self.Interactor.GetEventPosition()

        # If the user clicked on the CAD model...
        if self.ChooseRenderer() == '3D':


            # Do a pick with our picker object
            
            retval = self.Picker.Pick(clickcoords[0],clickcoords[1],0,self.Renderer)

            # If something was successfully picked, find out what it was...
            if retval != 0:

                pickedpoints = self.Picker.GetPickedPositions()

                # If more than 1 point is within the picker's tolerance,
                # use the one closest to the camera (this is most intuitive)
                npoints = pickedpoints.GetNumberOfPoints()
                dist_fromcam = []
                campos = self.Camera3D.GetPosition()

                for i in range(npoints):
                    point = pickedpoints.GetPoint(i)
                    dist_fromcam.append(np.sqrt( (campos[0] - point[0])**2 + (campos[1] - point[1])**2 + (campos[2] - point[2])**2 ))

                _, idx = min((val, idx) for (idx, val) in enumerate(dist_fromcam))

                pickcoords = pickedpoints.GetPoint(idx)
                coords_3d = pickcoords

                self.update_2D_from_3D(pickcoords)

            else:
                return
        # Click on 2D image.
        else:
            im_shape = self.gui_window.raycaster.fitresults.transform.get_display_shape()
            im_coords = self.DisplayToImageCoords(clickcoords)
            if im_coords[0] < 0 or im_coords[1] < 0 or im_coords[0] > im_shape[0] or im_coords[1] > im_shape[1]:
                return

            visible = [True] * self.gui_window.raycaster.fitresults.nfields
            self.clear_cursors_2D()
            self.place_cursor_2D(im_coords)
            raydata = self.gui_window.raycaster.raycast_pixels([im_coords[0]],[im_coords[1]])
            coords_3d = raydata.ray_end_coords[0]
            if raydata.get_ray_lengths()[0] < self.gui_window.cadmodel.max_ray_length-1e-3:
                show = True
            else:
                show = False

            self.place_cursor_3D(coords_3d,show=show)
            coords_2d = [[im_coords]]

            if self.gui_window.raycaster.fitresults.nfields > 1:
                field = self.gui_window.raycaster.fitresults.fieldmask[int(im_coords[1]),int(im_coords[0])]
                image_pos = self.gui_window.raycaster.fitresults.project_points([coords_3d],CheckVisible=True,VisibilityMargin=2e-3,RayCaster=self.gui_window.raycaster)
                for i in range(self.gui_window.raycaster.fitresults.nfields):
                    if i != field and np.all(np.isfinite(image_pos[i][0])):
                        self.place_cursor_2D(image_pos[i][0])
                    if not np.all(np.isfinite(image_pos[i][0])):
                        visible[i] = False

                coords_2d = image_pos

            self.gui_window.update_position_info(coords_2d,coords_3d,visible)
        if self.flyto3d:
            self.Interactor.FlyTo(self.Renderer,coords_3d)
        self.update_sightlines(self.gui_window.show_los_checkbox.isChecked())
        self.gui_window.refresh_vtk()

        


    def update_2D_from_3D(self,coords_3d):

        sightline = False
        intersection_coords = None

        visible = [False] * self.gui_window.raycaster.fitresults.nfields
        self.clear_cursors_2D()          
        # Find where the cursor(s) is/are in 2D.
        image_pos_nocheck = self.gui_window.raycaster.fitresults.project_points([coords_3d])

        image_pos = self.gui_window.raycaster.fitresults.project_points([coords_3d],CheckVisible=True,VisibilityMargin=2e-3,RayCaster=self.gui_window.raycaster)
        for i in range(len(image_pos)):
            if np.any(np.isnan(image_pos_nocheck[i][0])):
                visible[i] = False
                continue
            raydata = self.gui_window.raycaster.raycast_pixels([image_pos_nocheck[i][0][0]],[image_pos_nocheck[i][0][1]])
            sightline = True
            visible[i] =True
            if np.any(np.isnan(image_pos[i][0])):
                visible[i] = False
                intersection_coords = raydata.ray_end_coords[0]

            self.place_cursor_2D(image_pos_nocheck[i][0],visible=visible[i])
                

        self.place_cursor_3D(coords_3d,intersection_coords,outside_fov = not np.any(visible))

        coords_2d = image_pos_nocheck

        self.gui_window.refresh_vtk()

        self.gui_window.update_position_info(coords_2d,coords_3d,visible)



    def place_cursor_3D(self,coords,intersection_coords=None,show=True,outside_fov=None):

        if intersection_coords is None:
            intersection_coords = coords
            visible = True
        else:
            visible = False

        if outside_fov:
            visible = False

        if self.cursor3D is None:
            self.cursor3D = [vtk.vtkCursor3D(),vtk.vtkPolyDataMapper(),vtk.vtkActor(),visible]


            # Some setup of the cursor
            self.cursor3D[0].XShadowsOff()
            self.cursor3D[0].YShadowsOff()
            self.cursor3D[0].ZShadowsOff()
            self.cursor3D[0].OutlineOff()
            self.cursor3D[0].SetTranslationMode(1)
            # Mapper setup
            self.cursor3D[1].SetInputConnection(self.cursor3D[0].GetOutputPort())
            # Actor setup
            self.cursor3D[2].SetMapper(self.cursor3D[1])
            # Add new cursor to screen
            self.Renderer.AddActor(self.cursor3D[2])


        # Cursor position and style
        self.cursor3D[0].SetFocalPoint(coords[0],coords[1],coords[2])
        self.cursor3D[3] = visible
        self.Set3DCursorStyle()

        if show:
            self.cursor3D[2].VisibilityOn()
        else:
            self.cursor3D[2].VisibilityOff()
         

    def set_view_to_cursor3d(self):
        if self.cursor3D is not None:
            cursorpos = np.array(self.cursor3D[0].GetFocalPoint())
            pupilpos = np.array(self.gui_window.raycaster.fitresults.get_pupilpos(field=0))
            campos = (cursorpos - pupilpos)
            campos = campos / np.sqrt(np.sum(campos**2))
            campos = cursorpos - campos
            self.Camera3D.SetPosition(campos)
            self.Camera3D.SetFocalPoint(cursorpos)
            self.Camera3D.SetViewAngle(20)
            self.Camera3D.SetViewUp(0,0,1)
            self.Set3DCursorStyle()
            self.gui_window.refresh_vtk()


    def update_sightlines(self,show_sightlines=True):

        visible_linewidth = 3
        invisible_linewidth = 2
        line_dash_pattern = 0xf0f0

        for sightline in self.sightlines:
            self.Renderer.RemoveActor(sightline[2])
        self.sightlines = []

        if show_sightlines:
            for field,cursor in enumerate(self.cursor2D):
                raydata = self.gui_window.raycaster.raycast_pixels([cursor[3][0]],[cursor[3][1]])
                if raydata.get_ray_lengths()[0] < self.gui_window.cadmodel.max_ray_length-1e-3:
                    self.sightlines.append((vtk.vtkLineSource(),vtk.vtkPolyDataMapper(),vtk.vtkActor()))
                    pupilpos = self.gui_window.raycaster.fitresults.get_pupilpos(field=field)
                    self.sightlines[-1][0].SetPoint1(pupilpos[0],pupilpos[1],pupilpos[2])
                    self.sightlines[-1][0].SetPoint2(raydata.ray_end_coords[0][0],raydata.ray_end_coords[0][1],raydata.ray_end_coords[0][2])
                    self.sightlines[-1][1].SetInputConnection(self.sightlines[-1][0].GetOutputPort())
                    self.sightlines[-1][2].SetMapper(self.sightlines[-1][1])
                    self.sightlines[-1][2].GetProperty().SetLineWidth(visible_linewidth)
                    self.Renderer.AddActor(self.sightlines[-1][2])

                    self.sightlines.append((vtk.vtkLineSource(),vtk.vtkPolyDataMapper(),vtk.vtkActor()))
                    endpos = raydata.ray_end_coords[0] - pupilpos
                    endpos = endpos / np.sqrt(np.sum(endpos**2))
                    endpos = pupilpos + self.gui_window.cadmodel.max_ray_length * endpos

                    self.sightlines[-1][0].SetPoint1(endpos[0],endpos[1],endpos[2])
                    self.sightlines[-1][0].SetPoint2(raydata.ray_end_coords[0][0],raydata.ray_end_coords[0][1],raydata.ray_end_coords[0][2])
                    self.sightlines[-1][1].SetInputConnection(self.sightlines[-1][0].GetOutputPort())
                    self.sightlines[-1][2].SetMapper(self.sightlines[-1][1])
                    self.sightlines[-1][2].GetProperty().SetLineWidth(invisible_linewidth)
                    self.sightlines[-1][2].GetProperty().SetLineStipplePattern(0xf0f0)
                    self.sightlines[-1][2].GetProperty().SetLineStippleRepeatFactor(1)
                    self.Renderer.AddActor(self.sightlines[-1][2])

                else:
                    self.sightlines.append((vtk.vtkLineSource(),vtk.vtkPolyDataMapper(),vtk.vtkActor()))
                    pupilpos = self.gui_window.raycaster.fitresults.get_pupilpos(field=field)
                    self.sightlines[-1][0].SetPoint1(pupilpos[0],pupilpos[1],pupilpos[2])
                    self.sightlines[-1][0].SetPoint2(raydata.ray_end_coords[0][0],raydata.ray_end_coords[0][1],raydata.ray_end_coords[0][2])
                    self.sightlines[-1][1].SetInputConnection(self.sightlines[-1][0].GetOutputPort())
                    self.sightlines[-1][2].SetMapper(self.sightlines[-1][1])
                    self.sightlines[-1][2].GetProperty().SetLineWidth(invisible_linewidth)
                    self.sightlines[-1][2].GetProperty().SetLineStipplePattern(0xf0f0)
                    self.sightlines[-1][2].GetProperty().SetLineStippleRepeatFactor(1)
                    self.Renderer.AddActor(self.sightlines[-1][2])

        self.gui_window.refresh_vtk()



    def place_cursor_2D(self,im_coords,visible=True):
        # Create new cursor, mapper and actor
        self.cursor2D.append((vtk.vtkCursor3D(),vtk.vtkPolyDataMapper(),vtk.vtkActor(),im_coords,visible))
        
        # Some setup of the cursor
        self.cursor2D[-1][0].OutlineOff()
        self.cursor2D[-1][0].AxesOn()
        self.cursor2D[-1][0].XShadowsOff()
        self.cursor2D[-1][0].YShadowsOff()
        self.cursor2D[-1][0].ZShadowsOff()
        self.cursor2D[-1][0].TranslationModeOn()

        imshape = self.Image.transform.get_display_shape()

        worldpos = [im_coords[0],imshape[1] - im_coords[1],0.1]
        self.cursor2D[-1][0].SetFocalPoint(worldpos)

        # Mapper setup
        self.cursor2D[-1][1].SetInputConnection(self.cursor2D[-1][0].GetOutputPort())

        # Actor setup
        self.cursor2D[-1][2].SetMapper(self.cursor2D[-1][1])

        # Add new cursor to screen
        self.Renderer_2D.AddActor(self.cursor2D[-1][2])

        self.Set2DCursorStyle()




    def clear_cursors_2D(self):
        for cursor in self.cursor2D:
            self.Renderer_2D.RemoveActor(cursor[2])
        self.cursor2D = []
        self.update_sightlines()   



    def ZoomIn(self,obj,event):

        if self.ChooseRenderer() == '3D':

                # If ctrl + scroll, change the camera FOV
                if self.Interactor.GetControlKey():
                    self.Camera3D.SetViewAngle(max(self.Camera3D.GetViewAngle()*0.9,1))

                # Otherwise, move the camera forward.
                else:
                    orig_dist = self.Camera3D.GetDistance()
                    self.Camera3D.SetDistance(0.3)
                    self.Camera3D.Dolly(1.5)
                    self.Camera3D.SetDistance(orig_dist)

                self.Set3DCursorStyle()
                self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewAngle())
        else:

                # Re-position the image camera to keep the image point under the mouse pointer fixed when zooming


                zoom_ratio = 1 + 0.2/self.ZoomLevel                 # The zoom ratio we will have

                # Current camera position and scale
                campos = self.Camera2D.GetPosition()
                camscale = self.Camera2D.GetParallelScale() * 2.


                # Where is the zoom target in world coordinates?

                # Zoom coordinates in window pixels
                zoomcoords = list(self.Interactor.GetEventPosition())
                zoomcoords[0] = zoomcoords[0]

                # Position of current centre from to where we're zooming
                zoomcoords = ( (zoomcoords[0] - (self.WinSize[0]/4.))/(self.WinSize[0]/2.) * camscale * (float(self.WinSize[0])/2.)/float(self.WinSize[1]) + campos[0],
                               (zoomcoords[1] - self.WinSize[1]/2.)/self.WinSize[1] * camscale + campos[1] )

                # Vector from zoom point to current camera centre
                zoomvec = ( campos[0] - zoomcoords[0] , campos[1] - zoomcoords[1] )

                # Now we move the camera along the line bwteen the current camera centre and zoom position
                newxc = zoomvec[0]/zoom_ratio + zoomcoords[0]
                newyc = zoomvec[1]/zoom_ratio + zoomcoords[1]

                # Actually move the camera
                self.Camera2D.SetPosition((newxc,newyc,1.))
                self.Camera2D.SetFocalPoint((newxc,newyc,0.))

                # Actually zoom in.
                self.ZoomLevel = self.ZoomLevel + 0.2
                self.Camera2D.SetParallelScale(self.zoom_ref_scale / self.ZoomLevel)
                self.Set2DCursorStyle()

        self.gui_window.refresh_vtk()



    def ZoomOut(self,obj,event):

        if self.ChooseRenderer() == '3D':

                # If ctrl + scroll, change the camera FOV
                if self.Interactor.GetControlKey():
                    self.Camera3D.SetViewAngle(min(self.Camera3D.GetViewAngle()*1.1,110.))

                # Otherwise, move the camera backward.
                else:
                    orig_dist = self.Camera3D.GetDistance()
                    self.Camera3D.SetDistance(0.3)
                    self.Camera3D.Dolly(0.75)
                    self.Camera3D.SetDistance(orig_dist)

                # Update cursor sizes so they're all well visible:
                self.Set3DCursorStyle()
                self.gui_window.update_viewport_info(self.Camera3D.GetPosition(),self.get_view_target(),self.Camera3D.GetViewAngle())
        else:
                # Zoom out smoothly until the whole image is visible
                if self.ZoomLevel > 1.:

                    zoom_ratio = 0.2/(self.ZoomLevel**2 - 1.2*self.ZoomLevel + 0.2)

                    campos = self.Camera2D.GetPosition()

                    zoomvec = ( self.zoom_ref_cc[0] - campos[0] , self.zoom_ref_cc[1] - campos[1] )

                    self.Camera2D.SetPosition((campos[0] + zoomvec[0] * zoom_ratio, campos[1] +  zoomvec[1] * zoom_ratio, 1.))
                    self.Camera2D.SetFocalPoint((campos[0] + zoomvec[0] * zoom_ratio, campos[1] +  zoomvec[1] * zoom_ratio, 0.))

                    self.ZoomLevel = self.ZoomLevel - 0.2
                    self.Camera2D.SetParallelScale(self.zoom_ref_scale / self.ZoomLevel)
                    self.Set2DCursorStyle()

        self.gui_window.refresh_vtk()





    # Function to check if the user is interacting with the CAD or image view
    def ChooseRenderer(self):
        coords = self.Interactor.GetEventPosition()
        poked_renderer = self.Interactor.FindPokedRenderer(coords[0],coords[1])
        if poked_renderer == self.Renderer_2D:
                ren_name = '2D'
        else:
                ren_name = '3D'
        return ren_name





    # Set the visual style of cursors on the CAD view
    def Set3DCursorStyle(self):

        if self.cursor3D is None:
            return

        # Cursor appearance settings - mess with this as per your tastes.
        # The actual size & line width numbers correspond to 3D cursor side length in CAD units for a 75 degree FOV.
        focus_size = 0.03

        focus_linewidth = 4
        occluded_linewidth=4

        focus_colour = (0.,0.8,0)
        hidden_colour = (0.8,0.8,0)

        # Cursor size scales with camera FOV to maintain size on screen.
        focus_size = focus_size * (self.Camera3D.GetViewAngle()/75)

        point = self.cursor3D[0].GetFocalPoint()
        campos = self.Camera3D.GetPosition()
        dist_to_cam = np.sqrt( (campos[0] - point[0])**2 + (campos[1] - point[1])**2 + (campos[2] - point[2])**2 )


        self.cursor3D[0].SetModelBounds([point[0]-focus_size*dist_to_cam,point[0]+focus_size*dist_to_cam,point[1]-focus_size*dist_to_cam,point[1]+focus_size*dist_to_cam,point[2]-focus_size*dist_to_cam,point[2]+focus_size*dist_to_cam])


        if self.cursor3D[3]:
            self.cursor3D[2].GetProperty().SetLineWidth(focus_linewidth)
            self.cursor3D[2].GetProperty().SetColor(focus_colour)
        else:
            self.cursor3D[2].GetProperty().SetLineWidth(occluded_linewidth)
            self.cursor3D[2].GetProperty().SetColor(hidden_colour)


    # Similar to Set3DCursorStyle but for image points
    def Set2DCursorStyle(self):
        
        camscale = self.Camera2D.GetParallelScale()
        focus_size = 0.03 * camscale

        focus_linewidth = 3
        occluded_linewidth=3

        focus_colour = (0.,0.8,0)
        hidden_colour = (0.8,0.8,0)

        for cursor in self.cursor2D:
            pos = cursor[0].GetFocalPoint()
            cursor[0].SetModelBounds([pos[0]-focus_size,pos[0]+focus_size,pos[1]-focus_size,pos[1]+focus_size,0.,0.])
            

            if cursor[4]:
                cursor[2].GetProperty().SetLineWidth(focus_linewidth)
                cursor[2].GetProperty().SetColor(focus_colour)
            else:
                cursor[2].GetProperty().SetLineWidth(occluded_linewidth)
                cursor[2].GetProperty().SetColor(hidden_colour)



    # Adjust 2D image size and cursor positions if the window is resized
    def OnWindowSizeAdjust(self,obg=None,event=None):
        # This is to stop this function erroneously running before
        # the interactor starts (apparently that was a thing??)
        if self.Interactor is not None:

                if self.Image is not None:

                    newWinSize = self.Window.GetSize()
                    winaspect = (float(newWinSize[0])/2)/float(newWinSize[1])

                    bounds = self.ImageActor.GetBounds()
                    xc = bounds[0] + bounds[1] / 2
                    yc = bounds[2] + bounds[3] / 2
                    ye = bounds[3] - bounds[2]
                    xe = bounds[1] - bounds[0]

                    im_aspect = xe / ye

                    if winaspect >= im_aspect:
                        # Base new zero size on y dimension
                        self.zoom_ref_scale = 0.5*ye
                    else:
                        self.zoom_ref_scale = 0.5*xe/winaspect


                    self.Camera2D.SetParallelScale(self.zoom_ref_scale / self.ZoomLevel)
                    self.WinSize = newWinSize



    
    # Function to convert display coordinates to pixel coordinates on the camera image
    def DisplayToImageCoords(self,DisplayCoords):

        camyscale = self.Camera2D.GetParallelScale() * 2.
        camxscale = camyscale * (self.WinSize[0]/2.)/self.WinSize[1]
        cc = self.Camera2D.GetFocalPoint()
        ImCoords = [ (( DisplayCoords[0] - self.WinSize[0]/4 ) / (self.WinSize[0]/2.)) * camxscale + cc[0], (DisplayCoords[1] - self.WinSize[1]/2.)/self.WinSize[1] * camyscale + cc[1] ]

        ImCoords[1] = self.Image.transform.get_display_shape()[1] - ImCoords[1]

        return tuple(ImCoords)



    # Function to convert image pixel coordinates to display coordinates
    def ImageToDisplayCoords(self,ImCoords):

        camyscale = self.Camera2D.GetParallelScale() * 2.
        camxscale = camyscale * (self.WinSize[0]/2.)/self.WinSize[1]
        cc = self.Camera2D.GetFocalPoint()

        DisplayCoords = ( float(ImCoords[0] - cc[0]) / camxscale * self.WinSize[0]/2. + self.WinSize[0]/4. , float(ImCoords[1] - cc[1]) / camyscale*self.WinSize[1] + self.WinSize[1]/2.)


        return DisplayCoords






    def set_view_to_fit(self,field=0):

        orig_dist = self.Camera3D.GetDistance()
        self.Camera3D.SetPosition(self.gui_window.raycaster.fitresults.get_pupilpos(field=field))
        self.Camera3D.SetFocalPoint(self.gui_window.raycaster.fitresults.get_pupilpos(field=field) + self.gui_window.raycaster.fitresults.get_los_direction(self.ImageOriginalSize[0]/2,self.ImageOriginalSize[1]/2,ForceField=field))
        self.Camera3D.SetDistance(orig_dist)
        
        im_aspect = float(self.Image.transform.get_display_shape()[0])/float(self.Image.transform.get_display_shape()[1])
        win_aspect = float(self.WinSize[0]/2.)/float(self.WinSize[1])
        if im_aspect > win_aspect:
            self.Camera3D.UseHorizontalViewAngleOn()
            self.Camera3D.SetViewAngle(self.gui_window.raycaster.fitresults.get_fov(field=field)[0])
        else:
            self.Camera3D.UseHorizontalViewAngleOff()
            self.Camera3D.SetViewAngle(self.gui_window.raycaster.fitresults.get_fov(field=field)[1])
        
        self.Camera3D.SetViewUp(-1.*self.gui_window.raycaster.fitresults.get_cam_to_lab_rotation(field)[:,1])

        self.Set3DCursorStyle()
        
        self.gui_window.refresh_vtk()


    def get_view_target(self,zoom=False):


        campos = self.Camera3D.GetPosition()
        view_dir =  self.Camera3D.GetDirectionOfProjection()

        self.view_target = ( campos[0] + view_dir[0] , campos[1] + view_dir[1] , campos[2] + view_dir[2] )

        return self.view_target




    # Custom mouse move event to enable middle click panning on both
    # CAD and image views.
    def mouse_move(self,obj,event):

        # If the mouse is over the 3D renderer, invoke its normal mouse move callback
        # and make sure the 2D image stops panning.
        if self.ChooseRenderer() == '3D':
            self.im_dragging = False
            self.OnMouseMove()
        else:
            if self.im_dragging and self.ZoomLevel > 1:

                lastXYpos = self.Interactor.GetLastEventPosition() 
                xypos = self.Interactor.GetEventPosition()
                camscale = self.Camera2D.GetParallelScale() * 2
                oldpos = self.Camera2D.GetPosition()
                deltaX = (xypos[0] - lastXYpos[0])/(self.WinSize[0]/2.) * camscale * (self.WinSize[0]/2.)/self.WinSize[1]
                deltaY = (xypos[1] - lastXYpos[1])/self.WinSize[1] * camscale

                newY = oldpos[1] - deltaY
                newX = oldpos[0] - deltaX

                
                # Make sure we don't pan outside the image.
                im_bounds = self.ImageActor.GetBounds()
                xcamscale = (camscale * (self.WinSize[0]/2.)/self.WinSize[1])
                if newX + xcamscale/2. > im_bounds[1]:
                    newX = im_bounds[1] - xcamscale/2.
                elif newX - xcamscale/2. < im_bounds[0]:
                    newX = im_bounds[0] + xcamscale/2.

                if newY + camscale/2. > im_bounds[3]:
                    newY = im_bounds[3] - camscale/2.
                elif newY - camscale/2. < im_bounds[2]:
                    newY = im_bounds[2] + camscale/2.
                
                # Move image camera
                self.Camera2D.SetPosition((newX, newY,1.))
                self.Camera2D.SetFocalPoint((newX,newY,0.))

                self.gui_window.refresh_vtk(im_only=True)