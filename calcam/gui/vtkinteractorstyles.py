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
import numpy as np


class CalcamInterActorStyle3D(vtk.vtkInteractorStyleTerrain):
 
    def __init__(self,parent=None,viewport_callback=None,resize_callback=None,newpick_callback=None,cursor_move_callback=None,focus_changed_callback=None,refresh_callback=None):
        
        # Set callbacks for all the mouse controls
        self.AddObserver("LeftButtonPressEvent",self.on_left_click)
        self.AddObserver("RightButtonPressEvent",self.right_press)
        self.AddObserver("RightButtonReleaseEvent",self.right_release)
        self.AddObserver("MiddleButtonPressEvent",self.middle_press)
        self.AddObserver("MiddleButtonReleaseEvent",self.middle_release)
        self.AddObserver("MouseWheelForwardEvent",self.zoom_in)
        self.AddObserver("MouseWheelBackwardEvent",self.zoom_out)
        self.AddObserver("MouseMoveEvent",self.on_mouse_move)
        self.viewport_callback = viewport_callback
        self.pick_callback = newpick_callback
        self.resize_callback = resize_callback
        self.newpick_callback = newpick_callback
        self.cursor_move_callback = cursor_move_callback
        self.cursor_changed_callback = focus_changed_callback
        self.refresh_callback = refresh_callback


    # Do various initial setup things, most of which can't be done at the time of __init__
    def init(self):

        # Get the interactor object
        self.interactor = self.GetInteractor()

        # Some other objects from higher up which I need access to
        self.vtkwindow = self.interactor.GetRenderWindow()
        renderers = self.vtkwindow.GetRenderers()
        renderers.InitTraversal()
        self.renderer = renderers.GetNextItemAsObject()
        self.camera = self.renderer.GetActiveCamera()

        self.SetAutoAdjustCameraClippingRange(False)

        # Turn off any VTK responses to keyboard input (all necessary keyboard shortcuts etc are done in Q)
        self.interactor.RemoveObservers('KeyPressEvent')
        self.interactor.RemoveObservers('CharEvent')

        # Add observer for catching window resizing
        self.vtkwindow.AddObserver("ModifiedEvent",self.on_resize)


        # Create a picker
        self.picker = vtk.vtkCellPicker()
        self.interactor.SetPicker(self.picker)

        # Variables
        self.cursors = {}
        self.next_cursor_id = 0
        self.focus_cursor = None
        self.legend = None
        self.xsection_coords = None
    

        # We will use this for converting from 3D to screen coords.
        self.vtk_coord_transformer = vtk.vtkCoordinate()
        self.vtk_coord_transformer.SetCoordinateSystemToWorld()



    # Middle click + drag to pan
    def middle_press(self,obj,event):
        self.orig_dist = self.camera.GetDistance()
        self.camera.SetDistance(0.5)
        self.OnMiddleButtonDown()


    def middle_release(self,obj,event):
        self.OnMiddleButtonUp()
        self.camera.SetDistance(self.orig_dist)
        self.on_cam_moved()


    # On the CAD view, right click+drag to rotate (usually on left button in this interactorstyle)
    def right_press(self,obj,event):
        self.orig_dist = self.camera.GetDistance()
        self.camera.SetDistance(0.01)
        self.OnLeftButtonDown()


    def right_release(self,obj,event):
        self.OnLeftButtonUp()
        self.camera.SetDistance(self.orig_dist)
        self.on_cam_moved()



    def zoom_in(self,obj,event):

        # If ctrl + scroll, change the camera FOV
        if self.interactor.GetControlKey():
            self.camera.SetViewAngle(max(self.camera.GetViewAngle()*0.9,1))

        # Otherwise, move the camera forward.
        else:
            orig_dist = self.camera.GetDistance()
            self.camera.SetDistance(0.3)
            self.camera.Dolly(1.5)
            self.camera.SetDistance(orig_dist)

        # Update cursor sizes depending on their distance from the camera,
        # so they're all comfortably visible and clickable.
        self.on_cam_moved()
   


    def zoom_out(self,obj,event):

        # If ctrl + scroll, change the camera FOV
        if self.interactor.GetControlKey():
            self.camera.SetViewAngle(min(self.camera.GetViewAngle()*1.1,110.))

        # Otherwise, move the camera backward.
        else:
            orig_dist = self.camera.GetDistance()
            self.camera.SetDistance(0.3)
            self.camera.Dolly(0.75)
            self.camera.SetDistance(orig_dist)

        # Update cursor sizes so they're all well visible:
        self.on_cam_moved()


    def on_cam_moved(self):
        self.update_cursor_style(refresh=False)
        self.update_clipping()

        if self.viewport_callback is not None:
            self.viewport_callback()


    def set_xsection(self,xsection_coords):

        if xsection_coords is not None:
            self.xsection_coords = xsection_coords
        else:
            self.xsection_coords = None

        self.update_clipping()
    
    def get_xsection(self):
        return self.xsection_coords



    def get_cursor_coords(self,cursor_id):

        return self.cursors[cursor_id]['cursor3d'].GetFocalPoint()


    def update_clipping(self):

        self.renderer.ResetCameraClippingRange()

        if self.xsection_coords is not None:
            normal_range = self.camera.GetClippingRange()
            cam_to_xsec = self.xsection_coords - np.array(self.camera.GetPosition())
            cam_view_dir = self.camera.GetDirectionOfProjection()
            dist = max(normal_range[0],np.dot(cam_to_xsec,cam_view_dir))
            self.camera.SetClippingRange(dist,normal_range[1])

        if self.refresh_callback is not None:
            self.refresh_callback()


    # Left click to move a point or add a new point
    def on_left_click(self,obj,event):

        ctrl_pressed = self.interactor.GetControlKey()

        # These will be the variables we return. If the user clicked in free space they will stay None.
        clicked_cursor = None
        pickcoords = None

        # Do a pick with our picker object
        clickcoords = self.interactor.GetEventPosition()
        retval = self.picker.Pick(clickcoords[0],clickcoords[1],0,self.renderer)

        # If something was successfully picked, find out what it was...
        if retval != 0:

            pickedpoints = self.picker.GetPickedPositions()

            # If more than 1 point is within the picker's tolerance,
            # use the one closest to the camera (this is most intuitive)
            npoints = pickedpoints.GetNumberOfPoints()
            dist_fromcam = []
            campos = self.camera.GetPosition()

            for i in range(npoints):
                point = pickedpoints.GetPoint(i)
                dist_fromcam.append(np.sqrt( (campos[0] - point[0])**2 + (campos[1] - point[1])**2 + (campos[2] - point[2])**2 ))

            _, idx = min((val, idx) for (idx, val) in enumerate(dist_fromcam))

            pickcoords = pickedpoints.GetPoint(idx)

            # If the picked point is within 1.5x the cursor radius of any existing point,
            # say that the user clicked on that point
            dist = 1e7
            for cid,cursor in self.cursors.items():

                if cid == self.focus_cursor:
                    continue

                # Get the on-screen position of this cursor
                self.vtk_coord_transformer.SetValue(cursor['cursor3d'].GetFocalPoint())
                cursorpos = self.vtk_coord_transformer.GetComputedDisplayValue(self.renderer)
                
                # See if it's close enough to the click to be considered 'clicked'
                cursortol = 7 # Number of pixels the click has to be within

                dist_from_cursor = np.sqrt( (cursorpos[0] - clickcoords[0])**2 + (cursorpos[1] - clickcoords[1])**2 )
                if dist_from_cursor < cursortol and dist_from_cursor < dist:
                        clicked_cursor = cid
                        dist = dist_from_cursor



            # If they clicked a model position...
            if clicked_cursor is None and (self.focus_cursor is None or ctrl_pressed) and self.newpick_callback is not None:
                
                self.newpick_callback(pickcoords)

            elif clicked_cursor is None  and self.focus_cursor is not None:

                self.cursors[self.focus_cursor]['cursor3d'].SetFocalPoint(pickcoords)
                self.update_cursor_style()

                if self.cursor_move_callback is not None:
                    self.cursor_move_callback(pickcoords)

            elif clicked_cursor is not None:

                self.set_cursor_focus(clicked_cursor)

                if self.focus_changed_callback is not None:
                    self.focus_changed_callback(clicked_cursor)

                if self.cursor_move_callback is not None:
                    self.cursor_move_callback(self.cursors[clicked_cursor]['cursor3d'].GetFocalPoint())

            if self.refresh_callback is not None:
                self.refresh_callback()


    def update_cursor_style(self,refresh=True):

        campos = self.camera.GetPosition()
        for cid,cursor in self.cursors.items():

            position = cursor['cursor3d'].GetFocalPoint()

            dist_to_cam = np.sqrt( (campos[0] - position[0])**2 + (campos[1] - position[1])**2 + (campos[2] - position[2])**2 )

            if cid == self.focus_cursor:
                colour = (0,0.8,0)
                linewidth = 3
                size = 0.025
            else:
                if cursor['colour'] is not None:
                    colour = cursor['colour']
                else:
                    colour = (0.8,0,0)
                linewidth = 2
                size = 0.0125

            # Cursor size scales with camera FOV to maintain size on screen.
            size = size * (self.camera.GetViewAngle()/75)

            cursor['cursor3d'].SetModelBounds([position[0]-size*dist_to_cam,position[0]+size*dist_to_cam,position[1]-size*dist_to_cam,position[1]+size*dist_to_cam,position[2]-size*dist_to_cam,position[2]+size*dist_to_cam])
            cursor['actor'].GetProperty().SetColor(colour)
            cursor['actor'].GetProperty().SetLineWidth(linewidth)

        if self.refresh_callback is not None and refresh:
            self.refresh_callback()



    # Defocus cursors for a given point pair
    def set_cursor_focus(self,cursor_id):

        if cursor_id is not None:
            if cursor_id not in self.cursors.keys():
                raise ValueError('No cursor with ID {:d}'.format(cursor_id))

        self.focus_cursor = cursor_id
        self.update_cursor_style()




    def add_cursor(self,coords,change_focus=True):

        # Create new cursor, mapper and actor
        new_cursor_id = self.next_cursor_id
        self.cursors[new_cursor_id] = {'cursor3d':vtk.vtkCursor3D(),'actor':vtk.vtkActor(),'colour':None}
        self.next_cursor_id += 1

        # Some setup of the cursor
        self.cursors[new_cursor_id]['cursor3d'].XShadowsOff()
        self.cursors[new_cursor_id]['cursor3d'].YShadowsOff()
        self.cursors[new_cursor_id]['cursor3d'].ZShadowsOff()
        self.cursors[new_cursor_id]['cursor3d'].OutlineOff()
        self.cursors[new_cursor_id]['cursor3d'].SetTranslationMode(1)
        self.cursors[new_cursor_id]['cursor3d'].SetFocalPoint(coords[0],coords[1],coords[2])


        # Mapper setup
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.cursors[new_cursor_id]['cursor3d'].GetOutputPort())
    
        # Actor setup
        self.cursors[new_cursor_id]['actor'].SetMapper(mapper)

        # Add new cursor to screen
        self.renderer.AddActor(self.cursors[new_cursor_id]['actor'])

        if change_focus:
            self.focus_cursor = new_cursor_id

        self.update_cursor_style()

        return new_cursor_id
  

    def remove_cursor(self,cursor_id):

        try:
            cursor = self.cursors.pop(cursor_id)
            self.renderer.RemoveActor(cursor['actor'])
            if cursor_id == self.focus_cursor:
                self.focus_cursor = None
                self.update_cursor_style()
        except KeyError:
            raise ValueError('No cursor with ID {:d}'.format(cursor_id))


    def on_resize(self,obj=None,event=None):

        vtk_size = self.vtkwindow.GetSize()

        # Sizing of the legend
        if self.legend is not None:

            legend_offset_y = 0.02
            legend_scale = 0.03

            legend_offset_x = legend_offset_y*vtk_size[1] / vtk_size[0]
            legend_pad_y = 20./vtk_size[1]
            legend_pad_x = 20./vtk_size[0]

            legend_height = legend_pad_y + legend_scale * self.n_legend_items
            abs_height = legend_scale * vtk_size[1]
            width_per_char = abs_height * 0.5
            legend_width = legend_pad_x + (width_per_char * self.longest_name)/vtk_size[0]

            self.legend.GetPosition2Coordinate().SetCoordinateSystemToNormalizedDisplay()
            self.legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()

            # Set Legend Size
            self.legend.GetPosition2Coordinate().SetValue(legend_width,legend_height )
            # Set Legend position
            self.legend.GetPositionCoordinate().SetValue(1 - legend_offset_x - legend_width, legend_offset_y)


        if self.resize_callback is not None:
            self.resize_callback(vtk_size)


    def on_mouse_move(self,obj=None,event=None):

        self.OnMouseMove()
        if self.xsection_coords is not None:
            self.update_clipping()