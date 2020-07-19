'''
* Copyright 2015-2018 European Atomic Energy Community (EURATOM)
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


import copy

from .core import *
from .vtkinteractorstyles import CalcamInteractorStyle2D, CalcamInteractorStyle3D
from ..render import render_cam_view
from ..coordtransformer import CoordTransformer
from ..raycast import raycast_sightlines
from ..image_enhancement import enhance_image

type_description = {'alignment': 'Manual Alignment', 'fit':'Point pair fitting','virtual':'Virtual'}

# Main calcam window class for actually creating calibrations.
class ImageAnalyser(CalcamGUIWindow):
 
    def __init__(self, app, parent = None):

        # GUI initialisation
        CalcamGUIWindow.init(self,'image_analyser.ui',app,parent)

        # Start up with no CAD model
        self.cadmodel = None
        self.calibration = None

        # Set up VTK
        self.qvtkwidget_3d = qt.QVTKRenderWindowInteractor(self.vtkframe_3d)
        self.vtkframe_3d.layout().addWidget(self.qvtkwidget_3d)
        self.interactor3d = CalcamInteractorStyle3D(refresh_callback=self.refresh_3d,viewport_callback=self.update_viewport_info,resize_callback=self.update_vtk_size,newpick_callback=self.update_from_3d,cursor_move_callback=lambda cid,coords: self.update_from_3d(coords))
        self.interactor3d.allow_focus_change = False
        self.qvtkwidget_3d.SetInteractorStyle(self.interactor3d)
        self.renderer_3d = vtk.vtkRenderer()
        self.renderer_3d.SetBackground(0, 0, 0)
        self.qvtkwidget_3d.GetRenderWindow().AddRenderer(self.renderer_3d)
        self.camera_3d = self.renderer_3d.GetActiveCamera()

        self.qvtkwidget_2d = qt.QVTKRenderWindowInteractor(self.vtkframe_2d)
        self.vtkframe_2d.layout().addWidget(self.qvtkwidget_2d)
        self.interactor2d = CalcamInteractorStyle2D(refresh_callback=self.refresh_2d,newpick_callback=lambda x: self.update_from_2d([x]),cursor_move_callback=lambda cid,coords: self.update_from_2d(coords))
        self.interactor2d.allow_focus_change = False
        self.qvtkwidget_2d.SetInteractorStyle(self.interactor2d)
        self.renderer_2d = vtk.vtkRenderer()
        self.renderer_2d.SetBackground(0, 0, 0)
        self.qvtkwidget_2d.GetRenderWindow().AddRenderer(self.renderer_2d)
        self.camera_2d = self.renderer_2d.GetActiveCamera()

        self.image_geometry = None

        self.coords_3d = None

        self.populate_models()
        


        # Disable image transform buttons if we have no image
        self.image_settings.hide()

        self.tabWidget.setTabEnabled(2,False)
        self.tabWidget.setTabEnabled(3,False)


        # Callbacks for GUI elements
        self.image_sources_list.currentIndexChanged.connect(self.build_imload_gui)
        self.viewlist.itemSelectionChanged.connect(self.change_cad_view)
        self.camX.valueChanged.connect(self.change_cad_view)
        self.camY.valueChanged.connect(self.change_cad_view)
        self.camZ.valueChanged.connect(self.change_cad_view)
        self.tarX.valueChanged.connect(self.change_cad_view)
        self.tarY.valueChanged.connect(self.change_cad_view)
        self.tarZ.valueChanged.connect(self.change_cad_view)
        self.cam_roll.valueChanged.connect(self.change_cad_view)
        self.camFOV.valueChanged.connect(self.change_cad_view)
        self.load_model_button.clicked.connect(self.load_model)
        self.model_name.currentIndexChanged.connect(self.populate_model_variants)
        self.feature_tree.itemChanged.connect(self.update_checked_features)
        self.load_image_button.clicked.connect(self.load_image)
        self.load_calib_button.clicked.connect(self.load_calib)
        self.reset_view_button.clicked.connect(lambda: self.set_view_from_calib(self.calibration,0))
        self.cursor_closeup_button.clicked.connect(self.set_view_to_cursor)
        self.cal_props_button.clicked.connect(self.show_calib_info)
        self.viewport_load_calib.clicked.connect(self.load_viewport_calib)

        self.sightline_checkbox.toggled.connect(lambda: self.update_from_3d(self.coords_3d))

        self.overlay_checkbox.toggled.connect(self.toggle_overlay)

        self.enhance_checkbox.stateChanged.connect(self.toggle_enhancement)

        self.image = None
        self.viewport_calibs = DodgyDict()

        self.control_sensitivity_slider.valueChanged.connect(lambda x: self.interactor3d.set_control_sensitivity(x*0.01))
        self.rmb_rotate.toggled.connect(self.interactor3d.set_rmb_rotate)
        self.interactor3d.set_control_sensitivity(self.control_sensitivity_slider.value()*0.01)



        # Populate image sources list and tweak GUI layout for image loading.
        self.imload_inputs = {}
        self.image_load_options.layout().setColumnMinimumWidth(0,100)

        self.image_sources = self.config.get_image_sources()
        index = -1
        for i,imsource in enumerate(self.image_sources):
            self.image_sources_list.addItem(imsource.display_name)
            if imsource.display_name == self.config.default_image_source:
                index = i

        self.image_sources_list.setCurrentIndex(index)

        self.cursor_ids = {'3d':None,'2d':{'visible':None,'hidden':None}}
        self.sightline_actors = []

        # Start the GUI!
        self.show()
        self.interactor2d.init()
        self.interactor3d.init()
        self.qvtkwidget_3d.GetRenderWindow().GetInteractor().Initialize()
        self.qvtkwidget_2d.GetRenderWindow().GetInteractor().Initialize()
        self.interactor3d.on_resize()



    def update_cursor_position(self,cursor_id,position):
        
        #info = 'Cursor location: ' + self.cadmodel.format_coord(position).replace('\n',' | ')

        pass

        #self.statusbar.showMessage(info)



    def update_cursor_position(self,ctype,cid,pos):
        pass

    def add_cursor(self,ctype,pos):

        if self.cursor_ids == [None,None]:

            if ctype == '3d':
                self.cursor_ids[0] = self.interactor3d.add_cursor(pos)

                im_pos = self.calibration.project_points()



    def update_from_3d(self,coords_3d):

        # Sight line drawing style
        visible_linewidth = 3
        visible_colour = (0,0.8,0)
        invisible_linewidth = 1
        invisible_colour = (0.8,0,0)

        if self.calibration is not None and self.image is not None and coords_3d is not None:

            self.cursor_closeup_button.setEnabled(True)

            # Clear anything which already exists
            if self.cursor_ids['3d'] is not None:
                self.interactor3d.remove_cursor(self.cursor_ids['3d'])
                self.cursor_ids['3d'] = None

            if self.cursor_ids['2d']['visible'] is not None:
                self.interactor2d.remove_active_cursor(self.cursor_ids['2d']['visible'])
                self.cursor_ids['2d']['visible'] = None

            if self.cursor_ids['2d']['hidden'] is not None:
                self.interactor2d.remove_active_cursor(self.cursor_ids['2d']['hidden'])
                self.cursor_ids['2d']['hidden'] = None

            for actor in self.sightline_actors:
                self.interactor3d.remove_extra_actor(actor)
            self.sightline_actors = []


            sightline = False
            intersection_coords = None

            visible = [False] * self.calibration.n_subviews
          
            # Find where the cursor(s) is/are in 2D.
            image_pos_nocheck = self.calibration.project_points([coords_3d],coords='original')

            image_pos = self.calibration.project_points([coords_3d],check_occlusion_with=self.cadmodel,occlusion_tol=1e-2,coords='original')

            for i in range(len(image_pos)):
                image_pos[i][0][:] = self.calibration.geometry.original_to_display_coords(*image_pos[i][0])
                image_pos_nocheck[i][0][:] = self.calibration.geometry.original_to_display_coords(*image_pos_nocheck[i][0])

                if np.any(np.isnan(image_pos_nocheck[i])):
                    visible[i] = False
                    continue

                raydata = raycast_sightlines(self.calibration,self.cadmodel,image_pos_nocheck[i][0,0],image_pos_nocheck[i][0,1],verbose=False,force_subview=i)
                sightline = True
                visible[i] =True

                if np.any(np.isnan(image_pos[i])):
                    visible[i] = False
                    intersection_coords = raydata.ray_end_coords

                if visible[i]:

                    if self.cursor_ids['2d']['visible'] is None:
                        self.cursor_ids['2d']['visible'] = self.interactor2d.add_active_cursor(image_pos_nocheck[i][0,:],run_move_callback=False)
                    else:
                        self.interactor2d.add_active_cursor(image_pos_nocheck[i][0,:],run_move_callback=False,add_to=self.cursor_ids['2d']['visible'])

                    if self.sightline_checkbox.isChecked():
                        linesource = vtk.vtkLineSource()
                        pupilpos = self.calibration.get_pupilpos(subview=i)
                        linesource.SetPoint1(pupilpos[0],pupilpos[1],pupilpos[2])
                        linesource.SetPoint2(coords_3d)
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(linesource.GetOutputPort())
                        actor = vtk.vtkActor()
                        actor.SetMapper(mapper)
                        actor.GetProperty().SetLineWidth(visible_linewidth)
                        actor.GetProperty().SetColor(visible_colour)
                        self.sightline_actors.append(actor)
                        self.interactor3d.add_extra_actor(actor)

                        model_extent = self.cadmodel.get_extent()
                        model_size = model_extent[1::2] - model_extent[::2]
                        max_ray_length = model_size.max() * 4
                        sldir = coords_3d - pupilpos
                        sldir = sldir / np.sqrt(np.sum(sldir**2))
                        rayend = pupilpos + sldir*max_ray_length

                        linesource = vtk.vtkLineSource()
                        linesource.SetPoint1(coords_3d)
                        linesource.SetPoint2(rayend)
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(linesource.GetOutputPort())
                        actor = vtk.vtkActor()
                        actor.SetMapper(mapper)
                        actor.GetProperty().SetLineWidth(invisible_linewidth)
                        actor.GetProperty().SetColor(invisible_colour)
                        self.sightline_actors.append(actor)
                        self.interactor3d.add_extra_actor(actor)

                else:
                    if self.cursor_ids['2d']['hidden'] is None:
                        self.cursor_ids['2d']['hidden'] = self.interactor2d.add_active_cursor(image_pos_nocheck[i][0,:],run_move_callback=False)
                    else:
                        self.interactor2d.add_active_cursor(image_pos_nocheck[i][0,:],run_move_callback=False,add_to=self.cursor_ids['2d']['hidden'])

                    if self.sightline_checkbox.isChecked():
                        linesource = vtk.vtkLineSource()
                        pupilpos = self.calibration.get_pupilpos(subview=i)
                        linesource.SetPoint1(*pupilpos)
                        linesource.SetPoint2(*intersection_coords)
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(linesource.GetOutputPort())
                        actor = vtk.vtkActor()
                        actor.SetMapper(mapper)
                        actor.GetProperty().SetLineWidth(invisible_linewidth)
                        actor.GetProperty().SetColor(visible_colour)
                        self.sightline_actors.append(actor)
                        self.interactor3d.add_extra_actor(actor)

                        linesource = vtk.vtkLineSource()
                        linesource.SetPoint1(*intersection_coords)
                        linesource.SetPoint2(*coords_3d)
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(linesource.GetOutputPort())
                        actor = vtk.vtkActor()
                        actor.SetMapper(mapper)
                        actor.GetProperty().SetLineWidth(invisible_linewidth)
                        actor.GetProperty().SetColor(invisible_colour)
                        self.sightline_actors.append(actor)
                        self.interactor3d.add_extra_actor(actor)
                    
            self.cursor_ids['3d'] = self.interactor3d.add_cursor(coords_3d)

            if self.cursor_ids['2d']['visible'] is not None:
                self.interactor2d.set_cursor_focus(self.cursor_ids['2d']['visible'])
                self.interactor3d.set_cursor_focus(self.cursor_ids['3d'])



            self.coords_2d = image_pos_nocheck
            self.coords_3d = coords_3d

            self.update_position_info(self.coords_2d,self.coords_3d,visible)

            


    def update_from_2d(self,coords_2d):

        if self.calibration is not None and self.image is not None and self.cadmodel is not None:
            for i,coords in enumerate(coords_2d):
                if coords is not None and np.any(coords != self.coords_2d[i]):
                    raydata = raycast_sightlines(self.calibration,self.cadmodel,coords[0],coords[1],coords='Display')
                    self.update_from_3d(raydata.ray_end_coords[0,:])
                    return




    def load_calib(self):

        opened_calib = self.object_from_file('calibration')

        if opened_calib is not None:
            if None in opened_calib.view_models:
                raise UserWarning('The selected calibration file does not contain a full set of calibration parameters. Only calibration files containing all calibration parameters can be used.')
            
            if self.image is not None:

                ioshape = self.image_geometry.get_original_shape()
                opened_calib.set_detector_window( (self.image_geometry.offset[0],self.image_geometry.offset[1],ioshape[0],ioshape[1]) )

            self.calibration = opened_calib
            self.calib_name.setText(os.path.split(self.calibration.filename)[1].replace('.ccc',''))
            self.cal_props_button.setEnabled(True)

            self.overlay_checkbox.setEnabled(True)
            self.reset_view_button.setEnabled(True)
            self.overlay = None

            if self.enhance_checkbox.isChecked():
                self.interactor2d.set_image(enhance_image(self.calibration.geometry.original_to_display_image(self.image)))
            else:
                self.interactor2d.set_image(self.calibration.geometry.original_to_display_image(self.image))

            self.interactor2d.set_subview_lookup(self.calibration.n_subviews,self.calibration.subview_lookup)

            if opened_calib.cad_config is not None:
                cconfig = opened_calib.cad_config
                if self.cadmodel is not None and self.cadmodel.machine_name == cconfig['model_name'] and self.cadmodel.model_variant == cconfig['model_variant']:
                    keep_model = True
                else:
                    keep_model = False

                if keep_model:
                    try:
                        self.cadmodel.enable_only(cconfig['enabled_features'])
                    except Exception as e:
                        if 'Unknown feature' not in str(e):
                            raise
                    self.update_feature_tree_checks()
                else:
                    cconfig = opened_calib.cad_config
                    load_model = True
                    try:
                        name_index = sorted(self.model_list.keys()).index(cconfig['model_name'])
                        self.model_name.setCurrentIndex(name_index)
                        variant_index = self.model_list[ cconfig['model_name'] ][1].index(cconfig['model_variant'])
                        self.model_variant.setCurrentIndex(variant_index)
                    except ValueError:
                        self.model_name.setCurrentIndex(-1)
                        load_model=False


                    if load_model:
                        try:
                            self.load_model(featurelist=cconfig['enabled_features'])
                        except Exception as e:
                            self.cadmodel = None
                            if 'Unknown feature' not in str(e):
                                raise          
                    
                    # I'm not sure why I have to call this twice to get it to work properly.
                    # On the first call it almost works, but not quite accurately. Then
                    # on the second call onwards it's fine. Should be investigated further.
                    self.set_view_from_calib(self.calibration,0)
                    self.set_view_from_calib(self.calibration,0)

            if self.overlay_checkbox.isChecked():
                self.overlay_checkbox.setChecked(False)
                self.overlay_checkbox.setChecked(True)

            if self.cursor_ids['3d'] is not None:
                self.update_from_3d(self.coords_3d)
            else:
                self.coords_2d = [None] * self.calibration.n_subviews
                


    def unload_calib(self):
        
        if self.calibration is not None:
            self.calibration = None
            self.calib_name.setText('No Calibration Loaded.')
            self.overlay_checkbox.setChecked(False)
            self.overlay_checkbox.setEnabled(False)
            self.reset_view_button.setEnabled(False)
            self.overlay = None
            self.cal_props_button.setEnabled(False)
            
            if self.cursor_ids['2d']['visible'] is not None:
                self.interactor2d.remove_active_cursor(self.cursor_ids['2d']['visible'])
                self.cursor_ids['2d']['visible'] = None
                
            if self.cursor_ids['2d']['hidden'] is not None:
                self.interactor2d.remove_active_cursor(self.cursor_ids['2d']['hidden'])            
                self.cursor_ids['2d']['hidden'] = None
                
                
    def set_view_to_cursor(self):
        cursorpos = self.interactor3d.get_cursor_coords(self.cursor_ids['3d'])

        pupilpos = self.calibration.get_pupilpos(subview=0)
        campos_vect = (pupilpos - cursorpos)
        campos_vect = campos_vect / np.sqrt(np.sum(campos_vect**2))

        self.camera_3d.SetFocalPoint(cursorpos)
        self.camera_3d.SetViewAngle(60)

        self.camera_3d.SetPosition(cursorpos + 0.3*campos_vect)

        self.update_viewport_info()
        self.cam_roll.setValue(0)
        self.refresh_3d()


    def on_model_load(self):
        # Enable the other tabs!
        self.tabWidget.setTabEnabled(2,True)
        #self.tabWidget.setTabEnabled(2,True)
        #self.tabWidget.setTabEnabled(3,True)
        if self.image is not None:
            self.tabWidget.setTabEnabled(3,True)

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.statusbar.showMessage('Generating model octree...')
        self.cadmodel.get_cell_locator()
        self.statusbar.clearMessage()
        self.app.restoreOverrideCursor()


    def on_load_image(self,newim):

        image = newim['image_data']
        
        # If the array isn't already 8-bit int, make it 8-bit int...
        if image.dtype != np.uint8:
            # If we're given a higher bit-depth integer, it's easy to downcast it.
            if image.dtype == np.uint16 or image.dtype == np.int16:
                image = np.uint8(image/2**8)
            elif image.dtype == np.uint32 or image.dtype == np.int32:
                image = np.uint8(image/2**24)
            elif image.dtype == np.uint64 or image.dtype == np.int64:
                image = np.uint8(image/2**56)


            # Otherwise, scale it in a floating point way to its own max & min
            # and strip out any transparency info (since we can't be sure of the scale used for transparency)
            else:

                if image.min() < 0:
                    image = image - image.min()

                if len(image.shape) == 3:
                    if image.shape[2] == 4:
                        image = image[:,:,:-1]

                image = np.uint8(255.*(image - image.min())/(image.max() - image.min()))

        self.image_geometry = CoordTransformer(offset=newim['image_offset'],paspect=newim['pixel_aspect'])
        self.image_geometry.set_image_shape(newim['image_data'].shape[1],newim['image_data'].shape[0],coords=newim['coords'])
        self.image_geometry.set_transform_actions(newim['transform_actions'])

        if self.calibration is not None:

            ioshape = self.image_geometry.get_original_shape()
            self.calibration.set_detector_window( (self.image_geometry.offset[0],self.image_geometry.offset[1],ioshape[0],ioshape[1]) )


        if newim['coords'].lower() == 'original':
            self.image = image
        elif self.calibration is not None:
            self.image = self.calibration.geometry.display_to_original_image(image)

        try:
            transformer = self.calibration.geometry
        except AttributeError:
            transformer = self.image_geometry


        self.interactor2d.set_image(transformer.original_to_display_image(self.image))


        if self.calibration is not None:
            self.interactor2d.set_subview_lookup(self.calibration.n_subviews,self.calibration.subview_lookup)

        self.image_settings.show()
            
        if self.enhance_checkbox.isChecked():
            self.enhance_checkbox.setChecked(False)
            self.enhance_checkbox.setChecked(True)

        
        self.update_image_info_string(self.image,transformer)
        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()

        if self.cadmodel is not None:
            self.tabWidget.setTabEnabled(3,True)

        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)
            self.overlay_checkbox.setChecked(True)
        


    def toggle_overlay(self,show=None):

        if show is None:
            if self.overlay_checkbox.isEnabled():
                self.overlay_checkbox.setChecked(not self.overlay_checkbox.isChecked())

        elif show:

            if self.overlay is None:

                oversampling = 1.
                self.statusbar.showMessage('Rendering wireframe overlay...')
                self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
                self.app.processEvents()
                try:
                    orig_colours = self.cadmodel.get_colour()
                    self.cadmodel.set_wireframe(True)
                    self.cadmodel.set_colour((0,0,1))
                    self.overlay = render_cam_view(self.cadmodel,self.calibration,transparency=True,verbose=False,aa=2)
                    self.cadmodel.set_colour(orig_colours)
                    self.cadmodel.set_wireframe(False)


                    if np.max(self.overlay) == 0:
                        dialog = qt.QMessageBox(self)
                        dialog.setStandardButtons(qt.QMessageBox.Ok)
                        dialog.setWindowTitle('Calcam - Information')
                        dialog.setTextFormat(qt.Qt.RichText)
                        dialog.setText('Wireframe overlay image is blank.')
                        dialog.setInformativeText('This usually means the fit is wildly wrong.')
                        dialog.setIcon(qt.QMessageBox.Information)
                        dialog.exec_()
                        
                
                except:
                    self.interactor2d.set_overlay_image(None)
                    self.statusbar.clearMessage()
                    self.overlay_checkbox.setChecked(False) 
                    self.app.restoreOverrideCursor()
                    raise


                self.statusbar.clearMessage()
                self.app.restoreOverrideCursor()


            self.interactor2d.set_overlay_image(self.overlay)
            self.refresh_2d()

        else:
            self.interactor2d.set_overlay_image(None)
   



    def toggle_enhancement(self,check_state):

        # Enable / disable adaptive histogram equalisation
        if check_state == qt.Qt.Checked:
            image = enhance_image(self.image)
        else:
            image = self.image

        if self.calibration is not None:
            transformer = self.calibration.geometry
        else:
            transformer = self.image_geometry

        self.interactor2d.set_image(transformer.original_to_display_image(image),hold_position=True)
        if self.calibration is not None:
            self.interactor2d.set_subview_lookup(self.calibration.n_subviews,self.calibration.subview_lookup)




    def on_change_cad_features(self):
        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.statusbar.showMessage('Updating model octree...')
        self.cadmodel.get_cell_locator()
        self.statusbar.clearMessage()
        self.app.restoreOverrideCursor()


    def update_position_info(self,coords_2d,coords_3d,visible):

        model_extent = self.cadmodel.get_extent()
        model_size = model_extent[1::2] - model_extent[::2]
        max_ray_length = model_size.max() * 4

        cadinfo_str = self.cadmodel.format_coord(coords_3d)

        sightline_info_string = ''

        iminfo_str = ''

        sightline_fieldnames_str = ''

        impos_fieldnames_str = ''

        self.cursor_closeup_button.setEnabled(False)

        for field_index in range(self.calibration.n_subviews):

            if field_index > 0:
                impos_fieldnames_str = impos_fieldnames_str + '<br><br>'
                sightline_fieldnames_str = sightline_fieldnames_str + '<br><br>'
                iminfo_str = iminfo_str + '<br><br>'
                sightline_info_string = sightline_info_string + '<br><br>'

            sightline_exists = False

            
            impos_fieldnames_str = impos_fieldnames_str + '[{:s}]&nbsp;'.format( self.calibration.subview_names[field_index] )

            if np.any(np.isnan(coords_2d[field_index][0])):
                iminfo_str = iminfo_str + ' Cursor outside field of view.'
            elif not visible[field_index]:
                iminfo_str = iminfo_str +  ' Cursor hidden from view.'
            else:
                iminfo_str = iminfo_str + ' X,Y : ( {:.0f} , {:.0f} ) px'.format(coords_2d[field_index][0][0],coords_2d[field_index][0][1])
                sightline_exists = True
                self.cursor_closeup_button.setEnabled(True)


            sightline_fieldnames_str = sightline_fieldnames_str + '[{:s}]&nbsp;'.format( self.calibration.subview_names[field_index] )
            if sightline_exists:
                sightline_fieldnames_str = sightline_fieldnames_str + '<br><br>'
                pupilpos = self.calibration.get_pupilpos(subview=field_index)
                sightline = coords_3d - pupilpos
                sdir = sightline / np.sqrt(np.sum(sightline**2))


                sightline_info_string = sightline_info_string + ' Origin X,Y,Z : ( {:.3f} , {:.3f} , {:.3f} )<br>'.format(pupilpos[0],pupilpos[1],pupilpos[2])
                sightline_info_string = sightline_info_string + ' Direction X,Y,Z : ( {:.3f} , {:.3f} , {:.3f} )<br>'.format(sdir[0],sdir[1],sdir[2])
                if np.sqrt(np.sum(sightline**2)) < (max_ray_length-1e-3):
                    sightline_info_string = sightline_info_string  +' Distance to camera : {:.3f} m'.format(np.sqrt(np.sum(sightline**2)))
                else:
                    sightline_info_string = sightline_info_string + ' Sight line does not inersect CAD model.'
                    cadinfo_str = ' Sight line does not intersect CAD model.'
            else:
                sightline_info_string = sightline_info_string + ' No line-of-sight to cursor<br>'

        if self.calibration.n_subviews > 1:
            self.impos_fieldnames.setText(impos_fieldnames_str)
            self.impos_fieldnames.show()
            self.sightline_fieldnames.setText(sightline_fieldnames_str)
            self.sightline_fieldnames.show()
        else:
            self.impos_fieldnames.hide()
            self.sightline_fieldnames.hide()

        self.impos_info.setText(iminfo_str)
        self.sightline_info.setText(sightline_info_string)
        self.cadpos_info.setText(cadinfo_str)