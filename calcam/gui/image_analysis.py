import cv2
import copy

from .core import *
from .vtkinteractorstyles import CalcamInteractorStyle2D, CalcamInteractorStyle3D
from ..calibration import Calibration
from ..render import render_cam_view
from ..coordtransformer import CoordTransformer
from ..raytrace import raycast_sightlines, check_visible

type_description = {'alignment': 'Manual Alignment', 'fit':'Point pair fitting','virtual':'Virtual'}

# Main calcam window class for actually creating calibrations.
class ImageAnalyserWindow(CalcamGUIWindow):
 
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
        self.im_flipud.clicked.connect(self.transform_image)
        self.im_fliplr.clicked.connect(self.transform_image)
        self.im_rotate_button.clicked.connect(self.transform_image)
        self.im_reset.clicked.connect(self.transform_image)
        self.im_y_stretch_button.clicked.connect(self.transform_image)
        self.load_calib_button.clicked.connect(self.load_calib)
        self.reset_view_button.clicked.connect(lambda: self.set_view_from_calib(self.calibration,0))
        self.cursor_closeup_button.clicked.connect(self.set_view_to_cursor)

        self.sightline_checkbox.toggled.connect(lambda: self.update_from_3d(self.coords_3d))

        self.overlay_checkbox.toggled.connect(self.toggle_overlay)

        self.hist_eq_checkbox.stateChanged.connect(self.toggle_hist_eq)

        self.original_image = None

        self.control_sensitivity_slider.valueChanged.connect(lambda x: self.interactor3d.set_control_sensitivity(x*0.01))
        self.rmb_rotate.toggled.connect(self.interactor3d.set_rmb_rotate)
        self.interactor3d.set_control_sensitivity(self.control_sensitivity_slider.value()*0.01)

        # If we have an old version of openCV, histo equilisation won't work :(
        cv2_version = float('.'.join(cv2.__version__.split('.')[:2]))
        cv2_micro_version = int(cv2.__version__.split('.')[2].split('-')[0])
        if cv2_version < 2.4 or (cv2_version == 2.4 and cv2_micro_version < 6):
            self.hist_eq_checkbox.setEnabled(False)
            self.hist_eq_checkbox.setToolTip('Requires OpenCV 2.4.6 or newer; you have {:s}'.format(cv2.__version__))



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

        if self.calibration is not None and self.original_image is not None and coords_3d is not None:

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
            image_pos_nocheck = self.calibration.project_points([coords_3d])

            image_pos = self.calibration.project_points([coords_3d],check_occlusion_by=self.cadmodel,occlusion_tol=1e-2)
            for i in range(len(image_pos)):
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
                        linesource.SetPoint1(pupilpos[0],pupilpos[1],pupilpos[2])
                        linesource.SetPoint2(intersection_coords)
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(linesource.GetOutputPort())
                        actor = vtk.vtkActor()
                        actor.SetMapper(mapper)
                        actor.GetProperty().SetLineWidth(invisible_linewidth)
                        actor.GetProperty().SetColor(visible_colour)
                        self.sightline_actors.append(actor)
                        self.interactor3d.add_extra_actor(actor)

                        linesource = vtk.vtkLineSource()
                        linesource.SetPoint1(intersection_coords)
                        linesource.SetPoint2(coords_3d)
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

        if self.calibration is not None and self.original_image is not None:
            for i,coords in enumerate(coords_2d):
                if coords is not None and np.any(coords != self.coords_2d[i]):
                    raydata = raycast_sightlines(self.calibration,self.cadmodel,coords[0],coords[1])
                    self.update_from_3d(raydata.ray_end_coords)
                    return



    def update_image_info_string(self):

        if np.any(self.image_geometry.get_display_shape() != self.image_geometry.get_original_shape()):
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP) [ As Displayed ]<br>{3:d} x {4:d} pixels ({5:.1f} MP) [ Raw Data ]<br>'.format(self.image_geometry.get_display_shape()[0],self.image_geometry.get_display_shape()[1],np.prod(self.image_geometry.get_display_shape()) / 1e6 ,self.image_geometry.get_original_shape()[0],self.image_geometry.get_original_shape()[1],np.prod(self.image_geometry.get_original_shape()) / 1e6 )
        else:
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP)<br>'.format(self.image_geometry.get_display_shape()[0],self.image_geometry.get_display_shape()[1],np.prod(self.image_geometry.get_display_shape()) / 1e6 )
        
        if len(self.original_image.shape) == 2:
            info_str = info_str + 'Monochrome'
        elif len(self.original_image.shape) == 3 and self.original_image.shape[2] == 3:
            info_str = info_str + 'RGB Colour'

        self.image_info.setText(info_str)




    def load_calib(self):

        opened_calib = self.object_from_file('calibration')

        if opened_calib is not None:
            if None in opened_calib.view_models:
                raise UserWarning('The selected calibration file does not contain a full set of calibration parameters. Only calibration files containing all calibration parameters can be used.')

            self.calibration = opened_calib
            self.calib_name.setText(os.path.split(self.calibration.filename)[1].replace('.ccc',''))
            imshape = self.calibration.geometry.get_display_shape()
            self.calib_im_size.setText('{:d} x {:d} pixels'.format(imshape[0],imshape[1]))
            self.calib_type.setText(type_description[self.calibration._type])

            self.overlay_checkbox.setEnabled(True)
            self.reset_view_button.setEnabled(True)
            self.overlay = None

            self.interactor2d.set_subview_lookup(self.calibration.n_subviews,self.calibration.subview_lookup)

            if opened_calib.cad_config is not None:
                cconfig = opened_calib.cad_config
                if self.cadmodel is not None and self.cadmodel.machine_name == cconfig['model_name'] and self.cadmodel.model_variant == cconfig['model_variant']:
                    keep_model = True
                else:
                    keep_model = False

                if keep_model:
                    self.cadmodel.enable_only(cconfig['enabled_features'])
                else:
                    cconfig = opened_calib.cad_config
                    load_model = True
                    try:
                        name_index = sorted(self.model_list.keys()).index(cconfig['model_name'])
                        self.model_name.setCurrentIndex(name_index)
                    except ValueError:
                        self.model_name.setCurrentIndex(-1)
                        load_model=False
                    try:    
                        variant_index = self.model_list[ cconfig['model_name'] ][1].index(cconfig['model_variant'])
                        self.model_variant.setCurrentIndex(variant_index)
                    except ValueError:
                        self.model_name.setCurrentIndex(-1)
                        load_model=False

                    if load_model:
                        self.load_model(featurelist=cconfig['enabled_features'])            

            self.set_view_from_calib(self.calibration,0)

            if self.cursor_ids['3d'] is not None:
                self.update_from_3d(self.coords_3d)
            else:
                self.coords_2d = [None] * self.calibration.n_subviews
                
            if self.original_image is not None:
                self.image_settings.setEnabled(True)
            else:
                self.image_settings.setEnabled(False)


    def set_view_to_cursor(self):
        cursorpos = self.interactor3d.get_cursor_coords(self.cursor_ids['3d'])
        pupilpos = self.calibration.get_pupilpos(subview=0)
        campos_vect = (pupilpos - cursorpos)
        campos_vect = campos_vect / np.sqrt(np.sum(campos_vect**2))

        self.camera_3d.SetFocalPoint(cursorpos)
        self.camera_3d.SetViewAngle(60)

        front_pos = cursorpos + 0.5*campos_vect
        if check_visible(front_pos,cursorpos,self.cadmodel):
            self.camera_3d.SetPosition(front_pos)
        else:
            self.camera_3d.SetPosition(cursorpos - 0.5*campos_vect)

        self.update_viewport_info()
        self.cam_roll.setValue(0)
        self.refresh_3d()


    def on_model_load(self):
        # Enable the other tabs!
        self.tabWidget.setTabEnabled(2,True)
        #self.tabWidget.setTabEnabled(2,True)
        #self.tabWidget.setTabEnabled(3,True)
        if self.original_image is not None:
            self.tabWidget.setTabEnabled(3,True)

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.statusbar.showMessage('Generating model octree...')
        self.cadmodel.get_cell_locator()
        self.statusbar.clearMessage()
        self.app.restoreOverrideCursor()


    def on_load_image(self,newim):


        self.image_geometry = CoordTransformer()
        self.image_geometry.set_pixel_aspect(newim['pixel_aspect'],relative_to='original')
        self.image_geometry.set_transform_actions(newim['transform_actions'])
        self.image_geometry.set_image_shape(newim['image_data'].shape[1],newim['image_data'].shape[0],coords=newim['coords'])

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

        if newim['coords'].lower() == 'original':
            self.original_image = image
        else:
            self.original_image = self.image_geometry.display_to_original_image(image)

        self.interactor2d.set_image(self.image_geometry.original_to_display_image(self.original_image))

        if self.calibration is not None:
            self.interactor2d.set_subview_lookup(self.calibration.n_subviews,self.calibration.subview_lookup)

        self.image_settings.show()
        
        if self.calibration is not None:
            self.image_settings.setEnabled(True)
        else:
            self.image_settings.setEnabled(False)
            
        if self.hist_eq_checkbox.isChecked():
            self.hist_eq_checkbox.setChecked(False)
            self.hist_eq_checkbox.setChecked(True)


        self.update_image_info_string()
        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()

        if self.cadmodel is not None:
            self.tabWidget.setTabEnabled(3,True)

        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)
            self.overlay_checkbox.setChecked(True)





    def transform_image(self,data):

        # First, back up the point pair locations in original coordinates.
        orig_coords = []

        if self.sender() is self.im_flipud:
            self.image_geometry.add_transform_action('flip_up_down')

        elif self.sender() is self.im_fliplr:
            self.image_geometry.add_transform_action('flip_left_right')

        elif self.sender() is self.im_rotate_button:
            self.image_geometry.add_transform_action('rotate_clockwise_{:d}'.format(self.im_rotate_angle.value()))

        elif self.sender() is self.im_y_stretch_button:
            self.image_geometry.set_pixel_aspect(self.im_y_stretch_factor.value(),absolute=False)
 
        elif self.sender() is self.im_reset:
            self.image_geometry.transform_actions = []
            self.image_geometry.pixel_aspectratio = 1


        # Update the image and point pairs
        self.interactor2d.set_image(self.image_geometry.original_to_display_image(self.original_image),n_subviews = self.calibration.n_subviews,subview_lookup=self.calibration.subview_lookup)

        if self.hist_eq_checkbox.isChecked():
            self.hist_eq_checkbox.setChecked(False)
            self.hist_eq_checkbox.setChecked(True)
 
        self.update_image_info_string()


        


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
   





    def toggle_hist_eq(self,check_state):

        im_out = self.image_geometry.original_to_display_image(self.original_image)

        # Enable / disable adaptive histogram equalisation
        if check_state == qt.Qt.Checked:
            hist_equaliser = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            if len(im_out.shape) == 2:
                im_out = hist_equaliser.apply(im_out.astype('uint8'))
            elif len(im_out.shape) > 2:
                for channel in range(3):
                    im_out[:,:,channel] = hist_equaliser.apply(im_out.astype('uint8')[:,:,channel]) 

        self.interactor2d.set_image(im_out,n_subviews = self.calibration.n_subviews,subview_lookup=self.calibration.subview_lookup,hold_position=True)




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