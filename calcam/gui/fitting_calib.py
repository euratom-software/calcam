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

from collections import deque

from scipy.ndimage.measurements import center_of_mass as CoM

from .core import *
from .vtkinteractorstyles import CalcamInteractorStyle2D, CalcamInteractorStyle3D
from ..calibration import Calibration, Fitter, ImageUpsideDown
from ..pointpairs import PointPairs
from ..render import render_cam_view
from .. import misc
from ..image_enhancement import enhance_image, scale_to_8bit
from ..movement import manual_movement

max_undo_depth = 20

# Main calcam window class for actually creating calibrations.
class FittingCalib(CalcamGUIWindow):

    def __init__(self, app, parent = None, load_file=None):

        # GUI initialisation
        CalcamGUIWindow.init(self,'fitting_calib.ui',app,parent)

        # Start up with no CAD model
        self.cadmodel = None
        self.calibration = Calibration(cal_type='fit')

        # Set up VTK
        self.qvtkwidget_3d = qt.QVTKRenderWindowInteractor(self.vtkframe_3d)
        self.vtkframe_3d.layout().addWidget(self.qvtkwidget_3d)
        self.interactor3d = CalcamInteractorStyle3D(refresh_callback=self.refresh_3d,viewport_callback=self.update_viewport_info,cursor_move_callback=self.update_cursor_position,newpick_callback=self.new_point_3d,focus_changed_callback=lambda x: self.change_point_focus('3d',x),resize_callback=self.update_vtk_size,pre_move_callback=self.record_undo_state)
        self.qvtkwidget_3d.SetInteractorStyle(self.interactor3d)
        self.renderer_3d = vtk.vtkRenderer()
        self.renderer_3d.SetBackground(0, 0, 0)
        self.qvtkwidget_3d.GetRenderWindow().AddRenderer(self.renderer_3d)
        self.camera_3d = self.renderer_3d.GetActiveCamera()

        self.qvtkwidget_2d = qt.QVTKRenderWindowInteractor(self.vtkframe_2d)
        self.vtkframe_2d.layout().addWidget(self.qvtkwidget_2d)
        self.interactor2d = CalcamInteractorStyle2D(refresh_callback=self.refresh_2d,newpick_callback = self.new_point_2d,cursor_move_callback=self.update_cursor_position,focus_changed_callback=lambda x: self.change_point_focus('2d',x),pre_move_callback=self.record_undo_state)
        self.qvtkwidget_2d.SetInteractorStyle(self.interactor2d)
        self.renderer_2d = vtk.vtkRenderer()
        self.renderer_2d.SetBackground(0, 0, 0)
        self.qvtkwidget_2d.GetRenderWindow().AddRenderer(self.renderer_2d)
        self.camera_2d = self.renderer_2d.GetActiveCamera()

        self.populate_models()



        # Disable image transform buttons if we have no image
        self.image_settings.hide()

        self.tabWidget.setTabEnabled(2,False)
        self.tabWidget.setTabEnabled(3,False)
        self.tabWidget.setTabEnabled(4,False)


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
        self.load_pointpairs_button.clicked.connect(self.load_pointpairs)
        self.fitted_points_checkbox.toggled.connect(self.toggle_reprojected)
        self.overlay_checkbox.toggled.connect(self.update_overlay)
        self.enhance_checkbox.stateChanged.connect(self.toggle_enhancement)
        self.im_define_splitFOV.clicked.connect(self.edit_split_field)
        self.pixel_size_checkbox.toggled.connect(self.update_pixel_size)
        self.pixel_size_box.valueChanged.connect(self.update_pixel_size)
        self.load_chessboard_button.clicked.connect(self.modify_chessboard_constraints)
        self.chessboard_checkbox.toggled.connect(self.toggle_chessboard_constraints)
        self.load_intrinsics_calib_button.clicked.connect(self.modify_intrinsics_calib)
        self.intrinsics_calib_checkbox.toggled.connect(self.toggle_intrinsics_calib)
        self.viewport_load_calib.clicked.connect(self.load_viewport_calib)
        self.action_save.triggered.connect(self.save_calib)
        self.action_save_as.triggered.connect(lambda: self.save_calib(saveas=True))
        self.action_open.triggered.connect(self.load_calib)
        self.action_new.triggered.connect(self.reset)
        self.action_cal_info.triggered.connect(self.show_calib_info)
        self.overlay_colour_button.clicked.connect(self.change_overlay_colour)
        self.comparison_overlay_colour_button.clicked.connect(self.change_comparison_colour)
        self.open_comparison_calib.clicked.connect(self.select_comparison_calib)
        self.comparison_overlay_checkbox.toggled.connect(self.update_overlay)
        self.overlay_type.currentIndexChanged.connect(self.update_overlay)
        self.comparison_overlay_type.currentIndexChanged.connect(self.update_overlay)
        self.comparison_overlay_opacity_slider.valueChanged.connect(self.change_comparison_colour)

        self.control_sensitivity_slider.valueChanged.connect(lambda x: self.interactor3d.set_control_sensitivity(x*0.01))
        self.rmb_rotate.toggled.connect(self.interactor3d.set_rmb_rotate)
        self.interactor3d.set_control_sensitivity(self.control_sensitivity_slider.value()*0.01)

        self.del_pp_button.clicked.connect(self.remove_current_pointpair)
        self.clear_points_button.clicked.connect(self.clear_pointpairs)

        self.points_undo_button.clicked.connect(self.pointpairs_undo)

        self.overlay_opacity_slider.valueChanged.connect(self.change_overlay_colour)

        # Set up some keyboard shortcuts
        # It is done this way in 3 lines per shortcut to avoid segfaults on some configurations
        sc = qt.QShortcut(qt.QKeySequence(qt.QKeySequence.Delete),self)
        sc.setContext(qt.Qt.WindowShortcut)
        sc.activated.connect(self.remove_current_pointpair)

        sc = qt.QShortcut(qt.QKeySequence("Ctrl+F"),self)
        sc.setContext(qt.Qt.WindowShortcut)
        sc.activated.connect(self.do_fit)

        sc = qt.QShortcut(qt.QKeySequence("Ctrl+P"),self)
        sc.setContext(qt.Qt.WindowShortcut)
        sc.activated.connect(self.toggle_reprojected)

        sc = qt.QShortcut(qt.QKeySequence(qt.QKeySequence.Undo),self)
        sc.setContext(qt.Qt.WindowShortcut)
        sc.activated.connect(self.pointpairs_undo)

        # Odds & sods
        self.pixel_size_box.setSuffix(u' \u00B5m')

        self.viewport_calibs = DodgyDict()
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

        self.chessboard_pointpairs = []

        self.point_pairings = []
        self.selected_pointpair = None

        self.fitters = []
        self.fit_settings_widgets = []

        self.fit_overlay = None
        self.comp_overlay = None

        self.chessboard_history = []

        self.fit_results = []

        self.filename = None

        self.waiting_pointpairs = None

        self.pointpairs_history = deque(maxlen=max_undo_depth)

        self.n_points = [ [0,0] ] # One list per sub-field, which is [extrinsics_data, intrinsics_data]

        self.intrinsics_calib = None

        self.fit_initted = False

        self.overlay_opacity_slider.setValue(self.config.main_overlay_colour[3]*100)
        self.overlay_appearance_controls.hide()
        self.comparison_overlay_appearance.hide()

        # Start the GUI!
        self.show()
        self.interactor2d.init()
        self.interactor3d.init()
        self.qvtkwidget_3d.GetRenderWindow().GetInteractor().Initialize()
        self.qvtkwidget_2d.GetRenderWindow().GetInteractor().Initialize()
        self.interactor3d.on_resize()

        if load_file is not None:
            self.load_calib(load_file)


    def modify_intrinsics_calib(self):
        loaded_calib = self.object_from_file('calibration')

        if loaded_calib is not None:
            self.intrinsics_calib = loaded_calib
            if self.intrinsics_calib_checkbox.isChecked():
                self.intrinsics_calib_checkbox.setChecked(False)

            self.intrinsics_calib_checkbox.setChecked(True)
            self.unsaved_changes = True


    def toggle_intrinsics_calib(self,on):

        if on and self.intrinsics_calib is None:
            self.modify_intrinsics_calib()
            if self.intrinsics_calib is None:
                self.intrinsics_calib_checkbox.setChecked(False)
        self.reset_fit()
        self.update_pointpairs()
        self.update_n_points()
        self.unsaved_changes = True


    def select_comparison_calib(self):

        cal = self.object_from_file('calibration')

        if cal is not None:

            if all([vm is None for vm in cal.view_models]):
                raise UserWarning('The selected calibration file does not contain a camera model so cannot be used for comparison.')

            curr_shape = self.calibration.geometry.get_display_shape()
            comp_shape = cal.geometry.get_display_shape()
            if curr_shape != comp_shape:
                raise UserWarning('The selected calibration has different image dimensions ({:d} x {:d}) to the current calibration ({:d} x {:d}), so cannot be used to compare!'.format(comp_shape[0],comp_shape[1],curr_shape[0],curr_shape[1]))

            self.comp_calib = cal
            self.comparison_overlay_checkbox.setEnabled(True)
            self.comparison_overlay_checkbox.setChecked(True)
            self.comparison_name.setText(('...' + cal.filename[-25:]) if len(cal.filename) > 28 else cal.filename)

        elif self.comp_calib is None:
            self.comparison_overlay_checkbox.setChecked(False)
            self.comparison_overlay_checkbox.setEnabled(False)



            oversampling = int(np.ceil(min(1000 / np.array(self.calibration.geometry.get_display_shape()))))

            fname = cal.filename

            self.statusbar.showMessage('Rendering wireframe overlay...')
            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.app.processEvents()

            orig_colours = self.cadmodel.get_colour()
            if self.comparison_overlay_type.currentIndex() == 0:
                self.cadmodel.set_wireframe(True)
            self.cadmodel.set_colour((1,1,1))
            self.comp_overlay = render_cam_view(self.cadmodel,cal,transparency=True,verbose=False,aa=2,oversampling=oversampling)
            self.cadmodel.set_colour(orig_colours)
            self.cadmodel.set_wireframe(False)

            self.statusbar.clearMessage()
            self.app.restoreOverrideCursor()
            self.comparison_overlay_checkbox.setEnabled(True)
            self.comparison_overlay_checkbox.setChecked(True)


    def on_close(self):
        self.qvtkwidget_2d.close()
        self.qvtkwidget_3d.close()


    def reset(self,keep_cadmodel=False):

        self.fit_initted = False

        if not keep_cadmodel:
            self.tabWidget.setTabEnabled(2,False)
            if self.cadmodel is not None:
                self.cadmodel.remove_from_renderer(self.renderer_3d)
                self.cadmodel.unload()
                self.feature_tree.blockSignals(True)
                self.feature_tree.clear()
                self.feature_tree.blockSignals(False)
                self.cadmodel = None

        self.clear_pointpairs()

        self.interactor2d.set_image(None)

        self.calibration = Calibration(cal_type='fit')

        self.image_settings.hide()

        self.comp_overlay = None
        self.comparison_overlay_checkbox.setChecked(False)
        self.comparison_overlay_checkbox.setEnabled(False)
        self.comparison_name.setText('No comparison calibration loaded')

        self.tabWidget.setTabEnabled(3,False)
        self.tabWidget.setTabEnabled(4,False)

        self.filename = None
        self.setWindowTitle('Calcam Calibration Tool (Point Fitting)')

        self.chessboard_pointpairs = []
        self.chessboard_checkbox.setChecked(False)
        self.intrinsics_calib = None
        self.intrinsics_calib_checkbox.setChecked(False)

        self.refresh_2d()
        self.refresh_3d()
        self.unsaved_changes = False


    def change_overlay_colour(self):

        if self.sender() is self.overlay_opacity_slider:
            new_colour = self.config.main_overlay_colour[:3]

        else:
            old_colour = self.config.main_overlay_colour
            new_colour = self.pick_colour(init_colour=old_colour)

        if new_colour is not None:

            new_colour = new_colour + [self.overlay_opacity_slider.value() / 100]
            self.config.main_overlay_colour = new_colour

            self.update_overlay()


    def change_comparison_colour(self):

        if self.sender() is self.comparison_overlay_opacity_slider:
            new_colour = self.config.second_overlay_colour[:3]

        else:
            old_colour = self.config.second_overlay_colour
            new_colour = self.pick_colour(init_colour=old_colour)

        if new_colour is not None:

            new_colour = new_colour + [self.comparison_overlay_opacity_slider.value()/100]
            self.config.second_overlay_colour = new_colour

            self.update_overlay()


    def reset_fit(self,subview=None):

        if subview is None:
            if not any(self.calibration.view_models):
                return
            self.calibration.view_models = [None] * self.calibration.n_subviews
            self.calibration.history['fit'] = [None] * self.calibration.n_subviews
        else:
            if self.calibration.view_models[subview] is None:
                return
            self.calibration.view_models[subview] = None
            self.calibration.history['fit'][subview] = None

        self.fit_overlay = None
        self.update_fit_results(show_points = self.fitted_points_checkbox.isChecked())

        self.unsaved_changes = True


    def update_cursor_position(self,cursor_id,new_position):

        self.unsaved_changes = True
        self.update_cursor_info()
        self.update_pointpairs()


    def record_undo_state(self):

        pp = self.calibration.pointpairs
        hist = self.calibration.history['pointpairs']
        try:
            coords3d = self.interactor3d.get_cursor_coords(self.point_pairings[self.selected_pointpair][0])
        except Exception:
            coords3d = None

        self.pointpairs_history.append((pp,hist,coords3d))
        self.points_undo_button.setEnabled(True)


    def new_point_2d(self,im_coords):

        self.record_undo_state()

        if self.selected_pointpair is not None:
            if self.point_pairings[self.selected_pointpair][1] is None:
                self.point_pairings[self.selected_pointpair][1] = self.interactor2d.add_active_cursor(im_coords)
                self.interactor2d.set_cursor_focus(self.point_pairings[self.selected_pointpair][1])
                self.update_pointpairs()
                self.update_n_points()
                return

        self.point_pairings.append( [None,self.interactor2d.add_active_cursor(im_coords)] )
        self.interactor2d.set_cursor_focus(self.point_pairings[-1][1])
        self.interactor3d.set_cursor_focus(None)
        self.selected_pointpair = len(self.point_pairings) - 1
        self.update_cursor_info()

        self.update_n_points()


    def new_point_3d(self,coords):

        self.record_undo_state()
        if self.selected_pointpair is not None:
            if self.point_pairings[self.selected_pointpair][0] is None:
                self.point_pairings[self.selected_pointpair][0] = self.interactor3d.add_cursor(coords)
                self.interactor3d.set_cursor_focus(self.point_pairings[self.selected_pointpair][0])
                self.update_pointpairs()
                self.update_n_points()
                return

        self.point_pairings.append( [self.interactor3d.add_cursor(coords),None] )
        self.interactor3d.set_cursor_focus(self.point_pairings[-1][0])
        self.interactor2d.set_cursor_focus(None)
        self.selected_pointpair = len(self.point_pairings) - 1
        self.update_cursor_info()

        self.update_n_points()




    def change_point_focus(self,sender,new_focus):

        if self.selected_pointpair is not None:
            if None in self.point_pairings[self.selected_pointpair]:
                if self.point_pairings[self.selected_pointpair][0] is not None:
                    self.interactor3d.remove_cursor(self.point_pairings[self.selected_pointpair][0])
                if self.point_pairings[self.selected_pointpair][1] is not None:
                    self.interactor2d.remove_active_cursor(self.point_pairings[self.selected_pointpair][1])
                self.point_pairings.remove(self.point_pairings[self.selected_pointpair])

        if sender == '3d':
            for i,pointpair in enumerate(self.point_pairings):
                if pointpair[0] == new_focus:
                    self.interactor2d.set_cursor_focus(pointpair[1])
                    self.selected_pointpair = i
        elif sender == '2d':
            for i,pointpair in enumerate(self.point_pairings):
                if pointpair[1] == new_focus:
                    self.interactor3d.set_cursor_focus(pointpair[0])
                    self.selected_pointpair = i

        self.update_cursor_info()


    def update_cursor_info(self):

        if self.selected_pointpair is not None:
            if self.point_pairings[self.selected_pointpair][0] is not None:
                object_coords = self.interactor3d.get_cursor_coords(self.point_pairings[self.selected_pointpair][0])
            else:
                object_coords = None

            if self.point_pairings[self.selected_pointpair][1] is not None:
                image_coords = self.interactor2d.get_cursor_coords(self.point_pairings[self.selected_pointpair][1])
            else:
                image_coords = None

            info_string = ''

            if object_coords is not None and self.cadmodel is not None:
                info_string = info_string + '<span style=" text-decoration: underline;">CAD Point<br></span>' + self.cadmodel.format_coord(object_coords).replace('\n','<br>') + '<br><br>'

            if image_coords is not None:
                info_string = info_string + '<span style=" text-decoration: underline;">Image Point(s)</span><br>'

                for i,point in enumerate(image_coords):
                    if point is not None:
                        if sum(x is not None for x in image_coords) > 1:
                            info_string = info_string + self.calibration.subview_names[i] + ': '.replace(' ','&nbsp;')

                        info_string = info_string + '( {:.0f} , {:.0f} ) px'.format(point[0],point[1]).replace(' ','&nbsp;')

                        if i < len(image_coords) - 1:
                            info_string = info_string + '<br>'


            self.del_pp_button.setEnabled(True)

        else:
            info_string = 'No selection'
            self.del_pp_button.setEnabled(False)

        self.point_info_text.setText(info_string)



    def on_model_load(self):
        # Enable the other tabs!
        self.tabWidget.setTabEnabled(2,True)
        self.update_fit_results()
        #self.tabWidget.setTabEnabled(2,True)
        #self.tabWidget.setTabEnabled(3,True)


    def on_load_image(self,newim):

        imshape = newim['image_data'].shape[1::-1]

        # If we already have an image in the calibration, check if the new one is the same size.
        try:
            # If the new image is the same size, assume all its metadata are the same as the existing image.
            if imshape == tuple(self.calibration.geometry.get_display_shape()) or imshape == tuple(self.calibration.geometry.get_original_shape()):

                if self.calibration.geometry.get_display_shape() == self.calibration.geometry.get_original_shape():
                    if 'coords' not in newim:
                        newim['coords'] = 'display'
                elif imshape == tuple(self.calibration.geometry.get_display_shape()):
                    newim['coords'] = 'display'
                else:
                    newim['coords'] = 'original'

                newim['subview_names'] = self.calibration.subview_names
                newim['subview_mask'] = self.calibration.get_subview_mask(coords=newim['coords'])
                newim['transform_actions'] = self.calibration.geometry.transform_actions
                newim['pixel_aspect'] = self.calibration.geometry.pixel_aspectratio
                newim['pixel_size'] = self.calibration.pixel_size

            else:
                # If the new image is the wrong size, we have to start a new calibration.
                msg = 'The selected image has different dimensions ({:d}x{:d}) to the current one so cannot be loaded without starting a new calibration.<br><br>Do you want to start a new calibration based on the loaded image?'.format(imshape[0],imshape[1])
                reply = qt.QMessageBox.question(self, 'Start new calibration?', msg, qt.QMessageBox.Yes, qt.QMessageBox.Cancel)

                if reply == qt.QMessageBox.Cancel:
                    return
                elif reply == qt.QMessageBox.Yes:
                    self.reset()

            old_image = self.calibration.get_image(coords='Display')

        # A TypeError signifies no image in the existing calibration
        except TypeError:
            old_image = None
            pass

        if newim['pixel_size'] is not None:
            self.pixel_size_checkbox.setChecked(True)
            self.pixel_size_box.setValue(newim['pixel_size']*1e6)
        else:
            self.pixel_size_checkbox.setChecked(False)

        self.calibration.set_image( scale_to_8bit(newim['image_data']) , newim['source'],subview_mask = newim['subview_mask'], transform_actions = newim['transform_actions'],coords=newim['coords'],subview_names=newim['subview_names'],pixel_aspect=newim['pixel_aspect'],pixel_size=newim['pixel_size'],offset=newim['image_offset'] )

        self.interactor2d.set_image(self.calibration.get_image(coords='Display'),n_subviews = self.calibration.n_subviews,subview_lookup = self.calibration.subview_lookup)

        oshape = self.calibration.geometry.get_original_shape()
        self.detector_window = (int(self.calibration.geometry.offset[0]),int(self.calibration.geometry.offset[1]),int(oshape[0]),int(oshape[1]))

        self.image_settings.show()
        if self.enhance_checkbox.isChecked():
            self.enhance_checkbox.setChecked(False)
            self.enhance_checkbox.setChecked(True)

        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)
            self.overlay_checkbox.setChecked(True)
        elif self.comparison_overlay_checkbox.isChecked():
            self.comparison_overlay_checkbox.setChecked(False)
            self.comparison_overlay_checkbox.setChecked(True)

        if not self.fit_initted:
            self.init_fitting()

        self.unsaved_changes = True

        self.update_image_info_string(newim['image_data'],self.calibration.geometry)

        if old_image is not None and len(self.point_pairings) > 0:
            self.app.restoreOverrideCursor()
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
            dialog.setWindowTitle('Adjust points all at once?')
            dialog.setTextFormat(qt.Qt.RichText)
            dialog.setText('Would you like to open the image movement tool to adjust the calibration points all at once?')
            dialog.setIcon(qt.QMessageBox.Question)
            retcode = dialog.exec()

            if retcode == dialog.Yes:
                pos_lim = np.array(self.calibration.geometry.get_display_shape()) - 0.5
                pp_to_remove = []
                movement = manual_movement(old_image,self.calibration.get_image(coords='Display'),parent_window=self)
                if movement is not None:
                    self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
                    for pp in self.point_pairings:
                        orig_coords = self.interactor2d.get_cursor_coords(pp[1])
                        new_coords = [None] * len(orig_coords)
                        for subview in range(len(orig_coords)):
                            if orig_coords[subview] is not None:
                                new_coords[subview] = movement.ref_to_moved_coords(*orig_coords[subview])
                                if np.all( np.array(new_coords[subview]) >= 0) and np.all( np.array(new_coords[subview]) < pos_lim):
                                    self.interactor2d.set_cursor_coords(pp[1],new_coords[subview],subview=subview)
                                else:
                                    new_coords[subview] = None
                        if all(p is None for p in new_coords):
                            pp_to_remove.append(pp)

                    for pp in pp_to_remove:
                        if pp[0] is not None:
                            self.interactor3d.remove_cursor(pp[0])
                        if pp[1] is not None:
                            self.interactor2d.remove_active_cursor(pp[1])

                        r_index = self.point_pairings.index(pp)
                        if self.selected_pointpair == r_index:
                            if len(self.point_pairings) > 1:
                                self.selected_pointpair = (self.selected_pointpair - 1) % len(self.point_pairings)
                                self.interactor2d.set_cursor_focus(self.point_pairings[self.selected_pointpair][1])
                                self.interactor3d.set_cursor_focus(self.point_pairings[self.selected_pointpair][0])
                            else:
                                self.selected_pointpair = None
                                self.interactor3d.set_cursor_focus(None)
                                self.interactor2d.set_cursor_focus(None)

                        self.point_pairings.remove(pp)

                    self.app.restoreOverrideCursor()
                    self.init_fitting()


    def change_fit_params(self,fun,state):

        fun(state)
        self.fit_enable_check()


    def init_fitting(self,reset_fit=True,reset_options=False):

        self.fit_initted = False

        if reset_fit:
            self.reset_fit()

        # Build the GUI to show fit options, according to the number of fields.
        current_tab = self.subview_tabs.currentIndex()
        self.subview_tabs.clear()

        # List of settings widgets (for showing / hiding when changing model)
        self.perspective_settings = []
        self.fit_settings_widgets = []
        self.fisheye_settings = []
        self.fit_buttons = []
        self.fit_results = []

        if reset_options or len(self.fitters) != self.calibration.n_subviews:
            self.fitters = []

        self.fit_results_widgets = []
        self.view_to_fit_buttons = []

        for field in range(self.calibration.n_subviews):

            if reset_options or len(self.fitters) != self.calibration.n_subviews:
                self.fitters.append(Fitter())

            new_tab = qt.QWidget()
            new_layout = qt.QVBoxLayout()

            options_groupbox = qt.QGroupBox('Fit Options')
            options_layout = qt.QGridLayout()

            # Selection of model
            widgetlist = [qt.QRadioButton('Rectilinear Lens'),qt.QRadioButton('Fisheye Lens')]

            if int(cv2.__version__[0]) < 3:
                widgetlist[1].setEnabled(False)
                widgetlist[1].setToolTip('Requires OpenCV 3')

            widgetlist[0].toggled.connect(self.change_dist_model)
            widgetlist[1].toggled.connect(self.change_dist_model)
            sub_widget = qt.QWidget()
            sub_layout = qt.QHBoxLayout()
            sub_widget.setLayout(sub_layout)
            sub_layout.addWidget(widgetlist[0])
            sub_layout.addWidget(widgetlist[1])
            sub_layout.setContentsMargins(0,0,0,0)
            options_layout.addWidget(sub_widget)


            # Settings for perspective model
            #---------------------------------
            self.perspective_settings.append( qt.QWidget() )
            perspective_settings_layout = qt.QVBoxLayout()
            perspective_settings_layout.setContentsMargins(0,0,0,0)
            self.perspective_settings[-1].setLayout(perspective_settings_layout)

            widgetlist = widgetlist + [qt.QCheckBox('Disable k1'),qt.QCheckBox('Disable k2'),qt.QCheckBox('Disable k3')]

            sub_widget = qt.QWidget()
            sub_layout = qt.QHBoxLayout()
            sub_widget.setLayout(sub_layout)
            sub_layout.addWidget(widgetlist[-3])
            sub_layout.addWidget(widgetlist[-2])
            sub_layout.addWidget(widgetlist[-1])
            sub_layout.setContentsMargins(0,0,0,0)
            perspective_settings_layout.addWidget(sub_widget)

            widgetlist[-3].setChecked(self.fitters[field].fixk1)
            widgetlist[-3].toggled.connect(lambda state,field=field: self.change_fit_params(self.fitters[field].fix_k1,state))
            widgetlist[-2].setChecked(self.fitters[field].fixk2)
            widgetlist[-2].toggled.connect(lambda state,field=field: self.change_fit_params(self.fitters[field].fix_k2,state))
            widgetlist[-1].setChecked(self.fitters[field].fixk3)
            widgetlist[-1].toggled.connect(lambda state,field=field: self.change_fit_params(self.fitters[field].fix_k3,state))
            widgetlist.append(qt.QCheckBox('Disable tangential distortion'))
            perspective_settings_layout.addWidget(widgetlist[-1])
            widgetlist[-1].setChecked(self.fitters[field].disabletangentialdist)
            widgetlist[-1].toggled.connect(lambda state,field=field: self.change_fit_params(self.fitters[field].fix_tangential,state))
            widgetlist.append(qt.QCheckBox('Fix Fx = Fy'))
            widgetlist[-1].setChecked(self.fitters[field].fixaspectratio)
            widgetlist[-1].toggled.connect(lambda state,field=field: self.change_fit_params(self.fitters[field].fix_aspect,state))
            perspective_settings_layout.addWidget(widgetlist[-1])


            # ------- End of perspective settings -----------------

            # Settings for fisheye model
            #---------------------------------
            self.fisheye_settings.append( qt.QWidget() )
            fisheye_settings_layout = qt.QVBoxLayout()
            fisheye_settings_layout.setContentsMargins(0,0,0,0)
            self.fisheye_settings[-1].setLayout(fisheye_settings_layout)

            widgetlist = widgetlist + [qt.QCheckBox('Disable k1'),qt.QCheckBox('Disable k2'),qt.QCheckBox('Disable k3'),qt.QCheckBox('Disable k4')]

            sub_widget = qt.QWidget()
            sub_layout = qt.QGridLayout()
            sub_widget.setLayout(sub_layout)
            sub_layout.addWidget(widgetlist[-4],0,0)
            sub_layout.addWidget(widgetlist[-3],0,1)
            sub_layout.addWidget(widgetlist[-2],0,2)
            sub_layout.addWidget(widgetlist[-1],1,0)
            sub_layout.setContentsMargins(0,0,0,0)
            fisheye_settings_layout.addWidget(sub_widget)

            widgetlist[-4].setChecked(self.fitters[field].fixk1)
            widgetlist[-4].toggled.connect(lambda state,field=field: self.change_fit_params(self.fitters[field].fix_k1,state))
            widgetlist[-3].setChecked(self.fitters[field].fixk2)
            widgetlist[-3].toggled.connect(lambda state,field=field: self.change_fit_params(self.fitters[field].fix_k2,state))
            widgetlist[-2].setChecked(self.fitters[field].fixk3)
            widgetlist[-2].toggled.connect(lambda state,field=field: self.change_fit_params(self.fitters[field].fix_k3,state))
            widgetlist[-1].setChecked(self.fitters[field].fixk4)
            widgetlist[-1].toggled.connect(lambda state,field=field: self.change_fit_params(self.fitters[field].fix_k4,state))

            for widgetno in [-4,-3,-2,-1]:
                widgetlist[widgetno].toggled.connect(self.fit_enable_check)

            # ------- End of fisheye settings -----------------


            options_layout.addWidget(self.perspective_settings[-1])
            options_layout.addWidget(self.fisheye_settings[-1])

            options_groupbox.setLayout(options_layout)

            fit_button = qt.QPushButton('Do Fit')

            fit_button.clicked.connect(self.do_fit)
            #fit_button.setEnabled(False)
            options_layout.addWidget(fit_button)
            self.fit_buttons.append(fit_button)

            self.fit_settings_widgets.append(widgetlist)

            new_layout.addWidget(options_groupbox)

            results_groupbox = qt.QGroupBox('Fit Results')
            results_layout = qt.QGridLayout()

            self.fit_results.append(results_groupbox)
            results_groupbox.setHidden(True)

            model = self.fitters[field].model
            widgetlist[0].setChecked(model == 'rectilinear')
            widgetlist[1].setChecked(model == 'fisheye')


            # Build GUI to show the fit results, according to the number of fields.

            widgets = [ qt.QLabel('Fit RMS residual = ') , qt.QLabel('Parameter names'),  qt.QLabel('Parameter values'), qt.QPushButton('Set CAD view to match fit')]
            self.view_to_fit_buttons.append(widgets[-1])
            widgets[1].setAlignment(qt.Qt.AlignRight)
            widgets[3].clicked.connect(self.set_fit_viewport)
            results_layout.addWidget(widgets[0],0,0,1,-1)
            results_layout.addWidget(widgets[1],1,0)
            results_layout.addWidget(widgets[2],1,1)
            results_layout.addWidget(widgets[3],2,0,1,-1)
            self.fit_results_widgets.append(widgets)
            results_layout.setColumnMinimumWidth(0,90)
            results_groupbox.setLayout(results_layout)

            new_layout.addWidget(results_groupbox)

            vspace = qt.QSpacerItem(20, 0, qt.QSizePolicy.Minimum, qt.QSizePolicy.Expanding)
            new_layout.addItem(vspace)

            new_tab.setLayout(new_layout)
            self.subview_tabs.addTab(new_tab,self.calibration.subview_names[field])


            for fitter in self.fitters:
                fitter.set_image_shape(self.calibration.geometry.get_display_shape())

            self.tabWidget.setTabEnabled(3,True)
            self.tabWidget.setTabEnabled(4,True)

        if current_tab < self.subview_tabs.count():
            self.subview_tabs.setCurrentIndex(current_tab)

        self.fit_initted = True



    def transform_image(self,data):

        # First, back up the point pair locations in original coordinates.
        orig_pointpairs = self.calibration.geometry.display_to_original_pointpairs(self.calibration.pointpairs)

        self.pointpairs_history.clear()
        self.points_undo_button.setEnabled(False)

        for i in range(len(self.chessboard_pointpairs)):
            self.chessboard_pointpairs[i][0] = self.calibration.geometry.display_to_original_image(self.chessboard_pointpairs[i][0])
            self.chessboard_pointpairs[i][1] = self.calibration.geometry.display_to_original_pointpairs(self.chessboard_pointpairs[i][1])

        if self.sender() is self.im_flipud:
            self.calibration.geometry.add_transform_action('flip_up_down')

        elif self.sender() is self.im_fliplr:
            self.calibration.geometry.add_transform_action('flip_left_right')

        elif self.sender() is self.im_rotate_button:
            self.calibration.geometry.add_transform_action('rotate_clockwise_{:d}'.format(self.im_rotate_angle.value()))

        elif self.sender() is self.im_y_stretch_button:
            self.calibration.geometry.set_pixel_aspect(self.im_y_stretch_factor.value(),absolute=False)

        elif self.sender() is self.im_reset:
            self.calibration.geometry.set_transform_actions([])
            self.calibration.geometry.set_pixel_aspect(1)

        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)

        if self.comparison_overlay_checkbox.isChecked():
            self.comparison_overlay_checkbox.setChecked(False)

        if self.fitted_points_checkbox.isChecked():
            self.fitted_points_checkbox.setChecked(False)


        # Update the image and point pairs
        self.interactor2d.set_image(self.calibration.get_image(coords='Display'),n_subviews = self.calibration.n_subviews,subview_lookup=self.calibration.subview_lookup)
        if orig_pointpairs is not None:
            self.load_pointpairs(pointpairs = self.calibration.geometry.original_to_display_pointpairs(orig_pointpairs),history=self.calibration.history['pointpairs'],force_clear=True,clear_fit=False)

        for i in range(len(self.chessboard_pointpairs)):
            self.chessboard_pointpairs[i][0] = self.calibration.geometry.original_to_display_image(self.chessboard_pointpairs[i][0])
            self.chessboard_pointpairs[i][1] = self.calibration.geometry.original_to_display_pointpairs(self.chessboard_pointpairs[i][1])

        if self.enhance_checkbox.isChecked():
            self.enhance_checkbox.setChecked(False)
            self.enhance_checkbox.setChecked(True)


        self.update_image_info_string(self.calibration.get_image(),self.calibration.geometry)
        self.init_fitting()
        self.unsaved_changes = True


    def load_pointpairs(self,data=None,pointpairs=None,src=None,history=None,force_clear=None,clear_fit=True,include_in_undo=True):

        if include_in_undo:
            self.record_undo_state()

        if pointpairs is None:
            pointpairs = self.object_from_file('pointpairs')

        if pointpairs is not None:

            if pointpairs.n_subviews != self.calibration.n_subviews:
                raise UserWarning('These point pairs cannot be used with the current image because they contain a different number of sub-views ({:d} in point pairs vs. {:d} in image).'.format(pointpairs.n_subviews,self.calibration.n_subviews))

            try:
                history = pointpairs.history
            except AttributeError:
                pass
            try:
                src = pointpairs.src
            except AttributeError:
                pass

            if history is None and src is None:
                raise Exception('History or source of the loaded point pairs must be specified!')

            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.fitted_points_checkbox.setChecked(False)
            self.overlay_checkbox.setChecked(False)

            if (self.pointpairs_clear_before_load.isChecked() or force_clear) and force_clear != False:
                self.clear_pointpairs(include_in_undo=False)

            for i in range(len(pointpairs.object_points)):

                cursorid_2d = None
                for j in range(len(pointpairs.image_points[i])):
                    if pointpairs.image_points[i][j] is not None:
                        if np.min(pointpairs.image_points[i][j]) >= 0 and pointpairs.image_points[i][j][0] < self.calibration.geometry.get_display_shape()[0] and pointpairs.image_points[i][j][1] < self.calibration.geometry.get_display_shape()[1]:
                            if cursorid_2d is None:
                                cursorid_2d = self.interactor2d.add_active_cursor(pointpairs.image_points[i][j])
                            else:
                                self.interactor2d.add_active_cursor(pointpairs.image_points[i][j],add_to=cursorid_2d)

                if cursorid_2d is not None:
                    cursorid_3d = self.interactor3d.add_cursor(pointpairs.object_points[i])
                    self.point_pairings.append([cursorid_3d,cursorid_2d])

            self.update_pointpairs(src=src,history=history,clear_fit=clear_fit)
            self.update_n_points()
            self.update_cursor_info()
            self.app.restoreOverrideCursor()
            self.unsaved_changes = True



    def toggle_reprojected(self,show=None):

        if show is None:
            if self.fitted_points_checkbox.isEnabled():
                self.fitted_points_checkbox.setChecked(not self.fitted_points_checkbox.isChecked())

        elif show:
            self.overlay_checkbox.setChecked(False)
            points = self.calibration.pointpairs.object_points
            projected = self.calibration.project_points(points)
            for point_list in projected:
                for point in point_list:
                    self.interactor2d.add_passive_cursor(point)
        else:
            self.interactor2d.clear_passive_cursors()


    def remove_current_pointpair(self):

        if self.selected_pointpair is not None:

            self.record_undo_state()

            pp_to_remove = self.point_pairings.pop(self.selected_pointpair)

            if len(self.point_pairings) > 0:
                self.selected_pointpair = (self.selected_pointpair - 1) % len(self.point_pairings)
                self.interactor2d.set_cursor_focus(self.point_pairings[self.selected_pointpair][1])
                self.interactor3d.set_cursor_focus(self.point_pairings[self.selected_pointpair][0])
            else:
                self.selected_pointpair = None
                self.interactor3d.set_cursor_focus(None)
                self.interactor2d.set_cursor_focus(None)

            if pp_to_remove[0] is not None:
                self.interactor3d.remove_cursor(pp_to_remove[0])
            if pp_to_remove[1] is not None:
                self.interactor2d.remove_active_cursor(pp_to_remove[1])

            self.update_cursor_info()
            self.update_pointpairs()
            self.update_n_points()

    def pointpairs_undo(self):

        try:
            prev_pointpairs,pp_history,selected_coords = self.pointpairs_history.pop()
            if len(self.pointpairs_history) == 0:
                self.points_undo_button.setEnabled(False)
        except IndexError:
            raise UserWarning('Reached the end of undo history - nothing more to undo.')

        self.load_pointpairs(pointpairs=prev_pointpairs,history=pp_history,force_clear=True,include_in_undo=False)


        for cid in self.interactor3d.cursors.keys():
            if self.interactor3d.get_cursor_coords(cid) == selected_coords:
                self.interactor3d.set_cursor_focus(cid)
                self.change_point_focus('3d',cid)
                break

    def clear_pointpairs(self,include_in_undo=True):

        if include_in_undo:
            self.record_undo_state()

        self.interactor3d.set_cursor_focus(None)
        self.interactor2d.set_cursor_focus(None)
        self.selected_pointpair = None

        for pp in self.point_pairings:
            if pp[0] is not None:
                self.interactor3d.remove_cursor(pp[0])
            if pp[1] is not None:
                self.interactor2d.remove_active_cursor(pp[1])

        self.point_pairings = []

        self.reset_fit()
        self.update_n_points()


    def fit_enable_check(self):

        # This avoids raising errors if this function is called when we have no
        # fit options GUI.
        if len(self.fit_settings_widgets) < self.calibration.n_subviews:
            return


        # Check whether or not we have enough points to enable the fit button.
        for i in range(self.calibration.n_subviews):
            enable = True

            # We need at least 4 extrinsics points and at least as many total points as free parameters
            if self.n_points[i][0] < 4 or np.sum(self.n_points[i]) < 6:
                enable = False

            self.fit_buttons[i].setEnabled(enable)
            if enable:
                self.fit_buttons[i].setToolTip('Do fit')
            else:
                self.fit_buttons[i].setToolTip('Cannot fit: more free parameters than point pair data.')



    def update_pointpairs(self,src=None,history=None,clear_fit=True):

        pp = PointPairs()

        for pointpair in self.point_pairings:
            if pointpair[0] is not None and pointpair[1] is not None:
                pp.add_pointpair(self.interactor3d.get_cursor_coords(pointpair[0]) , self.interactor2d.get_cursor_coords(pointpair[1]) )

        if clear_fit and self.calibration.pointpairs is not None:
            for subview,same in enumerate(pp == self.calibration.pointpairs):
                if not same:
                    self.reset_fit(subview=subview)

        if pp.get_n_points() > 0:

            self.calibration.set_pointpairs(pp,src=src,history=history)

            for subview in range(self.calibration.n_subviews):
                self.fitters[subview].set_pointpairs(self.calibration.pointpairs,subview=subview)
                self.fitters[subview].clear_intrinsics_pointpairs()

        else:
            self.calibration.set_pointpairs(None)


        # Add the intrinsics constraints
        self.calibration.clear_intrinsics_constraints()

        if self.intrinsics_calib_checkbox.isChecked():
            self.calibration.add_intrinsics_constraints(calibration=self.intrinsics_calib)
            for subview in range(self.calibration.n_subviews):
                self.fitters[subview].add_intrinsics_pointpairs(self.intrinsics_calib.pointpairs,subview=subview)
                for ic in self.intrinsics_calib.intrinsics_constraints:
                    self.fitters[subview].add_intrinsics_pointpairs(ic[1])

        if self.chessboard_checkbox.isChecked():
            for n,chessboard_constraint in enumerate(self.chessboard_pointpairs):
                self.calibration.add_intrinsics_constraints(image=chessboard_constraint[0],im_history=self.chessboard_history[n][0],pointpairs = chessboard_constraint[1],pp_history=self.chessboard_history[n][1])
                for subview in range(self.calibration.n_subviews):
                    self.fitters[subview].add_intrinsics_pointpairs(chessboard_constraint[1],subview=subview)


        self.unsaved_changes = True



    def update_fit_results(self,show_points=True):

        self.overlay_checkbox.setChecked(False)
        self.overlay_checkbox.setEnabled(False)
        self.fitted_points_checkbox.setChecked(False)
        self.fitted_points_checkbox.setEnabled(False)

        if not self.fit_initted:
            return

        for subview in range(self.calibration.n_subviews):

            if self.calibration.view_models[subview] is None:
                self.fit_results[subview].hide()
                continue

            # Put the results in to the GUI
            params = self.calibration.view_models[subview]

            # Get CoM of this field on the chip
            ypx,xpx = CoM( self.calibration.get_subview_mask(coords='Display') == subview)

            # Line of sight at the field centre
            los_centre = params.get_los_direction(xpx,ypx)
            fov = self.calibration.get_fov(subview)

            pupilpos = self.calibration.get_pupilpos(subview=subview)

            widgets = self.fit_results_widgets[subview]

            if self.calibration.view_models[subview].model == 'rectilinear':
                widgets[0].setText( '<b>RMS Fit Residual: {: .1f} pixels<b>'.format(params.reprojection_error) )
                widgets[1].setText( ' : <br>'.join( [  'Pupil position' ,
                                                    'View direction' ,
                                                    'Field of view',
                                                    'Focal length' ,
                                                    'Optical centre' ,
                                                    'Distortion coeff. k1' ,
                                                    'Distortion coeff. k2' ,
                                                    'Distortion coeff. k3' ,
                                                    'Distortion coeff. p1' ,
                                                    'Distortion coeff. p2' ,
                                                    ''
                                                    ] ) )
                if self.calibration.pixel_size is not None:
                    fx = params.cam_matrix[0,0] * self.calibration.pixel_size*1e3
                    fy = params.cam_matrix[1,1] * self.calibration.pixel_size*1e3
                    fl_units = 'mm'
                else:
                    fx = params.cam_matrix[0,0]
                    fy = params.cam_matrix[1,1]
                    fl_units = 'px'


                widgets[2].setText( '<br>'.join( [ '( {: .3f} , {: .3f} , {: .3f} ) m'.format(pupilpos[0],pupilpos[1],pupilpos[2]).replace(' ','&nbsp;') ,
                                                '( {: .3f} , {: .3f} , {: .3f} )'.format(los_centre[0],los_centre[1],los_centre[2]).replace(' ','&nbsp;') ,
                                                '{:.1f}\xb0 x {:.1f}\xb0 '.format(fov[0],fov[1]).replace(' ','&nbsp;') ,
                                                "{0:.1f} {2:s} x {1:.1f} {2:s}".format(fx,fy,fl_units).replace(' ','&nbsp;') ,
                                                "( {: .0f} , {: .0f} )".format(params.cam_matrix[0,2], params.cam_matrix[1,2]).replace(' ','&nbsp;') ,
                                                "{: 5.4f}".format(params.kc[0][0]).replace(' ','&nbsp;') ,
                                                "{: 5.4f}".format(params.kc[0][1]).replace(' ','&nbsp;') ,
                                                "{: 5.4f}".format(params.kc[0][4]).replace(' ','&nbsp;') ,
                                                "{: 5.4f}".format(params.kc[0][2]).replace(' ','&nbsp;') ,
                                                "{: 5.4f}".format(params.kc[0][3]).replace(' ','&nbsp;') ,
                                                ''
                                                ] ) )
            elif params.model == 'fisheye':
                widgets[0].setText( '<b>RMS Fit Residual: {: .1f} pixels<b>'.format(params.reprojection_error) )
                widgets[1].setText( ' : <br>'.join( [  'Pupil position' ,
                                                    'View direction' ,
                                                    'Field of view',
                                                    'Focal length' ,
                                                    'Optical centre' ,
                                                    'Distortion coeff. k1' ,
                                                    'Distortion coeff. k2' ,
                                                    'Distortion coeff. k3' ,
                                                    'Distortion coeff. k4' ,
                                                    ''
                                                    ] ) )
                if self.calibration.pixel_size is not None:
                    fx = params.cam_matrix[0,0] * self.calibration.pixel_size*1e3
                    fy = params.cam_matrix[1,1] * self.calibration.pixel_size*1e3
                    fl_units = 'mm'
                else:
                    fx = params.cam_matrix[0,0]
                    fy = params.cam_matrix[1,1]
                    fl_units = 'px'

                widgets[2].setText( '<br>'.join( [ '( {: .3f} , {: .3f} , {: .3f} ) m'.format(pupilpos[0],pupilpos[1],pupilpos[2]).replace(' ','&nbsp;') ,
                                                '( {: .3f} , {: .3f} , {: .3f} )'.format(los_centre[0],los_centre[1],los_centre[2]).replace(' ','&nbsp;') ,
                                                '{:.1f}\xb0 x {:.1f}\xb0 '.format(fov[0],fov[1]).replace(' ','&nbsp;') ,
                                                "{0:.1f} {2:s} x {1:.1f} {2:s}".format(fx,fy,fl_units).replace(' ','&nbsp;') ,
                                                "( {: .0f} , {: .0f} )".format(params.cam_matrix[0,2], params.cam_matrix[1,2]).replace(' ','&nbsp;') ,
                                                "{: 5.4f}".format(params.kc[0]).replace(' ','&nbsp;') ,
                                                "{: 5.4f}".format(params.kc[1]).replace(' ','&nbsp;') ,
                                                "{: 5.4f}".format(params.kc[2]).replace(' ','&nbsp;') ,
                                                "{: 5.4f}".format(params.kc[3]).replace(' ','&nbsp;') ,
                                                ''
                                                ] ) )
            if self.cadmodel is not None:
                widgets[3].setEnabled(True)
                self.overlay_checkbox.setEnabled(True)

            self.fit_results[subview].show()
            self.fitted_points_checkbox.setEnabled(True)

        if self.fitted_points_checkbox.isEnabled() and show_points:
            self.fitted_points_checkbox.setChecked(True)




    def do_fit(self):

        subview = self.subview_tabs.currentIndex()

        # If this was called via a keyboard shortcut, we may be in no position to do a fit.
        if not self.fit_buttons[subview].isEnabled():
            return

        self.update_pointpairs()

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.fitted_points_checkbox.setChecked(False)
        self.overlay_checkbox.setChecked(False)
        self.fit_overlay = None

        # Do the fit!
        self.statusbar.showMessage('Performing calibration fit...')
        try:
            self.calibration.set_fit(subview, self.fitters[subview].do_fit())
            self.statusbar.clearMessage()
        except ImageUpsideDown:
            self.show_msgbox('The fitted calibration indicates that the images is upside down! Try rotating the image 180 degrees and fitting again.')
            self.app.restoreOverrideCursor()
            return

        self.update_fit_results()

        self.app.restoreOverrideCursor()
        if self.tabWidget.isHidden():
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Close)
            dialog.setWindowTitle('Calcam - Fit Results')
            dialog.setTextFormat(qt.Qt.RichText)
            dialog.setText(str(self.pointpicker.FitResults).replace('\n','<br>'))
            dialog.setIcon(qt.QMessageBox.Information)
            dialog.exec()

        self.unsaved_changes = True


    def render_overlay_image(self,calibration,wireframe):

        oversampling = int(np.ceil(min(1000/np.array(calibration.geometry.get_display_shape()))))

        self.statusbar.showMessage('Rendering CAD image overlay...')
        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.app.processEvents()

        orig_colours = self.cadmodel.get_colour()

        self.cadmodel.set_wireframe(wireframe)
        self.cadmodel.set_colour((1, 1, 1))
        image = render_cam_view(self.cadmodel, calibration, transparency=True, verbose=False, aa=2,oversampling=oversampling)
        self.cadmodel.set_colour(orig_colours)
        self.cadmodel.set_wireframe(False)

        self.statusbar.clearMessage()
        self.app.restoreOverrideCursor()

        return image



    def update_overlay(self):

        # Clear the existing overlay image to force it to re-render if the user has changed between solid / wireframe
        if self.sender() is self.overlay_type:
            self.fit_overlay = None

        if self.sender() is self.comparison_overlay_type:
            self.comp_overlay = None

        # Show or hide extra controls depending on what overlays are enabled
        self.overlay_appearance_controls.setVisible(self.overlay_checkbox.isChecked())
        self.comparison_overlay_appearance.setVisible(self.comparison_overlay_checkbox.isChecked())

        overlay_ims = []

        if self.overlay_checkbox.isChecked():

            self.fitted_points_checkbox.setChecked(False)

            if self.fit_overlay is None:
                self.fit_overlay = self.render_overlay_image(self.calibration,self.overlay_type.currentIndex() == 0)

                if np.max(self.fit_overlay) == 0:
                    self.show_msgbox('CAD model overlay result is a blank image.','This usually means the fit is wildly wrong.')

            # Apply desired colour (stored as a list in self.config.main_overlay_colour)
            overlay_ims.append( (self.fit_overlay * np.tile(np.array(self.config.main_overlay_colour)[np.newaxis,np.newaxis,:],self.fit_overlay.shape[:2] + (1,))).astype(np.uint8) )


        if self.comparison_overlay_checkbox.isChecked():

            self.comparison_overlay_appearance.show()

            if self.comp_overlay is None:
                self.comp_overlay = self.render_overlay_image(self.comp_calib,self.comparison_overlay_type.currentIndex() == 0)

            # Apply desired colour (stored as a list in self.config.second_overlay_colour)
            overlay_ims.append( (self.comp_overlay * np.tile(np.array(self.config.second_overlay_colour)[np.newaxis, np.newaxis, :],self.fit_overlay.shape[:2] + (1,))).astype(np.uint8) )

        self.interactor2d.set_overlay_image(overlay_ims)

        self.refresh_2d()



    def update_n_points(self):

        self.n_points = []

        chessboard_points = 0
        intcal_points = 0

        msg = ''

        for subview in range(self.calibration.n_subviews):
            if subview > 0:
                msg = msg + '\n'

            npts = 0
            for pp in self.point_pairings:
                if pp[0] is not None and pp[1] is not None:
                    im_coords = self.interactor2d.get_cursor_coords(pp[1])

                    if im_coords[subview] is not None:
                        npts += 1

            self.n_points.append([npts,0])

            if self.calibration.n_subviews > 1:
                msg = msg + '{:s}: {:d} point pairs'.format(self.calibration.subview_names[subview],npts)
            else:
                msg = msg + '{:d} point pairs'.format(npts)


        self.n_pointpairs_text.setText(msg)

        if len(self.point_pairings) > 0:
            self.clear_points_button.setEnabled(1)
        else:
            self.clear_points_button.setEnabled(0)


        for chessboard_constraint in self.chessboard_pointpairs:
            for subview in range(self.calibration.n_subviews):
                if self.chessboard_checkbox.isChecked():
                    self.n_points[subview][1] = self.n_points[subview][1] + chessboard_constraint[1].get_n_points(subview) - 6
                chessboard_points = chessboard_points + chessboard_constraint[1].get_n_points(subview)

        if self.intrinsics_calib is not None:
            for subview in range(self.calibration.n_subviews):
                if self.intrinsics_calib_checkbox.isChecked():
                    self.n_points[subview][1] = self.n_points[subview][1] + self.intrinsics_calib.pointpairs.get_n_points(subview) - 6
                intcal_points = intcal_points + self.intrinsics_calib.pointpairs.get_n_points(subview)

                for intrinsics_constraint in self.intrinsics_calib.intrinsics_constraints:
                    if self.intrinsics_calib_checkbox.isChecked():
                        self.n_points[subview][1] = self.n_points[subview][1] + intrinsics_constraint[1].get_n_points(subview) - 6
                    intcal_points = intcal_points + intrinsics_constraint[1].get_n_points(subview)

        if chessboard_points > 0:
            self.chessboard_checkbox.setText('Chessboard Images ({:d} points)'.format(chessboard_points))
        else:
            self.chessboard_checkbox.setText('Chessboard Images (None Loaded)')

        if intcal_points > 0:
            self.intrinsics_calib_checkbox.setText('Existing Calibration ({:d} points)'.format(intcal_points))
        else:
            self.intrinsics_calib_checkbox.setText('Existing Calibration (None Loaded)')

        self.fit_enable_check()
        self.unsaved_changes = True


    def save_calib(self,saveas=False):

        if self.calibration.subview_mask is None:
            raise UserWarning('Nothing to save! You need to load a camera image to calibrate before you can save anything in this tool.')

        if saveas:
            # Back up then clear the current filename if we want to force a new filename choice.
            orig_filename = self.filename
            self.filename = None

        # Save file dialog box
        if self.filename is None:
            self.filename = self.get_save_filename('calibration')

        if self.filename is not None:

            if self.cadmodel is not None:
                viewport = {'cam_x':self.camX.value(),'cam_y':self.camY.value(),'cam_z':self.camZ.value(),'tar_x':self.tarX.value(),'tar_y':self.tarY.value(),'tar_z':self.tarZ.value(),'fov':self.camFOV.value(),'roll':self.cam_roll.value()}
                self.calibration.cad_config = {'model_name':self.cadmodel.machine_name , 'model_variant':self.cadmodel.model_variant , 'enabled_features':self.cadmodel.get_enabled_features(),'viewport':viewport}

            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.statusbar.showMessage('Saving...')

            try:
                self.calibration.save(self.filename)
            except PermissionError:
                raise UserWarning('Could not write to {:s}: permission denied.'.format(self.filename))

            self.unsaved_changes = False
            self.action_save.setEnabled(True)
            self.setWindowTitle('Calcam Calibration Tool (Point Fitting) - {:s}'.format(os.path.split(self.filename)[-1][:-4]))
            self.statusbar.clearMessage()
            self.app.restoreOverrideCursor()

        elif saveas:
            # Restore original filename if we didn't end up finishing the Save As...
            self.filename = orig_filename




    def load_calib(self,filename=None):

        try:
            opened_calib = Calibration(filename)
        except:
            opened_calib = self.object_from_file('calibration',maintain_window=False)

        if opened_calib is None:
            return

        if opened_calib._type == 'alignment':
            raise UserWarning('The selected calibration is an alignment calibration and cannot be edited in this tool. Please open it with the alignment calibration editor instead.')
        elif opened_calib._type == 'virtual':
            raise UserWarning('The selected calibration is a virtual calibration and cannot be edited in this tool. Please open it in the virtual calibration editor instead.')

        if opened_calib.cad_config is not None:
            cconfig = opened_calib.cad_config
            if self.cadmodel is not None and self.cadmodel.machine_name == cconfig['model_name'] and self.cadmodel.model_variant == cconfig['model_variant']:
                keep_model = True
            else:
                keep_model = False
        else:
            keep_model = False

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.reset(keep_cadmodel=keep_model)

        # Basic setup
        self.filename = opened_calib.filename
        self.setWindowTitle('Calcam Calibration Tool (Point Fitting) - {:s}'.format(os.path.split(self.filename)[-1][:-4]))

        # Load the image
        for imsource in self.image_sources:
            if imsource.display_name == 'Calcam Calibration':
                try:
                    self.load_image(newim = imsource.get_image_function(self.filename))
                except Exception as e:
                    if 'does not contain an image' in str(e):
                        self.interactor2d.set_image(opened_calib.get_subview_mask(coords='Display')*0,n_subviews=opened_calib.n_subviews,subview_lookup=opened_calib.subview_lookup)
                    else:
                        raise

        self.calibration = opened_calib

        if not self.fit_initted:
            self.init_fitting(reset_fit=False)

        self.interactor2d.set_subview_lookup(self.calibration.n_subviews,self.calibration.subview_lookup)

        self.chessboard_pointpairs = self.calibration.intrinsics_constraints
        self.chessboard_history = self.calibration.history['intrinsics_constraints']

        if len(self.chessboard_pointpairs) > 0:
            self.chessboard_checkbox.blockSignals(True)
            self.chessboard_checkbox.setChecked(True)
            self.chessboard_checkbox.blockSignals(False)


        # Load the point pairs
        if opened_calib.pointpairs is not None:
            self.load_pointpairs(pointpairs=opened_calib.pointpairs,history=opened_calib.history['pointpairs'],force_clear=False,clear_fit=False)


        # Load the appropriate CAD model, if we know what that is
        if opened_calib.cad_config is not None:
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

            self.camX.setValue(cconfig['viewport']['cam_x'])
            self.camY.setValue(cconfig['viewport']['cam_y'])
            self.camZ.setValue(cconfig['viewport']['cam_z'])
            self.tarX.setValue(cconfig['viewport']['tar_x'])
            self.tarY.setValue(cconfig['viewport']['tar_y'])
            self.tarZ.setValue(cconfig['viewport']['tar_z'])
            self.camFOV.setValue(cconfig['viewport']['fov'])
            self.cam_roll.setValue(cconfig['viewport']['roll'])

        for field in range(len(self.fit_settings_widgets)):
            if opened_calib.view_models[field] is not None:

                if opened_calib.view_models[field].model == 'rectilinear':
                    self.fit_settings_widgets[field][0].setChecked(True)
                    widget_index_start = 2
                    widget_index_end = 6
                elif opened_calib.view_models[field].model == 'fisheye':
                    self.fit_settings_widgets[field][1].setChecked(True)
                    widget_index_start = 7
                    widget_index_end = 10


                for widget in self.fit_settings_widgets[field][widget_index_start:widget_index_end+1]:
                    widget.setChecked(False)
                    if str(widget.text()) in opened_calib.view_models[field].fit_options:
                        widget.setChecked(True)



        if self.calibration.readonly:
            self.action_save.setEnabled(False)
        else:
            self.action_save.setEnabled(True)

        self.update_n_points()
        self.update_fit_results()
        self.pointpairs_history.clear()
        self.points_undo_button.setEnabled(False)
        self.app.restoreOverrideCursor()
        self.unsaved_changes = False


    def toggle_enhancement(self,check_state):

        image = self.calibration.get_image(coords='display')

        # Enable / disable adaptive histogram equalisation
        if self.enhance_checkbox.isChecked():
            image = enhance_image(image)

        self.interactor2d.set_image(image,n_subviews = self.calibration.n_subviews,subview_lookup=self.calibration.subview_lookup,hold_position=True)



    def edit_split_field(self):

        dialog = ImageMaskDialog(self,self.interactor2d.get_image())
        result = dialog.exec()

        if result == 1:
            self.calibration.set_subview_mask(dialog.fieldmask,subview_names=dialog.field_names,coords='Display')
            self.interactor2d.set_subview_lookup(self.calibration.n_subviews,self.calibration.subview_lookup)
            self.init_fitting()
            self.unsaved_changes = True
            self.update_n_points()
            self.pointpairs_history.clear()
            self.points_undo_button.setEnabled(False)

        del dialog



    def change_dist_model(self,choice):

        for field in range(len(self.fit_settings_widgets)):
            if self.sender() == self.fit_settings_widgets[field][0]:
                self.perspective_settings[field].show()
                self.fisheye_settings[field].hide()
                self.fitters[field].set_model('rectilinear')
            elif self.sender() == self.fit_settings_widgets[field][1]:
                self.perspective_settings[field].hide()
                self.fisheye_settings[field].show()
                self.fitters[field].set_model('fisheye')

        self.update_n_points()


    def update_pixel_size(self):

        if self.pixel_size_checkbox.isChecked():
            self.pixel_size_box.setEnabled(True)
            self.calibration.pixel_size = self.pixel_size_box.value() / 1e6
        else:
            self.pixel_size_box.setEnabled(False)
            self.calibration.pixel_size = None

        self.update_fit_results()
        self.unsaved_changes = True


    def set_fit_viewport(self):
        subview = self.view_to_fit_buttons.index(self.sender())

        self.set_view_from_calib(self.calibration,subview)



    def modify_chessboard_constraints(self):

        dialog = ChessboardDialog(self)
        dialog.exec()

        # Resizing the window + 1 pixel then immediately back again after the chessboard dialog is closed
        # is a workaround for Issue #65, the root cause of which is a mystery to me. I shake my fist at VTK.
        size = self.size()
        self.resize(size.width(),size.height()+1)
        self.app.processEvents()
        self.resize(size.width(),size.height())
        self.app.processEvents()

        if dialog.results != []:
            self.chessboard_pointpairs = dialog.results
            point_history = ['Auto-detected corners of {:d}x{:d} square chessboard pattern with {:.1f}mm squares.'.format(dialog.chessboard_squares_x.value(),dialog.chessboard_squares_y.value(),dialog.chessboard_square_size.value()),None]
            self.chessboard_history = [ ('Chessboard image from {:s}, loaded by {:s} on {:s} at {:s}.'.format(fname,misc.username,misc.hostname,misc.get_formatted_time()),point_history) for fname in dialog.filenames]
            if self.chessboard_checkbox.isChecked():
                self.chessboard_checkbox.setChecked(False)
            self.chessboard_checkbox.setChecked(True)




    def toggle_chessboard_constraints(self,on):

        if on and len(self.chessboard_pointpairs) == 0:
            self.modify_chessboard_constraints()
            if len(self.chessboard_pointpairs) == 0:
                self.chessboard_checkbox.setChecked(False)

        self.reset_fit()
        self.update_pointpairs()
        self.update_n_points()
        self.fit_enable_check()
