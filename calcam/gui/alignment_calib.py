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

from .core import *
from .vtkinteractorstyles import CalcamInteractorStyle3D
from ..calibration import Calibration
from ..image_enhancement import enhance_image, scale_to_8bit

# View designer window.
# This allows creation of FitResults objects for a 'virtual' camera.
class AlignmentCalib(CalcamGUIWindow):

    def __init__(self, app, parent = None, load_file=None):

        # GUI initialisation
        CalcamGUIWindow.init(self,'alignment_calib_editor.ui',app,parent)

        # Start up with no CAD model
        self.cadmodel = None

        # Set up VTK
        self.qvtkwidget_3d = qt.QVTKRenderWindowInteractor(self.vtk_frame)
        self.vtk_frame.layout().addWidget(self.qvtkwidget_3d,0,0)
        self.interactor3d = CalcamInteractorStyle3D(refresh_callback=self.refresh_3d,viewport_callback=self.update_viewport_info,resize_callback=self.update_vtk_size)
        self.qvtkwidget_3d.SetInteractorStyle(self.interactor3d)
        self.renderer_3d = vtk.vtkRenderer()
        self.renderer_3d.SetBackground(0, 0, 0)
        self.qvtkwidget_3d.GetRenderWindow().AddRenderer(self.renderer_3d)
        self.camera_3d = self.renderer_3d.GetActiveCamera()


        self.populate_models()


        # Synthetic camera object to store the results
        self.calibration = Calibration(cal_type='alignment')
        self.chessboard_fit = None

        self.filename = None
        self.original_image = None

        self.manual_exc = True

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
        self.load_model_button.clicked.connect(self._load_model)
        self.model_name.currentIndexChanged.connect(self.populate_model_variants)
        self.feature_tree.itemChanged.connect(self.update_checked_features)
        self.feature_tree.itemSelectionChanged.connect(self.update_cadtree_selection)
        self.load_image_button.clicked.connect(self.load_image)
        self.im_flipud.clicked.connect(self.transform_image)
        self.rendertype_edges.toggled.connect(self.toggle_wireframe)
        self.im_fliplr.clicked.connect(self.transform_image)
        self.im_rotate_button.clicked.connect(self.transform_image)
        self.im_reset.clicked.connect(self.transform_image)
        self.im_y_stretch_button.clicked.connect(self.transform_image)
        self.calcam_intrinsics.clicked.connect(self.update_intrinsics)
        self.chessboard_intrinsics.clicked.connect(self.update_intrinsics)
        self.pinhole_intrinsics.clicked.connect(self.update_intrinsics)
        self.pixel_size_checkbox.toggled.connect(self.update_intrinsics)
        self.pixel_size_box.valueChanged.connect(self.update_intrinsics)
        self.focal_length_box.valueChanged.connect(self.update_intrinsics)
        self.load_chessboard_button.clicked.connect(self.update_chessboard_intrinsics)
        self.load_intrinsics_button.clicked.connect(self.update_intrinsics)
        self.load_extrinsics_button.clicked.connect(self.load_viewport_calib)
        self.cad_colour_reset_button.clicked.connect(self.set_cad_colour)
        self.cad_colour_choose_button.clicked.connect(self.set_cad_colour)
        self.im_opacity_slider.valueChanged.connect(self.update_overlay)
        self.enhance.clicked.connect(self.update_overlay)
        self.edge_detect.clicked.connect(self.update_overlay)
        self.no_effect.clicked.connect(self.update_overlay)
        self.im_edge_colour_button.clicked.connect(self.update_edge_colour)
        self.edge_threshold_1.valueChanged.connect(self.update_overlay)
        self.edge_threshold_2.valueChanged.connect(self.update_overlay)
        self.image_mask_button.clicked.connect(self.edit_masking)

        self.control_sensitivity_slider.valueChanged.connect(lambda x: self.interactor3d.set_control_sensitivity(x*0.01))
        self.rmb_rotate.toggled.connect(self.interactor3d.set_rmb_rotate)
        self.interactor3d.set_control_sensitivity(self.control_sensitivity_slider.value()*0.01)

        self.pixel_size_box.setSuffix(u' \u00B5m')

        self.action_save.triggered.connect(self.save)
        self.action_save_as.triggered.connect(lambda: self.save(saveas=True))
        self.action_open.triggered.connect(self.open_calib)
        self.action_new.triggered.connect(self.reset)
        self.action_cal_info.triggered.connect(self.show_calib_info)

        self.viewport_calibs = DodgyDict()

        self.image_settings.hide()
        self.image_display_settings.hide()
        self.edge_detect_settings.hide()

        self.tabWidget.setTabEnabled(2,False)

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

        self.edge_detect_colour = (1,0,0)
        self.view_aspect = None
        self.vtk_aspect = None
        self.intrinsics_calib = None

        self.extrinsics_src = None

        self.calibration = None


        # If we have an old version of openCV, histo equilisation won't work :(
        cv2_version = float('.'.join(cv2.__version__.split('.')[:2]))
        cv2_micro_version = int(cv2.__version__.split('.')[2].split('-')[0])
        if cv2_version < 2.4 or (cv2_version == 2.4 and cv2_micro_version < 6):
            self.edge_detect.setEnabled(False)
            self.edge_detect.setToolTip('Requires OpenCV 2.4.6 or newer; you have {:s}'.format(cv2.__version__))

        self.fov_enabled = False
        self.viewdir_at_cc = True

        # Start the GUI!
        self.show()

        self.interactor3d.init()
        self.qvtkwidget_3d.GetRenderWindow().GetInteractor().Initialize()
        self.reset()

        if load_file is not None:
            self.open_calib(load_file)



    def _load_model(self):
        self.load_model(hold_view = self.cadmodel is not None or self.calibration.filename is not None)
        if self.calibration.image is not None:
            self.tabWidget.setTabEnabled(2,True)
        self.update_intrinsics()

    def on_close(self):
        self.qvtkwidget_3d.close()

    def reset(self,keep_cadmodel=False):

        if not keep_cadmodel:
            if self.cadmodel is not None:
                self.cadmodel.remove_from_renderer(self.renderer_3d)
                self.cadmodel.unload()
                self.feature_tree.blockSignals(True)
                self.feature_tree.clear()
                self.feature_tree.blockSignals(False)
                self.cadmodel = None

        self.pinhole_intrinsics.setChecked(True)
        self.pixel_size_checkbox.setChecked(False)

        self.interactor3d.set_overlay_image(None)

        self.calibration = Calibration(cal_type='alignment')

        self.tabWidget.setTabEnabled(2,False)

        self.filename = None
        self.setWindowTitle('Calcam Calibration Tool (Manual Alignment)')

        self.chessboard_fit = None
        self.intrinsics_calib = None

        self.update_intrinsics()

        self.refresh_3d()
        self.unsaved_changes = False


    def edit_masking(self):

        dialog = ImageMaskDialog(self,self.calibration.get_image(),allow_subviews=False)
        result = dialog.exec()

        if result == 1:
            self.calibration.set_subview_mask(dialog.fieldmask,subview_names=dialog.field_names,coords='Display')
            self.unsaved_changes = True

        del dialog

    def update_edge_colour(self):

        col = self.pick_colour(self.edge_detect_colour)

        if col is not None:
            self.edge_detect_colour = col
            self.update_overlay()

    def toggle_wireframe(self,wireframe):

        if self.cadmodel is not None:

            self.cadmodel.set_wireframe( wireframe )

            self.refresh_3d()


    def update_viewport_info(self,keep_selection=False):

        CalcamGUIWindow.update_viewport_info(self,keep_selection)
        self.unsaved_changes = True

        if self.original_image is not None:

            if self.pinhole_intrinsics.isChecked():

                fov = 3.14159 * self.camera_3d.GetViewAngle() / 180.
                f = self.calibration.geometry.get_display_shape()[1]/(2*np.tan(fov/2.))
                if self.pixel_size_checkbox.isChecked():
                        f = f * self.pixel_size_box.value() / 1e3

                self.focal_length_box.blockSignals(True)
                self.focal_length_box.setValue(f)
                self.focal_length_box.blockSignals(False)


    def open_calib(self,filename=None):

        try:
            opened_calib = Calibration(filename)
        except:
            opened_calib = self.object_from_file('calibration')

        if opened_calib is None:
            return

        if opened_calib._type == 'fit':
            raise UserWarning('The selected calibration is a point-pair fitting calibration and cannot be edited in this tool. Please open it with the point fitting calibration tool instead.')
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
        self.reset(keep_cadmodel = keep_model)

        # Basic setup
        self.filename = opened_calib.filename
        self.setWindowTitle('Calcam Calibration Tool (Manual Alignment) - {:s}'.format(os.path.split(self.filename)[-1][:-4]))

        # Load the image
        for imsource in self.image_sources:
            if imsource.display_name == 'Calcam Calibration':
                try:
                    self.load_image(newim = imsource.get_image_function(self.filename))
                except Exception as e:
                    self.show_msgbox(e)

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
                        self.load_model(featurelist=cconfig['enabled_features'],hold_view=True)
                    except Exception as e:
                        self.cadmodel = None
                        if 'Unknown feature' not in str(e):
                            raise


        self.calibration = opened_calib

        if self.calibration.readonly:
            self.action_save.setEnabled(False)
        else:
            self.action_save.setEnabled(True)

        # Load the intrinsics
        if opened_calib.intrinsics_type == 'pinhole':
            fl = opened_calib.view_models[0].cam_matrix[0,0]
            if opened_calib.pixel_size is not None:
                fl = fl * opened_calib.pixel_size  * 1000

                self.pixel_size_box.setValue(opened_calib.pixel_size*1e6)
                self.pixel_size_checkbox.setChecked(True)
            else:
                self.pixel_size_checkbox.setChecked(False)

            self.focal_length_box.setValue(fl)
            self.pinhole_intrinsics.setChecked(True)

        elif opened_calib.intrinsics_type == 'chessboard':
            self.chessboard_fit = opened_calib.view_models[0]
            self.chessboard_pointpairs = opened_calib.intrinsics_constraints
            self.chessboard_src = opened_calib.history['intrinsics']
            self.chessboard_intrinsics.setChecked(True)

        elif opened_calib.intrinsics_type == 'calibration':
            self.intrinsics_calib = self.calibration
            self.calcam_intrinsics.setChecked(True)

        self.update_intrinsics()

        self.set_view_from_calib(self.calibration,0)

        self.app.restoreOverrideCursor()
        if self.cadmodel is not None and self.calibration.image is not None:
            self.tabWidget.setTabEnabled(2,True)

        self.unsaved_changes = False


    def update_intrinsics(self):

        if self.calibration.image is None:
            return

        nx,ny = self.calibration.geometry.get_original_shape()
        self.calibration.set_subview_mask(np.zeros((ny,nx),dtype=np.int8), coords='Original')

        if self.sender() is self.load_intrinsics_button:
            self.intrinsics_calib = None

        if self.pixel_size_checkbox.isChecked():
            self.pixel_size_box.setEnabled(True)
            self.calibration.pixel_size = self.pixel_size_box.value() / 1e6
        else:
            self.pixel_size_box.setEnabled(False)
            self.calibration.pixel_size = None

        if self.calcam_intrinsics.isChecked():
            self.interactor3d.zoom_enabled = False
            self.load_chessboard_button.setEnabled(False)
            self.load_intrinsics_button.setEnabled(True)
            self.focal_length_box.setEnabled(False)

            if self.intrinsics_calib is None:
                self.intrinsics_calib = self.object_from_file('calibration')

            if self.intrinsics_calib is not None:
                if len(self.intrinsics_calib.view_models) != 1:
                    self.intrinsics_calib = None
                    self.current_intrinsics_combobox.setChecked(True)
                    raise UserWarning('This calibration has multiple sub-fields, which is not supported by the manual alignment calibration tool.')

                self.calibration.set_calib_intrinsics(self.intrinsics_calib,update_hist_recursion = not (self.intrinsics_calib is self.calibration))
                self.calibration.set_subview_mask(self.intrinsics_calib.get_subview_mask(coords='Original'),coords='Original')
                self.current_intrinsics_combobox = self.calcam_intrinsics
            else:
                self.current_intrinsics_combobox.setChecked(True)


        elif self.pinhole_intrinsics.isChecked():
            self.interactor3d.zoom_enabled = True

            if self.pixel_size_checkbox.isChecked():
                if self.focal_length_box.suffix() == ' px':
                    self.focal_length_box.blockSignals(True)
                    self.focal_length_box.setValue( self.focal_length_box.value() * self.pixel_size_box.value() / 1e3 )
                    self.focal_length_box.blockSignals(False)
                    self.focal_length_box.setSuffix(' mm')
            else:
                if self.focal_length_box.suffix() == ' mm':
                    self.focal_length_box.blockSignals(True)
                    self.focal_length_box.setValue( 1e3 * self.focal_length_box.value() / self.pixel_size_box.value() )
                    self.focal_length_box.blockSignals(False)
                    self.focal_length_box.setSuffix(' px')

            self.load_chessboard_button.setEnabled(False)
            self.focal_length_box.setEnabled(True)

            nx,ny = self.calibration.geometry.get_display_shape()

            f = self.focal_length_box.value()
            if self.pixel_size_checkbox.isChecked():
                f = 1e3 * f / self.pixel_size_box.value()

            self.calibration.set_pinhole_intrinsics(fx=f,fy=f)
            self.current_intrinsics_combobox = self.pinhole_intrinsics

        elif self.chessboard_intrinsics.isChecked():
            self.interactor3d.zoom_enabled = False
            self.load_chessboard_button.setEnabled(True)
            self.load_intrinsics_button.setEnabled(False)
            self.focal_length_box.setEnabled(False)

            if self.chessboard_fit is None:
                self.update_chessboard_intrinsics()
                if self.chessboard_fit is None:
                    self.current_intrinsics_combobox.setChecked(True)
                return
            else:
                self.calibration.set_chessboard_intrinsics(self.chessboard_fit,self.chessboard_pointpairs,self.chessboard_src)
                self.current_intrinsics_combobox = self.chessboard_intrinsics


        self.update_overlay()

        mat = self.calibration.get_cam_matrix()
        n = self.calibration.geometry.get_display_shape()
        wcx = -2.*(mat[0,2] - n[0]/2.) / float(n[0])
        wcy = 2.*(mat[1,2] - n[1]/2.) / float(n[1])
        self.camera_3d.SetWindowCenter(wcx,wcy)
        fov = 360*np.arctan( float(n[1]) / (2*mat[1,1]))/3.14159
        self.camera_3d.SetViewAngle(fov)
        self.unsaved_changes = True



    def on_load_image(self,newim):

        if newim['pixel_size'] is not None:
            self.pixel_size_checkbox.setChecked(True)
            self.pixel_size_box.setValue(newim['pixel_size']*1e6)
        else:
            self.pixel_size_checkbox.setChecked(False)

        self.calibration.set_image( scale_to_8bit(newim['image_data']) , newim['source'],subview_mask = newim['subview_mask'], transform_actions = newim['transform_actions'],coords=newim['coords'],subview_names=newim['subview_names'],pixel_aspect=newim['pixel_aspect'],pixel_size=newim['pixel_size'],offset=newim['image_offset'] )

        imshape = self.calibration.geometry.get_display_shape()
        self.interactor3d.force_aspect = float( imshape[1] ) / float( imshape[0] )

        # This is a slight hack - we have to resize the window slightly
        # and resize it back again to get VTK to redraw the background.
        # More elegant solution needed here really.
        size = self.size()
        self.resize(size.width()+1,size.height())
        self.refresh_3d()
        self.resize(size.width(),size.height())

        self.calibration.view_models = [None] * self.calibration.n_subviews
        self.fit_timestamps = [None] * self.calibration.n_subviews

        if self.calcam_intrinsics.isChecked() and np.any(np.array(self.intrinsics_calib.geometry.get_original_shape()) != np.array(self.calibration.geometry.get_original_shape())):
            self.show_msgbox('The current calibration intrinsics are the wrong shape for this image. The current intrinsics will be reset.')
            self.intrinsics_calib = None
            self.pinhole_intrinsics.setChecked(True)

        self.update_intrinsics()

        self.image_settings.show()
        self.image_display_settings.show()

        if self.cadmodel is not None:
            self.tabWidget.setTabEnabled(2,True)

        self.update_image_info_string(newim['image_data'],self.calibration.geometry)
        self.update_overlay()
        self.unsaved_changes = True


    def update_overlay(self):

        self.overlay_image = self.calibration.undistort_image( self.calibration.get_image(coords='display') )

        if self.enhance.isChecked() or self.edge_detect.isChecked():
            self.overlay_image = enhance_image(self.overlay_image)

        if self.edge_detect.isChecked():

            self.edge_detect_settings.show()
            self.label_18.hide()
            self.im_opacity_slider.hide()
            if self.edge_threshold_1.value() > self.edge_threshold_2.value():
                if self.sender() is self.edge_threshold_1:
                    self.edge_threshold_2.setValue(self.edge_threshold_1.value())
                else:
                    self.edge_threshold_1.setValue(self.edge_threshold_2.value())

            blurred_im = cv2.GaussianBlur(self.overlay_image.astype('uint8'),(5,5),0)
            edge_im = cv2.Canny(blurred_im,self.edge_threshold_1.value(),self.edge_threshold_2.value())
            self.overlay_image = np.zeros(edge_im.shape + (4,),dtype='uint8')
            self.overlay_image[:,:,0] = edge_im[:,:]*self.edge_detect_colour[0]
            self.overlay_image[:,:,1] = edge_im[:,:]*self.edge_detect_colour[1]
            self.overlay_image[:,:,2] = edge_im[:,:]*self.edge_detect_colour[2]
            self.overlay_image[:,:,3] = 255*(edge_im > 0)
        else:
            self.edge_detect_settings.hide()
            self.im_opacity_slider.show()
            self.label_18.show()

        if len(self.overlay_image.shape) < 3:
            self.overlay_image = np.tile(self.overlay_image[:,:,np.newaxis],(1,1,3))

        if self.overlay_image.shape[2] == 3:
            alpha = np.uint8(np.ones(self.overlay_image.shape[:2])*255)
            self.overlay_image = np.dstack([self.overlay_image,alpha])

        if self.im_opacity_slider.isEnabled():
            alpha = self.im_opacity_slider.value()/10.
        else:
            alpha = 1.

        self.interactor3d.set_overlay_image( (self.overlay_image * np.tile([1,1,1,alpha],self.overlay_image.shape[:2] + (1,))).astype('uint8'))

        self.refresh_3d()


    def transform_image(self,data):

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
            self.calibration.geometry.set_pixel_aspect(1.)

        # Update the image and point pairs
        imshape = self.calibration.geometry.get_display_shape()
        self.interactor3d.force_aspect = float(imshape[1]) / float(imshape[0])

        self.update_image_info_string(self.calibration.get_image(),self.calibration.geometry)

        self.update_overlay()
        self.unsaved_changes = True





    def save(self,saveas=False):

        if self.calibration.view_models[0] is None:
            raise UserWarning('Nothing to save! You need to load a camera image to calibrate before you can save anything in this tool.')

        if saveas:
            orig_filename = self.filename
            self.filename = None

        if self.filename is None:
            self.filename = self.get_save_filename('calibration')

        if self.filename is not None:

            self.update_extrinsics()

            if self.cadmodel is not None:
                viewport = {'cam_x':self.camX.value(),'cam_y':self.camY.value(),'cam_z':self.camZ.value(),'tar_x':self.tarX.value(),'tar_y':self.tarY.value(),'tar_z':self.tarZ.value(),'fov':self.camera_3d.GetViewAngle(),'roll':self.cam_roll.value()}
                self.calibration.cad_config = {'model_name':self.cadmodel.machine_name , 'model_variant':self.cadmodel.model_variant , 'enabled_features':self.cadmodel.get_enabled_features(),'viewport':viewport }


            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.statusbar.showMessage('Saving...')
            try:
                self.calibration.save(self.filename)
            except PermissionError:
                raise UserWarning('Could not write to {:s}: permission denied.'.format(self.filename))
            self.unsaved_changes = False
            self.statusbar.clearMessage()
            self.app.restoreOverrideCursor()

            self.action_save.setEnabled(True)
            self.setWindowTitle('Calcam Calibration Tool (Manual Alignment) - {:s}'.format(os.path.split(self.filename)[-1][:-4]))

        elif saveas:
            self.filename = orig_filename
