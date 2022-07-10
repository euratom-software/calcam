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

import cv2

from .core import *
from .vtkinteractorstyles import CalcamInteractorStyle3D
from ..calibration import Calibration

# View designer window.
# This allows creation of FitResults objects for a 'virtual' camera.
class VirtualCalib(CalcamGUIWindow):
 
    def __init__(self, app, parent = None,load_file=None):

        # GUI initialisation
        CalcamGUIWindow.init(self,'virtual_calib_editor.ui',app,parent)

        # Start up with no CAD model
        self.cadmodel = None

        # Set up VTK
        self.qvtkwidget_3d = qt.QVTKRenderWindowInteractor(self.vtk_frame)
        self.vtk_frame.layout().addWidget(self.qvtkwidget_3d,0,0)
        self.interactor3d = CalcamInteractorStyle3D(refresh_callback=self.refresh_3d,viewport_callback=self.update_viewport_info)
        self.qvtkwidget_3d.SetInteractorStyle(self.interactor3d)
        self.renderer_3d = vtk.vtkRenderer()
        self.renderer_3d.SetBackground(0, 0, 0)
        self.qvtkwidget_3d.GetRenderWindow().AddRenderer(self.renderer_3d)
        self.camera_3d = self.renderer_3d.GetActiveCamera()


        self.populate_models()


        # Synthetic camera object to store the results
        self.calibration = Calibration(cal_type='virtual')
        self.chessboard_fit = None

        self.filename = None

        self.manual_exc = True

        # Callbacks for GUI elements
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
        self.calcam_intrinsics.clicked.connect(self.update_intrinsics)
        self.chessboard_intrinsics.clicked.connect(self.update_intrinsics)
        self.pinhole_intrinsics.clicked.connect(self.update_intrinsics)
        self.pixel_size_box.valueChanged.connect(self.update_intrinsics)
        self.x_pixels_box.valueChanged.connect(self.update_intrinsics)
        self.y_pixels_box.valueChanged.connect(self.update_intrinsics)
        self.focal_length_box.valueChanged.connect(self.update_intrinsics)
        self.fov_box.valueChanged.connect(self.update_intrinsics)
        self.load_chessboard_button.clicked.connect(self.update_chessboard_intrinsics)
        self.load_intrinsics_button.clicked.connect(self.update_intrinsics)
        self.load_extrinsics_button.clicked.connect(self.load_viewport_calib)
        self.pixel_size_box.setSuffix(u' \u00B5m')

        self.control_sensitivity_slider.valueChanged.connect(lambda x: self.interactor3d.set_control_sensitivity(x*0.01))
        self.rmb_rotate.toggled.connect(self.interactor3d.set_rmb_rotate)
        self.interactor3d.set_control_sensitivity(self.control_sensitivity_slider.value()*0.01)

        self.action_save.triggered.connect(self.save)
        self.action_save_as.triggered.connect(lambda: self.save(saveas=True))
        self.action_open.triggered.connect(self.open_calib)
        self.action_new.triggered.connect(self.reset)
        self.action_cal_info.triggered.connect(self.show_calib_info)

        self.viewport_calibs = DodgyDict()
        self.intrinsics_calib = None

        self.fov_enabled = False
        self.viewdir_at_cc = True

        self.extrinsics_src = None

        self.calibration = Calibration(cal_type='virtual')

        # Start the GUI!
        self.show()

        self.interactor3d.init()
        self.qvtkwidget_3d.GetRenderWindow().GetInteractor().Initialize()
        self.update_intrinsics()

        if load_file is not None:
            self.open_calib(load_file)
        


    def update_viewport_info(self,keep_selection=False):

        CalcamGUIWindow.update_viewport_info(self,keep_selection)
        if self.cadmodel is not None:
            self.unsaved_changes = True

        if self.pinhole_intrinsics.isChecked():

            fov = 3.14159 * self.camera_3d.GetViewAngle() / 180.
            f = self.calibration.geometry.get_display_shape()[1]/(2*np.tan(fov/2.))
            f = f * self.pixel_size_box.value() / 1e3

            self.focal_length_box.setValue(f)


    def on_close(self):
        self.qvtkwidget_3d.close()
        

    def update_intrinsics(self):

        if self.sender() is self.load_intrinsics_button:
            self.intrinsics_calib = None


        nx,ny = self.calibration.geometry.get_image_shape(coords='Original')
        if nx is not None and ny is not None:
            self.calibration.set_subview_mask(np.zeros((ny,nx), dtype=np.int8), coords='Original')

        if self.calcam_intrinsics.isChecked():
            self.interactor3d.zoom_enabled = False
            self.load_chessboard_button.setEnabled(False)
            self.load_intrinsics_button.setEnabled(True)
            self.pinhole_params_box.hide()
            
            if self.intrinsics_calib is None:
                self.intrinsics_calib = self.object_from_file('calibration')
                if self.intrinsics_calib is not None:
                    if len(self.intrinsics_calib.view_models) != 1:
                        self.intrinsics_calib = None
                        raise UserWarning('This calibration has multiple sub-fields, but only single sub-field views are supported in virtual calibrations. Sorry.')
                        self.current_intrinsics_combobox.setChecked(True)
                    else:
                        self.calibration.set_calib_intrinsics(self.intrinsics_calib,update_hist_recursion = not (self.intrinsics_calib is self.calibration) )
                        self.calibration.set_subview_mask(self.intrinsics_calib.get_subview_mask(coords='Original'),coords='Original')
                        self.current_intrinsics_combobox = self.calcam_intrinsics
                        self.calcam_intrinsics_fname.setText(os.path.split(self.intrinsics_calib.filename)[-1][:-4])
                        self.calcam_intrinsics_fname.show()

                else:
                    self.current_intrinsics_combobox.setChecked(True)
                
            else:
                self.calcam_intrinsics_fname.show()
                self.calibration.set_calib_intrinsics(self.intrinsics_calib,update_hist_recursion = not (self.intrinsics_calib is self.calibration))
                self.current_intrinsics_combobox = self.calcam_intrinsics

        elif self.pinhole_intrinsics.isChecked():
            self.calcam_intrinsics_fname.hide()
            self.interactor3d.zoom_enabled = True
            self.load_chessboard_button.setEnabled(False)
            self.load_intrinsics_button.setEnabled(False)
            self.pinhole_params_box.show()

            if self.sender() is self.fov_box:
                self.focal_length_box.blockSignals(True)
                fl = self.y_pixels_box.value() * self.pixel_size_box.value() * 1e-3 / (2*np.tan(self.fov_box.value()/360 * np.pi))
                self.focal_length_box.setValue(fl)
                self.focal_length_box.blockSignals(False)
            elif self.sender() is self.focal_length_box:
                self.fov_box.blockSignals(True)
                fov = 360*np.arctan(self.y_pixels_box.value() * self.pixel_size_box.value() * 1e-3 / (2*self.focal_length_box.value())) / np.pi
                self.fov_box.setValue(fov)
                self.fov_box.blockSignals(False)

            nx = self.x_pixels_box.value()
            ny = self.y_pixels_box.value()
            f = 1e3 * self.focal_length_box.value() / self.pixel_size_box.value()

            self.calibration.set_pinhole_intrinsics(fx=f,fy=f,cx=nx/2.,cy=ny/2.,nx=nx,ny=ny)
            self.calibration.pixel_size = self.pixel_size_box.value() * 1e-6
            self.current_intrinsics_combobox = self.pinhole_intrinsics

        elif self.chessboard_intrinsics.isChecked():
            self.calcam_intrinsics_fname.hide()
            self.interactor3d.zoom_enabled = False
            self.load_chessboard_button.setEnabled(True)
            self.load_intrinsics_button.setEnabled(False)
            self.pinhole_params_box.hide()

            if self.chessboard_fit is None:
                self.update_chessboard_intrinsics(pass_calib=False)
                if self.chessboard_fit is None:
                    self.current_intrinsics_combobox.click()
                return
            else:
                self.calibration.set_chessboard_intrinsics(self.chessboard_fit,self.chessboard_pointpairs,self.chessboard_src)
                self.current_intrinsics_combobox = self.chessboard_intrinsics

        old_aspect = self.interactor3d.force_aspect

        imshape = self.calibration.geometry.get_display_shape()

        aspect = float(imshape[1]) / float(imshape[0])
        aspect_changed = True
        if old_aspect is not None:
            if np.abs(old_aspect - aspect) < 1e-2:
                aspect_changed = False

        self.interactor3d.force_aspect = aspect

        mat = self.calibration.get_cam_matrix()
        n = self.calibration.geometry.get_display_shape()
        wcx = -2.*(mat[0,2] - n[0]/2.) / float(n[0])
        wcy = 2.*(mat[1,2] - n[1]/2.) / float(n[1])

        self.camera_3d.SetWindowCenter(wcx,wcy)
        self.camera_3d.SetViewAngle(self.calibration.get_fov()[1])

        # This is a slight hack - we have to resize the window slightly
        # and resize it back again to get VTK to redraw the background.
        # More elegant solution needed here really.
        if aspect_changed:
            size = self.size()
            self.resize(size.width()+1,size.height())
            self.refresh_3d()
            self.resize(size.width(),size.height())


        # For a reason I don't fully understand, to avoid having partly non-updating
        # drawing in the VTK frame if the aspect ratio is changed, we have to resize
        # the window. So resize and and un-resize it again.
        winsize = self.size()
        self.resize(winsize.width()+1,winsize.height())
        self.resize(winsize.width(),winsize.height())

        self.refresh_3d()
        if self.cadmodel is not None:
            self.unsaved_changes = True



    def save(self,saveas=False):

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
            self.action_save.setEnabled(True)
            self.setWindowTitle('Calcam Virtual Calibration Tool - {:s}'.format(os.path.split(self.filename)[-1][:-4]))
            self.statusbar.clearMessage()
            self.app.restoreOverrideCursor()

        elif saveas:
            self.filename = orig_filename
      


    def open_calib(self,filename=None):

        try:
            opened_calib = Calibration(filename)
        except:
            opened_calib = self.object_from_file('calibration')

        if opened_calib is None:
            return
        
        if opened_calib._type == 'fit':
            raise UserWarning('The selected calibration is a point-pair fitting calibration and cannot be edited in this tool. Please open it with the point fitting calibration tool instead.')
        elif opened_calib._type == 'alignment':
            raise UserWarning('The selected calibration is an alignment calibration and cannot be edited in this tool. Please open it with the alignment calibration editor instead.')

        if opened_calib.cad_config is not None:
            cconfig = opened_calib.cad_config
            if self.cadmodel is not None and self.cadmodel.machine_name == cconfig['model_name'] and self.cadmodel.model_variant == cconfig['model_variant']:
                keep_model = True
            else:
                keep_model = False
        else:
            keep_model = True

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.reset(keep_cadmodel = keep_model)

        # Basic setup
        self.filename = opened_calib.filename
        self.setWindowTitle('Calcam Virtual Calibration Tool - {:s}'.format(os.path.split(self.filename)[-1][:-4]))

        
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

        # Load the intrinsics
        if opened_calib.intrinsics_type == 'pinhole':

            if opened_calib.pixel_size is not None:
                fl = opened_calib.view_models[0].cam_matrix[0,0] * opened_calib.pixel_size  * 1000
                self.pixel_size_box.setValue(opened_calib.pixel_size * 1e6)
            else:
                fl = opened_calib.view_models[0].cam_matrix[0,0] * self.pixel_size_box.value()  / 1000

            im_size = opened_calib.geometry.get_display_shape()

            self.x_pixels_box.setValue(im_size[0])
            self.y_pixels_box.setValue(im_size[1])

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

        if self.calibration.readonly:
            self.action_save.setEnabled(False)
        else:
            self.action_save.setEnabled(True)

        self.app.restoreOverrideCursor()
        self.unsaved_changes = False


    def _load_model(self):
        self.load_model(hold_view = self.cadmodel is not None or self.calibration.filename is not None)


    def reset(self,keep_cadmodel=False):

        if not keep_cadmodel:
            if self.cadmodel is not None:
                self.cadmodel.remove_from_renderer(self.renderer_3d)
                self.cadmodel.unload()
                self.feature_tree.blockSignals(True)
                self.feature_tree.clear()
                self.feature_tree.blockSignals(False)
                self.cadmodel = None


        self.calibration = Calibration(cal_type='virtual')

        self.filename = None
        self.setWindowTitle('Calcam Virtual Calibration Tool')

        self.chessboard_fit = None
        self.intrinsics_calib = None
        self.pinhole_intrinsics.setChecked(True)

        self.unsaved_changes = False
        self.refresh_3d()