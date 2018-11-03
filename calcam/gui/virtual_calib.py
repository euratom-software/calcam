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
from ..calibration import Calibration,Fitter

# View designer window.
# This allows creation of FitResults objects for a 'virtual' camera.
class VirtualCalib(CalcamGUIWindow):
 
    def __init__(self, app, parent = None):

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
        self.virtual_calib = Calibration(cal_type='virtual')
        self.chessboard_fit = None

        self.filename = None

        # Callbacks for GUI elements
        self.viewlist.itemSelectionChanged.connect(self.change_cad_view)
        self.camX.valueChanged.connect(self.change_cad_view)
        self.camY.valueChanged.connect(self.change_cad_view)
        self.camZ.valueChanged.connect(self.change_cad_view)
        self.tarX.valueChanged.connect(self.change_cad_view)
        self.tarY.valueChanged.connect(self.change_cad_view)
        self.tarZ.valueChanged.connect(self.change_cad_view)
        self.cam_roll.valueChanged.connect(self.change_cad_view)
        self.load_model_button.clicked.connect(self.load_model)
        self.model_name.currentIndexChanged.connect(self.populate_model_variants)
        self.feature_tree.itemChanged.connect(self.update_checked_features)
        self.calcam_intrinsics.clicked.connect(self.update_intrinsics)
        self.chessboard_intrinsics.clicked.connect(self.update_intrinsics)
        self.pinhole_intrinsics.clicked.connect(self.update_intrinsics)
        self.pixel_size_box.valueChanged.connect(self.update_intrinsics)
        self.x_pixels_box.valueChanged.connect(self.update_intrinsics)
        self.y_pixels_box.valueChanged.connect(self.update_intrinsics)
        self.focal_length_box.valueChanged.connect(self.update_intrinsics)
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

        self.viewport_calibs = DodgyDict()
        self.intrinsics_calib = None

        # Start the GUI!
        self.show()

        self.interactor3d.init()
        self.update_intrinsics()
        self.qvtkwidget_3d.GetRenderWindow().GetInteractor().Initialize()
        # Warn the user if we don't have any CAD models
        if self.model_list == {}:
            warn_no_models(self)



    def update_viewport_info(self,keep_selection=False):

        CalcamGUIWindow.update_viewport_info(self,keep_selection)
        self.unsaved_changes = True

        if self.pinhole_intrinsics.isChecked():

            fov = 3.14159 * self.camera_3d.GetViewAngle() / 180.
            f = self.virtual_calib.geometry.get_display_shape()[1]/(2*np.tan(fov/2.))
            f = f * self.pixel_size_box.value() / 1e3

            self.focal_length_box.setValue(f)


    def update_intrinsics(self):

        if self.sender() is self.load_intrinsics_button:
            self.intrinsics_calib = None

        if self.calcam_intrinsics.isChecked():
            self.interactor3d.zoom_enabled = False
            self.load_chessboard_button.setEnabled(False)
            self.load_intrinsics_button.setEnabled(True)
            self.x_pixels_box.setEnabled(False)
            self.y_pixels_box.setEnabled(False)
            self.pixel_size_box.setEnabled(False)
            self.focal_length_box.setEnabled(False)
            
            if self.intrinsics_calib is None:
                self.intrinsics_calib = self.object_from_file('calibration')
                if self.intrinsics_calib is not None:
                    if len(self.intrinsics_calib.view_models) != 1:
                        self.intrinsics_calib = None
                        raise UserWarning('This calibration has multiple sub-fields; no worky; sorry.')

                self.virtual_calib.set_calib_intrinsics(self.intrinsics_calib)

        elif self.pinhole_intrinsics.isChecked():
            self.interactor3d.zoom_enabled = True
            self.load_chessboard_button.setEnabled(False)
            self.load_intrinsics_button.setEnabled(False)
            self.x_pixels_box.setEnabled(True)
            self.y_pixels_box.setEnabled(True)
            self.pixel_size_box.setEnabled(True)
            self.focal_length_box.setEnabled(True)

            nx = self.x_pixels_box.value()
            ny = self.y_pixels_box.value()
            f = 1e3 * self.focal_length_box.value() / self.pixel_size_box.value()

            self.virtual_calib.set_pinhole_intrinsics(fx=f,fy=f,cx=nx/2.,cy=ny/2.,nx=nx,ny=ny)
            self.virtual_calib.pixel_size = self.pixel_size_box.value()

        elif self.chessboard_intrinsics.isChecked():
            self.interactor3d.zoom_enabled = False
            self.load_chessboard_button.setEnabled(True)
            self.load_intrinsics_button.setEnabled(False)
            self.x_pixels_box.setEnabled(False)
            self.y_pixels_box.setEnabled(False)
            self.pixel_size_box.setEnabled(False)
            self.focal_length_box.setEnabled(False)

            if self.chessboard_fit is None:
                self.update_chessboard_intrinsics()

            if self.chessboard_fit is not None:
                self.virtual_calib.set_chessboard_intrinsics(self.chessboard_fit,self.chessboard_pointpairs,self.chessboard_src)
                self.current_intrinsics_combobox = self.chessboard_intrinsics

        old_aspect = self.interactor3d.force_aspect

        aspect = float(self.virtual_calib.geometry.y_pixels) / float(self.virtual_calib.geometry.x_pixels)
        aspect_changed = True
        if old_aspect is not None:
            if np.abs(old_aspect - aspect) < 1e-2:
                aspect_changed = False

        self.interactor3d.force_aspect = aspect

        cc = self.virtual_calib.get_cc()
        n = self.virtual_calib.geometry.get_display_shape()
        wcx = -2.*(cc[0] - n[0]/2.) / float(n[0])
        wcy = 2.*(cc[1] - n[1]/2.) / float(n[1])
        self.camera_3d.SetWindowCenter(wcx,wcy)
        self.camera_3d.SetViewAngle(self.virtual_calib.get_fov()[1])

        # This is a slight hack - we have to resize the window slightly
        # and resize it back again to get VTK to redraw the background.
        # More elegant solution needed here really.
        if aspect_changed:
            size = self.size()
            self.resize(size.width()+1,size.height())
            self.refresh_3d()
            self.resize(size.width(),size.height())

        self.refresh_3d()
        self.unsaved_changes = True


    def update_chessboard_intrinsics(self):

        dialog = ChessboardDialog(self,modelselection=True)
        dialog.exec_()

        if dialog.results != []:
            chessboard_pointpairs = dialog.results
            if dialog.perspective_model.isChecked():
                fitter = Fitter('perspective')
            elif dialog.fisheye_model.isChecked():
                fitter = Fitter('fisheye')
            fitter.set_pointpairs(chessboard_pointpairs[0][1])
            for chessboard_im in chessboard_pointpairs[1:]:
                fitter.add_intrinsics_pointpairs(chessboard_im[1])

            self.chessboard_pointpairs = chessboard_pointpairs
            self.chessboard_fit = fitter.do_fit()
            self.chessboard_src = dialog.chessboard_source

        del dialog




    def save(self,saveas=False):

        if saveas:
            orig_filename = self.filename
            self.filename = None

        if self.filename is None:
            self.filename = self.get_save_filename('calibration')

        if self.filename is not None:

            # First we have to add the extrinsics to the calibration object
            campos = np.matrix(self.camera_3d.GetPosition())
            camtar = np.matrix(self.camera_3d.GetFocalPoint())

            # We need to pass the view up direction to set_exirtinsics, but it isn't kept up-to-date by
            # the VTK camera. So here we explicitly ask for it to be updated then pass the correct
            # version to set_extrinsics, but then reset it back to what it was, to avoid ruining 
            # the mouse interaction.
            cam_roll = self.camera_3d.GetRoll()
            self.camera_3d.OrthogonalizeViewUp()
            upvec = np.array(self.camera_3d.GetViewUp())
            self.camera_3d.SetViewUp(0,0,1)
            self.camera_3d.SetRoll(cam_roll)

            self.virtual_calib.set_extrinsics(campos,upvec,camtar = camtar)

            if self.cadmodel is not None:
                self.virtual_calib.cad_config = {'model_name':self.cadmodel.machine_name , 'model_variant':self.cadmodel.model_variant , 'enabled_features':self.cadmodel.get_enabled_features(),'viewport':[self.camX.value(),self.camY.value(),self.camZ.value(),self.tarX.value(),self.tarY.value(),self.tarZ.value(),self.camera_3d.GetViewAngle()] }


            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.statusbar.showMessage('Saving...')
            self.virtual_calib.save(self.filename)
            self.unsaved_changes = False
            self.statusbar.clearMessage()
            self.app.restoreOverrideCursor()

        elif saveas:
            self.filename = orig_filename



    def open_calib(self):

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
            keep_model = False

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.reset(keep_cadmodel = keep_model)

        # Basic setup
        self.filename = opened_calib.filename


        # Load the appropriate CAD model, if we know what that is
        if opened_calib.cad_config is not None:
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


        self.calibration = opened_calib

        # Load the intrinsics
        if opened_calib.intrinsics_type == 'pinhole':
            fl = opened_calib.view_models[0].cam_matrix[0,0] * opened_calib.pixel_size  / 1000

            self.pixel_size_box.setValue(opened_calib.pixel_size)
            im_size = opened_calib.geometry.get_display_shape()
            self.x_pixels_box.setValue(im_size[0])
            self.y_pixels_box.setValue(im_size[1])

            self.focal_length_box.setValue(fl)
            self.pinhole_intrinsics.setChecked(True)

        elif opened_calib.intrinsics_type == 'chessboard':
            self.chessboard_fit = opened_calib.view_models[0]
            self.chessboard_pointpairs = opened_calib.intrinsics_constraints
            self.chessboard_source = opened_calib.history['intrinsics']
            self.chessboard_intrinsics.setChecked(True)

        elif opened_calib.intrinsics_type == 'calibration':
            self.intrinsics_calib = self.crlibration
            self.calcam_intrinsics.setChecked(True)

        self.update_intrinsics()

        self.set_view_from_calib(self.calibration,0)

        if self.calibration.readonly:
            self.action_save.setEnabled(False)
        else:
            self.action_save.setEnabled(True)

        self.app.restoreOverrideCursor()
        self.unsaved_changes = False


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

        self.chessboard_fit = None
        self.intrinsics_calib = None

        self.unsaved_changes = False
        self.refresh_3d()


    def change_cad_view(self):

        if self.sender() is self.viewlist:
            items = self.viewlist.selectedItems()
            if len(items) > 0:
                view_item = items[0]
            else:
                return

            if view_item.parent() is self.views_root_model:

                view = self.cadmodel.get_view( str(view_item.text(0)))

                # Set to that view
                self.camera_3d.SetPosition(view['cam_pos'])
                self.camera_3d.SetFocalPoint(view['target'])
                self.camera_3d.SetViewUp(0,0,1)
                self.interactor3d.set_xsection(None)
                self.interactor3d.set_roll(view['roll'])

            elif view_item.parent() is self.views_root_results or view_item.parent() in self.viewport_calibs.keys():

                view,subfield = self.viewport_calibs[(view_item)]
                
                if subfield is not None:
                    self.set_view_from_calib(view,subfield)

                return

        self.update_viewport_info(keep_selection=True)


    def set_view_from_calib(self,calibration,subfield):

        viewmodel = calibration.view_models[subfield]

        self.camera_3d.SetPosition(viewmodel.get_pupilpos())
        self.camera_3d.SetFocalPoint(viewmodel.get_pupilpos() + viewmodel.get_los_direction(calibration.get_cc(subview=subfield)[0],calibration.get_cc(subview=subfield)[1]))
        
        self.camera_3d.SetViewUp(-1.*viewmodel.get_cam_to_lab_rotation()[:,1])
        self.interactor3d.set_xsection(None)       

        self.update_viewport_info(keep_selection=True)

        self.interactor3d.update_clipping()

        self.refresh_3d()