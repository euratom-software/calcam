import cv2
import copy

from .core import *
from .vtkinteractorstyles import CalcamInteractorStyle3D
from ..calibration import Calibration, Fitter
from ..render import get_image_actor

# View designer window.
# This allows creation of FitResults objects for a 'virtual' camera.
class AlignmentCalibWindow(CalcamGUIWindow):
 
    def __init__(self, app, parent = None):

        # GUI initialisation
        CalcamGUIWindow.init(self,'alignment_calib_editor.ui',app,parent)

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
        self.calibration = Calibration(cal_type='alignment')
        self.chessboard_fit = None

        self.filename = None
        self.original_image = None

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
        self.load_model_button.clicked.connect(self.load_model)
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
        self.hist_eq.clicked.connect(self.update_overlay)
        self.edge_detect.clicked.connect(self.update_overlay)
        self.no_effect.clicked.connect(self.update_overlay)
        self.im_edge_colour_button.clicked.connect(self.update_edge_colour)
        self.edge_threshold_1.valueChanged.connect(self.update_overlay)
        self.edge_threshold_2.valueChanged.connect(self.update_overlay)

        self.control_sensitivity_slider.valueChanged.connect(lambda x: self.interactor3d.set_control_sensitivity(x*0.01))
        self.rmb_rotate.toggled.connect(self.interactor3d.set_rmb_rotate)
        self.interactor3d.set_control_sensitivity(self.control_sensitivity_slider.value()*0.01)

        self.pixel_size_box.setSuffix(u' \u00B5m')

        self.action_save.triggered.connect(self.save)
        self.action_save_as.triggered.connect(lambda: self.save(saveas=True))
        self.action_open.triggered.connect(self.open_calib)
        self.action_new.triggered.connect(self.reset)

        self.viewport_calibs = DodgyDict()

        self.image_settings.hide()
        self.image_display_settings.hide()
        self.edge_detect_settings.hide()

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

        # Start the GUI!
        self.show()

        self.interactor3d.init()
        self.update_intrinsics()
        self.qvtkwidget_3d.GetRenderWindow().GetInteractor().Initialize()


    def reset(self,keep_cadmodel=False):

        if not keep_cadmodel:
            if self.cadmodel is not None:
                self.cadmodel.remove_from_renderer(self.renderer_3d)
                self.cadmodel.unload()
                self.feature_tree.blockSignals(True)
                self.feature_tree.clear()
                self.feature_tree.blockSignals(False)
                self.cadmodel = None

        self.interactor3d.set_overlay_image(None)

        self.calibration = Calibration(cal_type='alignment')

        self.tabWidget.setTabEnabled(3,False)

        self.filename = None

        self.chessboard_fit = None
        self.intrinsics_calib = None

        self.refresh_3d()
        self.unsaved_changes = False



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


    def open_calib(self):

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

        # Load the image
        for imsource in self.image_sources:
            if imsource.display_name == 'Calcam Calibration':
                self.load_image(newim = imsource.get_image_function(self.filename))

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
            fl = opened_calib.view_models[0].cam_matrix[0,0]
            if opened_calib.pixel_size is not None:
                fl = fl * opened_calib.pixel_size  / 1000

                self.pixel_size_box.setValue(opened_calib.pixel_size)
                self.pixel_size_checkbox.setChecked(True)
            else:
                self.pixel_size_checkbox.setChecked(False)

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

        self.app.restoreOverrideCursor()
        self.unsaved_changes = False

    def update_intrinsics(self):

        if self.original_image is None:
            return

        if self.sender() is self.load_intrinsics_button:
            self.intrinsics_calib = None

        if self.pixel_size_checkbox.isChecked():
            self.pixel_size_box.setEnabled(True)
        else:
            self.pixel_size_box.setEnabled(False)

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
                        raise UserWarning('This calibration has multiple sub-fields; no worky; sorry.')
                    
                    self.calibration.set_calib_intrinsics(self.intrinsics_calib)
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

            #fov = 3.14159 * self.camera_3d.GetViewAngle() / 180.
            f = self.focal_length_box.value()
            if self.pixel_size_checkbox.isChecked():
                f = 1e3 * f / self.pixel_size_box.value()

            self.calibration.set_pinhole_intrinsics(fx=f,fy=f,cx=nx/2.,cy=ny/2.,nx=nx,ny=ny)
            self.current_intrinsics_combobox = self.pinhole_intrinsics

        elif self.chessboard_intrinsics.isChecked():
            self.interactor3d.zoom_enabled = False
            self.load_chessboard_button.setEnabled(True)
            self.load_intrinsics_button.setEnabled(False)
            self.focal_length_box.setEnabled(False)

            if self.chessboard_fit is None:
                self.update_chessboard_intrinsics()

            if self.chessboard_fit is not None:
                self.calibration.set_chessboard_intrinsics(self.chessboard_fit,self.chessboard_pointpairs,self.chessboard_source)
                self.current_intrinsics_combobox = self.chessboard_intrinsics
            else:
                self.current_intrinsics_combobox.setChecked(True)

        self.update_overlay()

        cc = self.calibration.get_cc()
        n = self.calibration.geometry.get_display_shape()
        wcx = -2.*(cc[0] - n[0]/2.) / float(n[0])
        wcy = 2.*(cc[1] - n[1]/2.) / float(n[1])
        self.camera_3d.SetWindowCenter(wcx,wcy)
        fov = 360*np.arctan( float(n[1]) / (2*self.calibration.view_models[0].cam_matrix[1,1]))/3.14159
        self.camera_3d.SetViewAngle(fov)
        self.unsaved_changes = True



    def load_image(self,data=None,newim=None):

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.statusbar.showMessage('Loading image...')

        if newim is None:
            # Gather up the required input arguments from the image load gui
            imload_options = {}
            for arg_name,option in self.imload_inputs.items():
                imload_options[arg_name] = option[1]()
                if qt.qt_ver == 4:
                    if type(imload_options[arg_name]) == qt.QString:
                        imload_options[arg_name] = str(imload_options[arg_name])

            newim = self.imsource.get_image_function(**imload_options)
            self.config.default_image_source = self.imsource.display_name

        # Some checking, user prompting etc should go here
        keep_points = False

        self.original_image = newim['image_data']

        if 'subview_mask' in newim:
            self.original_subview_mask = newim['subview_mask']
        else:
            self.original_subview_mask = np.zeros(self.original_image.shape[:2],dtype=np.uint8)

        if 'subview_names' in newim:
            subview_names = newim['subview_names']
        else:
            subview_names = []

        if 'transform_actions' in newim:
            transform_actions = newim['transform_actions']
        else:
            transform_actions = ''

        if 'pixel_size' in newim:
            self.pixel_size_checkbox.setChecked(True)
            self.pixel_size_box.setValue(newim['pixel_size'])

        self.calibration.set_image( self.original_image , newim['source'], subview_mask = self.original_subview_mask, transform_actions = transform_actions,coords='Original',subview_names=subview_names )

        self.interactor3d.force_aspect = float( self.original_image.shape[0] ) / float( self.original_image.shape[1] )

        # This is a slight hack - we have to resize the window slightly
        # and resize it back again to get VTK to redraw the background.
        # More elegant solution needed here really.
        size = self.size()
        self.resize(size.width()+1,size.height())
        self.refresh_3d()
        self.resize(size.width(),size.height())

        self.calibration.view_models = [None] * self.calibration.n_subviews
        self.fit_timestamps = [None] * self.calibration.n_subviews 

        self.update_intrinsics()

        self.image_settings.show()
        self.image_display_settings.show()

        self.update_image_info_string()
        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()
        self.unsaved_changes = True


    def update_overlay(self):

        self.overlay_image = self.calibration.undistort_image( self.calibration.geometry.original_to_display_image(self.original_image) )

        if self.hist_eq.isChecked() or self.edge_detect.isChecked():
            hist_equaliser = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            if len(self.overlay_image.shape) == 2:
                self.overlay_image = hist_equaliser.apply(self.overlay_image.astype('uint8'))
            elif len(self.overlay_image.shape) > 2:
                for channel in range(3):
                    self.overlay_image[:,:,channel] = hist_equaliser.apply(self.overlay_image.astype('uint8')[:,:,channel]) 

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
            self.calibration.geometry.transform_actions = []
            self.calibration.geometry.pixel_aspectratio = 1



        # Update the image and point pairs
        newim = self.calibration.geometry.original_to_display_image(self.original_image)
        self.calibration.set_image(newim,self.calibration.history['image'],subview_mask = self.calibration.geometry.original_to_display_image(self.original_subview_mask),transform_actions = self.calibration.geometry.transform_actions, pixel_aspect = self.calibration.geometry.pixel_aspectratio)

        self.interactor3d.force_aspect = float(newim.shape[0]) / float(newim.shape[1])
 
        self.update_image_info_string()

        self.update_overlay()
        self.unsaved_changes = True



    def update_chessboard_intrinsics(self):

        dialog = ChessboardDialog(self,modelselection=True)
        dialog.exec_()

        if dialog.results is not None:
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
            self.chessboard_source = dialog.chessboard_source

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
            self.camera_3d.OrthogonalizeViewUp()
            upvec = np.array(self.camera_3d.GetViewUp())

            self.calibration.set_extrinsics(campos,upvec,camtar = camtar)

            if self.cadmodel is not None:
                self.calibration.cad_config = {'model_name':self.cadmodel.machine_name , 'model_variant':self.cadmodel.model_variant , 'enabled_features':self.cadmodel.get_enabled_features(),'viewport':[self.camX.value(),self.camY.value(),self.camZ.value(),self.tarX.value(),self.tarY.value(),self.tarZ.value(),self.camera_3d.GetViewAngle()] }


            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.statusbar.showMessage('Saving...')
            self.calibration.save(self.filename)
            self.unsaved_changes = False
            self.statusbar.clearMessage()
            self.app.restoreOverrideCursor()

        elif saveas:
            self.filename = orig_filename



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