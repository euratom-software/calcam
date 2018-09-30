import cv2

from .core import *
from .vtkinteractorstyles import CalcamInteractorStyle3D
from ..calibration import Calibration
from ..render import get_image_actor

# View designer window.
# This allows creation of FitResults objects for a 'virtual' camera.
class AlignmentCalibWindow(CalcamGUIWindow):
 
    def __init__(self, app, parent = None):

        # GUI initialisation
        CalcamGUIWindow.init(self,'alignment_calib_editor.ui',app,parent)

        self.action_new.setIcon( app.style().standardIcon(qt.QStyle.SP_FileIcon) )
        self.action_open.setIcon( app.style().standardIcon(qt.QStyle.SP_DialogOpenButton) )
        self.action_save.setIcon( app.style().standardIcon(qt.QStyle.SP_DialogSaveButton) )

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

        self.pixel_size_box.setSuffix(u' \u00B5m')

        self.action_save.triggered.connect(self.save)
        self.action_save_as.triggered.connect(lambda: self.save(saveas=True))

        self.viewport_calibs = DodgyDict()

        self.image_settings.hide()
        self.image_display_settings.hide()
        self.edge_detect_settings.hide()

        # Populate image sources list and tweak GUI layout for image loading.
        self.imload_inputs = {}
        self.image_load_options.layout().setColumnMinimumWidth(0,100)

        self.image_sources = self.config.get_image_sources()
        for imsource in self.image_sources:
            self.image_sources_list.addItem(imsource['display_name'])
        self.image_sources_list.setCurrentIndex(0)

        self.edge_detect_colour = (1,0,0)
        self.view_aspect = None
        self.vtk_aspect = None
        self.intrinsics_calib = None

        # Start the GUI!
        self.show()

        self.interactor3d.init()
        self.update_intrinsics()
        self.qvtkwidget_3d.GetRenderWindow().GetInteractor().Initialize()
        # Warn the user if we don't have any CAD models
        if self.model_list == {}:
            warn_no_models(self)


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

        if self.original_image is not None:

            CalcamGUIWindow.update_viewport_info(self,keep_selection)

            if self.pinhole_intrinsics.isChecked():

                fov = 3.14159 * self.camera_3d.GetViewAngle() / 180.
                f = self.calibration.geometry.get_display_shape()[1]/(2*np.tan(fov/2.))
                f = f * self.pixel_size_box.value() / 1e3

                self.focal_length_box.setValue(f)



    def update_intrinsics(self,redraw=True):

        if self.original_image is None:
            return

        if self.sender() is self.load_intrinsics_button:
            self.intrinsics_calib = None

        if self.calcam_intrinsics.isChecked():
            self.interactor3d.zoom_enabled = False
            self.load_chessboard_button.setEnabled(False)
            self.load_intrinsics_button.setEnabled(True)
            self.pixel_size_box.setEnabled(False)
            self.focal_length_box.setEnabled(False)
            
            if self.intrinsics_calib is None:
                self.intrinsics_calib = self.object_from_file('calibration')
                if self.intrinsics_calib is not None:
                    if len(self.intrinsics_calib.view_models) != 1:
                        self.intrinsics_calib = None
                        raise UserWarning('This calibration has multiple sub-fields; no worky; sorry.')

            self.calibration.set_calib_intrinsics(self.intrinsics_calib)

        elif self.pinhole_intrinsics.isChecked():
            self.interactor3d.zoom_enabled = True
            self.load_chessboard_button.setEnabled(False)
            self.focal_length_box.setEnabled(True)

            nx,ny = self.calibration.geometry.get_display_shape()

            fov = 3.14159 * self.camera_3d.GetViewAngle() / 180.
            f = ny/(2*np.arctan(fov/2.))

            self.calibration.set_pinhole_intrinsics(fx=f,fy=f,cx=nx/2.,cy=ny/2.,nx=nx,ny=ny)

        elif self.chessboard_intrinsics.isChecked():
            self.interactor3d.zoom_enabled = False
            self.load_chessboard_button.setEnabled(True)
            self.load_intrinsics_button.setEnabled(False)
            self.pixel_size_box.setEnabled(False)
            self.focal_length_box.setEnabled(False)

            if self.chessboard_fit is None:
                self.update_chessboard_intrinsics()

            if self.chessboard_fit is not None:
                self.calibration.set_chessboard_intrinsics(self.chessboard_fit,self.chessboard_pointpairs)
                self.current_intrinsics_combobox = self.chessboard_intrinsics

        self.update_overlay()

        cc = self.calibration.get_cc()
        n = self.calibration.geometry.get_display_shape()
        wcx = -2.*(cc[0] - n[0]/2.) / float(n[0])
        wcy = 2.*(cc[1] - n[1]/2.) / float(n[1])
        self.camera_3d.SetWindowCenter(wcx,wcy)
        fov = 360*np.arctan( float(n[1]) / (2*self.calibration.view_models[0].cam_matrix[1,1]))/3.14159
        self.camera_3d.SetViewAngle(fov)



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

            newim = self.imsource['get_image_function'](**imload_options)

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

        self.calibration.set_image( self.original_image , subview_mask = self.original_subview_mask, transform_actions = transform_actions,coords='Original',subview_names=subview_names )

        self.interactor3d.force_aspect = float( self.original_image.shape[1] ) / float( self.original_image.shape[0] )

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


        self.calibration.history.append( (int(time.time()), self.config.username,self.config.hostname,'Image loaded from {:s}'.format(newim['from'])) )

        self.update_image_info_string()
        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()


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
            self.im_opacity_slider.setEnabled(False)
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
            self.im_opacity_slider.setEnabled(True)

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
        self.calibration.set_image(newim,subview_mask = self.calibration.geometry.original_to_display_image(self.original_subview_mask),transform_actions = self.calibration.geometry.transform_actions, pixel_aspect = self.calibration.geometry.pixel_aspectratio)

        self.interactor3d.force_aspect = float(newim.shape[1]) / float(newim.shape[0])
 
        self.update_image_info_string()

        self.update_overlay()


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

            self.calibration.set_extrinsics(campos,upvec,camtar = camtar)

            if self.cadmodel is not None:
                self.calibration.cad_config = {'model_name':self.cadmodel.machine_name , 'model_variant':self.cadmodel.model_variant , 'enabled_features':self.cadmodel.get_enabled_features(),'viewport':[self.camX.value(),self.camY.value(),self.camZ.value(),self.tarX.value(),self.tarY.value(),self.tarZ.value(),self.camera_3d.GetViewAngle()] }


            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.statusbar.showMessage('Saving...')
            self.calibration.save(self.filename)
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