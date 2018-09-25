import cv2
import time
from scipy.ndimage.measurements import center_of_mass as CoM
import matplotlib.cm
import copy

from .core import *
from .vtkinteractorstyles import CalcamInteractorStyle2D, CalcamInteractorStyle3D
from ..calibration import Calibration, Fitter
from ..pointpairs import PointPairs
from ..render import render_cam_view,get_image_actor

# Main calcam window class for actually creating calibrations.
class FittingCalibrationWindow(CalcamGUIWindow):
 
    def __init__(self, app, parent = None):

        # GUI initialisation
        CalcamGUIWindow.init(self,'fitting_calib.ui',app,parent)

        # Some messing with background colours to fix annoying QT behaviour
        #self.scrollArea.setStyleSheet("QScrollArea {background-color:transparent;}");
        #self.scrollArea.viewport().setStyleSheet(".QWidget {background-color:transparent;}");

        self.action_new.setIcon( app.style().standardIcon(qt.QStyle.SP_FileIcon) )
        self.action_open.setIcon( app.style().standardIcon(qt.QStyle.SP_DialogOpenButton) )
        self.action_save.setIcon( app.style().standardIcon(qt.QStyle.SP_DialogSaveButton) )

        # Start up with no CAD model
        self.cadmodel = None
        self.calibration = Calibration()
        self.calibration.calib_type = 'fit'

        # Set up VTK
        self.qvtkwidget_3d = qt.QVTKRenderWindowInteractor(self.vtkframe_3d)
        self.vtkframe_3d.layout().addWidget(self.qvtkwidget_3d)
        self.interactor3d = CalcamInteractorStyle3D(refresh_callback=self.refresh_3d,viewport_callback=self.update_viewport_info,cursor_move_callback=self.update_cursor_position,newpick_callback=self.new_point_3d,focus_changed_callback=lambda x: self.change_point_focus('3d',x),resize_callback=self.update_vtk_size)
        self.qvtkwidget_3d.SetInteractorStyle(self.interactor3d)
        self.renderer_3d = vtk.vtkRenderer()
        self.renderer_3d.SetBackground(0, 0, 0)
        self.qvtkwidget_3d.GetRenderWindow().AddRenderer(self.renderer_3d)
        self.camera_3d = self.renderer_3d.GetActiveCamera()

        self.qvtkwidget_2d = qt.QVTKRenderWindowInteractor(self.vtkframe_2d)
        self.vtkframe_2d.layout().addWidget(self.qvtkwidget_2d)
        self.interactor2d = CalcamInteractorStyle2D(refresh_callback=self.refresh_2d,newpick_callback = self.new_point_2d,cursor_move_callback=self.update_cursor_position,focus_changed_callback=lambda x: self.change_point_focus('2d',x))
        self.qvtkwidget_2d.SetInteractorStyle(self.interactor2d)
        self.renderer_2d = vtk.vtkRenderer()
        self.renderer_2d.SetBackground(0, 0, 0)
        self.qvtkwidget_2d.GetRenderWindow().AddRenderer(self.renderer_2d)
        self.camera_2d = self.renderer_2d.GetActiveCamera()

        self.populate_models()
        


        # Disable image transform buttons if we have no image
        self.image_settings.hide()
        #self.fit_results.hide()

        self.tabWidget.setTabEnabled(2,False)
        self.tabWidget.setTabEnabled(3,False)
        self.tabWidget.setTabEnabled(4,False)


        self.point_update_timestamp = None
        self.fit_timestamps = []


        # Callbacks for GUI elements
        self.image_sources_list.currentIndexChanged.connect(self.build_imload_gui)
        self.viewlist.itemSelectionChanged.connect(self.change_cad_view)
        self.camX.valueChanged.connect(self.change_cad_view)
        self.camY.valueChanged.connect(self.change_cad_view)
        self.camZ.valueChanged.connect(self.change_cad_view)
        self.tarX.valueChanged.connect(self.change_cad_view)
        self.tarY.valueChanged.connect(self.change_cad_view)
        self.tarZ.valueChanged.connect(self.change_cad_view)
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
        #self.fit_button.clicked.connect(self.do_fit)
        self.fitted_points_checkbox.toggled.connect(self.toggle_reprojected)
        self.overlay_checkbox.toggled.connect(self.toggle_overlay)
        #self.save_fit_button.clicked.connect(self.save_fit)
        #self.save_points_button.clicked.connect(self.save_points)
        self.hist_eq_checkbox.stateChanged.connect(self.toggle_hist_eq)
        self.im_define_splitFOV.clicked.connect(self.edit_split_field)
        #self.pointpairs_load_name.currentIndexChanged.connect(self.update_load_pp_button_status)
        self.pixel_size_checkbox.toggled.connect(self.update_fitopts_gui)
        self.pixel_size_box.valueChanged.connect(self.update_pixel_size)
        #self.toggle_controls_button.clicked.connect(self.toggle_controls)
        self.load_chessboard_button.clicked.connect(self.modify_chessboard_constraints)
        self.chessboard_checkbox.toggled.connect(self.toggle_chessboard_constraints)
        self.load_intrinsics_calib_button.clicked.connect(self.modify_intrinsics_calib)
        self.intrinsics_calib_checkbox.toggled.connect(self.toggle_intrinsics_calib)

        self.action_save.triggered.connect(self.save_calib)
        self.action_save_as.triggered.connect(lambda: self.save_calib(saveas=True))
        self.action_open.triggered.connect(self.load_calib)
        self.action_new.triggered.connect(self.reset)


        self.del_pp_button.clicked.connect(self.remove_current_pointpair)
        self.clear_points_button.clicked.connect(self.clear_pointpairs)

        # If we have an old version of openCV, histo equilisation won't work :(
        cv2_version = float('.'.join(cv2.__version__.split('.')[:2]))
        cv2_micro_version = int(cv2.__version__.split('.')[2].split('-')[0])
        if cv2_version < 2.4 or (cv2_version == 2.4 and cv2_micro_version < 6):
            self.hist_eq_checkbox.setEnabled(False)
            self.hist_eq_checkbox.setToolTip('Requires OpenCV 2.4.6 or newer; you have {:s}'.format(cv2.__version__))

        # Set up some keyboard shortcuts
        # It is done this way in 3 lines per shortcut to avoid segfaults on some configurations
        sc = qt.QShortcut(qt.QKeySequence("Del"),self)
        sc.setContext(qt.Qt.ApplicationShortcut)
        sc.activated.connect(self.remove_current_pointpair)

        sc = qt.QShortcut(qt.QKeySequence("Ctrl+F"),self)
        sc.setContext(qt.Qt.ApplicationShortcut)
        sc.activated.connect(self.do_fit)

        sc = qt.QShortcut(qt.QKeySequence("Ctrl+P"),self)
        sc.setContext(qt.Qt.ApplicationShortcut)
        sc.activated.connect(self.toggle_reprojected)

        sc = qt.QShortcut(qt.QKeySequence("Ctrl+O"),self)
        sc.setContext(qt.Qt.ApplicationShortcut)
        sc.activated.connect(self.toggle_overlay)

        # Odds & sods
        self.pixel_size_box.setSuffix(u' \u00B5m')
        #self.save_fit_button.setEnabled(False)


        # Populate image sources list and tweak GUI layout for image loading.
        self.imload_inputs = {}
        self.image_load_options.layout().setColumnMinimumWidth(0,100)

        self.image_sources = self.config.get_image_sources()
        for imsource in self.image_sources:
            self.image_sources_list.addItem(imsource['display_name'])
        self.image_sources_list.setCurrentIndex(0)

        self.chessboard_pointpairs = []
        
        self.point_pairings = []
        self.selected_pointpair = None

        self.fitters = []
        self.fit_settings_widgets = []

        self.fit_overlay = None

        self.fit_results = []

        self.filename = None

        self.n_points = [ [0,0] ] # One list per sub-field, which is [extrinsics_data, intrinsics_data]

        self.intrinsics_calib = None

        # Start the GUI!
        self.show()
        self.interactor2d.init()
        self.interactor3d.init()
        self.qvtkwidget_3d.GetRenderWindow().GetInteractor().Initialize()
        self.qvtkwidget_2d.GetRenderWindow().GetInteractor().Initialize()
        self.interactor3d.on_resize()


    def modify_intrinsics_calib(self):
        loaded_calib = self.object_from_file('calibration')
        if loaded_calib is not None:
            self.intrinsics_calib = loaded_calib
            self.intrinsics_calib_checkbox.setChecked(True)


    def toggle_intrinsics_calib(self,on):

        if on and self.intrinsics_calib is None:
            self.modify_intrinsics_calib()
            if self.intrinsics_calib is None:
                self.intrinsics_calib_checkbox.setChecked(False)

        self.update_n_points()



    def reset(self):

        # Start up with no CAD model
        if self.cadmodel is not None:
            self.cadmodel.remove_from_renderer(self.renderer_3d)
            self.cadmodel.unload()
            self.feature_tree.blockSignals(True)
            self.feature_tree.clear()
            self.feature_tree.blockSignals(False)
            self.cadmodel = None

        self.clear_pointpairs()

        self.interactor2d.set_image(None)

        self.calibration = Calibration()
        self.calibration.calib_type = 'fit'
        # Disable image transform buttons if we have no image
        self.image_settings.hide()
        #self.fit_results.hide()

        self.tabWidget.setTabEnabled(2,False)
        self.tabWidget.setTabEnabled(3,False)
        self.tabWidget.setTabEnabled(4,False)

        self.point_update_timestamp = None
        self.fit_timestamps = []

        self.filename = None

        self.chessboard_pointpairs = []
        self.chessboard_checkbox.setChecked(False)
        self.intrinsics_calib = None
        self.intrinsics_calib_checkbox.setChecked(False)

        self.refresh_2d()
        self.refresh_3d()


    def reset_fit(self,reset_options=True):

        self.fit_overlay = None
        self.fitted_points_checkbox.setChecked(False)
        self.fitted_points_checkbox.setEnabled(False)
        self.overlay_checkbox.setChecked(False)
        self.overlay_checkbox.setEnabled(False)
        self.rebuild_image_gui(reset_fitters=reset_options)
        self.calibration.view_models = [None] * self.calibration.n_subviews

    def update_cursor_position(self,cursor_id,position):
        
        #info = 'Cursor location: ' + self.cadmodel.format_coord(position).replace('\n',' | ')

        self.point_update_timestamp = time.time()
        self.calibration.view_models = [None] * self.calibration.n_subviews
        self.fit_timestamps = [None] * self.calibration.n_subviews

        #self.statusbar.showMessage(info)



    def new_point_2d(self,im_coords):

        if self.selected_pointpair is not None:
            if self.point_pairings[self.selected_pointpair][1] is None:
                self.point_pairings[self.selected_pointpair][1] = self.interactor2d.add_active_cursor(im_coords)
                self.interactor2d.set_cursor_focus(self.point_pairings[self.selected_pointpair][1])
                self.update_n_points()
                return

        self.point_pairings.append( [None,self.interactor2d.add_active_cursor(im_coords)] )
        self.interactor2d.set_cursor_focus(self.point_pairings[-1][1])
        self.interactor3d.set_cursor_focus(None)
        self.selected_pointpair = len(self.point_pairings) - 1

        self.update_n_points()


    def new_point_3d(self,coords):

        if self.selected_pointpair is not None:
            if self.point_pairings[self.selected_pointpair][0] is None:
                self.point_pairings[self.selected_pointpair][0] = self.interactor3d.add_cursor(coords)
                self.interactor3d.set_cursor_focus(self.point_pairings[self.selected_pointpair][0])
                self.update_n_points()
                return

        self.point_pairings.append( [self.interactor3d.add_cursor(coords),None] )
        self.interactor3d.set_cursor_focus(self.point_pairings[-1][0])
        self.interactor2d.set_cursor_focus(None)
        self.selected_pointpair = len(self.point_pairings) - 1

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
            object_coords = self.interactor3d.get_cursor_coords(self.point_pairings[self.selected_pointpair][0])
            image_coords = self.interactor2d.get_cursor_coords(self.point_pairings[self.selected_pointpair][1])

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
        #self.tabWidget.setTabEnabled(2,True)
        #self.tabWidget.setTabEnabled(3,True)


    def load_image(self,data=None,newim=None):

        # By default we assume we don't know the pixel size
        self.pixel_size_checkbox.setChecked(False)

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

        self.calibration.view_models = [None] * self.calibration.n_subviews
        self.fit_timestamps = [None] * self.calibration.n_subviews 

        self.interactor2d.set_image(self.calibration.geometry.original_to_display_image(newim['image_data']),n_subviews = self.calibration.n_subviews,subview_lookup = self.calibration.subview_lookup)

        self.image_settings.show()
        if self.hist_eq_checkbox.isChecked():
            self.hist_eq_checkbox.setChecked(False)
            self.hist_eq_checkbox.setChecked(True)

        self.rebuild_image_gui()


        if keep_points:

            if self.overlay_checkbox.isChecked():
                self.overlay_checkbox.setChecked(False)
                self.overlay_checkbox.setChecked(True)

            if self.fitted_points_checkbox.isChecked():
                self.fitted_points_checkbox.setChecked(False)
                self.fitted_points_checkbox.setChecked(True)

        else:

            self.fitted_points_checkbox.setChecked(False)
            self.overlay_checkbox.setChecked(False)

        self.calibration.history.append( (int(time.time()), self.config.username,self.config.hostname,'Image loaded from {:s}'.format(newim['from'])) )

        self.update_image_info_string()
        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()


    def change_fit_params(self,fun,state):

        fun(state)
        self.fit_enable_check()


    def rebuild_image_gui(self,reset_fitters = True):

        # Build the GUI to show fit options, according to the number of fields.
        self.subview_tabs.clear()

        # List of settings widgets (for showing / hiding when changing model)
        self.perspective_settings = []
        self.fit_settings_widgets = []
        self.fisheye_settings = []
        self.fit_buttons = []
        self.fit_results = []

        if reset_fitters:
            self.fitters = []

        if self.fitters == []:
            reset_fitters = True

        self.fit_results_widgets = []
        self.view_to_fit_buttons = []

        for field in range(self.calibration.n_subviews):
            
            if reset_fitters:
                self.fitters.append(Fitter())

            new_tab = qt.QWidget()
            new_layout = qt.QVBoxLayout()

            options_groupbox = qt.QGroupBox('Fit Options')
            options_layout = qt.QGridLayout()

            # Selection of model
            widgetlist = [qt.QRadioButton('Perspective Model'),qt.QRadioButton('Fisheye Model')]
        
            if int(cv2.__version__[0]) < 3:
                widgetlist[1].setEnabled(False)
                widgetlist[1].setToolTip('Requires OpenCV 3')

            widgetlist[0].setChecked(True)
            widgetlist[0].toggled.connect(self.update_fitopts_gui)
            widgetlist[1].toggled.connect(self.update_fitopts_gui)
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

            widgetlist[-1].setChecked(True)
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
            widgetlist[-1].toggled.connect(lambda state,field=field: self.change_fit_params(self.fitters[field].fix_4,state))

            for widgetno in [-4,-3,-2,-1]:
                widgetlist[widgetno].toggled.connect(self.fit_enable_check)

            # ------- End of fisheye settings -----------------


            options_layout.addWidget(self.perspective_settings[-1])
            options_layout.addWidget(self.fisheye_settings[-1])
            widgetlist[0].setChecked(True)
            self.fisheye_settings[-1].hide()
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

            #self.fit_results.hide()
            self.tabWidget.setTabEnabled(3,True)
            self.tabWidget.setTabEnabled(4,True)


        # Set pixel size, if the image knows its pixel size.
        if self.calibration.pixel_size is not None:
            self.pixel_size_box.setValue(self.calibration.pixel_size * 1.e6)
            self.pixel_size_checkbox.setChecked(True)


    def update_image_info_string(self):

        if np.any(self.calibration.geometry.get_display_shape() != self.calibration.geometry.get_original_shape()):
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP) [ As Displayed ]<br>{3:d} x {4:d} pixels ({5:.1f} MP) [ Raw Data ]<br>'.format(self.calibration.geometry.get_display_shape()[0],self.calibration.geometry.get_display_shape()[1],np.prod(self.calibration.geometry.get_display_shape()) / 1e6 ,self.calibration.geometry.get_original_shape()[0],self.calibration.geometry.get_original_shape()[1],np.prod(self.calibration.geometry.get_original_shape()) / 1e6 )
        else:
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP)<br>'.format(self.calibration.geometry.get_display_shape()[0],self.calibration.geometry.get_display_shape()[1],np.prod(self.calibration.geometry.get_display_shape()) / 1e6 )
        
        if len(self.calibration.image.shape) == 2:
            info_str = info_str + 'Monochrome'
        elif len(self.calibration.image.shape) == 3 and self.calibration.image.shape[2] == 3:
            info_str = info_str + 'RGB Colour'

        self.image_info.setText(info_str)


    def browse_for_file(self,name_filter,target_textbox=None):

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(0)
        filedialog.setFileMode(1)
        filedialog.setWindowTitle('Select File')
        filedialog.setNameFilter(name_filter)
        filedialog.setLabelText(3,'Select')
        filedialog.exec_()
        if filedialog.result() == 1:

            if target_textbox is not None:
                target_textbox.setText(str(filedialog.selectedFiles()[0]))
            else:
                return


    def transform_image(self,data):

        # First, back up the point pair locations in original coordinates.
        orig_coords = []
        
        # Loop over sub-fields
        for _,cursor_id in self.point_pairings:
            orig_coords.append([])
            for coords in self.interactor2d.get_cursor_coords(cursor_id):
                orig_coords[-1].append(self.calibration.geometry.display_to_original_coords(coords[0],coords[1]))


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


        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)

        if self.fitted_points_checkbox.isChecked():
            self.fitted_points_checkbox.setChecked(False)

        # Transform all the point pairs in to the new coordinates
        for i,(_,cursor_id) in enumerate(self.point_pairings):
            old_coords = orig_coords[i]
            new_coords = []
            for coords in old_coords:
                new_coords.append( self.calibration.geometry.original_to_display_coords(*coords ) )
            self.interactor2d.set_cursor_coords(cursor_id,new_coords)


        # Update the image and point pairs
        self.calibration.set_image(self.calibration.geometry.original_to_display_image(self.original_image),subview_mask = self.calibration.geometry.original_to_display_image(self.original_subview_mask),transform_actions = self.calibration.geometry.transform_actions, pixel_aspect = self.calibration.geometry.pixel_aspectratio)
        self.interactor2d.set_image(self.calibration.geometry.original_to_display_image(self.original_image),n_subviews = self.calibration.n_subviews,subview_lookup=self.calibration.subview_lookup)
        self.update_pointpairs()

        if self.hist_eq_checkbox.isChecked():
            self.hist_eq_checkbox.setChecked(False)
            self.hist_eq_checkbox.setChecked(True)
 
        self.update_image_info_string()
        self.rebuild_image_gui()


    def load_pointpairs(self,data=None,pointpairs=None):

        if pointpairs is None:
            pointpairs = self.object_from_file('pointpairs')

        if pointpairs is not None:

            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.fitted_points_checkbox.setChecked(False)
            self.overlay_checkbox.setChecked(False)

            if self.pointpairs_clear_before_load.isChecked():
                self.clear_pointpairs()

            for i in range(len(pointpairs.object_points)):
                cursorid_3d = self.interactor3d.add_cursor(pointpairs.object_points[i])

                cursorid_2d = None
                for j in range(len(pointpairs.image_points[i])):
                    if pointpairs.image_points[i][j] is not None:
                        if cursorid_2d is None:
                            cursorid_2d = self.interactor2d.add_active_cursor(pointpairs.image_points[i][j])
                        else:
                            self.interactor2d.add_active_cursor(pointpairs.image_points[i][j],add_to=cursorid_2d)

                self.point_pairings.append([cursorid_3d,cursorid_2d])

            self.update_pointpairs()
            self.update_n_points()
            self.update_cursor_info()
            self.app.restoreOverrideCursor()



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
            self.update_n_points()


    def clear_pointpairs(self):

        self.interactor3d.set_cursor_focus(None)
        self.interactor2d.set_cursor_focus(None)
        self.selected_pointpair = None

        for pp in self.point_pairings:
            if pp[0] is not None:
                self.interactor3d.remove_cursor(pp[0])
            if pp[1] is not None:
                self.interactor2d.remove_active_cursor(pp[1])

        self.point_pairings = []

        self.reset_fit(reset_options=False)
        self.update_n_points()


    def fit_enable_check(self):

        # This avoids raising errors if this function is called when we have no
        # fit options GUI.
        if len(self.fit_settings_widgets) == 0:
            return

        # Check whether or not we have enough points to enable the fit button.
        for i,fitter in enumerate(self.fitters):
            enable = True

            # We need at least 4 extrinsics points and at least as many total points as free parameters
            if self.n_points[i][0] < 4 or np.sum(self.n_points[i]) < 6:
                enable = False

            self.fit_buttons[i].setEnabled(enable)
            if enable:
                self.fit_buttons[i].setToolTip('Do fit')
            else:
                self.fit_buttons[i].setToolTip('Cannot fit: more free parameters than point pair data.')



    def update_pointpairs(self):

        pp = PointPairs()

        for pointpair in self.point_pairings:
            if pointpair[0] is not None and pointpair[1] is not None:
                pp.add_pointpair(self.interactor3d.get_cursor_coords(pointpair[0]) , self.interactor2d.get_cursor_coords(pointpair[1]) )

        pp.image_shape = self.calibration.geometry.get_display_shape()

        if pp.get_n_points() > 0:
            self.calibration.set_pointpairs(pp)
        else:
            self.calibration.set_pointpairs(None)

        # Add the intrinsics constraints
        self.calibration.clear_intrinsics_constraints()

        for subview in range(self.calibration.n_subviews):
            self.fitters[subview].set_pointpairs(self.calibration.pointpairs,subview=subview)
            self.fitters[subview].clear_intrinsics_pointpairs()

        if self.intrinsics_calib_checkbox.isChecked():
            self.calibration.add_intrinsics_constraints(calibration=self.intrinsics_calib)
            for subview in range(self.calibration.n_subviews):
                self.fitters[subview].add_intrinsics_pointpairs(self.intrinsics_calib.pointpairs,subview=subview)

        if self.chessboard_checkbox.isChecked():
            for chessboard_constraint in self.chessboard_pointpairs:
                self.calibration.add_intrinsics_constraints(image=chessboard_constraint[0],pointpairs = chessboard_constraint[1]) 
                for subview in range(self.calibration.n_subviews):
                     self.fitters[subview].add_intrinsics_pointpairs(chessboard_constraint[1],subview=subview)


               

    def update_fit_results(self):

        for subview in range(self.calibration.n_subviews):

            if self.calibration.view_models[subview] is None:
                continue

            # Put the results in to the GUI
            params = self.calibration.view_models[subview]

            # Get CoM of this field on the chip
            ypx,xpx = CoM( (self.calibration.subview_mask + 1) * (self.calibration.subview_mask == subview) )

            # Line of sight at the field centre
            los_centre = params.get_los_direction(xpx,ypx)
            fov = self.calibration.get_fov(subview)

            pupilpos = self.calibration.get_pupilpos(subview=subview)

            widgets = self.fit_results_widgets[subview]

            if self.calibration.view_models[subview].model == 'perspective':
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
                                                   "{: 5.4f}".format(params.k1).replace(' ','&nbsp;') ,
                                                   "{: 5.4f}".format(params.k2).replace(' ','&nbsp;') ,
                                                   "{: 5.4f}".format(params.k3).replace(' ','&nbsp;') ,
                                                   "{: 5.4f}".format(params.k4).replace(' ','&nbsp;') ,
                                                   ''
                                                   ] ) )                
            if self.cadmodel is not None:
                widgets[3].setEnabled(True)
            else:
                widgets[3].setEnabled(False)

     
            if self.cadmodel is None:
                self.overlay_checkbox.setEnabled(False)
            else:
                self.overlay_checkbox.setEnabled(True)

            self.fit_results[subview].show()
            self.fitted_points_checkbox.setEnabled(True)
            self.fitted_points_checkbox.setChecked(True)       



    def do_fit(self):

        for i,button in enumerate(self.fit_buttons):
            if self.sender() is button:
                subview = i

        # If this was called via a keyboard shortcut, we may be in no position to do a fit.
        #if not self.fit_button.isEnabled():
        #    return

        self.update_pointpairs()

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.fitted_points_checkbox.setChecked(False)
        self.overlay_checkbox.setChecked(False)
        self.fit_overlay = None

        # Do the fit!
        self.fit_timestamps[subview] = time.time()
        self.statusbar.showMessage('Performing calibration fit...')
        self.calibration.view_models[subview] = self.fitters[subview].do_fit()
        self.statusbar.clearMessage()


        self.fit_changed = True

        self.update_fit_results()

        self.app.restoreOverrideCursor()
        if self.tabWidget.isHidden():
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Close)
            dialog.setWindowTitle('Calcam - Fit Results')
            dialog.setTextFormat(qt.Qt.RichText)
            dialog.setText(str(self.pointpicker.FitResults).replace('\n','<br>'))
            dialog.setIcon(qt.QMessageBox.Information)
            dialog.exec_()



    def toggle_overlay(self,show=None):

        if show is None:
            if self.overlay_checkbox.isEnabled():
                self.overlay_checkbox.setChecked(not self.overlay_checkbox.isChecked())

        elif show:

            if self.fit_overlay is None:

                oversampling = 1.
                self.statusbar.showMessage('Rendering wireframe overlay...')
                self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
                self.app.processEvents()
                try:
                    orig_colours = self.cadmodel.get_colour()
                    self.cadmodel.set_wireframe(True)
                    self.cadmodel.set_colour((0,0,1))
                    self.fit_overlay = render_cam_view(self.cadmodel,self.calibration,transparency=True,verbose=False,aa=2)
                    self.cadmodel.set_colour(orig_colours)
                    self.cadmodel.set_wireframe(False)


                    if np.max(self.fit_overlay) == 0:
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


            self.interactor2d.set_overlay_image(self.fit_overlay)
            self.fitted_points_checkbox.setChecked(False)
            self.refresh_2d()

        else:
            self.interactor2d.set_overlay_image(None)
   


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

        if chessboard_points > 0:
            self.chessboard_checkbox.setText('Chessboard Images ({:d} points)'.format(chessboard_points))
        else:
            self.chessboard_checkbox.setText('Chessboard Images (None Loaded)')

        if intcal_points > 0:
            self.intrinsics_calib_checkbox.setText('Existing Calibration ({:d} points)'.format(intcal_points))
        else:
            self.intrinsics_calib_checkbox.setText('Existing Calibration (None Loaded)')

        self.fit_enable_check()


    def save_calib(self,saveas=False):

        if saveas:
            orig_filename = self.filename
            self.filename = None

        if self.filename is None:
            self.filename = self.get_save_filename('calibration')
        
        if self.filename is not None:

            if self.point_update_timestamp is not None:
                self.calibration.history.append((int(self.point_update_timestamp),self.config.username,self.config.hostname,'Point pairs edited'))
            for subview in range(self.calibration.n_subviews):
                if self.fit_timestamps[subview] is not None:
                    self.calibration.history.append((int(self.fit_timestamps[subview]),self.config.username,self.config.hostname,'Fit performed for {:s}'.format(self.calibration.subview_names[subview])))

            if self.cadmodel is not None:
                self.calibration.cad_config = {'model_name':self.cadmodel.machine_name , 'model_variant':self.cadmodel.model_variant , 'enabled_features':self.cadmodel.get_enabled_features(),'viewport':[self.camX.value(),self.camY.value(),self.camZ.value(),self.tarX.value(),self.tarY.value(),self.tarZ.value(),self.camFOV.value()] }

            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.statusbar.showMessage('Saving...')
            self.calibration.save(self.filename)
            self.statusbar.clearMessage()
            self.app.restoreOverrideCursor()

        elif saveas:
            self.filename = orig_filename


    def load_calib(self):

        # Clear any existing stuff
        opened_calib = self.object_from_file('calibration')

        if opened_calib is None:
            return
        
        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.reset()

        # Basic setup
        self.filename = opened_calib.filename

        # Load the image
        for imsource in self.image_sources:
            if imsource['display_name'] == 'Calcam Calibration':
                self.load_image(newim = imsource['get_image_function'](self.filename))

        # Load the point pairs
        self.load_pointpairs(pointpairs=opened_calib.pointpairs)

        # Load the appropriate CAD model, if we know what that is
        if opened_calib.cad_config is not None:
            cconfig = opened_calib.cad_config
            name_index = sorted(self.model_list.keys()).index(cconfig['model_name'])
            self.model_name.setCurrentIndex(name_index)
            variant_index = self.model_list[ cconfig['model_name'] ][1].index(cconfig['model_variant'])
            self.model_variant.setCurrentIndex(variant_index)

            self.load_model(featurelist=cconfig['enabled_features'])
            self.camX.setValue(cconfig['viewport'][0])
            self.camY.setValue(cconfig['viewport'][1])
            self.camZ.setValue(cconfig['viewport'][2])
            self.tarX.setValue(cconfig['viewport'][3])
            self.tarY.setValue(cconfig['viewport'][4])
            self.tarZ.setValue(cconfig['viewport'][5])
            self.camFOV.setValue(cconfig['viewport'][6])

        for field in range(len(self.fit_settings_widgets)):
            if opened_calib.view_models[field] is not None:

                if opened_calib.view_models[field].model == 'perspective':
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

        self.calibration = opened_calib
        self.interactor2d.subview_lookup = self.calibration.subview_lookup

        for constraint in self.calibration.intrinsics_constraints:
            if constraint.__class__ is Calibration:
                self.intrinsics_calib = constraint
                self.intrinsics_calib_checkbox.setChecked(True)
            else:
                if len(self.chessboard_pointpairs) == 0:
                    self.chessboard_pointpairs = []
                self.chessboard_pointpairs.append(constraint)
                self.chessboard_checkbox.setChecked(True)

        self.update_n_points()
        self.update_fit_results()
        self.app.restoreOverrideCursor()

    def toggle_hist_eq(self,check_state):

        im_out = self.calibration.geometry.original_to_display_image(self.original_image)

        # Enable / disable adaptive histogram equalisation
        if check_state == qt.Qt.Checked:
            hist_equaliser = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            if len(im_out.shape) == 2:
                im_out = hist_equaliser.apply(im_out.astype('uint8'))
            elif len(im_out.shape) > 2:
                for channel in range(3):
                    im_out[:,:,channel] = hist_equaliser.apply(im_out.astype('uint8')[:,:,channel]) 

        self.interactor2d.set_image(im_out,n_subviews = self.calibration.n_subviews,subview_lookup=self.calibration.subview_lookup,hold_position=True)


        
    def closeEvent(self,event):

        self.on_close()


    def edit_split_field(self):

        dialog = SplitFieldDialog(self,self.calibration.image)
        result = dialog.exec_()
        if result == 1:
            self.calibration.set_subview_mask(dialog.fieldmask,subview_names=dialog.field_names)
            self.interactor2d.n_subviews = self.calibration.n_subviews
            self.rebuild_image_gui()

        del dialog


    def build_imload_gui(self,index):

        layout = self.image_load_options.layout()
        for widgets,_ in self.imload_inputs.values():
            for widget in widgets:
                layout.removeWidget(widget)
                widget.close()

        #layout = qt.QGridLayout(self.image_load_options)
        self.imsource = self.image_sources[index]

        self.imload_inputs = {}

        row = 0
        for option in self.imsource['get_image_arguments']:

            labelwidget = qt.QLabel(option['gui_label'] + ':')
            layout.addWidget(labelwidget,row,0)

            if option['type'] == 'filename':
                button = qt.QPushButton('Browse...')
                button.setMaximumWidth(80)
                layout.addWidget(button,row+1,1)
                fname = qt.QLineEdit()
                button.clicked.connect(lambda : self.browse_for_file(option['filter'],fname))                
                if 'default' in option:
                    fname.setText(option['default'])
                layout.addWidget(fname,row,1)
                self.imload_inputs[option['arg_name']] = ([labelwidget,button,fname],fname.text )
                row = row + 2
            elif option['type'] == 'float':
                valbox = qt.QDoubleSpinBox()
                valbox.setButtonSymbols(qt.QAbstractSpinBox.NoButtons)
                if 'limits' in option:
                    valbox.setMinimum(option['limits'][0])
                    valbox.setMaximum(option['limits'][1])
                if 'default' in option:
                    valbox.setValue(option['default'])
                if 'decimals' in option:
                    valbox.setDecimals(option['decimals'])
                layout.addWidget(valbox,row,1)
                self.imload_inputs[option['arg_name']] = ([labelwidget,valbox],valbox.value )
                row = row + 1
            elif option['type'] == 'int':
                valbox = qt.QSpinBox()
                valbox.setButtonSymbols(qt.QAbstractSpinBox.NoButtons)
                if 'limits' in option:
                    valbox.setMinimum(option['limits'][0])
                    valbox.setMaximum(option['limits'][1])
                if 'default' in option:
                    valbox.setValue(option['default'])
                layout.addWidget(valbox,row,1)
                self.imload_inputs[option['arg_name']] = ([labelwidget,valbox],valbox.value )
                row = row + 1
            elif option['type'] == 'string':
                ted = qt.QLineEdit()
                if 'default' in option:
                    ted.setText(option['default'])
                layout.addWidget(ted,row,1)
                self.imload_inputs[option['arg_name']] = ([labelwidget,ted],ted.text )
                row = row + 1
            elif option['type'] == 'bool':
                checkbox = qt.QCheckBox()
                if 'default' in option:
                    checkbox.setChecked(option['default'])
                layout.addWidget(checkbox,row,1)
                self.imload_inputs[option['arg_name']] = ([labelwidget,checkbox],checkbox.isChecked )
                row = row + 1
            elif option['type'] == 'choice':
                cb = qt.QComboBox()
                set_ind = -1
                for i,it in enumerate(option['choices']):
                    cb.addItem(it)
                    if 'default' in option:
                        if option['default'] == it:
                            set_ind = i
                cb.setCurrentIndex(set_ind)
                layout.addWidget(cb,row,1)
                self.imload_inputs[option['arg_name']] = ([labelwidget,cb],cb.currentText) 
                row = row + 1


    
    def update_fitopts_gui(self,choice):

        if self.sender() == self.pixel_size_checkbox:
            if choice:
                self.pixel_size_box.setEnabled(True)
                self.update_pixel_size()
                '''
                for field in range(self.calibration.n_subviews):
                    self.fit_settings_widgets[field][10].setSuffix(' mm')
                    self.fit_settings_widgets[field][10].setValue(self.fit_settings_widgets[field][10].value() * self.image.pixel_size*1e3)
                '''
            else:
                self.pixel_size_box.setEnabled(False)
                self.update_pixel_size()
                '''
                for field in range(self.calibration.n_subviews):
                    self.fit_settings_widgets[field][10].setSuffix(' px')
                    self.fit_settings_widgets[field][10].setValue(self.fit_settings_widgets[field][10].value() / (self.image.pixel_size*1e3))
                '''
    

        
        for field in range(len(self.fit_settings_widgets)):
            if self.sender() == self.fit_settings_widgets[field][0]:
                self.perspective_settings[field].show()
                self.fisheye_settings[field].hide()
            elif self.sender() == self.fit_settings_widgets[field][1]:
                self.perspective_settings[field].hide()
                self.fisheye_settings[field].show()
            elif self.sender() == self.fit_settings_widgets[field][7]:
                self.fit_settings_widgets[field][8].setEnabled(choice)
                self.fit_settings_widgets[field][9].setEnabled(choice)

        self.fit_enable_check()


    def update_pixel_size(self):
        if self.pixel_size_checkbox.isChecked():
            self.calibration.pixel_size = self.pixel_size_box.value() / 1e6
        else:
            self.calibration.pixel_size = None


    def set_fit_viewport(self):
        subview = self.view_to_fit_buttons.index(self.sender())

        self.set_view_from_calib(self.calibration,subview)



    def modify_chessboard_constraints(self):

        dialog = ChessboardDialog(self)
        dialog.exec_()

        if dialog.results != []:
            self.chessboard_pointpairs = copy.deepcopy(dialog.results)
            self.chessboard_checkbox.setChecked(True)

        del dialog


    def toggle_chessboard_constraints(self,on):
        
        if on and len(self.chessboard_pointpairs) == 0:
            self.modify_chessboard_constraints()
            if len(self.chessboard_pointpairs) == 0:
                self.chessboard_checkbox.setChecked(False)

        self.update_n_points()
        self.fit_enable_check()




class SplitFieldDialog(qt.QDialog):

    def __init__(self, parent, image):

        # GUI initialisation
        qt.QDialog.__init__(self, parent)
        qt.uic.loadUi(os.path.join(guipath,'split_field_dialog.ui'), self)

        self.image = image
        self.field_names_boxes = []
        self.parent = parent
        
        # Set up VTK
        self.qvtkwidget = qt.QVTKRenderWindowInteractor(self.vtkframe)
        self.vtkframe.layout().addWidget(self.qvtkwidget)
        self.interactor = CalcamInteractorStyle2D(refresh_callback = self.refresh_2d,newpick_callback=self.on_pick,cursor_move_callback=self.update_fieldmask)
        self.qvtkwidget.SetInteractorStyle(self.interactor)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.qvtkwidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkInteractor = self.qvtkwidget.GetRenderWindow().GetInteractor()

        if self.parent.calibration.n_subviews == 1:
            self.no_split.setChecked(True)

        # Callbacks for GUI elements
        self.method_mask.toggled.connect(self.change_method)
        self.method_points.toggled.connect(self.change_method)
        self.no_split.toggled.connect(self.change_method)
        self.mask_alpha_slider.valueChanged.connect(self.update_mask_alpha)
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self.apply)
        self.mask_browse_button.clicked.connect(self.load_mask_image)

        self.points_options.hide()
        self.mask_options.hide()
        self.mask_image = None
        self.points = []

        self.update_mask_alpha(self.mask_alpha_slider.value())

        # Start the GUI!
        self.show()
        self.interactor.init()
        self.renderer.Render()
        self.vtkInteractor.Initialize()
        self.interactor.set_image(self.image)


    def refresh_2d(self):
        self.renderer.Render()
        self.qvtkwidget.update()


    def change_method(self,checkstate=None):

        if not checkstate:
            return

        if self.method_points.isChecked():
            self.points_options.show()
            self.mask_options.hide()
            self.mask_alpha_slider.setEnabled(True)
            self.mask_alpha_label.setEnabled(True)
            for point in self.points:
                point[1] = self.interactor.add_active_cursor(point[0])
            if len(self.points) == 2:
                self.interactor.set_cursor_focus(point[1])

        elif self.method_mask.isChecked():
            self.mask_options.show()
            self.points_options.hide()
            for point in self.points:
                self.interactor.remove_active_cursor(point[1])
                point[1] = None

            self.mask_alpha_slider.setEnabled(True)
            self.mask_alpha_label.setEnabled(True)
            self.update_fieldmask( mask = np.zeros(self.image.shape[:2],dtype=np.uint8) )
        else:
            self.mask_options.hide()
            self.points_options.hide()
            for point in self.points:
                self.interactor.remove_active_cursor(point[1])
                point[1] = None
            self.mask_alpha_slider.setEnabled(False)
            self.mask_alpha_label.setEnabled(False)
            self.update_fieldmask( mask = np.zeros(self.image.shape[:2],dtype=np.uint8) )

        self.refresh_2d()

    def update_mask_alpha(self,value):
        self.mask_alpha_label.setText('Mask Opacity: {:d}%'.format(value))
        self.overlay_opacity = int(2.55*value)
        if self.mask_image is not None:
            self.mask_image[:,:,3] = self.overlay_opacity
            self.interactor.set_overlay_image(self.mask_image)



    def update_fieldnames_gui(self,n_fields,colours,names):

        if n_fields > 1:
            self.fieldnames_box.show()
        else:
            self.fieldnames_box.hide()

        if n_fields != len(self.field_names_boxes):
            layout = self.fieldnames_box.layout()
            for widget in self.field_names_boxes:
                layout.removeWidget(widget)
            self.field_names_boxes = []
            if n_fields > 1:
                for field in range(n_fields):
                    self.field_names_boxes.append(qt.QLineEdit())
                    self.field_names_boxes[-1].setStyleSheet("background: rgb({:.0f}, {:.0f}, {:.0f}); color: rgb(255,255,255); border: 1px solid;".format(colours[field][0],colours[field][1],colours[field][2]))
                    self.field_names_boxes[-1].setMaximumWidth(150)
                    self.field_names_boxes[-1].setText(names[field])
                    layout.addWidget(self.field_names_boxes[-1])


    def apply(self):

        if self.fieldmask.max() == 0:
            self.field_names = ['Image']
        else:
            self.field_names = [str(namebox.text()) for namebox in self.field_names_boxes]

        self.done(1)



    def load_mask_image(self):

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(0)
        filedialog.setFileMode(1)
        filedialog.setWindowTitle('Select Mask File')
        filedialog.setNameFilter('Image Files (*.png *.bmp *.jp2 *.tiff *.tif)')
        filedialog.setLabelText(3,'Load')
        filedialog.exec_()
        if filedialog.result() == 1:
            mask_im = cv2.imread(str(filedialog.selectedFiles()[0]))

            if mask_im.shape[0] != self.image.shape[0] or mask_im.shape[1] != self.image.shape[1]:
                mask_im = self.parent.calibration.geometry.original_to_display_image(mask_im)

                if mask_im.shape[0] != self.image.shape[0] or mask_im.shape[1] != self.image.shape[1]:
                    raise UserWarning('The selected mask image is a different shape to the camera image! The mask image must be the same shape as the camera image.')

            self.update_fieldmask(mask=mask_im)


    def on_pick(self,im_coords):

        if self.method_points.isChecked() and len(self.points) < 2:
            self.points.append( [im_coords, self.interactor.add_active_cursor(im_coords) ] )
            if len(self.points) == 2:
                self.interactor.set_cursor_focus(self.points[-1][1])
            self.update_fieldmask()



    def update_fieldmask(self,cursor_moved=None,updated_position=None,mask=None):

        if cursor_moved is not None:
            for point in self.points:
                if cursor_moved == point[1]:
                    point[0] = updated_position[0]


        # If we're doing this from points...
        if mask is None:

            if len(self.points) == 2:

                points = [self.points[0][0],self.points[1][0]]
                n0 = np.argmin([points[0][0], points[1][0]])
                n1 = np.argmax([points[0][0], points[1][0]])

                m = (points[n1][1] - points[n0][1])/(points[n1][0] - points[n0][0])
                c = points[n0][1] - m*points[n0][0]

                x,y = np.meshgrid(np.linspace(0,self.image.shape[1]-1,self.image.shape[1]),np.linspace(0,self.image.shape[0]-1,self.image.shape[0]))
                
                ylim = m*x + c
                self.fieldmask = np.zeros(x.shape,int)

                self.fieldmask[y > ylim] = 1

                n_fields = self.fieldmask.max() + 1

            else:
                return

        # or if we're doing it from a mask..
        else:

            if len(mask.shape) > 2:
                for channel in range(mask.shape[2]):
                    mask[:,:,channel] = mask[:,:,channel] * 2**channel
                mask = np.sum(mask,axis=2)

            lookup = list(np.unique(mask))
            n_fields = len(lookup)

            if n_fields > 5:
                raise UserWarning('The loaded mask image contains {:d} different colours. This seems wrong.')

            for value in lookup:
                mask[mask == value] = lookup.index(value)

            self.fieldmask = np.uint8(mask)
        
        self.field_colours = []


        if n_fields > 1:
    
            colours = [matplotlib.cm.get_cmap('nipy_spectral')(i) for i in np.linspace(0.1,0.9,n_fields)]
            
            for colour in colours:
                self.field_colours.append( np.uint8(np.array(colour) * 255) )

            y,x = self.image.shape[:2]

            mask_image = np.zeros([y,x,4],dtype=np.uint8)

            for field in range(n_fields):
                inds = np.where(self.fieldmask == field)
                mask_image[inds[0],inds[1],0] = self.field_colours[field][0]
                mask_image[inds[0],inds[1],1] = self.field_colours[field][1]
                mask_image[inds[0],inds[1],2] = self.field_colours[field][2]

            mask_image[:,:,3] = self.overlay_opacity

            self.interactor.set_overlay_image(mask_image)
            self.mask_image = mask_image
        else:
            self.interactor.set_overlay_image(None)
            self.mask_image = None


        if len(self.field_names_boxes) != n_fields:
            names = []
            if n_fields == 2:

                    xpx = np.zeros(2)
                    ypx = np.zeros(2)

                    for field in range(n_fields):

                        # Get CoM of this field on the chip
                        x,y = np.meshgrid(np.linspace(0,self.image.shape[1]-1,self.image.shape[1]),np.linspace(0,self.image.shape[0]-1,self.image.shape[0]))
                        x[self.fieldmask != field] = 0
                        y[self.fieldmask != field] = 0
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


            self.update_fieldnames_gui(n_fields,self.field_colours,names)
