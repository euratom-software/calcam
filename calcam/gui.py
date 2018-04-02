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


'''
QT Window classes for CalCam, and convenience functions for starting the GUI
(at the end)

Written by Scott Silburn
'''

import sys
import vtk
import os
import numpy as np
import cv2
import gc
import traceback
from scipy.ndimage.measurements import center_of_mass as CoM
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import copy
import webbrowser
import subprocess

from . import qt_wrapper as qt, __version__, CADModel, paths, fitting, image, pointpairs, raytrace, render, image_filters, vtkinteractorstyles, raycast_sightlines
from .config import CalcamConfig
from .coordtransformer import CoordTransformer

cv2_version = float('.'.join(cv2.__version__.split('.')[:2]))
cv2_micro_version = int(cv2.__version__.split('.')[2].split('-')[0])

'''
We start off with various useful functions which are used across the different GUIs.
'''

# Handle exceptions with a dialog giving the user (hopefully) useful information about the error that occured.
def show_exception_box(self,excep_type,excep_value,tb):

    UserCode = False
    self.app.restoreOverrideCursor()
    self.statusbar.clearMessage()

    # First, see if we can blame code the user has written and plugged in.
    for traceback_line in traceback.format_exception(excep_type,excep_value,tb):
        if paths.machine_geometry in traceback_line or paths.image_sources in traceback_line:
            userexc_info = traceback_line
            UserCode = True

    if UserCode and excep_type != UserWarning:
        dialog = qt.QMessageBox(self)
        dialog.setStandardButtons(qt.QMessageBox.Ok)
        dialog.setTextFormat(qt.Qt.RichText)
        dialog.setWindowTitle('Calcam - User Code Error')
        ex_text = traceback.format_exception(excep_type,excep_value,tb)
        dialog.setText(ex_text[-1])
        
        dialog.setInformativeText('This unhandled exception was raised by user plugin code at:\n{:s}'.format(userexc_info.replace(',','\n')))#,''.join(ex_text[:-1])))
        dialog.setIcon(qt.QMessageBox.Warning)
        dialog.exec_()

    else:
        # I'm using user warnings for information boxes which need to be raised:
        if excep_type == UserWarning:
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Ok)
            dialog.setTextFormat(qt.Qt.RichText)
            dialog.setWindowTitle('Calcam')
            dialog.setText(str(excep_value))
            dialog.setIcon(qt.QMessageBox.Information)
            dialog.exec_()
        # Check if we've run out of memory:
        elif excep_type == MemoryError:
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Ok)
            dialog.setTextFormat(qt.Qt.RichText)
            dialog.setWindowTitle('Calcam - Memory Errror')
            text = 'Insufficient memory. '
            if sys.maxsize < 2**32:
                text = text + 'Switching to 64-bit python is highly recommended when working with large data!'
            dialog.setText(text)
            dialog.setInformativeText('Ran out of memory at:<br>'+ traceback.format_exception(excep_type,excep_value,tb)[1])
            dialog.setIcon(qt.QMessageBox.Warning)
            dialog.exec_()                
        # otherwise it's something really unexpected:
        else:
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Ok)
            dialog.setTextFormat(qt.Qt.RichText)
            dialog.setWindowTitle('Calcam - Error')
            dialog.setText('An unhandled exception has been raised. This is probably a bug in Calcam; please report it <a href="https://github.com/euratom-software/calcam/issues">here</a> and/or consider contributing a fix!')
            dialog.setInformativeText(''.join(traceback.format_exception(excep_type,excep_value,tb)))
            dialog.setIcon(qt.QMessageBox.Warning)
            dialog.exec_()


def warn_no_models(self):
    dialog = qt.QMessageBox(self)
    dialog.setStandardButtons(qt.QMessageBox.Ok)
    dialog.setTextFormat(qt.Qt.RichText)
    dialog.setWindowTitle('Calcam - No CAD Models')
    dialog.setText('No CAD model definitions were found. To define some, see the example at:<br><br>{:s}\Example.py'.format(paths.machine_geometry))
    dialog.setIcon(qt.QMessageBox.Information)
    dialog.exec_()


# We start off with some functions which are common to the various GUIs
def init_viewports_list(self):

    self.viewlist.clear()

    # Populate viewports list
    self.views_root_model = qt.QTreeWidgetItem(['Defined in CAD model'])
    self.views_root_results = qt.QTreeWidgetItem(['Calibration Results'])
    self.views_root_synthetic = qt.QTreeWidgetItem(['Virtual Calibrations'])
    self.views_results = []

    # Add views to list
    for view in self.cadmodel.views:
        qt.QTreeWidgetItem(self.views_root_model,[view[0]])

    self.viewlist.addTopLevelItem(self.views_root_model)
    self.viewlist.addTopLevelItem(self.views_root_results)
    self.viewlist.addTopLevelItem(self.views_root_synthetic)
    self.views_root_model.setExpanded(True)
    self.views_root_model.setFlags(qt.Qt.ItemIsEnabled)
    self.views_root_results.setFlags(qt.Qt.ItemIsEnabled)
    self.views_root_synthetic.setFlags(qt.Qt.ItemIsEnabled)

    for view in paths.get_save_list('FitResults'):
        self.views_results.append(qt.QTreeWidgetItem(self.views_root_results,[view]))
        try:
            res = fitting.CalibResults(view)
        except:
            self.views_results[-1].setDisabled(True)
            continue

        if res.nfields > 1:
            for name in res.field_names:
                qt.QTreeWidgetItem(self.views_results[-1],[name])

    for view in paths.get_save_list('VirtualCameras'):

        self.views_results.append(qt.QTreeWidgetItem(self.views_root_synthetic,[view]))
        try:
            res = fitting.VirtualCalib(view)
        except:
            self.views_results[-1].setDisabled(True)
            continue

        if res.nfields > 1:
            for name in res.field_names:
                qt.QTreeWidgetItem(self.views_results[-1],[name])

def init_model_settings(self):

    self.feature_tree.blockSignals(True)
    self.feature_tree.clear()

    # Populate CAD feature tree
    self.treeitem_machine = qt.QTreeWidgetItem([self.cadmodel.machine_name])
    self.treeitem_machine.setFlags(qt.Qt.ItemIsEnabled)
    self.feature_tree.addTopLevelItem(self.treeitem_machine)
    self.treeitem_machine.setExpanded(True)
    self.group_items = {}
    for feature in self.cadmodel.features:
        if feature[8] is None:
            newitem = qt.QTreeWidgetItem(self.treeitem_machine,[feature[0]])
            newitem.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
            if feature[3]:
                newitem.setCheckState(0,qt.Qt.Checked)
            else:
                newitem.setCheckState(0,qt.Qt.Unchecked)
        else:
            if feature[8] in self.group_items:
                newitem = qt.QTreeWidgetItem(self.group_items[feature[8]],[feature[0]])
                newitem.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                if feature[3]:
                    newitem.setCheckState(0,qt.Qt.Checked)
                else:
                    newitem.setCheckState(0,qt.Qt.Unchecked)
            else:
                self.group_items[feature[8]] = qt.QTreeWidgetItem(self.treeitem_machine,[feature[8]])
                self.group_items[feature[8]].setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                self.group_items[feature[8]].setCheckState(0,qt.Qt.Checked)
                self.group_items[feature[8]].setExpanded(True)
                newitem = qt.QTreeWidgetItem(self.group_items[feature[8]],[feature[0]])
                newitem.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                if feature[3]:
                    newitem.setCheckState(0,qt.Qt.Checked)
                else:
                    newitem.setCheckState(0,qt.Qt.Unchecked)

    for group in self.group_items.values():

        enabled_states = []
        for i in range(group.childCount()):
            if group.child(i).checkState(0) == qt.Qt.Checked:
                enabled_states.append(True)
            elif group.child(i).checkState(0) == qt.Qt.Unchecked:
                enabled_states.append(False)

        if not np.any(enabled_states):
            group.setCheckState(0,qt.Qt.Unchecked)
        else:
            if np.all(enabled_states):
                group.setCheckState(0,qt.Qt.Checked)
            else:
                group.setCheckState(0,qt.Qt.PartiallyChecked)

    try:

        self.selected_feature = str(self.feature_tree.currentItem().text(0))
        if self.selected_feature in self.group_tiems:
            feature = self.selected_feature
            self.selected_feature = []
            for i in range(self.group_items[self.selected_feature].childCount()):
                self.selected_feature.append(str(self.group_items[self.selected_feature].child(i).text(0)))
        else:
            self.selected_feature = [self.selected_feature]

    except AttributeError:
        self.selected_feature = None

    self.feature_tree.blockSignals(False)




# CAD viewer window.
# This allows viewing of the CAD model and overlaying raycasted sight-lines, etc.
class CADViewerWindow(qt.QMainWindow):
 
    def __init__(self, app, parent = None):

        # GUI initialisation
        qt.QMainWindow.__init__(self, parent)
        qt.uic.loadUi(os.path.join(paths.ui,'cad_viewer.ui'), self)

        self.setWindowIcon(qt.QIcon(os.path.join(paths.calcampath,'ui','icon.png')))

        self.app = app

        self.config = CalcamConfig()

        # See how big the screen is and open the window at an appropriate size
        desktopinfo = self.app.desktop()
        available_space = desktopinfo.availableGeometry(self)
        self.screensize = (available_space.width(),available_space.height())
        # Open the window with same aspect ratio as the screen, and no fewer than 500px tall.
        win_height = max(500,min(780,0.75*available_space.height()))
        win_width = win_height * available_space.width() / available_space.height() 
        self.resize(win_width,win_height)

         # Let's show helpful dialog boxes if we have unhandled exceptions:
        sys.excepthook = lambda *ex: show_exception_box(self,*ex)

        # Start up with no CAD model
        self.cadmodel = None

        # Set up VTK
        self.qvtkWidget = qt.QVTKRenderWindowInteractor(self.vtk_frame)
        self.vtk_frame.layout().addWidget(self.qvtkWidget,0,0,1,2)
        self.cadexplorer = vtkinteractorstyles.CADExplorer()
        self.qvtkWidget.SetInteractorStyle(self.cadexplorer)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.qvtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkInteractor = self.qvtkWidget.GetRenderWindow().GetInteractor()
        self.camera = self.renderer.GetActiveCamera()


        # Populate CAD model list
        self.model_list = self.config.get_cadmodels()

        self.model_name.addItems(sorted(self.model_list.keys()))
        self.model_name.setCurrentIndex(-1)
        self.load_model_button.setEnabled(0)

        # Callbacks for GUI elements
        self.viewlist.itemSelectionChanged.connect(self.change_cad_view)
        self.camX.valueChanged.connect(self.change_cad_view)
        self.camY.valueChanged.connect(self.change_cad_view)
        self.camZ.valueChanged.connect(self.change_cad_view)
        self.tarX.valueChanged.connect(self.change_cad_view)
        self.tarY.valueChanged.connect(self.change_cad_view)
        self.tarZ.valueChanged.connect(self.change_cad_view)
        self.camFOV.valueChanged.connect(self.change_cad_view)
        self.sightlines_list.itemChanged.connect(self.update_sightlines)
        self.load_model_button.clicked.connect(self.load_model)
        self.model_name.currentIndexChanged.connect(self.populate_model_variants)
        self.feature_tree.itemChanged.connect(self.update_checked_features)
        self.feature_tree.itemSelectionChanged.connect(self.update_cadtree_selection)
        self.xsection_checkbox.toggled.connect(self.toggle_cursor_xsection)
        self.sightline_opacity_slider.valueChanged.connect(self.update_sightlines)
        self.rendertype_edges.toggled.connect(self.toggle_wireframe)
        self.viewport_load_calib.clicked.connect(self.load_viewport_calib)
        self.sightlines_load_button.clicked.connect(self.update_sightlines)
        self.pick_sightlines_colour.clicked.connect(self.update_sightlines)
        self.sightlines_list.itemSelectionChanged.connect(self.update_selected_sightlines)
        self.sightline_type_volume.toggled.connect(self.update_sightlines)
        self.sightlines_legend_checkbox.toggled.connect(self.update_sightlines)
        self.render_button.clicked.connect(self.do_render)
        self.render_cam_view.toggled.connect(self.change_render_type)
        self.render_coords_combobox.currentIndexChanged.connect(self.update_render_coords)
        self.render_load_button.clicked.connect(self.load_render_result)
        self.cad_colour_reset_button.clicked.connect(self.set_cad_colour)
        self.cad_colour_choose_button.clicked.connect(self.set_cad_colour)
        self.save_view_button.clicked.connect(self.save_view_to_model)
        self.save_colours_button.clicked.connect(self.save_cad_colours)

        self.sightlines_legend = None
        self.render_calib = None

        self.model_actors = {}

        self.sightlines = DodgyDict()
        self.colour_q = []
        self.model_custom_colour = None
        self.viewport_calibs = DodgyDict()

        self.colourcycle = colourcycle()

        self.tabWidget.setTabEnabled(1,False)
        self.tabWidget.setTabEnabled(2,False)
        self.tabWidget.setTabEnabled(3,False)

        self.render_coords_text.setHidden(True)
        self.render_coords_combobox.setHidden(True)
        self.render_coords_combobox.setHidden(True)
        self.render_calib_description.setHidden(True)
        self.rendersettings_calib_label.setHidden(True)
        self.render_calib_namelabel.setHidden(True)
        self.render_load_button.setHidden(True)
        self.render_current_description.setHidden(False)



        # -------------------- Initialise View List ------------------
        self.viewlist.clear()

        # Populate viewports list
        self.views_root_model = qt.QTreeWidgetItem(['Defined in Model'])
        self.views_root_auto = qt.QTreeWidgetItem(['Auto Cross-Sections'])
        self.views_root_results = qt.QTreeWidgetItem(['From Calibrations'])


        # Auto type views
        item = qt.QTreeWidgetItem(self.views_root_auto,['Vertical cross-section thru cursor'])
        item.setFlags(qt.Qt.NoItemFlags)
        item = qt.QTreeWidgetItem(self.views_root_auto,['Horizontal cross-section thru cursor'])
        item.setFlags(qt.Qt.NoItemFlags)


        self.viewlist.addTopLevelItem(self.views_root_model)
        self.viewlist.addTopLevelItem(self.views_root_auto)
        self.viewlist.addTopLevelItem(self.views_root_results)
        self.views_root_model.setExpanded(True)
        self.views_root_auto.setExpanded(True)
        self.views_root_results.setExpanded(True)
        self.views_root_results.setHidden(True)
        self.views_root_model.setFlags(qt.Qt.ItemIsEnabled)
        self.views_root_auto.setFlags(qt.Qt.ItemIsEnabled)
        self.views_root_results.setFlags(qt.Qt.ItemIsEnabled)
        # ------------------------------------------------------------

        # Start the GUI!
        self.show()
        self.cadexplorer.DoInit(self.renderer,self)
        self.vtkInteractor.Initialize()

        # Warn the user if we don't have any CAD models
        if self.model_list == {}:
            warn_no_models(self)


 

    def toggle_cursor_xsection(self,onoff):

        if onoff:
            self.cadexplorer.set_xsection(self.cadexplorer.point[0].GetFocalPoint())
        else:
            self.cadexplorer.set_xsection(None)

        self.cadexplorer.update_clipping()
        self.refresh_vtk()


    def update_selected_sightlines(self):

        if len(self.sightlines_list.selectedItems()) > 0:

            self.sightlines_settings_box.setEnabled(True)
            first_sightlines = self.sightlines[self.sightlines_list.selectedItems()[0]]

            self.sightline_type_volume.blockSignals(True)
            if first_sightlines[2] == 'volume':
                self.sightline_type_volume.setChecked(True)
            else:
                self.sightline_type_lines.setChecked(True)
            self.sightline_type_volume.blockSignals(False)

            self.sightline_opacity_slider.blockSignals(True)
            self.sightline_opacity_slider.setValue(100*np.log(first_sightlines[1].GetProperty().GetOpacity()*100.)/np.log(100))

            self.sightline_opacity_slider.blockSignals(False)

        else:
            self.sightlines_settings_box.setEnabled(False)


    def load_viewport_calib(self):
        cals = object_from_file(self,'calibration',multiple=True)

        for cal in cals:
            listitem = qt.QTreeWidgetItem(self.views_root_results,[cal.name])
            if cal.nfields > 1:
                self.viewport_calibs[(listitem)] = (cal,None)
                listitem.setExpanded(True)
                for n,fieldname in enumerate(cal.field_names):
                    self.viewport_calibs[ (qt.QTreeWidgetItem(listitem,[fieldname])) ] = (cal,n) 
            else:
                self.viewport_calibs[(listitem)] = (cal,0)

        if len(cals) > 0:
            self.views_root_results.setHidden(False)


    def save_cad_colours(self):

        cols = self.cadmodel.get_colour()
        self.cadmodel.set_default_colour(cols)


    def change_render_type(self):

        if self.render_cam_view.isChecked():

            # Check if we have a calibration loaded and if not, load one.
            if self.render_calib is None:
                self.render_button.setEnabled(False)

            self.render_coords_text.setHidden(False)
            self.render_coords_combobox.setHidden(False)
            self.render_coords_combobox.setHidden(False)
            self.render_calib_description.setHidden(False)
            self.rendersettings_calib_label.setHidden(False)
            self.render_calib_namelabel.setHidden(False)
            self.render_load_button.setHidden(False)
            self.render_current_description.setHidden(True)
            self.update_render_coords()

        else:
            self.render_coords_text.setHidden(True)
            self.render_coords_combobox.setHidden(True)
            self.render_coords_combobox.setHidden(True)
            self.render_calib_description.setHidden(True)
            self.rendersettings_calib_label.setHidden(True)
            self.render_calib_namelabel.setHidden(True)
            self.render_load_button.setHidden(True)
            self.render_current_description.setHidden(False)
            self.render_resolution.setCurrentIndex(-1)
            self.render_button.setEnabled(True)
            self.cadexplorer.OnWindowSizeAdjust()


    def load_render_result(self):
        cal = object_from_file(self,'Calibration')
        if cal is not None:
            self.render_calib = cal
            self.render_calib_namelabel.setText(cal.name)
            self.render_button.setEnabled(True)
            self.update_render_coords()


    def update_render_coords(self):

        self.render_resolution.clear()
        if self.render_calib is not None:
            if self.render_coords_combobox.currentIndex() == 0:
                base_size = self.render_calib.image_display_shape                
                self.render_resolution.addItem('{:d} x {:d} (same as camera)'.format(base_size[0],base_size[1]))
                self.render_resolution.addItem('{:d} x {:d}'.format(base_size[0]*2,base_size[1]*2))
                self.render_resolution.addItem('{:d} x {:d}'.format(base_size[0]*4,base_size[1]*4))

            elif self.render_coords_combobox.currentIndex() == 1:
                base_size = [self.render_calib.transform.x_pixels,self.render_calib.transform.y_pixels]
                self.render_resolution.addItem('{:d} x {:d} (same as camera)'.format(base_size[0],base_size[1]))


    def save_view_to_model(self):

        if str(self.view_save_name.text()) in self.cadmodel.get_view_names():

            msg = 'A view with this name already exists in the model definition. Are you sure you want to over-write the existing one?'
            reply = qt.QMessageBox.question(self, 'Overwrite?', msg, qt.QMessageBox.Yes, qt.QMessageBox.No)

            if reply == qt.QMessageBox.No:
                return

        cam_pos = (self.camX.value(),self.camY.value(),self.camZ.value())
        target = (self.tarX.value(), self.tarY.value(), self.tarZ.value())
        fov = self.camFOV.value()
        xsection = self.cadexplorer.get_xsection()

        try:
            self.cadmodel.add_view(str(self.view_save_name.text()),cam_pos,target,fov,xsection)
            self.update_model_views()

        except:
            self.update_model_views()
            raise


    def update_model_views(self):

        self.views_root_model.setText(0,self.cadmodel.machine_name)
        self.views_root_model.takeChildren()

        # Add views to list
        for view in self.cadmodel.get_view_names():
            qt.QTreeWidgetItem(self.views_root_model,[view])



    def update_cadtree_selection(self):

        if len(self.feature_tree.selectedItems()) == 0:
            self.cad_colour_choose_button.setEnabled(False)
            self.cad_colour_reset_button.setEnabled(False)
        else:
            self.cad_colour_choose_button.setEnabled(True)
            self.cad_colour_reset_button.setEnabled(True)            


    def change_cad_view(self):


        if self.sender() is self.viewlist:
            items = self.viewlist.selectedItems()
            if len(items) > 0:
                view_item = items[0]
            else:
                return
  
            self.xsection_checkbox.setChecked(False)
            if view_item.parent() is self.views_root_model:

                view = self.cadmodel.get_view( str(view_item.text(0)))

                # Set to that view
                self.camera.SetViewAngle(view['y_fov'])
                self.camera.SetPosition(view['cam_pos'])
                self.camera.SetFocalPoint(view['target'])
                self.camera.SetViewUp(0,0,1)
                self.cadexplorer.set_xsection(view['xsection'])

            elif view_item.parent() is self.views_root_results or view_item.parent() in self.viewport_calibs.keys():

                view,subfield = self.viewport_calibs[(view_item)]
                if subfield is None:
                    return

                self.camera.SetPosition(view.get_pupilpos(field=subfield))
                self.camera.SetFocalPoint(view.get_pupilpos(field=subfield) + view.get_los_direction(view.image_display_shape[0]/2,view.image_display_shape[1]/2))
                self.camera.SetViewAngle(view.get_fov(field=subfield)[1])
                self.camera.SetViewUp(-1.*view.get_cam_to_lab_rotation(field=subfield)[:,1])
                self.cadexplorer.set_xsection(view['xsection'])               

            elif view_item.parent() is self.views_root_auto:

                if str(view_item.text(0)).lower() == 'horizontal cross-section thru cursor' and self.cadexplorer.point is not None:
                    self.camera.SetViewUp(0,1,0)
                    self.camera.SetPosition( (0.,0.,max(self.camZ.value(),self.cadexplorer.point[0].GetFocalPoint()[2]+1.)) )
                    self.camera.SetFocalPoint( (0.,0.,self.cadexplorer.point[0].GetFocalPoint()[2]-1.) )
                    self.xsection_checkbox.setChecked(True)

                elif str(view_item.text(0)).lower() == 'vertical cross-section thru cursor' and self.cadexplorer.point is not None:
                    self.camera.SetViewUp(0,0,1)
                    R_cursor = np.sqrt( self.cadexplorer.point[0].GetFocalPoint()[1]**2 + self.cadexplorer.point[0].GetFocalPoint()[0]**2 )
                    phi = np.arctan2(self.cadexplorer.point[0].GetFocalPoint()[1],self.cadexplorer.point[0].GetFocalPoint()[0])
                    phi_cam = phi - 3.14159/2.
                    R_cam = np.sqrt( self.camX.value()**2 + self.camY.value()**2 )
                    self.camera.SetPosition( (max(R_cam,R_cursor + 1) * np.cos(phi_cam), max(R_cam,R_cursor + 1) * np.sin(phi_cam), 0.) )
                    self.camera.SetFocalPoint( (0.,0.,0.) )
                    self.xsection_checkbox.setChecked(True)

                elif str(view_item.text(0)).lower() == 'centre current view on origin':
                    oldviewup = self.camera.GetViewUp()
                    self.camera.OrthogonalizeViewUp()
                    if np.argmax(self.camera.GetViewUp()) == 2:
                        self.camera.SetPosition( (self.camX.value(),self.camY.value(),0.) )
                        self.camera.SetViewUp((0.,0.,1.))
                    elif np.argmax(self.camera.GetViewUp()) == 1:
                        self.camera.SetPosition( (0.,0.,self.camZ.value()) )
                        self.camera.SetViewUp((0.,1.,0.))
                    self.camera.SetFocalPoint((0.,0.,-self.camZ.value()))


        else:
            self.camera.SetPosition((self.camX.value(),self.camY.value(),self.camZ.value()))
            self.camera.SetFocalPoint((self.tarX.value(),self.tarY.value(),self.tarZ.value()))
            self.camera.SetViewAngle(self.camFOV.value())

        self.update_viewport_info(keep_selection=True)

        self.cadexplorer.update_clipping()

        self.refresh_vtk()



    def colour_model_by_material(self):

            if self.cadmodel is not None:
                self.cadmodel.colour_by_material()
                self.refresh_vtk()


    def set_cad_colour(self):

        selected_features = []
        for treeitem in self.feature_tree.selectedItems():
            selected_features.append(self.cad_tree_items[treeitem])

        # Note: this does not mean nothing is selected;
        # rather it means the root of the model is selected!
        if None in selected_features:
            selected_features = None

        if self.sender() is self.cad_colour_choose_button:

            picked_colour = pick_colour(self,self.cadmodel.get_colour( selected_features )[0] )

            if picked_colour is not None:

                self.cadmodel.set_colour(picked_colour,selected_features)


        elif self.sender() is self.cad_colour_reset_button:

            self.cadmodel.reset_colour(selected_features)

        self.refresh_vtk()



    def toggle_wireframe(self,wireframe):
        
        if self.cadmodel is not None:

            self.cadmodel.set_wireframe( wireframe )

            self.refresh_vtk()



    def do_render(self):

        # Get the file name to save as from the user.
        dialog = qt.QFileDialog(self)
        dialog.setAcceptMode(1)
        dialog.setFileMode(0)
        dialog.setWindowTitle('Export High Res Image...')
        dialog.setNameFilter('PNG Image (*.png)')
        dialog.exec_()
        if dialog.result() == 1:
            filename = str(dialog.selectedFiles()[0])
            if not filename.endswith('.png'):
                filename = filename + '.png'
        else:
            return

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))


        # Values of extra AA and oversampling to use
        aa = (self.render_aa.currentIndex() + 1)
        oversampling=2**(self.render_resolution.currentIndex())
        use_transparency = self.render_bg_transparent.isChecked()


        # Render the current view, using the current renderer.
        if self.render_current_view.isChecked():
        
            # Actors which will be temporarily removed from the scene while we render
            # -----------------------------------------------------------------------
            temp_actors = []

            # For some reason having the legend completely ruins things,
            # so we turn it off.
            if self.sightlines_legend_checkbox.isChecked():
                temp_actors.append(self.cadexplorer.legend)

            # Get rid of the cursor unless the user said not to.
            if not self.render_include_cursor.isChecked():
                if self.cadexplorer.point is not None:
                    temp_actors.append( self.cadexplorer.point[2] )

            for actor in temp_actors:
                self.renderer.RemoveActor(actor)
            # -------------------------------------------------------------------------

            # Do the render
            im = render.render_hires(self.renderer,oversampling=oversampling,aa=aa,transparency=use_transparency)

            # Add back the temporarily removed actors
            for actor in temp_actors:
                self.renderer.AddActor(actor)

        # Render a calibrated camera's point of view
        elif self.render_cam_view.isChecked():

            # For render_cam_view we need to tell it what to include apart from the cad model.
            # So make a list of the other actors we need to add:
            extra_actors = []
            for listitem,sightlines in self.sightlines:
                if listitem.isChecked():
                    extra_actors.append(sightlines[1])

            im = render.render_cam_view(self.cadmodel,self.render_calib,extra_actors = extra_actors,oversampling=oversampling,aa=aa,transparency=use_transparency,verbose=False)
        

        # Save the image!
        im[:,:,:3] = im[:,:,2::-1]
        cv2.imwrite(filename,im)

        self.renderer.Render()

        self.app.restoreOverrideCursor()


    def populate_model_variants(self):

        model = self.model_list[str(self.model_name.currentText())]
        self.model_variant.clear()
        self.model_variant.addItems(model[1])
        self.model_variant.setCurrentIndex(model[1].index(model[2]))
        self.load_model_button.setEnabled(1)


    def update_checked_features(self,item):

            self.cadmodel.set_features_enabled(item.checkState(0) == qt.Qt.Checked,self.cad_tree_items[item])
            self.update_feature_tree_checks()

            self.refresh_vtk()

            for key,item in self.sightlines:
                recheck = False
                if key.checkState() == qt.Qt.Checked:
                    recheck = True
                    key.setCheckState(qt.Qt.Unchecked)
                    self.colourcycle.queue_colour(item[1].GetProperty().GetColor())
                item[1] = None
                if recheck:
                    key.setCheckState(qt.Qt.Checked)



    def update_feature_tree_checks(self):

        self.feature_tree.blockSignals(True)

        enabled_features = self.cadmodel.get_enabled_features()

        for qitem,feature in self.cad_tree_items:

            try:
                qitem.setCheckState(0,self.cadmodel.get_group_enable_state(feature))
            except KeyError:
                if feature in enabled_features:
                    qitem.setCheckState(0,qt.Qt.Checked)
                else:
                    qitem.setCheckState(0,qt.Qt.Unchecked)

        self.feature_tree.blockSignals(False)


    def load_model(self):

        # Dispose of the old model
        if self.cadmodel is not None:

            self.cadmodel.remove_from_renderer(self.renderer)
            self.cadmodel.unload()

            del self.cadmodel


        # Create a new one
        self.cadmodel = CADModel( str(self.model_name.currentText()) , str(self.model_variant.currentText()) , self.update_cad_status)


        if not self.cad_auto_load.isChecked():
            self.cadmodel.set_features_enabled(False)

        self.cadmodel.add_to_renderer(self.renderer)

        self.statusbar.showMessage('Setting up CAD model...')


        # -------------------------- Populate the model feature tree ------------------------------
        self.feature_tree.blockSignals(True)
        self.feature_tree.clear()

        self.cad_tree_items = DodgyDict()

        treeitem_top = qt.QTreeWidgetItem([self.cadmodel.machine_name])
        treeitem_top.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsSelectable)
        self.feature_tree.addTopLevelItem(treeitem_top)
        treeitem_top.setExpanded(True)

        self.cad_tree_items[treeitem_top] = None

        group_items = {}

        enabled_features = self.cadmodel.get_enabled_features()


        # We need to add the group items first, to make the tree look sensible:
        for feature in self.cadmodel.get_feature_list():
            namesplit = feature.split('/')
            if len(namesplit) > 1:
                if namesplit[0] not in group_items:
                    newitem = qt.QTreeWidgetItem(treeitem_top,[namesplit[0]])
                    newitem.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                    newitem.setExpanded(True)
                    self.cad_tree_items[newitem] = namesplit[0]
                    group_items[namesplit[0]] = newitem

        # Now go through and add the actual features
        for feature in self.cadmodel.get_feature_list():
            namesplit = feature.split('/')
            if len(namesplit) == 1:
                parent = treeitem_top
                featurename = feature
            else:
                parent = group_items[namesplit[0]]
                featurename = namesplit[1]

            newitem = qt.QTreeWidgetItem(parent, [featurename])
            newitem.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)

            self.cad_tree_items[newitem] = feature

        self.feature_tree.blockSignals(False)

        self.update_feature_tree_checks()
        # ---------------------------------------------------------------------------------------


        # Enable the other tabs!
        self.tabWidget.setTabEnabled(1,True)
        self.tabWidget.setTabEnabled(2,True)
        self.tabWidget.setTabEnabled(3,True)

        self.cad_colour_controls.setEnabled(True)
        self.cad_colour_reset_button.setEnabled(False)
        self.cad_colour_choose_button.setEnabled(False)


        # Make sure the light lights up the whole model without annoying shadows or falloff.
        light = self.renderer.GetLights().GetItemAsObject(0)
        light.PositionalOn()
        light.SetConeAngle(180)

        # Put the camera in some reasonable starting position
        self.camera.SetViewAngle(90)
        self.camera.SetViewUp((0,0,1))
        self.camera.SetFocalPoint(0,0,0)

        self.update_model_views()

        if self.cadmodel.initial_view is not None:
            for i in range(self.views_root_model.childCount()):
                if self.views_root_model.child(i).text(0) == self.cadmodel.initial_view:
                    self.views_root_model.child(i).setSelected(True)
                    break
        else:
            model_extent = self.cadmodel.get_extent()
            if np.abs(model_extent).max() > 0:
                self.camera.SetPosition(((model_extent[5] - model_extent[4])/2,(model_extent[2]+model_extent[3])/2,(model_extent[4]+model_extent[5])/2))
            else:
                self.camera.SetPosition((3.,0,0))

        self.statusbar.clearMessage()
        self.cadexplorer.update_clipping()

        self.refresh_vtk()

        self.app.restoreOverrideCursor()



    def mass_toggle_model(self):
        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        if self.sender() is self.enable_all_button:
            for i in range(self.treeitem_machine.childCount()):
               self.treeitem_machine.child(i).setCheckState(0,qt.Qt.Checked)

        elif self.sender() is self.disable_all_button:
            for i in range(self.treeitem_machine.childCount()):
                self.treeitem_machine.child(i).setCheckState(0,qt.Qt.Unchecked)
        self.app.restoreOverrideCursor()


    def update_viewport_info(self,keep_selection = False):

        campos = self.camera.GetPosition()
        camtar = self.camera.GetFocalPoint()
        fov = self.camera.GetViewAngle()

        self.camX.blockSignals(True)
        self.camY.blockSignals(True)
        self.camZ.blockSignals(True)
        self.tarX.blockSignals(True)
        self.tarY.blockSignals(True)
        self.tarZ.blockSignals(True)
        self.camFOV.blockSignals(True)

        self.camX.setValue(campos[0])
        self.camY.setValue(campos[1])
        self.camZ.setValue(campos[2])
        self.tarX.setValue(camtar[0])
        self.tarY.setValue(camtar[1])
        self.tarZ.setValue(camtar[2])
        self.camFOV.setValue(fov)

        self.camX.blockSignals(False)
        self.camY.blockSignals(False)
        self.camZ.blockSignals(False)
        self.tarX.blockSignals(False)
        self.tarY.blockSignals(False)
        self.tarZ.blockSignals(False)
        self.camFOV.blockSignals(False)

        if not keep_selection:
            self.viewlist.clearSelection()         



    def update_sightlines(self,data):

        if self.sender() is self.sightlines_load_button:

            cals = object_from_file(self,'Calibration',multiple=True)

            for cal in cals:
                # Add it to the sight lines list
                listitem = qt.QListWidgetItem(cal.name)
                
                listitem.setFlags(listitem.flags() | qt.Qt.ItemIsEditable | qt.Qt.ItemIsSelectable)
                listitem.setToolTip(cal.name)
                self.sightlines_list.addItem(listitem)
                if self.sightline_type_volume.isChecked():
                    actor_type = 'volume'
                elif self.sightline_type_lines.isChecked():
                    actor_type = 'lines'
                self.sightlines[listitem] = [cal,None,actor_type]
                listitem.setCheckState(qt.Qt.Checked)
                self.sightlines_list.setCurrentItem(listitem)


        elif self.sender() is self.sightlines_list:

            if data.checkState() == qt.Qt.Checked:
                
                if len(self.cadexplorer.sightline_actors) == 1:
                    self.sightlines_legend_checkbox.setChecked(True)

                if self.sightlines[data][1] is None:
                    self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
                    self.statusbar.showMessage('Ray casting camera sight lines...')
                    actor = render.get_fov_actor(self.cadmodel,self.sightlines[data][0],self.sightlines[data][2])
                    self.statusbar.clearMessage()
                    actor.GetProperty().SetColor(next(self.colourcycle))
                    actor.GetProperty().SetOpacity(100.**(self.sightline_opacity_slider.value()/100.)/100.)
                    self.sightlines[data][1] = actor
                    self.renderer.AddActor(actor)
                    self.app.restoreOverrideCursor()
                    self.cadexplorer.sightline_actors.append(actor)
                else:
                    self.renderer.AddActor(self.sightlines[data][1])
                    self.cadexplorer.sightline_actors.append(self.sightlines[data][1])

            else:
                self.renderer.RemoveActor(self.sightlines[data][1])
                self.cadexplorer.sightline_actors.remove(self.sightlines[data][1])
                if len(self.cadexplorer.sightline_actors) < 2:
                    self.sightlines_legend_checkbox.setChecked(False)

        elif self.sender() is self.sightline_opacity_slider:

            for item in self.sightlines_list.selectedItems():
                self.sightlines[item][1].GetProperty().SetOpacity( 100.**(data/100.)/100. )

        elif self.sender() is self.pick_sightlines_colour and len(self.sightlines_list.selectedItems()) > 0:

            picked_colour = pick_colour(self,self.sightlines[self.sightlines_list.selectedItems()[0]][1].GetProperty().GetColor())
            if picked_colour is not None:
                for item in self.sightlines_list.selectedItems():
                    self.sightlines[item][1].GetProperty().SetColor( picked_colour )

        elif self.sender() is self.sightline_type_volume:

            for item in self.sightlines_list.selectedItems():
                item.setCheckState(qt.Qt.Unchecked)
                self.colourcycle.queue_colour(self.sightlines[item][1].GetProperty().GetColor())
                self.sightlines[item][1] = None

                if self.sightline_type_volume.isChecked():
                    self.sightlines[item][2] = 'volume'
                elif self.sightline_type_lines.isChecked():
                    self.sightlines[item][2] = 'lines'

                item.setCheckState(qt.Qt.Checked)

            

        if self.sightlines_legend_checkbox.isChecked():
            
            legend_items = []
            for item in self.sightlines.keys():
                if item.checkState() == qt.Qt.Checked and self.sightlines[item][1] is not None:
                    legend_items.append( ( str(item.text()), self.sightlines[item][1].GetProperty().GetColor() ) )

            self.cadexplorer.set_legend(legend_items)  

        else:
            self.cadexplorer.set_legend([])

        self.qvtkWidget.update()


    def init_roi_list(self):

        self.roi_tree.clear()

        rois = paths.get_save_list('ROIs')

        roi_items = {}
        for roi in rois:
            if roi[1] is None:
                roi_items[roi[0]] = qt.QTreeWidgetItem([roi[0]])
                roi_items[roi[0]].setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                roi_items[roi[0]].setCheckState(0,qt.Qt.Unchecked)
                self.roi_tree.addTopLevelItem(roi_items[roi[0]])
            else:
                roi_items[roi[0]] = qt.QTreeWidgetItem(roi_items[roi[1]],[roi[0]])
                roi_items[roi[0]].setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                roi_items[roi[0]].setCheckState(0,qt.Qt.Unchecked)


    def update_rois(self,changed):

        self.roi_tree.blockSignals(True)

        # Propagate enable / disable down the tree
        if changed.childCount() > 0:
            for i in range(changed.childCount()):
                if changed.child(i).childCount() > 0:
                    for j in range(changed.child(i).childCount()):
                        roi = os.path.join(str(changed.text(0)),str(changed.child(i).text(0)),str(changed.child(i).child(j).text(0)))
                        if changed.checkState(0) == qt.Qt.Checked:
                            self.cadexplorer.add_roi(roi)
                            changed.child(i).child(j).setCheckState(0,qt.Qt.Checked)
                        else:
                            changed.child(i).child(j).setCheckState(0,qt.Qt.Unchecked)
                            self.cadexplorer.remove_roi(roi)
                else:
                    if changed.parent() is None:               
                        roi = os.path.join(str(changed.text(0)),str(changed.child(i).text(0)))
                    else:
                        roi = os.path.join(str(changed.parent().text(0)),str(changed.text(0)),str(changed.child(i).text(0)))

                if changed.checkState(0) == qt.Qt.Checked:
                    self.cadexplorer.add_roi(roi)
                    changed.child(i).setCheckState(0,qt.Qt.Checked)
                else:
                    changed.child(i).setCheckState(0,qt.Qt.Unchecked)
                    self.cadexplorer.remove_roi(roi)
        else:
            if changed.parent() is None:
                roi = str(changed.text(0))
            else:
                if changed.parent().parent() is None:               
                    roi = os.path.join(str(changed.parent().text(0)),str(changed.text(0)))
                else:
                    roi = os.path.join(str(changed.parent().parent().text(0)),str(changed.parent().text(0)),str(changed.text(0)))

            if changed.checkState(0) == qt.Qt.Checked:
                self.cadexplorer.add_roi(roi)
            else:
                self.cadexplorer.remove_roi(roi)


        # Propagate check marks up the tree
        if changed.parent() is not None:
            check_statuses = []
            for i in range(changed.parent().childCount()):
                if changed.parent().child(i).checkState(0) == qt.Qt.Checked:
                    check_statuses.append(1)
                elif changed.parent().child(i).checkState(0) == qt.Qt.Unchecked:
                    check_statuses.append(2)
                elif changed.parent().child(i).checkState(0) == qt.Qt.PartiallyChecked:
                    check_statuses.append(3)

            if len(list(set(check_statuses))) > 1:
                changed.parent().setCheckState(0,qt.Qt.PartiallyChecked)
            else:
                if check_statuses[0] == 1:
                    changed.parent().setCheckState(0,qt.Qt.Checked)
                elif check_statuses[0] == 2:
                    changed.parent().setCheckState(0,qt.Qt.Unchecked)
                else:
                    changed.parent().setCheckState(0,qt.Qt.PartiallyChecked)

            if changed.parent().parent() is not None:
                for i in range(changed.parent().parent().childCount()):
                    if changed.parent().parent().child(i).checkState(0) == qt.Qt.Checked:
                        check_statuses.append(1)
                    elif changed.parent().parent().child(i).checkState(0) == qt.Qt.Unchecked:
                        check_statuses.append(2)
                    elif changed.parent().parent().child(i).checkState(0) == qt.Qt.PartiallyChecked:
                        check_statuses.append(3)

                if len(list(set(check_statuses))) > 1:
                    changed.parent().parent().setCheckState(0,qt.Qt.PartiallyChecked)
                else:
                    if check_statuses[0] == 1:
                        changed.parent().parent().setCheckState(0,qt.Qt.Checked)
                    elif check_statuses[0] == 2:
                        changed.parent().parent().setCheckState(0,qt.Qt.Unchecked)
                    else:
                        changed.parent().parent().setCheckState(0,qt.Qt.PartiallyChecked)

        self.roi_tree.blockSignals(False)
        self.qvtkWidget.update()


    def update_cursor_position(self,position):
        info = 'Cursor location: ' + self.cadmodel.format_coord(position).replace('\n',' | ')
        self.statusbar.showMessage(info)

        self.xsection_checkbox.setEnabled(True)
        for i in range(self.views_root_auto.childCount()):
            self.views_root_auto.child(i).setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
        if self.xsection_checkbox.isChecked():
            self.cadexplorer.set_xsection(self.cadexplorer.point[0].GetFocalPoint())

    def refresh_vtk(self):
        self.renderer.Render()
        self.qvtkWidget.update()

    def update_cad_status(self,message):

        if message is not None:
            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.statusbar.showMessage(message)
            self.app.processEvents()
        else:
            self.app.restoreOverrideCursor()
            self.statusbar.clearMessage()
            self.app.processEvents()


    def closeEvent(self,event):

        if self.cadmodel is not None:
            self.cadmodel.remove_from_renderer(self.renderer)
            self.cadmodel.unload()

        # If we're exiting, put python'e exception handling back to normal.
        sys.excepthook = sys.__excepthook__




# Main calcam window class for actually creating calibrations.
class CalCamWindow(qt.QMainWindow):
 
    def __init__(self, app, parent = None):

        # GUI initialisation
        qt.QMainWindow.__init__(self, parent)
        qt.uic.loadUi(os.path.join(paths.ui,'calcam.ui'), self)

        self.setWindowIcon(qt.QIcon(os.path.join(paths.calcampath,'ui','icon.png')))

        self.app = app

        # See how big the screen is and open the window at an appropriate size
        desktopinfo = self.app.desktop()
        available_space = desktopinfo.availableGeometry(self)
        self.screensize = (available_space.width(),available_space.height())
        # Open the window with same aspect ratio as the screen, and no fewer than 500px tall.
        win_height = max(500,min(780,0.75*available_space.height()))
        win_width = win_height * available_space.width() / available_space.height() 
        self.resize(win_width,win_height)

        # Some messing with background colours to fix annoying QT behaviour
        self.scrollArea.setStyleSheet("QScrollArea {background-color:transparent;}");
        self.scrollArea.viewport().setStyleSheet(".QWidget {background-color:transparent;}");

        # Set up nice exception handling
        sys.excepthook = lambda *ex: show_exception_box(self,*ex)

        # Start up with no CAD model
        self.cadmodel = None

        # Set up VTK
        self.qvtkWidget = qt.QVTKRenderWindowInteractor(self.vtkframe)
        self.vtkframe.layout().addWidget(self.qvtkWidget)
        self.pointpicker = vtkinteractorstyles.PointPairPicker()
        self.qvtkWidget.SetInteractorStyle(self.pointpicker)
        self.renderer_cad = vtk.vtkRenderer()
        self.renderer_cad.SetBackground(0,0,0)
        self.renderer_cad.SetViewport(0,0,0.5,1)
        self.renderer_im = vtk.vtkRenderer()
        self.renderer_im.SetBackground(0,0,0)
        self.renderer_im.SetViewport(0.5,0,1,1)
        self.qvtkWidget.GetRenderWindow().AddRenderer(self.renderer_cad)
        self.qvtkWidget.GetRenderWindow().AddRenderer(self.renderer_im)
        self.vtkInteractor = self.qvtkWidget.GetRenderWindow().GetInteractor()
        self.camera = self.renderer_cad.GetActiveCamera()
        self.any_points = False
        self.pointpairs_changed = False
        self.fit_changed = False
        self.chessboard_pointpairs = None
        self.n_data = [0]


        # Populate CAD model list
        self.model_list = machine_geometry.get_available_models()

        self.model_name.addItems(sorted(self.model_list.keys()))
        self.model_name.setCurrentIndex(-1)
        self.load_model_button.setEnabled(0)


        # Disable image transform buttons if we have no image
        self.image_settings.hide()
        self.fit_results.hide()

        self.tabWidget.setTabEnabled(2,False)
        self.tabWidget.setTabEnabled(3,False)
        self.tabWidget.setTabEnabled(4,False)

        # Callbacks for GUI elements
        self.image_sources_list.currentIndexChanged.connect(self.build_imload_gui)
        self.enable_all_button.clicked.connect(self.mass_toggle_model)
        self.disable_all_button.clicked.connect(self.mass_toggle_model)
        self.viewlist.itemClicked.connect(self.change_cad_view)
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
        self.del_pp_button.clicked.connect(self.pointpicker.remove_current_pointpair)
        self.load_image_button.clicked.connect(self.load_image)
        self.im_flipud.clicked.connect(self.transform_image)
        self.im_fliplr.clicked.connect(self.transform_image)
        self.im_rotate_button.clicked.connect(self.transform_image)
        self.im_reset.clicked.connect(self.transform_image)
        self.im_y_stretch_button.clicked.connect(self.transform_image)
        self.load_pointpairs_button.clicked.connect(self.load_pointpairs)
        self.fit_button.clicked.connect(self.do_fit)
        self.fitted_points_checkbox.toggled.connect(self.toggle_reprojected)
        self.overlay_checkbox.toggled.connect(self.toggle_overlay)
        self.clear_points_button.clicked.connect(self.pointpicker.clear_all)
        self.save_fit_button.clicked.connect(self.save_fit)
        self.save_points_button.clicked.connect(self.save_points)
        self.hist_eq_checkbox.stateChanged.connect(self.toggle_hist_eq)
        self.im_define_splitFOV.clicked.connect(self.edit_split_field)
        self.pointpairs_load_name.currentIndexChanged.connect(self.update_load_pp_button_status)
        self.pixel_size_checkbox.toggled.connect(self.update_fitopts_gui)
        self.pixel_size_box.valueChanged.connect(self.update_pixel_size)
        self.toggle_controls_button.clicked.connect(self.toggle_controls)
        self.chessboard_button.clicked.connect(self.modify_chessboard_constraints)
        self.use_chessboard_checkbox.toggled.connect(self.toggle_chessboard_constraints)

        # If we have an old version of openCV, histo equilisation won't work :(
        if cv2_version < 2.4 or (cv2_version == 2.4 and cv2_micro_version < 6):
            self.hist_eq_checkbox.setEnabled(False)
            self.hist_eq_checkbox.setToolTip('Requires OpenCV 2.4.6 or newer; you have {:s}'.format(cv2.__version__))

        # Set up some keyboard shortcuts
        # It is done this way in 3 lines per shortcut to avoid segfaults on some configurations
        sc = qt.QShortcut(qt.QKeySequence("Del"),self)
        sc.setContext(qt.Qt.ApplicationShortcut)
        sc.activated.connect(self.pointpicker.remove_current_pointpair)

        sc = qt.QShortcut(qt.QKeySequence("Ctrl+F"),self)
        sc.setContext(qt.Qt.ApplicationShortcut)
        sc.activated.connect(self.do_fit)

        sc = qt.QShortcut(qt.QKeySequence("Ctrl+S"),self)
        sc.setContext(qt.Qt.ApplicationShortcut)
        sc.activated.connect(self.save_all)

        sc = qt.QShortcut(qt.QKeySequence("Ctrl+P"),self)
        sc.setContext(qt.Qt.ApplicationShortcut)
        sc.activated.connect(self.toggle_reprojected)

        sc = qt.QShortcut(qt.QKeySequence("Ctrl+O"),self)
        sc.setContext(qt.Qt.ApplicationShortcut)
        sc.activated.connect(self.toggle_overlay)

        # Odds & sods
        self.pixel_size_box.setSuffix(u' \u00B5m')
        self.save_fit_button.setEnabled(False)


        # Populate image sources list and tweak GUI layout for image loading.
        self.imload_inputs = []
        self.image_load_options.layout().setColumnMinimumWidth(0,100)
        for imsource in image.image_sources:
            self.image_sources_list.addItem(imsource.gui_display_name)
        self.image_sources_list.setCurrentIndex(0)
        

        # Start the GUI!
        self.show()
        self.pointpicker.DoInit(self.renderer_im,self.renderer_cad,self)
        self.vtkInteractor.Initialize()

        # Warn the user if we don't have any CAD models
        if self.model_list == {}:
            warn_no_models(self)


    def change_cad_view(self,view_item,init=False):


        if self.sender() is self.viewlist:
            if view_item.isDisabled() or view_item is self.views_root_results or view_item is self.views_root_synthetic or view_item is self.views_root_model:
                return

        if self.sender() is self.viewlist or init:


            if view_item.parent() in self.views_results:
                view = fitting.CalibResults(str(view_item.parent().text(0)))
                subfield = view.field_names.index(str(view_item.text(0)))

                self.camera.SetPosition(view.get_pupilpos(field=subfield))
                self.camera.SetFocalPoint(view.get_pupilpos(field=subfield) + view.get_los_direction(view.image_display_shape[0]/2,view.image_display_shape[1]/2))
                self.camera.SetViewAngle(view.get_fov(field=subfield)[1])
                self.camera.SetViewUp(-1.*view.get_cam_to_lab_rotation(field=subfield)[:,1])     
            elif view_item.parent() is self.views_root_model:
                self.cadmodel.set_default_view(str(view_item.text(0)))

                # Set to that view
                self.camera.SetViewAngle(self.cadmodel.cam_fov_default)
                self.camera.SetPosition(self.cadmodel.cam_pos_default)
                self.camera.SetFocalPoint(self.cadmodel.cam_target_default)
                self.camera.SetViewUp(0,0,1)

            elif view_item.parent() is self.views_root_results or self.views_root_synthetic:
                if view_item.parent() is self.views_root_results:
                    view = fitting.CalibResults(str(view_item.text(0)))
                else:
                    view = fitting.VirtualCalib(str(view_item.text(0)))

                if view.nfields > 1:
                    view_item.setExpanded(not view_item.isExpanded())
                    return

                self.camera.SetPosition(view.get_pupilpos())
                self.camera.SetFocalPoint(view.get_pupilpos() + view.get_los_direction(view.image_display_shape[0]/2,view.image_display_shape[1]/2))
                self.camera.SetViewAngle(view.get_fov()[1])
                self.camera.SetViewUp(-1.*view.get_cam_to_lab_rotation()[:,1])


        else:
            self.camera.SetPosition((self.camX.value(),self.camY.value(),self.camZ.value()))
            self.camera.SetFocalPoint((self.tarX.value(),self.tarY.value(),self.tarZ.value()))
            self.camera.SetViewAngle(self.camFOV.value())


        for i in range(len(self.pointpicker.ObjectPoints)):
            if self.pointpicker.ObjectPoints[i] is not None:
                self.pointpicker.Set3DCursorStyle(i,self.pointpicker.SelectedPoint == i,self.pointpicker.ImagePoints[i].count(None) < self.pointpicker.nFields)

        self.update_viewport_info(self.camera.GetPosition(),self.camera.GetFocalPoint(),self.camera.GetViewAngle())

        self.refresh_vtk()



    def populate_model_variants(self):

        model = self.model_list[str(self.model_name.currentText())]
        self.model_variant.clear()
        self.model_variant.addItems(model[1])
        self.model_variant.setCurrentIndex(model[2])
        self.load_model_button.setEnabled(1)

    def change_overlay_oversampling(self):

        if self.overlay_checkbox.isChecked():
            self.toggle_overlay(False)
            self.pointpicker.fit_overlay_actor = None
            self.toggle_overlay(True)
        else:
            self.pointpicker.fit_overlay_actor = None


    def update_checked_features(self,item):

            self.feature_tree.blockSignals(True)
            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            changed_feature = str(item.text(0))
            if changed_feature in self.group_items:
                feature = changed_feature
                changed_feature = []
                for i in range(self.group_items[feature].childCount()):
                    changed_feature.append(str(self.group_items[feature].child(i).text(0)))
                    self.group_items[feature].child(i).setCheckState(0,self.group_items[feature].checkState(0))
                    if self.group_items[feature].checkState(0) == qt.Qt.Checked:
                        self.group_items[feature].child(i).setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                    else:
                        self.group_items[feature].child(i).setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
            else:
                changed_feature = [changed_feature]
                feature = item.parent()
                if feature is not self.treeitem_machine:
                    checkstates = []
                    for i in range(feature.childCount()):
                        checkstates.append(feature.child(i).checkState(0))

                    if len(list(set(checkstates))) > 1:
                        feature.setCheckState(0,qt.Qt.PartiallyChecked)
                        feature.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                    else:
                        feature.setCheckState(0,checkstates[0])
                        if checkstates[0] == qt.Qt.Checked:
                            feature.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                        else:
                            feature.setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)

            if item.checkState(0) == qt.Qt.Checked:
                item.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                self.cadmodel.enable_features(changed_feature,self.renderer_cad)
            else:
                item.setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                self.cadmodel.disable_features(changed_feature,self.renderer_cad)

            self.app.restoreOverrideCursor()
            self.refresh_vtk()
            self.feature_tree.blockSignals(False)

            self.overlay_checkbox.setChecked(False)
            self.pointpicker.fit_overlay_actor = None



    def load_model(self):

        model = self.model_list[str(self.model_name.currentText())]

        # Dispose of the old model
        if self.cadmodel is not None:
            
            old_machine_name = self.cadmodel.machine_name
            old_enabled_features = self.cadmodel.get_enabled_features()

            for feature in self.model_actors.keys():
                actor = self.model_actors.pop(feature)
                self.renderer.RemoveActor(actor)
            
            del self.cadmodel
            self.tabWidget.setTabEnabled(2,True)
        else:
            old_machine_name = None

        # Create a new one
        exec('self.cadmodel = machine_geometry.' + model[0] + '("' + str(self.model_variant.currentText()) + '")')
        self.cadmodel.link_gui_window(self)

        if not self.cad_auto_load.isChecked():
            if self.cadmodel.machine_name == old_machine_name:
                self.cadmodel.enable_only(old_enabled_features)
            else:
                for feature in self.cadmodel.features:
                    self.cadmodel.disable_features(feature[0])


        for actor in self.cadmodel.get_vtkActors():
            self.renderer_cad.AddActor(actor)

        self.statusbar.showMessage('Setting up CAD model...')
        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))

        # Initialise the CAD model setup GUI
        init_model_settings(self)

        # Initialise other lists of things
        init_viewports_list(self)


        # Set selected CAD view to the model's default, if the machine has been changed (i.e. changing model variant will maintain the viewport)
        self.viewlist.clearSelection()
        if self.cadmodel.machine_name != old_machine_name:
            if old_machine_name is not None:
                self.pointpicker.clear_all()
            for i in range(self.views_root_model.childCount()):
                if str(self.views_root_model.child(i).text(0)) == self.cadmodel.default_view_name:
                    self.viewlist.setCurrentItem(self.views_root_model.child(i))
                    self.change_cad_view(self.views_root_model.child(i),init=True)

        if self.pointpicker.FitResults is not None:
            self.overlay_checkbox.setEnabled(True)
            for widgets in self.fit_results_widgets:
                widgets[3].setEnabled(True)

        self.pointpicker.PointPairs.machine_name = self.cadmodel.machine_name

        self.tabWidget.setTabEnabled(2,True)
        self.enable_all_button.setEnabled(1)
        self.disable_all_button.setEnabled(1)
        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()
        self.refresh_vtk()




    def mass_toggle_model(self):
        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        if self.sender() is self.enable_all_button:
            for i in range(self.treeitem_machine.childCount()):
               self.treeitem_machine.child(i).setCheckState(0,qt.Qt.Checked)
        elif self.sender() is self.disable_all_button:
            for i in range(self.treeitem_machine.childCount()):
                self.treeitem_machine.child(i).setCheckState(0,qt.Qt.Unchecked)
        self.app.restoreOverrideCursor()


    def update_viewport_info(self,campos,camtar,fov):

        self.camX.blockSignals(True)
        self.camY.blockSignals(True)
        self.camZ.blockSignals(True)
        self.tarX.blockSignals(True)
        self.tarY.blockSignals(True)
        self.tarZ.blockSignals(True)
        self.camFOV.blockSignals(True)

        self.camX.setValue(campos[0])
        self.camY.setValue(campos[1])
        self.camZ.setValue(campos[2])
        self.tarX.setValue(camtar[0])
        self.tarY.setValue(camtar[1])
        self.tarZ.setValue(camtar[2])
        self.camFOV.setValue(fov)

        self.camX.blockSignals(False)
        self.camY.blockSignals(False)
        self.camZ.blockSignals(False)
        self.tarX.blockSignals(False)
        self.tarY.blockSignals(False)
        self.tarZ.blockSignals(False)
        self.camFOV.blockSignals(False)



    def update_cursor_position(self,position):
        info = 'Cursor location: ' + self.cadmodel.get_position_info(position).replace('\n',' | ')
        self.statusbar.showMessage(info)

    def refresh_vtk(self,im_only=False):
        if not im_only:
            self.renderer_cad.Render()
        self.renderer_im.Render()
        self.qvtkWidget.update()


    def load_image(self,init_image=None):

        # By default we assume we don't know the pixel size
        self.pixel_size_checkbox.setChecked(False)
        self.save_fit_button.setEnabled(False)

        # Get a copy of the current image's field mask, if there is one
        if self.pointpicker.Image is not None:
            old_fieldmask = self.pointpicker.Image.fieldmask.copy()
            old_shape = self.pointpicker.Image.transform.get_display_shape()
        else:
            old_fieldmask = np.zeros(1)
            old_shape = None

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.statusbar.showMessage('Loading image...')

        # Gather up the required input arguments from the image load gui
        imload_options = []
        for option in self.imload_inputs:
            imload_options.append(option[1]())
            if qt.qt_ver == 4:
                if type(imload_options[-1]) == qt.QString:
                    imload_options[-1] = str(imload_options[-1])

        newim = self.imsource(*imload_options)

        self.image_settings.hide()
        self.tabWidget.setTabEnabled(3,False)
        self.tabWidget.setTabEnabled(4,False)

        existing_im_names = paths.get_save_list('Images')
        if newim.name in existing_im_names:
            testim = image.Image(newim.name)
            if not np.all(newim.data == testim.data):
                i = 0
                new_name = newim.name
                while new_name in existing_im_names:
                    i = i + 1
                    new_name = newim.name + '({:d})'.format(i)
                newim.name = new_name

        if old_shape is None:
            old_shape = newim.transform.get_display_shape()

        if newim.transform.get_display_shape()  != old_shape or (newim.fieldmask.max() != old_fieldmask.max()):
            self.pointpicker.clear_all()
            self.n_data = []
            for field in range(newim.n_fields):
                self.n_data.append(0)
            self.use_chessboard_checkbox.setChecked(False)
            self.chessboard_pointpairs = None
            self.chessboard_info.setText('No chessboard pattern images currently loaded.')
            keep_points = False
        else:
            self.use_chessboard_checkbox.setChecked(False)
            keep_points = True

        self.image = newim

        if self.image.n_fields > 1:
            self.use_chessboard_checkbox.setEnabled(False)
            self.chessboard_button.setEnabled(False)
            self.chessboard_info.setText('Cannot use chessboard images with split-field cameras.')
        else:
            self.use_chessboard_checkbox.setEnabled(True)
            self.chessboard_button.setEnabled(True)
            if self.chessboard_pointpairs is not None:
                self.chessboard_info.setText('{:d} chessboard pattern images loaded<br>Total additional points: {:d} '.format(len(self.chessboard_pointpairs),len(self.chessboard_pointpairs)*len(self.chessboard_pointpairs[0].objectpoints)))
            else:   
                self.chessboard_info.setText('No chessboard pattern images currently loaded.')



        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)

        self.pointpairs_save_name = self.image.name
        self.fit_save_name = self.image.name

        self.image_settings.show()
        self.hist_eq_checkbox.setCheckState(qt.Qt.Unchecked)
        self.pointpicker.init_image(self.image)

        self.rebuild_image_gui()
        self.populate_pointpairs_list()

        if keep_points:
            self.pointpicker.UpdateFromPPObject(False)

            if self.overlay_checkbox.isChecked():
                self.overlay_checkbox.setChecked(False)
                self.overlay_checkbox.setChecked(True)

            if self.fitted_points_checkbox.isChecked():
                self.fitted_points_checkbox.setChecked(False)
                self.fitted_points_checkbox.setChecked(True)

        self.update_image_info_string()
        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()


    def populate_pointpairs_list(self):
        pp_list = []
        for pp_save in paths.get_save_list('PointPairs'):

            try:
                pp = pointpairs.PointPairs(pp_save)
            except:
                continue
            if pp.n_fields == self.image.n_fields:
                if pp.image is None:
                    pp_list.append(pp_save)
                else:
                    if np.all(pp.image.transform.get_display_shape() == self.image.transform.get_display_shape()):
                        pp_list.append(pp_save)   

        self.pointpairs_load_name.clear()
        self.pointpairs_load_name.addItems(pp_list)
        self.pointpairs_load_name.setCurrentIndex(-1)





    def rebuild_image_gui(self):

        # Build the GUI to show fit options, according to the number of fields.
        self.fit_options_tabs.clear()

        # List of settings widgets (for showing / hiding when changing model)
        self.perspective_settings = []
        self.fit_settings_widgets = []
        self.fisheye_settings = []

        for field in range(self.image.n_fields):
            
            new_tab = qt.QWidget()
            new_layout = qt.QVBoxLayout()


            # Selection of model
            widgetlist = [qt.QRadioButton('Perspective Model'),qt.QRadioButton('Fisheye Model')]
        
            if int(fitting.cv2.__version__[0]) < 3:
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
            new_layout.addWidget(sub_widget)


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

            for widgetno in [-3,-2,-1]:
                widgetlist[widgetno].toggled.connect(self.fit_enable_check)

            widgetlist.append(qt.QCheckBox('Disable tangential distortion'))
            perspective_settings_layout.addWidget(widgetlist[-1])
            widgetlist[-1].clicked.connect(self.fit_enable_check)
            widgetlist.append(qt.QCheckBox('Fix Fx = Fy'))
            widgetlist[-1].clicked.connect(self.fit_enable_check)
            widgetlist[-1].setChecked(True)
            perspective_settings_layout.addWidget(widgetlist[-1])

            newWidgets = [qt.QCheckBox('Fix optical centre at: ('),qt.QDoubleSpinBox(),qt.QLabel(','), qt.QDoubleSpinBox(),qt.QLabel(')')]
            newWidgets[0].toggled.connect(self.update_fitopts_gui)
            widgetlist = widgetlist + [newWidgets[0],newWidgets[1],newWidgets[3]]
            newWidgets[1].setButtonSymbols(qt.QAbstractSpinBox.NoButtons)
            newWidgets[1].setDecimals(1)
            newWidgets[1].setMinimum(-self.image.transform.get_display_shape()[0]*10)
            newWidgets[1].setMaximum(self.image.transform.get_display_shape()[0]*10)
            newWidgets[1].setValue(self.image.transform.get_display_shape()[0]/2)
            newWidgets[1].setEnabled(False)
            newWidgets[3].setButtonSymbols(qt.QAbstractSpinBox.NoButtons)
            newWidgets[3].setMinimum(-self.image.transform.get_display_shape()[1]*10)
            newWidgets[3].setMaximum(self.image.transform.get_display_shape()[1]*10)
            newWidgets[3].setValue(self.image.transform.get_display_shape()[1]/2)
            newWidgets[3].setEnabled(False)
            newWidgets[3].setDecimals(1)

            sub_widget = qt.QWidget()
            sub_layout = qt.QHBoxLayout()
            sub_layout.setContentsMargins(0,0,0,0)
            sub_widget.setLayout(sub_layout)
            sub_layout.addWidget(newWidgets[0])
            sub_layout.addWidget(newWidgets[1])
            sub_layout.addWidget(newWidgets[2])
            sub_layout.addWidget(newWidgets[3])
            sub_layout.addWidget(newWidgets[4])
            perspective_settings_layout.addWidget(sub_widget)


            newWidgets = [qt.QLabel('Initial guess for focal length:'),qt.QSpinBox()]
            widgetlist = widgetlist + [newWidgets[1]]
            newWidgets[1].setMinimum(0)
            newWidgets[1].setMaximum(1e9)
            newWidgets[1].setButtonSymbols(qt.QAbstractSpinBox.NoButtons)
            newWidgets[1].setSuffix(' px')
            newWidgets[1].setValue(1000)   

            sub_widget = qt.QWidget()
            sub_layout = qt.QHBoxLayout()
            sub_layout.setContentsMargins(0,0,0,0)
            sub_widget.setLayout(sub_layout)
            sub_layout.addWidget(newWidgets[0])
            sub_layout.addWidget(newWidgets[1])
            perspective_settings_layout.addWidget(sub_widget)

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
            for widgetno in [-4,-3,-2,-1]:
                widgetlist[widgetno].toggled.connect(self.fit_enable_check)


            newWidgets = [qt.QLabel('Initial guess for focal length:'),qt.QSpinBox()]
            widgetlist = widgetlist + [newWidgets[1]]
            newWidgets[1].setMinimum(0)
            newWidgets[1].setMaximum(1e9)
            newWidgets[1].setButtonSymbols(qt.QAbstractSpinBox.NoButtons)
            newWidgets[1].setSuffix(' px')
            newWidgets[1].setValue(1000)

            sub_widget = qt.QWidget()
            sub_layout = qt.QHBoxLayout()
            sub_layout.setContentsMargins(0,0,0,0)
            sub_widget.setLayout(sub_layout)
            sub_layout.addWidget(newWidgets[0])
            sub_layout.addWidget(newWidgets[1])
            fisheye_settings_layout.addWidget(sub_widget)
            spacer = qt.QSpacerItem(20,10,qt.QSizePolicy.Minimum,qt.QSizePolicy.Expanding)
            fisheye_settings_layout.addItem(spacer)
            # ------- End of fisheye settings -----------------


            new_layout.addWidget(self.perspective_settings[-1])
            new_layout.addWidget(self.fisheye_settings[-1])
            widgetlist[0].setChecked(True)
            self.fisheye_settings[-1].hide()
            new_tab.setLayout(new_layout)
            self.fit_options_tabs.addTab(new_tab,self.image.field_names[field])

            self.fit_settings_widgets.append(widgetlist)


        # Build GUI to show the fit results, according to the number of fields.
        self.fit_results_widgets = []
        self.fit_results_tabs.clear()
        for field in range(self.image.n_fields):
            new_tab = qt.QWidget()
            new_layout = qt.QGridLayout()
            widgets = [ qt.QLabel('Fit RMS residual = ') , qt.QLabel('Parameter names'),  qt.QLabel('Parameter values'), qt.QPushButton('Set CAD view to match fit')]
            widgets[1].setAlignment(qt.Qt.AlignRight)
            widgets[3].clicked.connect(self.set_fit_viewport)
            new_layout.addWidget(widgets[0],0,0,1,-1)
            new_layout.addWidget(widgets[1],1,0)
            new_layout.addWidget(widgets[2],1,1)
            new_layout.addWidget(widgets[3],2,0,1,-1)
            self.fit_results_widgets.append(widgets)
            new_layout.setColumnMinimumWidth(0,90)
            new_tab.setLayout(new_layout)
            self.fit_results_tabs.addTab(new_tab,self.image.field_names[field])
        self.fit_results.hide()
        self.tabWidget.setTabEnabled(3,True)
        self.tabWidget.setTabEnabled(4,True)


        # Set pixel size, if the image knows its pixel size.
        if self.image.pixel_size is not None:
            self.pixel_size_box.setValue(self.image.pixel_size * 1.e6)
            self.pixel_size_checkbox.setChecked(True)


    def update_image_info_string(self):

        if np.any(self.image.transform.get_display_shape() != list(self.image.data.shape[1::-1])):
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP) [ As Displayed ]<br>{3:d} x {4:d} pixels ({5:.1f} MP) [ Raw Data ]<br>'.format(self.image.transform.get_display_shape()[0],self.image.transform.get_display_shape()[1],np.prod(self.image.transform.get_display_shape()) / 1e6 ,self.image.data.shape[1],self.image.data.shape[0],np.prod(self.image.data.shape[:2]) / 1e6 )
        else:
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP)<br>'.format(self.image.transform.get_display_shape()[0],self.image.transform.get_display_shape()[1],np.prod(self.image.transform.get_display_shape()) / 1e6 )
        
        if len(self.image.data.shape) == 2:
            info_str = info_str + 'Monochrome'
        elif len(self.image.data.shape) == 3 and self.image.data.shape[2] == 3:
            info_str = info_str + 'RGB Colour'
        elif len(self.image.data.shape) == 3 and self.image.data.shape[2] == 3:
            info_str = info_str + 'RGB Colour'

        self.image_info.setText(info_str)


    def browse_for_file(self):

        for i,option in enumerate(self.imload_inputs):
            if self.sender() in option[0]:
                filename_filter = self.imsource.gui_inputs[i]['filter']
                target_textbox = option[0][2]


        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(0)
        filedialog.setFileMode(1)
        filedialog.setWindowTitle('Select File')
        filedialog.setNameFilter(filename_filter)
        filedialog.setLabelText(3,'Select')
        filedialog.exec_()
        if filedialog.result() == 1:
            target_textbox.setText(str(filedialog.selectedFiles()[0]))


    def transform_image(self,data):

        # First, back up the point pair locations in original coordinates.
        x = []
        y = []
        # Loop over sub-fields
        for i in range(len(self.pointpicker.PointPairs.imagepoints)):
                    # Loop over points
                    for j in range(len(self.pointpicker.PointPairs.imagepoints[i])):
                        if self.pointpicker.PointPairs.imagepoints[i][j] is not None:
                                x.append(self.pointpicker.PointPairs.imagepoints[i][j][0])
                                y.append(self.pointpicker.PointPairs.imagepoints[i][j][1])

        x,y = self.image.transform.display_to_original_coords(x,y)

        if self.sender() is self.im_flipud:
            if len(self.image.transform.transform_actions) > 0:
                if self.image.transform.transform_actions[-1] == 'flip_up_down':
                    del self.image.transform.transform_actions[-1]
                else:
                    self.image.transform.transform_actions.append('flip_up_down')
            else:
                self.image.transform.transform_actions.append('flip_up_down')

        elif self.sender() is self.im_fliplr:
            if len(self.image.transform.transform_actions) > 0:
                if self.image.transform.transform_actions[-1] == 'flip_left_right':
                    del self.image.transform.transform_actions[-1]
                else:
                    self.image.transform.transform_actions.append('flip_left_right')
            else:
                self.image.transform.transform_actions.append('flip_left_right')

        elif self.sender() is self.im_rotate_button:
            if len(self.image.transform.transform_actions) > 0:
                if 'rotate_clockwise' in self.image.transform.transform_actions[-1]:
                    current_angle = int(self.image.transform.transform_actions[-1].split('_')[2])
                    del self.image.transform.transform_actions[-1]
                    new_angle = self.im_rotate_angle.value()
                    total_angle = current_angle + new_angle
                    if total_angle > 270:
                        total_angle = total_angle - 360

                    if new_angle > 0:
                        self.image.transform.transform_actions.append('rotate_clockwise_' + str(total_angle))
                else:
                    self.image.transform.transform_actions.append('rotate_clockwise_' + str(self.im_rotate_angle.value()))
            else:
                self.image.transform.transform_actions.append('rotate_clockwise_' + str(self.im_rotate_angle.value()))

        elif self.sender() is self.im_y_stretch_button:
            sideways = False
            for action in self.image.transform.transform_actions:
                if action.lower() in ['rotate_clockwise_90','rotate_clockwise_270']:
                    sideways = not sideways
            if sideways:
                self.image.transform.pixel_aspectratio = self.image.transform.pixel_aspectratio/self.im_y_stretch_factor.value()
            else:
                self.image.transform.pixel_aspectratio = self.image.transform.pixel_aspectratio*self.im_y_stretch_factor.value()

        elif self.sender() is self.im_reset:
            self.image.transform.transform_actions = []
            self.image.transform.pixel_aspectratio = 1

        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)
        self.pointpicker.fit_overlay_actor = None

        if self.fitted_points_checkbox.isChecked():
            self.fitted_points_checkbox.setChecked(False)

        # Transform all the point pairs in to the new coordinates
        x,y = self.image.transform.original_to_display_coords(x,y)
        ind = 0
        # Loop over sub-fields
        for i in range(len(self.pointpicker.PointPairs.imagepoints)):
                    # Loop over points
                    for j in range(len(self.pointpicker.PointPairs.imagepoints[i])):
                        if self.pointpicker.PointPairs.imagepoints[i][j] is not None:
                            self.pointpicker.PointPairs.imagepoints[i][j][0] = x[ind]
                            self.pointpicker.PointPairs.imagepoints[i][j][1] = y[ind]
                            ind = ind + 1

        # Update the image and point pairs
        self.pointpicker.init_image(self.image,hold_position=True)
        if self.use_chessboard_checkbox.isChecked():
            self.toggle_chessboard_constraints(True)

        self.pointpicker.UpdateFromPPObject(False) 
        self.update_image_info_string()
        self.populate_pointpairs_list()
        self.rebuild_image_gui()
        self.save_fit_button.setEnabled(False)

    def load_pointpairs(self):
        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.fitted_points_checkbox.setChecked(False)
        self.overlay_checkbox.setChecked(False)
        self.rebuild_image_gui()
        newPP = pointpairs.PointPairs(str(self.pointpairs_load_name.currentText()),image=self.pointpicker.Image)
        if self.cadmodel is not None:
            newPP.machine_name = self.cadmodel.machine_name
        newPP.image = self.pointpicker.Image
        self.pointpicker.PointPairs = newPP
        self.pointpicker.Image = newPP.image
        self.pointpicker.Fitter.set_PointPairs(self.pointpicker.PointPairs)
        self.pointpicker.UpdateFromPPObject(not self.pointpairs_clear_before_load.isChecked())
        self.pointpairs_changed = False
        self.fit_changed = False
        self.app.restoreOverrideCursor()


    def toggle_reprojected(self,show=None):

        if show is None:
            if self.fitted_points_checkbox.isEnabled():
                self.fitted_points_checkbox.setChecked(not self.fitted_points_checkbox.isChecked())

        elif show:
            self.overlay_checkbox.setChecked(False)
            self.pointpicker.ShowReprojectedPoints()
        else:
            self.pointpicker.HideReprojectedPoints()

    def fit_enable_check(self):


        # This avoids raising errors if this function is called when we have no
        # fit options GUI.
        if len(self.fit_settings_widgets) == 0:
            return

  
        enable = True

        # Check whether or not we have enough points to enable the fit button.
        for field in range(self.pointpicker.nFields):

            # If we're doing a perspective fit...
            if self.fit_settings_widgets[field][0].isChecked():

                free_params = 15
                free_params = free_params - self.fit_settings_widgets[field][2].isChecked()
                free_params = free_params - self.fit_settings_widgets[field][3].isChecked()
                free_params = free_params - self.fit_settings_widgets[field][4].isChecked()
                free_params = free_params - 2*self.fit_settings_widgets[field][5].isChecked()
                free_params = free_params - self.fit_settings_widgets[field][6].isChecked()
                free_params = free_params - 2*self.fit_settings_widgets[field][7].isChecked()

            # Or for a fisheye fit...
            elif self.fit_settings_widgets[field][1].isChecked():
                free_params = 14
                free_params = free_params - self.fit_settings_widgets[field][11].isChecked()
                free_params = free_params - self.fit_settings_widgets[field][12].isChecked()
                free_params = free_params - self.fit_settings_widgets[field][13].isChecked()
                free_params = free_params - self.fit_settings_widgets[field][14].isChecked()

            # And the award for most confusingly written if condition goes to...
            if not ( (self.n_data[field] > free_params and self.n_data[field] > 9) or (self.n_data[field] > 9 and self.use_chessboard_checkbox.isChecked()) ):
                enable = False


        self.fit_button.setEnabled(enable)
        if enable:
            self.fit_button.setToolTip('Do fit')
        else:
            self.fit_button.setToolTip('Not enough point pairs for a well constrained fit')


    def do_fit(self):

        # If this was called via a keyboard shortcut, we may be in no position to do a fit.
        if not self.fit_button.isEnabled():
            return

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.fitted_points_checkbox.setChecked(False)
        self.overlay_checkbox.setChecked(False)
        self.pointpicker.fit_overlay_actor = None

        # Put the fit parameters in to the fitter
        for field in range(self.pointpicker.nFields):
            # If we're doing a perspective fit...
            if self.fit_settings_widgets[field][0].isChecked():
                self.pointpicker.Fitter.set_model('perspective',field)
                self.pointpicker.Fitter.fixk1[field] = self.fit_settings_widgets[field][2].isChecked()
                self.pointpicker.Fitter.fixk2[field] = self.fit_settings_widgets[field][3].isChecked()
                self.pointpicker.Fitter.fixk3[field] = self.fit_settings_widgets[field][4].isChecked()
                self.pointpicker.Fitter.disabletangentialdist[field] = self.fit_settings_widgets[field][5].isChecked()
                self.pointpicker.Fitter.fixaspectratio[field] = self.fit_settings_widgets[field][6].isChecked()
                self.pointpicker.Fitter.fix_cc(self.fit_settings_widgets[field][7].isChecked(),field,self.fit_settings_widgets[field][8].value(),self.fit_settings_widgets[field][9].value())
            if self.fit_settings_widgets[field][1].isChecked():
                self.pointpicker.Fitter.set_model('fisheye',field)
                self.pointpicker.Fitter.fixk1[field] = self.fit_settings_widgets[field][11].isChecked()
                self.pointpicker.Fitter.fixk2[field] = self.fit_settings_widgets[field][12].isChecked()
                self.pointpicker.Fitter.fixk3[field] = self.fit_settings_widgets[field][13].isChecked()
                self.pointpicker.Fitter.fixk4[field] = self.fit_settings_widgets[field][14].isChecked()


        # Do the fit!
        self.statusbar.showMessage('Performing calibration fit...')
        self.pointpicker.FitResults = self.pointpicker.Fitter.do_fit()
        self.statusbar.clearMessage()
        # Put the results in to the GUI
        for field,params in enumerate(self.pointpicker.FitResults.fit_params):

            # Get CoM of this field on the chip
            ypx,xpx = CoM( (self.pointpicker.FitResults.fieldmask + 1) * (self.pointpicker.FitResults.fieldmask == field) )

            # Line of sight at the field centre
            los_centre = self.pointpicker.FitResults.get_los_direction(xpx,ypx)
            fov = self.pointpicker.FitResults.get_fov(field)

            pupilpos = self.pointpicker.FitResults.get_pupilpos(field=field)

            widgets = self.fit_results_widgets[field]
            if params.model == 'perspective':
                widgets[0].setText( '<b>RMS Fit Residual: {: .1f} pixels<b>'.format(params.rms_error) )
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
                if self.image.pixel_size is not None:
                    fx = params.cam_matrix[0,0] * self.image.pixel_size*1e3
                    fy = params.cam_matrix[1,1] * self.image.pixel_size*1e3
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
                                                   "{: 5.4f}".format(params.p1).replace(' ','&nbsp;') ,
                                                   "{: 5.4f}".format(params.p2).replace(' ','&nbsp;') ,
                                                   ''
                                                   ] ) )
            elif params.model == 'fisheye':
                widgets[0].setText( '<b>RMS Fit Residual: {: .1f} pixels<b>'.format(params.rms_error) )
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
                if self.image.pixel_size is not None:
                    fx = params.cam_matrix[0,0] * self.image.pixel_size*1e3
                    fy = params.cam_matrix[1,1] * self.image.pixel_size*1e3
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

        self.fit_results.show()
        self.fitted_points_checkbox.setEnabled(True)
        self.fitted_points_checkbox.setChecked(True)
        self.save_fit_button.setEnabled(True)
        self.fit_changed = True
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

            if self.pointpicker.fit_overlay_actor is None:

                oversampling = 1.
                self.statusbar.showMessage('Rendering wireframe overlay...')
                self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
                self.app.processEvents()
                try:
                    OverlayImage = image.from_array(render.render_cam_view(self.cadmodel,self.pointpicker.FitResults,Edges=True,Transparency=True,Verbose=False,EdgeColour=(0,0,1),oversampling=oversampling,ScreenSize=self.screensize))

                    self.pointpicker.fit_overlay_actor = OverlayImage.get_vtkActor()

                    self.pointpicker.Renderer_2D.AddActor(self.pointpicker.fit_overlay_actor)
                    self.pointpicker.fit_overlay_actor.SetPosition(0.,0.,0.05)

                    self.fitted_points_checkbox.setChecked(False)
                    self.refresh_vtk()


                    if np.max(OverlayImage.data) == 0:
                        dialog = qt.QMessageBox(self)
                        dialog.setStandardButtons(qt.QMessageBox.Ok)
                        dialog.setWindowTitle('Calcam - Information')
                        dialog.setTextFormat(qt.Qt.RichText)
                        dialog.setText('Wireframe overlay image is blank.')
                        dialog.setInformativeText('This usually means the fit is wildly wrong.')
                        dialog.setIcon(qt.QMessageBox.Information)
                        dialog.exec_()
                        

                except MemoryError:
                    self.pointpicker.fit_overlay_actor = None
                    dialog = qt.QMessageBox(self)
                    dialog.setStandardButtons(qt.QMessageBox.Ok)
                    dialog.setWindowTitle('Calcam - Memory Error')
                    dialog.setTextFormat(qt.Qt.RichText)
                    dialog.setText('Insufficient memory to render wireframe overlay.')
                    text = 'Try using a lower resolution setting for the overlay.'
                    if sys.maxsize < 2**32:
                        text = text + ' Switching to 64-bit python is highly recommended when working with large data.'
                    dialog.setInformativeText(text)
                    dialog.setIcon(qt.QMessageBox.Warning)
                    dialog.exec_()
                    self.overlay_checkbox.setChecked(False) 
                
                except:
                    self.pointpicker.fit_overlay_actor = None
                    self.statusbar.clearMessage()
                    self.overlay_checkbox.setChecked(False) 
                    self.app.restoreOverrideCursor()
                    raise


                self.statusbar.clearMessage()
                self.app.restoreOverrideCursor()

            else:

                self.pointpicker.Renderer_2D.AddActor(self.pointpicker.fit_overlay_actor)
                self.fitted_points_checkbox.setChecked(False)
                self.refresh_vtk()

        else:
            
            self.pointpicker.Renderer_2D.RemoveActor(self.pointpicker.fit_overlay_actor)
            self.refresh_vtk()   


    def update_n_points(self,n_pairs,n_unpaired,n_list):

        self.n_data = np.array(n_list) * 3
        n_pairs_string = str(n_pairs) + ' Point Pairs'
        if n_unpaired > 0:
            n_pairs_string = n_pairs_string + ' + ' + str(n_unpaired) + ' unpaired points'

        self.n_pointpairs_text.setText(n_pairs_string)

        if n_pairs > 0:
            self.save_points_button.setEnabled(1)
            self.clear_points_button.setEnabled(1)
        else:
            self.save_points_button.setEnabled(0)
            self.clear_points_button.setEnabled(0)

        if n_pairs > 0 or n_unpaired > 0:
            self.any_points = True
        else:
            self.any_points = False

        self.fit_enable_check()


    def update_current_points(self,object_coords,image_coords):

        info_string = ''

        if object_coords is not None and self.cadmodel is not None:
            info_string = info_string + '<span style=" text-decoration: underline;">CAD Point<br></span>' + self.cadmodel.get_position_info(object_coords).replace('\n','<br>') + '<br><br>'

        if image_coords is not None:
            info_string = info_string + '<span style=" text-decoration: underline;">Image Point(s)</span><br>'


            for i,point in enumerate(image_coords):
                info_string = info_string + '( {:.0f} , {:.0f} ) px'.format(point[0],point[1]).replace(' ','&nbsp;')
                if len(image_coords) > 1:
                    info_string = info_string + '  [' + self.image.field_names[i] + ']'.replace(' ','&nbsp;')
                if image_coords.index(point) < len(image_coords) - 1:
                    info_string = info_string + '<br>'
        if info_string == '':
            info_string = 'None'
            self.del_pp_button.hide()
        else:
            self.del_pp_button.show()

        self.point_info_text.setText(info_string)


    def update_fit_opts(self,state):

        if self.sender() is self.fix_k1:
            if state == qt.Qt.Checked:
                self.pointpicker.Fitter.fixk1 = True
            else:
                self.pointpicker.Fitter.fixk1 = False

        if self.sender() is self.fix_k2:
            if state == qt.Qt.Checked:
                self.pointpicker.Fitter.fixk2 = True
            else:
                self.pointpicker.Fitter.fixk2 = False

        if self.sender() is self.fix_k3:
            if state == qt.Qt.Checked:
                self.pointpicker.Fitter.fixk3 = True
            else:
                self.pointpicker.Fitter.fixk3 = False

        if self.sender() is self.disable_tangential_dist:
            if state == qt.Qt.Checked:
                self.pointpicker.Fitter.disabletangentialdist = True
            else:
                self.pointpicker.Fitter.disabletangentialdist = False

        if self.sender() is self.fix_aspect:
             if state == qt.Qt.Checked:
                self.pointpicker.Fitter.fixaspectratio = True
             else:
                self.pointpicker.Fitter.fixaspectratio = False

    def save_all(self):
        if self.pointpairs_changed and self.save_points_button.isEnabled():
            if self.fit_changed:
                if self.save_points(confirm=False):
                    dialog = qt.QMessageBox(self)
                    dialog.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
                    dialog.setWindowTitle('Calcam - Save Fit?')
                    dialog.setText('Point pairs saved successfully. Also save fit results now?')
                    dialog.setIcon(qt.QMessageBox.Information)
                    dialog.exec_()
                    if dialog.result() == 16384:
                        self.save_fit()   
            else:
                self.save_points()        
       
        elif self.save_fit_button.isEnabled():
            self.save_fit()


    def save_points(self,event=None,confirm=True):
        try:
            dialog = SaveAsDialog(self,'Point Pairs',self.pointpairs_save_name)
            dialog.exec_()
            if dialog.result() == 1:
                self.pointpairs_save_name = dialog.name
                del dialog
                self.image.save()
                self.pointpicker.PointPairs.save(self.pointpairs_save_name)
                self.pointpairs_changed = False
                if confirm:
                    dialog = qt.QMessageBox(self)
                    dialog.setStandardButtons(qt.QMessageBox.Ok)
                    dialog.setWindowTitle('Calcam - Save Complete')
                    dialog.setText('Point pairs saved successfully')
                    dialog.setIcon(qt.QMessageBox.Information)
                    dialog.exec_()
                return 1
            else:
                return 0

        except Exception as err:
            raise
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Ok)
            dialog.setWindowTitle('Calcam - Save Error')
            dialog.setText('Error saving point pairs:\n' + str(err))
            dialog.setIcon(qt.QMessageBox.Warning)
            dialog.exec_()
            return 0


    def save_fit(self):
        if self.pointpairs_changed:
            self.save_points()
        try:
            dialog = SaveAsDialog(self,'Fit Results',self.fit_save_name)
            dialog.exec_()
            if dialog.result() == 1:
                self.fit_save_name = dialog.name
                del dialog
                self.image.save()          
                self.pointpicker.FitResults.save(self.fit_save_name)
                self.fit_changed = False
                dialog = qt.QMessageBox(self)
                dialog.setStandardButtons(qt.QMessageBox.Ok)
                dialog.setWindowTitle('Calcam - Save Complete')
                dialog.setText('Fit results saved successfully.')
                dialog.setIcon(qt.QMessageBox.Information)
                dialog.exec_()
        except Exception as err:
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Ok)
            dialog.setWindowTitle('Calcam - Save Error')
            dialog.setText('Error saving fit results:\n' + str(err))
            dialog.setIcon(qt.QMessageBox.Warning)
            dialog.exec_()


    def update_cad_status(self,message):

        if message is not None:
            self.statusbar.showMessage(message)
            self.app.processEvents()
        else:
            self.statusbar.clearMessage()
            self.app.processEvents()


    def toggle_hist_eq(self,check_state):

        # Enable / disable adaptive histogram equalisation
        if check_state == qt.Qt.Checked:
            self.image.postprocessor = image_filters.hist_eq()
        else:
            self.image.postprocessor = None

        self.pointpicker.init_image(self.image,hold_position=True)
        if self.use_chessboard_checkbox.isChecked():
            self.toggle_chessboard_constraints(True)

        self.pointpicker.UpdateFromPPObject(False)

        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)
            self.overlay_checkbox.setChecked(True)

        if self.fitted_points_checkbox.isChecked():
            self.fitted_points_checkbox.setChecked(False)
            self.fitted_points_checkbox.setChecked(True)

        

    def closeEvent(self,event):

        changed_string = []
        if self.pointpairs_changed:
            changed_string.append('Point Pair Changes')
        if self.fit_changed:
            changed_string.append('Fit Results')

        if len(changed_string) > 0:
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Save|qt.QMessageBox.Discard|qt.QMessageBox.Cancel)
            dialog.setWindowTitle('Save changes?')
            dialog.setText('There are unsaved {:s}. Save these before exiting?'.format(' and '.join(changed_string)))
            dialog.setIcon(qt.QMessageBox.Information)
            choice = dialog.exec_()
            if choice == qt.QMessageBox.Save:
                self.save_all()
            elif choice == qt.QMessageBox.Cancel:
                event.ignore()
                return

        # If we're exiting, put python'e exception handling back to normal.
        sys.excepthook = sys.__excepthook__

        # Call the parent class' close event
        super(CalCamWindow,self).closeEvent(event)


    def edit_split_field(self):
        dialog = SplitFieldDialog(self,self.pointpicker.Image)
        result = dialog.exec_()
        if result == 1:
            self.pointpicker.clear_all()
            self.n_data = []
            for field in range(self.pointpicker.nFields):
                self.n_data.append(0)

            if dialog.fieldmask.max() > 0:
                self.use_chessboard_checkbox.setChecked(False)
                self.use_chessboard_checkbox.setEnabled(False)
                self.chessboard_button.setEnabled(False)
                self.chessboard_pointpairs = None
                self.chessboard_info.setText('Cannot use chessboard images with split-field cameras.')
            else:
                self.use_chessboard_checkbox.setEnabled(True)
                self.chessboard_button.setEnabled(True)
                if self.chessboard_pointpairs is not None:
                    self.chessboard_info.setText('{:d} chessboard pattern images loaded<br>Total additional points: {:d} '.format(len(self.chessboard_pointpairs),len(self.chessboard_pointpairs)*len(self.chessboard_pointpairs[0].objectpoints)))
                else:   
                    self.chessboard_info.setText('No chessboard pattern images currently loaded.')
            
            self.image.fieldmask = dialog.fieldmask.copy()
            self.image.n_fields = dialog.fieldmask.max() + 1
            self.image.field_names = dialog.field_names

            self.pointpicker.init_image(self.image)
            self.rebuild_image_gui()

        del dialog


    def build_imload_gui(self,index):

        layout = self.image_load_options.layout()
        for widgets,_ in self.imload_inputs:
            for widget in widgets:
                layout.removeWidget(widget)
                widget.close()

        #layout = qt.QGridLayout(self.image_load_options)
        self.imsource = image.image_sources[index]

        self.imload_inputs = []

        row = 0
        for option in self.imsource.gui_inputs:

            labelwidget = qt.QLabel(option['label'] + ':')
            layout.addWidget(labelwidget,row,0)

            if option['type'] == 'filename':
                button = qt.QPushButton('Browse...')
                button.clicked.connect(self.browse_for_file)
                button.setMaximumWidth(80)
                layout.addWidget(button,row+1,1)
                fname = qt.QLineEdit()
                if 'default' in option:
                    fname.setText(option['default'])
                layout.addWidget(fname,row,1)
                self.imload_inputs.append( ([labelwidget,button,fname],fname.text ) )
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
                self.imload_inputs.append( ([labelwidget,valbox],valbox.value ) )
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
                self.imload_inputs.append( ([labelwidget,valbox],valbox.value ) )
                row = row + 1
            elif option['type'] == 'string':
                ted = qt.QLineEdit()
                if 'default' in option:
                    ted.setText(option['default'])
                layout.addWidget(ted,row,1)
                self.imload_inputs.append( ([labelwidget,ted],ted.text ) )
                row = row + 1
            elif option['type'] == 'bool':
                checkbox = qt.QCheckBox()
                if 'default' in option:
                    checkbox.setChecked(option['default'])
                layout.addWidget(checkbox,row,1)
                self.imload_inputs.append( ([labelwidget,checkbox],checkbox.isChecked ) )
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
                self.imload_inputs.append( ([labelwidget,cb],cb.currentText) )
                row = row + 1

    def update_load_pp_button_status(self,index):
        if index > -1:
            self.load_pointpairs_button.setEnabled(True)
        else:
            self.load_pointpairs_button.setEnabled(False)


    def update_fitopts_gui(self,choice):

        if self.sender() == self.pixel_size_checkbox:
            if choice:
                self.pixel_size_box.setEnabled(True)
                self.update_pixel_size()
                for field in range(self.image.n_fields):
                    self.fit_settings_widgets[field][10].setSuffix(' mm')
                    self.fit_settings_widgets[field][10].setValue(self.fit_settings_widgets[field][10].value() * self.image.pixel_size*1e3)
                
            else:
                self.pixel_size_box.setEnabled(False)
                for field in range(self.image.n_fields):
                    self.fit_settings_widgets[field][10].setSuffix(' px')
                    self.fit_settings_widgets[field][10].setValue(self.fit_settings_widgets[field][10].value() / (self.image.pixel_size*1e3))
                self.update_pixel_size()


        
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
            self.image.pixel_size = self.pixel_size_box.value() / 1e6
        else:
            self.image.pixel_size = None


    def set_fit_viewport(self):
        for field in range(self.pointpicker.nFields):
            if self.sender() in self.fit_results_widgets[field]:
                self.pointpicker.set_view_to_fit(field)


    def toggle_controls(self):
        if self.tabWidget.isHidden():
            self.tabWidget.show()
            self.toggle_controls_button.setText('>> Hide Controls')
        else:
            self.tabWidget.hide()
            self.toggle_controls_button.setText('<< Show Controls')


    def modify_chessboard_constraints(self):

        dialog = ChessboardDialog(self)
        dialog.exec_()

        if dialog.pointpairs_result is not None:
            self.use_chessboard_checkbox.setChecked(False)
            self.chessboard_pointpairs = copy.deepcopy(dialog.pointpairs_result)
            self.use_chessboard_checkbox.setEnabled(True)
            self.use_chessboard_checkbox.setChecked(True)
            self.chessboard_info.setText('{:d} chessboard pattern images loaded<br>Total additional points: {:d} '.format(len(self.chessboard_pointpairs),len(self.chessboard_pointpairs)*len(self.chessboard_pointpairs[0].objectpoints)))

        del dialog


    def toggle_chessboard_constraints(self,on):
        
        if on:
            self.pointpicker.Fitter.add_intrinsics_pointpairs(self.chessboard_pointpairs)
        else:
            self.pointpicker.Fitter.clear_intrinsics_pointpairs()

        self.fit_enable_check()


# Convenience function for starting the CAD Viewer.
# This is the one the user should call.
def start_cad_viewer():

    app = qt.QApplication(sys.argv)
    CADViewerWindow(app)
    return app.exec_()

# Convenience function for starting  CalCam to do a calibration.
# This is the one the user should call.
def start_calcam():

    app = qt.QApplication(sys.argv)
    CalCamWindow(app)
    return app.exec_()

# Convenience function for starting  CalCam to do a calibration.
# This is the one the user should call.
def start_view_designer():

    app = qt.QApplication(sys.argv)
    ViewDesignerWindow(app)
    return app.exec_()


# Convenience function for starting  CalCam to do a calibration.
# This is the one the user should call.
def start_alignment_calib():

    app = qt.QApplication(sys.argv)
    AlignmentCalibWindow(app)
    return app.exec_()


# Convenience function for starting  CalCam to do image analysis.
# This is the one the user should call.
def start_image_analysis():

    app = qt.QApplication(sys.argv)
    ImageAnalyserWindow(app)
    return app.exec_()


class SplitFieldDialog(qt.QDialog):

    def __init__(self, parent, image):

        # GUI initialisation
        qt.QDialog.__init__(self, parent)
        qt.uic.loadUi(os.path.join(paths.ui,'split_field_dialog.ui'), self)

        self.image = image
        self.field_names_boxes = []
        self.parent = parent
        
        # Set up VTK
        self.qvtkWidget = qt.QVTKRenderWindowInteractor(self.vtkframe)
        self.vtkframe.layout().addWidget(self.qvtkWidget)
        self.splitfieldeditor = vtkinteractorstyles.SplitFieldEditor()
        self.qvtkWidget.SetInteractorStyle(self.splitfieldeditor)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.qvtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkInteractor = self.qvtkWidget.GetRenderWindow().GetInteractor()
        self.camera = self.renderer.GetActiveCamera()

        if self.image.n_fields == 1:
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
        self.splitfieldeditor.click_enabled = False


        # Start the GUI!
        self.show()
        self.splitfieldeditor.DoInit(self.renderer,self.image,self,self.mask_alpha_slider.value()/100.)
        self.renderer.Render()
        self.vtkInteractor.Initialize()
        self.splitfieldeditor.Renderer.AddActor2D(self.splitfieldeditor.ImageActor)
        self.splitfieldeditor.update_fieldmask(self.splitfieldeditor.Image.fieldmask,self.splitfieldeditor.Image.field_names)


    def change_method(self,obj=None,event=None):
        if self.method_points.isChecked():
            self.points_options.show()
            self.splitfieldeditor.click_enabled = True
            self.mask_options.hide()
            self.mask_alpha_slider.setEnabled(True)
            self.mask_alpha_label.setEnabled(True)
        elif self.method_mask.isChecked():
            self.mask_options.show()
            self.points_options.hide()
            while self.splitfieldeditor.SelectedPoint is not None:
                self.splitfieldeditor.DeletePoint(self.splitfieldeditor.SelectedPoint)

            self.splitfieldeditor.click_enabled = False
            self.mask_alpha_slider.setEnabled(True)
            self.mask_alpha_label.setEnabled(True)
        else:
            self.mask_options.hide()
            self.points_options.hide()
            while self.splitfieldeditor.SelectedPoint is not None:
                self.splitfieldeditor.DeletePoint(self.splitfieldeditor.SelectedPoint)
            self.splitfieldeditor.update_fieldmask(np.zeros(self.image.data.shape,dtype=np.uint8))
            self.mask_alpha_slider.setEnabled(False)
            self.mask_alpha_label.setEnabled(False)
        self.qvtkWidget.update()


    def update_mask_alpha(self,value):
        self.mask_alpha_label.setText('Mask Opacity: {:d}%'.format(value))
        self.splitfieldeditor.set_mask_opacity(value/100.)
        self.qvtkWidget.update()


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

 
        self.fieldmask = self.splitfieldeditor.fieldmask
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
            try:
                self.splitfieldeditor.update_fieldmask(mask=mask_im)
            except ValueError as e:
                if 'wrong shape' in str(e):
                    dialog = qt.QMessageBox(self)
                    dialog.setStandardButtons(qt.QMessageBox.Ok)
                    dialog.setTextFormat(qt.Qt.RichText)
                    dialog.setWindowTitle('Calcam - wrong mask shape')
                    dialog.setText("Selected mask image is the wrong shape ({:d}x{:d}) for this camera image so cannot be used.".format(mask_im.shape[1],mask_im.shape[0]))
                    original_shape = np.array([self.image.data.shape[1],self.image.data.shape[0]])
                    display_shape = np.array(self.image.transform.get_display_shape())
                    if np.any(original_shape != display_shape):
                        dialog.setInformativeText("This image requires a field mask of either {:d}x{:d} or {:d}x{:d} pixels.".format(original_shape[0],original_shape[1],display_shape[0],display_shape[1]))
                    else:
                        dialog.setInformativeText("A field mask of {:d}x{:d} pixels is required.".format(original_shape[0],original_shape[1]))

                    dialog.setIcon(qt.QMessageBox.Warning)
                    dialog.exec_()



class ChessboardDialog(qt.QDialog):

    def __init__(self, parent,modelselection=False):

        # GUI initialisation
        qt.QDialog.__init__(self, parent)
        qt.uic.loadUi(os.path.join(paths.ui,'chessboard_image_dialog.ui'), self)

        self.parent = parent
        try:
            self.image_transformer = self.parent.image.transform
        except:
            self.image_transformer = CoordTransformer()

        if not modelselection:
            self.model_options.hide()


        # Callbacks for GUI elements
        self.load_images_button.clicked.connect(self.load_images)
        self.detect_chessboard_button.clicked.connect(self.detect_corners)
        self.apply_button.clicked.connect(self.apply)
        self.cancel_button.clicked.connect(self.reject)
        self.next_im_button.clicked.connect(self.change_image)
        self.prev_im_button.clicked.connect(self.change_image)
        self.current_image = None

        if int(fitting.cv2.__version__[0]) < 3:
            self.fisheye_model.setEnabled(False)
            self.fisheye_model.setToolTip('Requires OpenCV 3')

        # Sort out pyplot setup for showing the images
        im_figure = plt.figure()
        self.mplwidget = FigureCanvas(im_figure)
        self.mplwidget.hide()
        self.im_control_bar.hide()
        self.imax = im_figure.add_subplot(111)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.imax.get_xaxis().set_visible(False)
        self.imax.get_yaxis().set_visible(False)
        self.image_frame.layout().addWidget(self.mplwidget,1)

        self.detection_run = False

        self.images = []
        self.filenames = []

        self.pointpairs_result = None

        # Start the GUI!
        self.show()



    def load_images(self):

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(0)
        filedialog.setFileMode(3)
        filedialog.setWindowTitle('Load chessboard images')
        filedialog.setNameFilter('Image Files (*.jpg *.jpeg *.png *.bmp *.jp2 *.tiff *.tif)')
        filedialog.setLabelText(3,'Load')
        filedialog.exec_()
        if filedialog.result() == 1:
            self.images = []
            self.filenames = []
            wrong_shape = []
            if self.image_transformer.x_pixels is None or self.image_transformer.y_pixels is None:
                expected_shape = None
            else:
                expected_shape = np.array([self.image_transformer.x_pixels,self.image_transformer.y_pixels])

            for n,fname in enumerate(filedialog.selectedFiles()):
                self.status_text.setText('<b>Loading image {:d} / {:d} ...'.format(n,len(filedialog.selectedFiles())))
                im = cv2.imread(str(fname))
                if expected_shape is None:
                    expected_shape = im.shape[1::-1]
                    self.im_shape = im.shape[:2]
                    wrong_shape.append(False)
                else:
                    wrong_shape.append(not np.all(expected_shape == im.shape[1::-1]))
                
                # OpenCV loads colour channels in BGR order.
                if len(im.shape) == 3:
                    im[:,:,:3] = im[:,:,3::-1]
                self.images.append(im)
                self.filenames.append(os.path.split(str(fname))[1])
            
            self.status_text.setText('')
            if np.all(wrong_shape):
                dialog = qt.QMessageBox(self)
                dialog.setStandardButtons(qt.QMessageBox.Ok)
                dialog.setTextFormat(qt.Qt.RichText)
                dialog.setWindowTitle('Calcam - Wrong image size')
                dialog.setText("Selected chessboard pattern images are the wrong size for this camera image and have not been loaded.")
                dialog.setInformativeText("Chessboard images of {:d} x {:d} pixels are required for this camera image.".format(expected_shape[0],expected_shape[1]))
                dialog.setIcon(qt.QMessageBox.Warning)
                dialog.exec_()
                self.images = []
                self.filenames = []
                return

            elif np.any(wrong_shape):
                dialog = qt.QMessageBox(self)
                dialog.setStandardButtons(qt.QMessageBox.Ok)
                dialog.setTextFormat(qt.Qt.RichText)
                dialog.setWindowTitle('Calcam - Wrong image size')
                dialog.setText("{:d} of the selected images are the wrong size for this camera image and were not loaded:".format(np.count_nonzero(wrong_shape)))
                dialog.setInformativeText('<br>'.join([ self.filenames[i] for i in range(len(self.filenames)) if wrong_shape[i] ]))
                dialog.setIcon(qt.QMessageBox.Warning)
                dialog.exec_()
                for i in range(len(self.images)-1,-1,-1):

                    if wrong_shape[i]:
                        del self.images[i]
                        del self.filenames[i]

            self.detection_run = False
            self.chessboard_status = [False for i in range(len(self.images))]
            self.current_image = 0
            self.update_image_display()
            self.mplwidget.show()
            self.im_control_bar.show()
            self.mplwidget.draw()
            self.detect_chessboard_button.setEnabled(True)


    def change_image(self):
        if self.sender() is self.next_im_button:
            self.current_image = (self.current_image + 1) % len(self.images)
        elif self.sender() is self.prev_im_button:
            self.current_image = (self.current_image - 1) % len(self.images)
        self.update_image_display()



    def detect_corners(self):

        self.parent.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.chessboard_status = []
        self.chessboard_points_2D = [np.zeros([ (self.chessboard_squares_x.value() - 1)*(self.chessboard_squares_y.value() - 1),2]) for i in range(len(self.images))]
        self.n_chessboard_points = (self.chessboard_squares_x.value() - 1, self.chessboard_squares_y.value() - 1 )
        for imnum in range(len(self.images)):
            self.status_text.setText('<b>Detecting chessboard pattern in image {:d} / {:d}...</b>'.format(imnum,len(self.images)))
            self.parent.app.processEvents()
            status,points = cv2.findChessboardCorners( self.images[imnum], self.n_chessboard_points, flags=cv2.CALIB_CB_ADAPTIVE_THRESH )
            self.chessboard_status.append(not status)
            if status:
                for j,point in enumerate(points):
                    self.chessboard_points_2D[imnum][j,:] = point[0]
        self.status_text.setText('')
        self.parent.app.restoreOverrideCursor()        
        if np.all(self.chessboard_status):
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Ok)
            dialog.setTextFormat(qt.Qt.RichText)
            dialog.setWindowTitle('Calcam - No Chessboards Detected')
            dialog.setText("No {:d} x {:d} square chessboard patterns were found in the images.".format(self.chessboard_squares_x.value(),self.chessboard_squares_y.value()))
            dialog.setInformativeText("Is the number of squares set correctly?")
            dialog.setIcon(qt.QMessageBox.Warning)
            dialog.exec_()
        elif np.any(self.chessboard_status):
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Ok)
            dialog.setTextFormat(qt.Qt.RichText)
            dialog.setWindowTitle('Calcam - Chessboard Detection')
            dialog.setText("A {:d} x {:d} square chessboard pattern could not be detected in the following {:d} of {:d} images, which will therefore not be included as additional chessboard constraints:".format(self.chessboard_squares_x.value(),self.chessboard_squares_y.value(),np.count_nonzero(self.chessboard_status),len(self.images)))
            dialog.setInformativeText('<br>'.join(['[#{:d}] '.format(i+1) + self.filenames[i] for i in range(len(self.filenames)) if self.chessboard_status[i] ]))
            dialog.setIcon(qt.QMessageBox.Warning)
            dialog.exec_()                

        self.chessboard_status = [not status for status in self.chessboard_status]
        self.detection_run = True
        self.update_image_display()
        if np.any(self.chessboard_status):
            self.apply_button.setEnabled(True)
            self.status_text.setText('<b>Chessboard patterns detected successfully in {:d} images. Click Apply to use these in Calcam.</b>'.format(np.count_nonzero(self.chessboard_status),len(self.images)))
        else:
            self.apply_button.setEnabled(False)
            self.status_text.setText('')




    def update_image_display(self):

        plt.hold(False)
        self.imax.imshow(self.images[self.current_image])
        plt.hold(True)
        if self.detection_run:
            if self.chessboard_status[self.current_image]:
                status_string = ' - Chessboard Detected OK'
            else:
                status_string = ' - Chessboard Detection FAILED'
        else:
            status_string = ''

        self.current_filename.setText('<html><head/><body><p align="center">{:s} [#{:d}/{:d}]{:s}</p></body></html>'.format(self.filenames[self.current_image],self.current_image+1,len(self.images),status_string))
        if self.chessboard_status[self.current_image]:
            xl = plt.xlim()
            yl = plt.ylim()
            self.imax.plot(self.chessboard_points_2D[self.current_image][:,0],self.chessboard_points_2D[self.current_image][:,1],color='lime',marker='o',linestyle='None')
            plt.xlim(xl)
            plt.ylim(yl)
        self.mplwidget.draw()        



    def apply(self):

        # List of pointpairs objects for the chessboard point pairs
        self.pointpairs_result = []

        chessboard_points_3D = []

        point_spacing = self.chessboard_square_size.value() * 1e-3

        try:
            fieldmask = self.parent.image.fieldmask
            imob = image.from_array(self.images[0])
        except:
            imob = image.from_array(self.images[0])
            fieldmask = imob.fieldmask

        n_fields = fieldmask.max() + 1

        # Create the chessboard coordinates in 3D space. OpenCV returns chessboard corners
        # along each row. 
        for j in range(self.n_chessboard_points[1]):
            for i in range(self.n_chessboard_points[0]):
                chessboard_points_3D.append( ( i * point_spacing , j * point_spacing, 0.) )

        # Loop over chessboard images
        for i in range(len(self.images)):

            # Skip images where no chessboard was found
            if not self.chessboard_status[i]:
                continue

            # Start a new pointpairs object for this image
            self.pointpairs_result.append(pointpairs.PointPairs())
            if len(self.pointpairs_result) == 1 and imob is not None:
                self.pointpairs_result[0].image = imob
            # We already have the 3D positions
            self.pointpairs_result[-1].objectpoints = chessboard_points_3D

            # Initialise image points
            self.pointpairs_result[-1].imagepoints = []

            # Get a neater looking reference to the chessboard corners for this image
            impoints = self.chessboard_points_2D[i]

            # Loop over chessboard points
            for point in range( np.prod(self.n_chessboard_points) ):
                self.pointpairs_result[-1].imagepoints.append([])

                # Populate coordinates for relevant field
                for field in range(n_fields):
                    if fieldmask[int(impoints[point,1]),int(impoints[point,0])] == field:
                        self.pointpairs_result[-1].imagepoints[-1].append([impoints[point,0], impoints[point,1]])
                    else:
                        self.pointpairs_result[-1].imagepoints[-1].append(None)


        # And close the window.
        self.done(1)



class SaveAsDialog(qt.QDialog):

    def __init__(self, parent, type, default_name = ''):

        # GUI initialisation
        qt.QDialog.__init__(self, parent)
        qt.uic.loadUi(os.path.join(paths.ui,'savename_dialog.ui'), self)

        self.label.setText('Enter name to save {:s}:'.format(type))

        self.name = default_name

        self.namebox.setText(default_name)
        self.namebox.textEdited.connect(self.updatename)

        # Start the GUI!
        self.show()


    def updatename(self,text):
        self.name = str(text)



# View designer window.
# This allows creation of FitResults objects for a 'virtual' camera.
class ViewDesignerWindow(qt.QMainWindow):
 
    def __init__(self, app, parent = None):

        # GUI initialisation
        qt.QMainWindow.__init__(self, parent)
        qt.uic.loadUi(os.path.join(paths.ui,'view_designer.ui'), self)

        self.setWindowIcon(qt.QIcon(os.path.join(paths.calcampath,'ui','icon.png')))

        self.app = app

        # See how big the screen is and open the window at an appropriate size
        desktopinfo = self.app.desktop()
        available_space = desktopinfo.availableGeometry(self)

        # Open the window with same aspect ratio as the screen, and no fewer than 500px tall.
        win_height = max(500,min(780,0.75*available_space.height()))
        win_width = win_height * available_space.width() / available_space.height() 
        self.resize(win_width,win_height)


        # Let's show helpful dialog boxes if we have unhandled exceptions:
        sys.excepthook = lambda *ex: show_exception_box(self,*ex)

        # Start up with no CAD model
        self.cadmodel = None

        # Set up VTK
        self.qvtkWidget = qt.QVTKRenderWindowInteractor(self.centralwidget)
        self.gridLayout.addWidget(self.qvtkWidget,1,0)
        self.viewdesigner = vtkinteractorstyles.ViewDesigner()
        self.qvtkWidget.SetInteractorStyle(self.viewdesigner)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.qvtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkInteractor = self.qvtkWidget.GetRenderWindow().GetInteractor()
        self.camera = self.renderer.GetActiveCamera()


        # Populate CAD model list
        self.model_list = machine_geometry.get_available_models()

        self.model_name.addItems(sorted(self.model_list.keys()))
        self.model_name.setCurrentIndex(-1)
        self.load_model_button.setEnabled(0)

        # Populate intrinsics list
        for resname in paths.get_save_list('FitResults'):
            try:
                res = fitting.CalibResults(resname)
                if res.nfields == 1:
                    self.load_intrinsics_combobox.addItem(resname)
            except:
                pass
        self.load_intrinsics_combobox.setCurrentIndex(0)

        # Synthetic camera object to store the results
        self.virtualcamera = fitting.VirtualCalib()
        self.chessboard_fit = None

        # Callbacks for GUI elements
        self.enable_all_button.clicked.connect(self.mass_toggle_model)
        self.disable_all_button.clicked.connect(self.mass_toggle_model)
        self.controls_dock_widget.topLevelChanged.connect(self.update_controls_docked)
        self.viewlist.currentIndexChanged.connect(self.change_cad_view)
        self.camX.valueChanged.connect(self.change_cad_view)
        self.camY.valueChanged.connect(self.change_cad_view)
        self.camZ.valueChanged.connect(self.change_cad_view)
        self.tarX.valueChanged.connect(self.change_cad_view)
        self.tarY.valueChanged.connect(self.change_cad_view)
        self.tarZ.valueChanged.connect(self.change_cad_view)
        self.load_model_button.clicked.connect(self.load_model)
        self.model_name.currentIndexChanged.connect(self.populate_model_variants)
        self.feature_tree.itemChanged.connect(self.update_checked_features)
        self.calcam_intrinsics.clicked.connect(self.update_intrinsics)
        self.chessboard_intrinsics.clicked.connect(self.update_intrinsics)
        self.pinhole_intrinsics.clicked.connect(self.update_intrinsics)
        self.load_intrinsics_combobox.currentIndexChanged.connect(self.update_intrinsics)
        self.pixel_size_box.valueChanged.connect(self.update_intrinsics)
        self.x_pixels_box.valueChanged.connect(self.update_intrinsics)
        self.y_pixels_box.valueChanged.connect(self.update_intrinsics)
        self.focal_length_box.valueChanged.connect(self.update_intrinsics)
        self.save_button.clicked.connect(self.save)
        self.load_chessboard_button.clicked.connect(self.update_chessboard_intrinsics)

        self.pixel_size_box.setSuffix(u' \u00B5m')

        # Start the GUI!
        self.show()
        self.viewdesigner.DoInit(self.renderer,self)
        self.update_intrinsics(redraw=False)
        self.vtkInteractor.Initialize()

        # Warn the user if we don't have any CAD models
        if self.model_list == {}:
            warn_no_models(self)


    def change_cad_view(self,view_index,init=False):

        if self.sender() is self.viewlist or init:
 
            try:
                if not init:
                    self.cadmodel.set_default_view(str(self.viewlist.currentText()))

                # Set to that view
                self.camera.SetPosition(self.cadmodel.cam_pos_default)
                self.camera.SetFocalPoint(self.cadmodel.cam_target_default)
                self.camera.SetViewUp(0,0,1)

            except:
                try:
                    view = fitting.CalibResults(str(self.viewlist.currentText()))

                    self.camera.SetPosition(view.get_pupilpos())
                    self.camera.SetFocalPoint(view.get_pupilpos() + view.get_los_direction(view.image_display_shape[0]/2,view.image_display_shape[1]/2))
                    self.camera.SetViewUp(-1.*view.get_cam_to_lab_rotation()[:,1])
                except:
                    self.viewlist.setCurrentIndex(0)

        else:
            self.camera.SetPosition((self.camX.value(),self.camY.value(),self.camZ.value()))
            self.camera.SetFocalPoint((self.tarX.value(),self.tarY.value(),self.tarZ.value()))

        self.update_viewport_info(self.camera.GetPosition(),self.camera.GetFocalPoint())

        self.refresh_vtk()


    def update_intrinsics(self,redraw=True):

        if self.calcam_intrinsics.isChecked():
            self.load_chessboard_button.setEnabled(False)
            self.load_intrinsics_combobox.setEnabled(True)
            self.x_pixels_box.setEnabled(False)
            self.y_pixels_box.setEnabled(False)
            self.pixel_size_box.setEnabled(False)
            self.focal_length_box.setEnabled(False)
            try:
                res = fitting.CalibResults(str(self.load_intrinsics_combobox.currentText()))
                self.virtualcamera.import_intrinsics(res)
                self.current_intrinsics_combobox = self.calcam_intrinsics
            except:
                self.load_intrinsics_combobox.setCurrentIndex(-1)
        
        elif self.pinhole_intrinsics.isChecked():
            self.load_chessboard_button.setEnabled(False)
            self.load_intrinsics_combobox.setEnabled(False)
            self.x_pixels_box.setEnabled(True)
            self.y_pixels_box.setEnabled(True)
            self.pixel_size_box.setEnabled(True)
            self.focal_length_box.setEnabled(True)

            x_pixels = self.x_pixels_box.value()
            y_pixels = self.y_pixels_box.value()

            cam_matrix = np.zeros([3,3])
            cam_matrix[0,0] = 1000. * self.focal_length_box.value() / self.pixel_size_box.value()
            cam_matrix[1,1] = cam_matrix[0,0]
            cam_matrix[2,2] = 1.
            cam_matrix[0,2] = x_pixels/2.
            cam_matrix[1,2] = y_pixels/2.

            rvec = np.zeros([1,3])
            tvec = np.zeros([1,3])

            virtual_fit_params = [0,cam_matrix,np.zeros([1,4]),rvec,tvec]

            self.virtualcamera.fit_params[0] = fitting.FieldFit('perspective',virtual_fit_params)
            self.virtualcamera.image_display_shape = (x_pixels,y_pixels)
            self.virtualcamera.fieldmask = np.zeros([y_pixels,x_pixels],dtype='uint8')
            self.virtualcamera.nfields = 1
            self.virtualcamera.transform = CoordTransformer()
            self.virtualcamera.transform.x_pixels = x_pixels
            self.virtualcamera.transform.y_pixels = y_pixels
            self.virtualcamera.field_names = ['Image']
            self.current_intrinsics_combobox = self.pinhole_intrinsics
        elif self.chessboard_intrinsics.isChecked():

            if self.chessboard_fit is None:
                self.update_chessboard_intrinsics()

            if self.chessboard_fit is not None:
                self.load_chessboard_button.setEnabled(True)
                self.virtualcamera.import_intrinsics(self.chessboard_fit)
                self.current_intrinsics_combobox = self.chessboard_intrinsics

        self.current_intrinsics_combobox.setChecked(True)
        self.camera.SetViewAngle(self.virtualcamera.get_fov()[1])
        if redraw:
            self.refresh_vtk()


    def init_viewports_chooser(self):

        self.viewlist.clear()

        # Add views to list
        self.viewlist.addItem('-- Defined in CAD Model --')
        for view in self.cadmodel.views:
            self.viewlist.addItem(view[0])

        self.viewlist.addItem(' ')
        self.viewlist.addItem('-- Calibration Results --')
        for view in paths.get_save_list('FitResults'):
            try:
                res = fitting.CalibResults(view)
                if res.nfields == 1:
                    self.viewlist.addItem(view)
            except:
                pass
        self.viewlist.setCurrentIndex(-1)



    def populate_model_variants(self):

        model = self.model_list[str(self.model_name.currentText())]
        self.model_variant.clear()
        self.model_variant.addItems(model[1])
        self.model_variant.setCurrentIndex(model[2])
        self.load_model_button.setEnabled(1)


    def update_checked_features(self,item):
            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.feature_tree.blockSignals(True)
            changed_feature = str(item.text(0))
            if changed_feature in self.group_items:
                feature = changed_feature
                changed_feature = []
                for i in range(self.group_items[feature].childCount()):
                    changed_feature.append(str(self.group_items[feature].child(i).text(0)))
                    self.group_items[feature].child(i).setCheckState(0,self.group_items[feature].checkState(0))
                    if self.group_items[feature].checkState(0) == qt.Qt.Checked:
                        self.group_items[feature].child(i).setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                    else:
                        self.group_items[feature].child(i).setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
            else:
                changed_feature = [changed_feature]
                feature = item.parent()
                if feature is not self.treeitem_machine:
                    checkstates = []
                    for i in range(feature.childCount()):
                        checkstates.append(feature.child(i).checkState(0))

                    if len(list(set(checkstates))) > 1:
                        feature.setCheckState(0,qt.Qt.PartiallyChecked)
                        feature.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                    else:
                        feature.setCheckState(0,checkstates[0])
                        if checkstates[0] == qt.Qt.Checked:
                            feature.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                        else:
                            feature.setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)

            if item.checkState(0) == qt.Qt.Checked:
                item.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                self.cadmodel.enable_features(changed_feature,self.renderer)
            else:
                item.setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                self.cadmodel.disable_features(changed_feature,self.renderer)


            self.refresh_vtk()
            self.feature_tree.blockSignals(False)
            self.app.restoreOverrideCursor()



    def load_model(self):

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        model = self.model_list[str(self.model_name.currentText())]

        # Dispose of the old model
        if self.cadmodel is not None:
            
            old_machine_name = self.cadmodel.machine_name
            old_enabled_features = self.cadmodel.get_enabled_features()

            for feature in self.model_actors.keys():
                actor = self.model_actors.pop(feature)
                self.renderer.RemoveActor(actor)
            
            del self.cadmodel
            self.tabWidget.setTabEnabled(2,False)

        else:
            old_machine_name = None

        # Create a new one
        exec('self.cadmodel = machine_geometry.' + model[0] + '("' + str(self.model_variant.currentText()) + '")')

        self.cadmodel.link_gui_window(self)

        if not self.cad_auto_load.isChecked():
            if self.cadmodel.machine_name == old_machine_name:
                self.cadmodel.enable_only(old_enabled_features)
            else:
                for feature in self.cadmodel.features:
                    self.cadmodel.disable_features(feature[0])

        features = self.cadmodel.get_enabled_features()
        for i,actor in self.cadmodel.get_vtkActors(features):
            self.model_actors[features[i]] = actor
            self.renderer.AddActor(actor)

        self.statusbar.showMessage('Setting up CAD model...')

        # Initialise the CAD model setup GUI
        init_model_settings(self)

        # Initialise other lists of things
        self.init_viewports_chooser()


        # Set selected CAD view to the model's default, if the machine has been changed (i.e. changing model variant will maintain the viewport)
        if self.cadmodel.machine_name != old_machine_name:
            self.change_cad_view(0,init=True)

        self.enable_all_button.setEnabled(1)
        self.disable_all_button.setEnabled(1)
        self.statusbar.clearMessage()
        self.refresh_vtk()
        self.app.restoreOverrideCursor()


    def mass_toggle_model(self):
        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        if self.sender() is self.enable_all_button:
            for i in range(self.treeitem_machine.childCount()):
               self.treeitem_machine.child(i).setCheckState(0,qt.Qt.Checked)
        elif self.sender() is self.disable_all_button:
            for i in range(self.treeitem_machine.childCount()):
                self.treeitem_machine.child(i).setCheckState(0,qt.Qt.Unchecked)
        self.app.restoreOverrideCursor()


    def update_controls_docked(self,floating_controls):

        if floating_controls:
            self.cadview_header.hide()
        else:
            self.cadview_header.show()

    def update_viewport_info(self,campos,camtar):
        self.camX.blockSignals(True)
        self.camY.blockSignals(True)
        self.camZ.blockSignals(True)
        self.tarX.blockSignals(True)
        self.tarY.blockSignals(True)
        self.tarZ.blockSignals(True)
        self.tarZ.blockSignals(True)

        self.camX.setValue(campos[0])
        self.camY.setValue(campos[1])
        self.camZ.setValue(campos[2])
        self.tarX.setValue(camtar[0])
        self.tarY.setValue(camtar[1])
        self.tarZ.setValue(camtar[2])

        self.camX.blockSignals(False)
        self.camY.blockSignals(False)
        self.camZ.blockSignals(False)
        self.tarX.blockSignals(False)
        self.tarY.blockSignals(False)
        self.tarZ.blockSignals(False)



    def refresh_vtk(self):
        self.renderer.Render()
        self.qvtkWidget.update()



    def update_cad_status(self,message):

        if message is not None:
            self.statusbar.showMessage(message)
            self.app.processEvents()
        else:
            self.statusbar.clearMessage()
            self.app.processEvents()


    def update_chessboard_intrinsics(self):

        dialog = ChessboardDialog(self,modelselection=True)
        dialog.exec_()

        if dialog.pointpairs_result is not None:
            chessboard_pointpairs = copy.deepcopy(dialog.pointpairs_result)
            fitter = fitting.Fitter()
            if dialog.perspective_model.isChecked():
                fitter.model = ['perspective']
            elif dialog.fisheye_model.isChecked():
                fitter.model = ['fisheye']
            fitter.set_PointPairs(chessboard_pointpairs[0])
            fitter.add_intrinsics_pointpairs(chessboard_pointpairs[1:])
            self.chessboard_fit = fitter.do_fit()

        del dialog





    def closeEvent(self,event):

        # If we're exiting, put python'e exception handling back to normal.
        sys.excepthook = sys.__excepthook__


    def save(self):

        # First we have to add the extrinsics to the calibration object
        campos = np.matrix([[self.camX.value()],[self.camY.value()],[self.camZ.value()]])
        camtar = np.matrix([[self.tarX.value()],[self.tarY.value()],[self.tarZ.value()]])

        # We need to pass the view up direction to set_exirtinsics, but it isn't kept up-to-date by
        # the VTK camera. So here we explicitly ask for it to be updated then pass the correct
        # version to set_extrinsics, but then reset it back to what it was, to avoid ruining 
        # the mouse interaction.
        cam_roll = self.camera.GetRoll()
        self.camera.OrthogonalizeViewUp()
        upvec = np.array(self.camera.GetViewUp())
        self.camera.SetViewUp(0,0,1)
        self.camera.SetRoll(cam_roll)

        self.virtualcamera.set_extrinsics(campos,upvec,camtar = camtar)

        # Now save!
        try:
            dialog = SaveAsDialog(self,'Camera Definition')
            dialog.exec_()
            if dialog.result() == 1:
                save_name = dialog.name
                del dialog         
                self.virtualcamera.save(save_name)
                dialog = qt.QMessageBox(self)
                dialog.setStandardButtons(qt.QMessageBox.Ok)
                dialog.setWindowTitle('Calcam - Save Complete')
                dialog.setText('Camera definition saved successfully.')
                dialog.setIcon(qt.QMessageBox.Information)
                dialog.exec_()

        except Exception as err:
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Ok)
            dialog.setWindowTitle('Calcam - Save Error')
            dialog.setText('Error saving camera definition:\n' + str(err))
            dialog.setIcon(qt.QMessageBox.Warning)
            dialog.exec_()


# View designer window.
# This allows creation of FitResults objects for a 'virtual' camera.
class AlignmentCalibWindow(qt.QMainWindow):
 
    def __init__(self, app, parent = None):

        # GUI initialisation
        qt.QMainWindow.__init__(self, parent)
        qt.uic.loadUi(os.path.join(paths.ui,'alignment_calib.ui'), self)

        self.setWindowIcon(qt.QIcon(os.path.join(paths.calcampath,'ui','icon.png')))

        self.app = app

        # See how big the screen is and open the window at an appropriate size
        desktopinfo = self.app.desktop()
        available_space = desktopinfo.availableGeometry(self)

        # Open the window with same aspect ratio as the screen, and no fewer than 500px tall.
        win_height = max(500,min(780,0.75*available_space.height()))
        win_width = win_height * available_space.width() / available_space.height() 
        self.resize(win_width,win_height)


        # Let's show helpful dialog boxes if we have unhandled exceptions:
        sys.excepthook = lambda *ex: show_exception_box(self,*ex)

        # Start up with no CAD model
        self.cadmodel = None

        # Set up VTK
        self.qvtkWidget = qt.QVTKRenderWindowInteractor(self.centralwidget)
        self.gridLayout.addWidget(self.qvtkWidget,1,0)
        self.viewdesigner = vtkinteractorstyles.ViewAligner()
        self.qvtkWidget.SetInteractorStyle(self.viewdesigner)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.qvtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkInteractor = self.qvtkWidget.GetRenderWindow().GetInteractor()
        self.camera = self.renderer.GetActiveCamera()


        # Populate CAD model list
        self.model_list = machine_geometry.get_available_models()

        self.model_name.addItems(sorted(self.model_list.keys()))
        self.model_name.setCurrentIndex(-1)
        self.load_model_button.setEnabled(0)

        self.image_settings.hide()


        # Synthetic camera object to store the results
        self.virtualcamera = fitting.CalibResults()
        self.chessboard_fit = None
        self.image = None

        # Callbacks for GUI elements
        self.image_sources_list.currentIndexChanged.connect(self.build_imload_gui)
        self.load_image_button.clicked.connect(self.load_image)
        self.im_flipud.clicked.connect(self.transform_image)
        self.im_fliplr.clicked.connect(self.transform_image)
        self.im_rotate_button.clicked.connect(self.transform_image)
        self.im_reset.clicked.connect(self.transform_image)
        self.im_y_stretch_button.clicked.connect(self.transform_image)
        self.enable_all_button.clicked.connect(self.mass_toggle_model)
        self.disable_all_button.clicked.connect(self.mass_toggle_model)
        self.controls_dock_widget.topLevelChanged.connect(self.update_controls_docked)
        self.viewlist.currentIndexChanged.connect(self.change_cad_view)
        self.camX.valueChanged.connect(self.change_cad_view)
        self.camY.valueChanged.connect(self.change_cad_view)
        self.camZ.valueChanged.connect(self.change_cad_view)
        self.tarX.valueChanged.connect(self.change_cad_view)
        self.tarY.valueChanged.connect(self.change_cad_view)
        self.tarZ.valueChanged.connect(self.change_cad_view)
        self.camRoll.valueChanged.connect(self.change_cad_view)
        self.load_model_button.clicked.connect(self.load_model)
        self.model_name.currentIndexChanged.connect(self.populate_model_variants)
        self.feature_tree.itemChanged.connect(self.update_checked_features)
        self.calcam_intrinsics.clicked.connect(self.update_intrinsics)
        self.chessboard_intrinsics.clicked.connect(self.update_intrinsics)
        self.pinhole_intrinsics.clicked.connect(self.update_intrinsics)
        self.load_intrinsics_combobox.currentIndexChanged.connect(self.update_intrinsics)
        self.focal_length_box.valueChanged.connect(self.update_intrinsics)
        self.cx_box.valueChanged.connect(self.update_intrinsics)
        self.cy_box.valueChanged.connect(self.update_intrinsics)
        self.pixel_size_box.valueChanged.connect(self.update_intrinsics)
        self.save_button.clicked.connect(self.save)
        self.load_chessboard_button.clicked.connect(self.update_chessboard_intrinsics)
        self.pixel_size_checkbox.toggled.connect(self.toggle_real_pixel_size)
        self.im_opacity_slider.valueChanged.connect(self.transform_image)
        self.hist_eq_checkbox.stateChanged.connect(self.toggle_hist_eq)

        self.pixel_size_box.setSuffix(u' \u00B5m')
        self.camRoll.setSuffix(u"\u00B0")
        self.tabWidget.setTabEnabled(2,False)

        # If we have an old version of openCV, histo equilisation won't work :(
        if cv2_version < 2.4 or (cv2_version == 2.4 and cv2_micro_version < 6):
            self.hist_eq_checkbox.setEnabled(False)
            self.hist_eq_checkbox.setToolTip('Requires OpenCV 2.4.6 or newer; you have {:s}'.format(cv2.__version__))

        # Populate image sources list and tweak GUI layout for image loading.
        self.imload_inputs = []
        self.image_load_options.layout().setColumnMinimumWidth(0,100)
        for imsource in image.image_sources:
            self.image_sources_list.addItem(imsource.gui_display_name)
        self.image_sources_list.setCurrentIndex(0)

        # Start the GUI!
        self.show()
        self.viewdesigner.DoInit(self.renderer,self)
        #self.update_intrinsics(redraw=False)
        self.vtkInteractor.Initialize()

        # Warn the user if we don't have any CAD models
        if self.model_list == {}:
            warn_no_models(self)

    def toggle_real_pixel_size(self,real_pixel_size):
        self.focal_length_box.blockSignals(True)
        if real_pixel_size:
            self.focal_length_box.setValue( self.focal_length_box.value() * self.pixel_size_box.value() / 1000)
            self.focal_length_box.setSuffix(' mm')
            self.pixel_size_box.setEnabled(True)
        else:
            self.focal_length_box.setValue( 1000 * self.focal_length_box.value() / self.pixel_size_box.value() )
            self.focal_length_box.setSuffix(' px')
            self.pixel_size_box.setEnabled(False)

        self.focal_length_box.blockSignals(False)

    def change_cad_view(self,view_index,init=False):

        if self.sender() is self.viewlist or init:
 
            try:
                if not init:
                    self.cadmodel.set_default_view(str(self.viewlist.currentText()))

                # Set to that view
                self.camera.SetPosition(self.cadmodel.cam_pos_default)
                self.camera.SetFocalPoint(self.cadmodel.cam_target_default)
                self.camera.SetViewUp(0,0,1)

            except:
                try:
                    view = fitting.CalibResults(str(self.viewlist.currentText()))

                    self.camera.SetPosition(view.get_pupilpos())
                    Cx = view.fit_params[0].cam_matrix[0,2]
                    Cy = view.fit_params[0].cam_matrix[1,2]
                    self.camera.SetFocalPoint(view.get_pupilpos() + view.get_los_direction(Cx,Cy,ForceField=0))
                    self.camera.SetViewUp(-1.*view.get_cam_to_lab_rotation()[:,1])
                except:
                    self.viewlist.setCurrentIndex(0)

        else:
            self.camera.SetPosition((self.camX.value(),self.camY.value(),self.camZ.value()))
            self.camera.SetFocalPoint((self.tarX.value(),self.tarY.value(),self.tarZ.value()))

            # View up vector is in the plane whos normal is the view direction,
            # and the angle between the up vector and the projection of Z on that plane is the camera roll.
            view_direction = self.camera.GetDirectionOfProjection()
            if np.abs(view_direction[2]) < 0.99:
                z_projection = np.array([ -view_direction[0]*view_direction[2], -view_direction[1]*view_direction[2],1-view_direction[2]**2 ])
            else:
                z_projection = np.array([ 1.-view_direction[0]**2, -view_direction[0]*view_direction[1],-view_direction[2]*view_direction[0]])
            upvec = rotate_3D(z_projection,view_direction,self.camRoll.value())
            self.camera.SetViewUp(upvec)
            roll = self.camera.GetRoll()
            self.camera.SetViewUp(0,0,1)
            self.camera.SetRoll(roll)

        self.update_viewport_info(self.camera.GetPosition(),self.camera.GetFocalPoint(),self.camera.GetViewUp())

        self.refresh_vtk()

    def update_image_info_string(self):

        if np.any(self.image.transform.get_display_shape() != list(self.image.data.shape[1::-1])):
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP) [ As Displayed ]<br>{3:d} x {4:d} pixels ({5:.1f} MP) [ Raw Data ]<br>'.format(self.image.transform.get_display_shape()[0],self.image.transform.get_display_shape()[1],np.prod(self.image.transform.get_display_shape()) / 1e6 ,self.image.data.shape[1],self.image.data.shape[0],np.prod(self.image.data.shape[:2]) / 1e6 )
        else:
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP)<br>'.format(self.image.transform.get_display_shape()[0],self.image.transform.get_display_shape()[1],np.prod(self.image.transform.get_display_shape()) / 1e6 )
        
        if len(self.image.data.shape) == 2:
            info_str = info_str + 'Monochrome'
        elif len(self.image.data.shape) == 3 and self.image.data.shape[2] == 3:
            info_str = info_str + 'RGB Colour'
        elif len(self.image.data.shape) == 3 and self.image.data.shape[2] == 3:
            info_str = info_str + 'RGB Colour'

        self.image_info.setText(info_str)

    def browse_for_file(self):

        for i,option in enumerate(self.imload_inputs):
            if self.sender() in option[0]:
                filename_filter = self.imsource.gui_inputs[i]['filter']
                target_textbox = option[0][2]


        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(0)
        filedialog.setFileMode(1)
        filedialog.setWindowTitle('Select File')
        filedialog.setNameFilter(filename_filter)
        filedialog.setLabelText(3,'Select')
        filedialog.exec_()
        if filedialog.result() == 1:
            target_textbox.setText(str(filedialog.selectedFiles()[0]))


    def build_imload_gui(self,index):

        layout = self.image_load_options.layout()
        for widgets,_ in self.imload_inputs:
            for widget in widgets:
                layout.removeWidget(widget)
                widget.close()

        #layout = qt.QGridLayout(self.image_load_options)
        self.imsource = image.image_sources[index]

        self.imload_inputs = []

        row = 0
        for option in self.imsource.gui_inputs:

            labelwidget = qt.QLabel(option['label'] + ':')
            layout.addWidget(labelwidget,row,0)

            if option['type'] == 'filename':
                button = qt.QPushButton('Browse...')
                button.clicked.connect(self.browse_for_file)
                button.setMaximumWidth(80)
                layout.addWidget(button,row+1,1)
                fname = qt.QLineEdit()
                if 'default' in option:
                    fname.setText(option['default'])
                layout.addWidget(fname,row,1)
                self.imload_inputs.append( ([labelwidget,button,fname],fname.text ) )
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
                self.imload_inputs.append( ([labelwidget,valbox],valbox.value ) )
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
                self.imload_inputs.append( ([labelwidget,valbox],valbox.value ) )
                row = row + 1
            elif option['type'] == 'string':
                ted = qt.QLineEdit()
                if 'default' in option:
                    ted.setText(option['default'])
                layout.addWidget(ted,row,1)
                self.imload_inputs.append( ([labelwidget,ted],ted.text ) )
                row = row + 1
            elif option['type'] == 'bool':
                checkbox = qt.QCheckBox()
                if 'default' in option:
                    checkbox.setChecked(option['default'])
                layout.addWidget(checkbox,row,1)
                self.imload_inputs.append( ([labelwidget,checkbox],checkbox.isChecked ) )
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
                self.imload_inputs.append( ([labelwidget,cb],cb.currentText) )
                row = row + 1

    def toggle_hist_eq(self,check_state):

        # Enable / disable adaptive histogram equalisation
        if check_state == qt.Qt.Checked:
            self.image.postprocessor = image_filters.hist_eq()
            self.image_original.postprocessor = image_filters.hist_eq()
        else:
            self.image.postprocessor = None
            self.image_original.postprocessor = None

        cx = self.virtualcamera.fit_params[0].cam_matrix[0,2]
        cy = self.virtualcamera.fit_params[0].cam_matrix[1,2]
        self.viewdesigner.init_image(self.image,opacity=self.im_opacity_slider.value() / 10.,cx=cx,cy=cy)
 

    def load_image(self,init_image=None):

        if self.image is None:
            reset_intrinsics = True
        else:
            reset_intrinsics = False

        # By default we assume we don't know the pixel size
        self.pixel_size_checkbox.setChecked(False)

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.statusbar.showMessage('Loading image...')

        # Gather up the required input arguments from the image load gui
        imload_options = []
        for option in self.imload_inputs:
            imload_options.append(option[1]())
            if qt.qt_ver == 4:
                if type(imload_options[-1]) == qt.QString:
                    imload_options[-1] = str(imload_options[-1])

        newim = self.imsource(*imload_options)

        self.image_settings.hide()

        existing_im_names = paths.get_save_list('Images')
        if newim.name in existing_im_names:
            testim = image.Image(newim.name)
            if not np.all(newim.data == testim.data):
                i = 0
                new_name = newim.name
                while new_name in existing_im_names:
                    i = i + 1
                    new_name = newim.name + '({:d})'.format(i)
                newim.name = new_name

        self.image_original = newim

        if self.virtualcamera.transform is None:
            reset_intrinsics = True
            self.image = self.image_original             
        elif self.virtualcamera.transform.x_pixels == self.image_original.transform.x_pixels and self.virtualcamera.transform.y_pixels == self.image_original.transform.y_pixels:
            if self.pinhole_intrinsics.isChecked() == False:
                self.image = self.virtualcamera.undistort_image(self.image_original)
            else:
                self.image = self.image_original
        else:
            reset_intrinsics = True
            self.image = self.image_original 


        self.image_settings.show()
        self.hist_eq_checkbox.setCheckState(qt.Qt.Unchecked)
        self.viewdesigner.init_image(self.image,opacity=self.im_opacity_slider.value() / 10.)

        self.update_image_info_string()
        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()
        self.tabWidget.setTabEnabled(2,True)

        if reset_intrinsics:
            self.cx_box.setValue(self.image.transform.get_display_shape()[0]/2)
            self.cy_box.setValue(self.image.transform.get_display_shape()[1]/2)
            self.pinhole_intrinsics.setChecked(True)
            self.populate_intrinsics_list()

        self.update_intrinsics(redraw=False)
        self.virtualcamera.transform = self.image.transform


    def populate_intrinsics_list(self):

            # Populate intrinsics list
            self.load_intrinsics_combobox.clear()
            for resname in paths.get_save_list('FitResults'):
                try:
                    res = fitting.CalibResults(resname)
                    if res.nfields == 1 and res.transform.get_display_shape() == self.image_original.transform.get_display_shape():
                        self.load_intrinsics_combobox.addItem(resname)
                except Exception as e:
                    pass
            self.load_intrinsics_combobox.setCurrentIndex(0)
            if self.load_intrinsics_combobox.count() > 0:
                self.calcam_intrinsics.setEnabled(True)
            else:
                self.calcam_intrinsics.setEnabled(False)  


    def update_intrinsics(self,redraw=True):

        if self.calcam_intrinsics.isChecked():
            self.load_chessboard_button.setEnabled(False)
            self.load_intrinsics_combobox.setEnabled(True)
            self.focal_length_box.setEnabled(False)
            self.cx_box.setEnabled(False)
            self.cy_box.setEnabled(False)
            try:
                res = fitting.CalibResults(str(self.load_intrinsics_combobox.currentText()))
                self.virtualcamera.import_intrinsics(res)
                self.current_intrinsics_combobox = self.calcam_intrinsics
            except:
                self.load_intrinsics_combobox.setCurrentIndex(-1)
            self.image = self.virtualcamera.undistort_image(self.image_original)
            cx = self.virtualcamera.fit_params[0].cam_matrix[0,2]
            cy = self.virtualcamera.fit_params[0].cam_matrix[1,2]
            f = np.mean([self.virtualcamera.fit_params[0].cam_matrix[0,0],self.virtualcamera.fit_params[0].cam_matrix[1,1]])
            if self.pixel_size_checkbox.isChecked():
                self.focal_length_box.setValue(f*self.pixel_size_box.value()/1000)
            else:
                self.focal_length_box.setValue(f)
            self.cx_box.setValue(cx)
            self.cy_box.setValue(cy)

            self.viewdesigner.init_image(self.image,opacity=self.im_opacity_slider.value() / 10.,cx=cx,cy=cy)
        
        elif self.pinhole_intrinsics.isChecked():
            self.load_chessboard_button.setEnabled(False)
            self.load_intrinsics_combobox.setEnabled(False)
            self.focal_length_box.setEnabled(True)
            self.cx_box.setEnabled(True)
            self.cy_box.setEnabled(True)

            x_pixels = self.image.transform.get_display_shape()[0]
            y_pixels = self.image.transform.get_display_shape()[1]

            if self.pixel_size_checkbox.isChecked():
                focal_length = 1000 * self.focal_length_box.value() / self.pixel_size_box.value()
            else:
                focal_length = self.focal_length_box.value()

            cam_matrix = np.zeros([3,3])
            cam_matrix[0,0] = focal_length
            cam_matrix[1,1] = focal_length
            cam_matrix[2,2] = 1.
            cam_matrix[0,2] = self.cx_box.value()
            cam_matrix[1,2] = self.cy_box.value()

            rvec = np.zeros([1,3])
            tvec = np.zeros([1,3])

            virtual_fit_params = [0,cam_matrix,np.zeros([1,4]),rvec,tvec]

            self.virtualcamera.fit_params[0] = fitting.FieldFit('perspective',virtual_fit_params)
            self.virtualcamera.image_display_shape = (x_pixels,y_pixels)
            self.virtualcamera.fieldmask = np.zeros([y_pixels,x_pixels],dtype='uint8')
            self.virtualcamera.nfields = 1
            self.virtualcamera.field_names = ['Image']
            self.current_intrinsics_combobox = self.pinhole_intrinsics
            self.image = self.image_original
            self.viewdesigner.init_image(self.image,opacity=self.im_opacity_slider.value() / 10.,cx=self.cx_box.value(),cy=self.cy_box.value())

        elif self.chessboard_intrinsics.isChecked():

            if self.chessboard_fit is None:
                self.update_chessboard_intrinsics()

            if self.chessboard_fit is not None:
                self.load_chessboard_button.setEnabled(True)
                self.virtualcamera.import_intrinsics(self.chessboard_fit)
                self.current_intrinsics_combobox = self.chessboard_intrinsics

            self.image = self.virtualcamera.undistort_image(self.image_original)
            cx = self.virtualcamera.fit_params[0].cam_matrix[0,2]
            cy = self.virtualcamera.fit_params[0].cam_matrix[1,2]
            self.viewdesigner.init_image(self.image,opacity=self.im_opacity_slider.value() / 10.,cx=cx,cy=cy)
            f = np.mean([self.virtualcamera.fit_params[0].cam_matrix[0,0],self.virtualcamera.fit_params[0].cam_matrix[1,1]])
            if self.pixel_size_checkbox.isChecked():
                self.focal_length_box.setValue(f*self.pixel_size_box.value()/1000)
            else:
                self.focal_length_box.setValue(f)
            self.cx_box.setValue(cx)
            self.cy_box.setValue(cy)
            self.focal_length_box.setEnabled(False)
            self.cx_box.setEnabled(False)
            self.cy_box.setEnabled(False)
            
        self.current_intrinsics_combobox.setChecked(True)
        self.camera.SetViewAngle(self.virtualcamera.get_fov(FullChipWithoutDistortion=True)[1])

        if redraw:
            self.refresh_vtk()


    def init_viewports_chooser(self):

        self.viewlist.clear()

        # Add views to list
        self.viewlist.addItem('-- Defined in CAD Model --')
        for view in self.cadmodel.views:
            self.viewlist.addItem(view[0])

        self.viewlist.addItem(' ')
        self.viewlist.addItem('-- Calibration Results --')
        for view in paths.get_save_list('FitResults'):
            try:
                res = fitting.CalibResults(view)
                if res.nfields == 1:
                    self.viewlist.addItem(view)
            except:
                pass
        self.viewlist.setCurrentIndex(-1)



    def populate_model_variants(self):

        model = self.model_list[str(self.model_name.currentText())]
        self.model_variant.clear()
        self.model_variant.addItems(model[1])
        self.model_variant.setCurrentIndex(model[2])
        self.load_model_button.setEnabled(1)


    def update_checked_features(self,item):
            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.feature_tree.blockSignals(True)
            changed_feature = str(item.text(0))
            if changed_feature in self.group_items:
                feature = changed_feature
                changed_feature = []
                for i in range(self.group_items[feature].childCount()):
                    changed_feature.append(str(self.group_items[feature].child(i).text(0)))
                    self.group_items[feature].child(i).setCheckState(0,self.group_items[feature].checkState(0))
                    if self.group_items[feature].checkState(0) == qt.Qt.Checked:
                        self.group_items[feature].child(i).setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                    else:
                        self.group_items[feature].child(i).setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
            else:
                changed_feature = [changed_feature]
                feature = item.parent()
                if feature is not self.treeitem_machine:
                    checkstates = []
                    for i in range(feature.childCount()):
                        checkstates.append(feature.child(i).checkState(0))

                    if len(list(set(checkstates))) > 1:
                        feature.setCheckState(0,qt.Qt.PartiallyChecked)
                        feature.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                    else:
                        feature.setCheckState(0,checkstates[0])
                        if checkstates[0] == qt.Qt.Checked:
                            feature.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                        else:
                            feature.setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)

            if item.checkState(0) == qt.Qt.Checked:
                item.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                self.cadmodel.enable_features(changed_feature,self.renderer)
            else:
                item.setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                self.cadmodel.disable_features(changed_feature,self.renderer)


            self.refresh_vtk()
            self.feature_tree.blockSignals(False)
            self.app.restoreOverrideCursor()



    def load_model(self):

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        model = self.model_list[str(self.model_name.currentText())]

        # Dispose of the old model
        if self.cadmodel is not None:
            
            old_machine_name = self.cadmodel.machine_name
            old_enabled_features = self.cadmodel.get_enabled_features()

            for actor in self.cadmodel.get_vtkActors():
                self.renderer.RemoveActor(actor)
            
            del self.cadmodel
            self.tabWidget.setTabEnabled(2,False)

        else:
            old_machine_name = None

        # Create a new one
        exec('self.cadmodel = machine_geometry.' + model[0] + '("' + str(self.model_variant.currentText()) + '")')

        self.cadmodel.link_gui_window(self)
        self.cadmodel.set_colour((1,0,0))
        self.cadmodel.edges = True
        if not self.cad_auto_load.isChecked():
            if self.cadmodel.machine_name == old_machine_name:
                self.cadmodel.enable_only(old_enabled_features)
            else:
                for feature in self.cadmodel.features:
                    self.cadmodel.disable_features(feature[0])

        for actor in self.cadmodel.get_vtkActors():
            self.renderer.AddActor(actor)

        self.statusbar.showMessage('Setting up CAD model...')

        # Initialise the CAD model setup GUI
        init_model_settings(self)

        # Initialise other lists of things
        self.init_viewports_chooser()


        # Set selected CAD view to the model's default, if the machine has been changed (i.e. changing model variant will maintain the viewport)
        if self.cadmodel.machine_name != old_machine_name:
            self.change_cad_view(0,init=True)

        self.enable_all_button.setEnabled(1)
        self.disable_all_button.setEnabled(1)
        self.statusbar.clearMessage()
        self.refresh_vtk()
        self.app.restoreOverrideCursor()


    def mass_toggle_model(self):
        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        if self.sender() is self.enable_all_button:
            for i in range(self.treeitem_machine.childCount()):
               self.treeitem_machine.child(i).setCheckState(0,qt.Qt.Checked)
        elif self.sender() is self.disable_all_button:
            for i in range(self.treeitem_machine.childCount()):
                self.treeitem_machine.child(i).setCheckState(0,qt.Qt.Unchecked)
        self.app.restoreOverrideCursor()


    def update_controls_docked(self,floating_controls):

        if floating_controls:
            self.cadview_header.hide()
        else:
            self.cadview_header.show()

    def update_viewport_info(self,campos,camtar,upvec):


        self.camX.blockSignals(True)
        self.camY.blockSignals(True)
        self.camZ.blockSignals(True)
        self.tarX.blockSignals(True)
        self.tarY.blockSignals(True)
        self.tarZ.blockSignals(True)
        self.tarZ.blockSignals(True)
        self.camRoll.blockSignals(True)

        self.camX.setValue(campos[0])
        self.camY.setValue(campos[1])
        self.camZ.setValue(campos[2])
        self.tarX.setValue(camtar[0])
        self.tarY.setValue(camtar[1])
        self.tarZ.setValue(camtar[2])

        view_direction = np.array(camtar) - np.array(campos)
        view_direction = view_direction / np.sqrt(np.sum(view_direction**2))
        if np.abs(view_direction[2]) < 0.99:
            z_projection = np.array([ -view_direction[0]*view_direction[2], -view_direction[1]*view_direction[2],1-view_direction[2]**2 ])
        else:
            z_projection = np.array([ 1.-view_direction[0]**2, -view_direction[0]*view_direction[1],-view_direction[2]*view_direction[0]])

        h_projection = np.cross(z_projection,view_direction)
        h_projection = h_projection / np.sqrt(np.sum(h_projection**2))
        z_projection = z_projection / np.sqrt(np.sum(z_projection**2))
        x = np.dot(upvec,z_projection)
        y = np.dot(upvec,h_projection)
        cam_roll = - np.arctan2(y,x) * 180 / 3.14159
        self.camRoll.setValue(cam_roll)

        self.camX.blockSignals(False)
        self.camY.blockSignals(False)
        self.camZ.blockSignals(False)
        self.tarX.blockSignals(False)
        self.tarY.blockSignals(False)
        self.tarZ.blockSignals(False)
        self.camRoll.blockSignals(False)



    def refresh_vtk(self):
        self.renderer.Render()
        self.qvtkWidget.update()



    def update_cad_status(self,message):

        if message is not None:
            self.statusbar.showMessage(message)
            self.app.processEvents()
        else:
            self.statusbar.clearMessage()
            self.app.processEvents()


    def update_chessboard_intrinsics(self):

        dialog = ChessboardDialog(self,modelselection=True)
        dialog.exec_()

        if dialog.pointpairs_result is not None:
            chessboard_pointpairs = copy.deepcopy(dialog.pointpairs_result)
            fitter = fitting.Fitter()
            if dialog.perspective_model.isChecked():
                fitter.model = ['perspective']
            elif dialog.fisheye_model.isChecked():
                fitter.model = ['fisheye']
            fitter.set_PointPairs(chessboard_pointpairs[0])
            fitter.add_intrinsics_pointpairs(chessboard_pointpairs[1:])
            self.chessboard_fit = fitter.do_fit()

        del dialog


    def transform_image(self,data):

        c_orig = self.image.transform.display_to_original_coords(self.cx_box.value(),self.cy_box.value())

        if self.sender() is self.im_flipud:
            if len(self.image.transform.transform_actions) > 0:
                if self.image.transform.transform_actions[-1] == 'flip_up_down':
                    del self.image.transform.transform_actions[-1]
                else:
                    self.image.transform.transform_actions.append('flip_up_down')
            else:
                self.image.transform.transform_actions.append('flip_up_down')


        elif self.sender() is self.im_fliplr:
            if len(self.image.transform.transform_actions) > 0:
                if self.image.transform.transform_actions[-1] == 'flip_left_right':
                    del self.image.transform.transform_actions[-1]
                else:
                    self.image.transform.transform_actions.append('flip_left_right')
            else:
                self.image.transform.transform_actions.append('flip_left_right')

        elif self.sender() is self.im_rotate_button:
            if len(self.image.transform.transform_actions) > 0:
                if 'rotate_clockwise' in self.image.transform.transform_actions[-1]:
                    current_angle = int(self.image.transform.transform_actions[-1].split('_')[2])
                    del self.image.transform.transform_actions[-1]
                    new_angle = self.im_rotate_angle.value()
                    total_angle = current_angle + new_angle
                    if total_angle > 270:
                        total_angle = total_angle - 360

                    if new_angle > 0:
                        self.image.transform.transform_actions.append('rotate_clockwise_' + str(total_angle))
                else:
                    self.image.transform.transform_actions.append('rotate_clockwise_' + str(self.im_rotate_angle.value()))
            else:
                self.image.transform.transform_actions.append('rotate_clockwise_' + str(self.im_rotate_angle.value()))

        elif self.sender() is self.im_y_stretch_button:
            sideways = False
            for action in self.image.transform.transform_actions:
                if action.lower() in ['rotate_clockwise_90','rotate_clockwise_270']:
                    sideways = not sideways
            if sideways:
                self.image.transform.pixel_aspectratio = self.image.transform.pixel_aspectratio/self.im_y_stretch_factor.value()
            else:
                self.image.transform.pixel_aspectratio = self.image.transform.pixel_aspectratio*self.im_y_stretch_factor.value()
            
        elif self.sender() is self.im_reset:
            self.image.transform.transform_actions = []
            self.image.transform.pixel_aspectratio = 1

        self.virtualcamera.transform = self.image.transform
        self.image_original.transform = self.image.transform


        c_new = self.image.transform.original_to_display_coords(c_orig[0],c_orig[1])
        self.cx_box.setValue(c_new[0])
        self.cy_box.setValue(c_new[1])
        
        # Update the image and point pairs
        self.viewdesigner.init_image(self.image,opacity=self.im_opacity_slider.value() / 10.,cx=c_new[0],cy=c_new[1])

        self.update_image_info_string()
        self.populate_intrinsics_list()


    def closeEvent(self,event):

        # If we're exiting, put python'e exception handling back to normal.
        sys.excepthook = sys.__excepthook__


    def save(self):

        # First we have to add the extrinsics to the calibration object
        campos = np.matrix([[self.camX.value()],[self.camY.value()],[self.camZ.value()]])
        camtar = np.matrix([[self.tarX.value()],[self.tarY.value()],[self.tarZ.value()]])

        # We need to pass the view up direction to set_exirtinsics, but it isn't kept up-to-date by
        # the VTK camera. So here we explicitly ask for it to be updated then pass the correct
        # version to set_extrinsics, but then reset it back to what it was, to avoid ruining 
        # the mouse interaction.
        cam_roll = self.camera.GetRoll()
        self.camera.OrthogonalizeViewUp()
        upvec = np.array(self.camera.GetViewUp())
        self.camera.SetViewUp(0,0,1)
        self.camera.SetRoll(cam_roll)

        self.virtualcamera.set_extrinsics(campos,upvec,camtar = camtar,opt_axis=True)

        # Now save!
        try:
            dialog = SaveAsDialog(self,'Camera Definition')
            dialog.exec_()
            if dialog.result() == 1:
                save_name = dialog.name
                del dialog         
                self.virtualcamera.save(save_name)
                dialog = qt.QMessageBox(self)
                dialog.setStandardButtons(qt.QMessageBox.Ok)
                dialog.setWindowTitle('Calcam - Save Complete')
                dialog.setText('Camera definition saved successfully.')
                dialog.setIcon(qt.QMessageBox.Information)
                dialog.exec_()

        except Exception as err:
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Ok)
            dialog.setWindowTitle('Calcam - Save Error')
            dialog.setText('Error saving camera definition:\n' + str(err))
            dialog.setIcon(qt.QMessageBox.Warning)
            dialog.exec_()


# Main calcam window class for actually creating calibrations.
class ImageAnalyserWindow(qt.QMainWindow):
 
    def __init__(self, app, parent = None):

        # GUI initialisation
        qt.QMainWindow.__init__(self, parent)
        qt.uic.loadUi(os.path.join(paths.ui,'image_analyser.ui'), self)

        self.setWindowIcon(qt.QIcon(os.path.join(paths.calcampath,'ui','icon.png')))

        self.app = app

        # See how big the screen is and open the window at an appropriate size
        desktopinfo = self.app.desktop()
        available_space = desktopinfo.availableGeometry(self)
        self.screensize = (available_space.width(),available_space.height())
        # Open the window with same aspect ratio as the screen, and no fewer than 500px tall.
        win_height = max(500,min(780,0.75*available_space.height()))
        win_width = win_height * available_space.width() / available_space.height() 
        self.resize(win_width,win_height)

        # Set up nice exception handling
        sys.excepthook = lambda *ex: show_exception_box(self,*ex)

        # Start up with no CAD model
        self.cadmodel = None

        # We'll be needing a ray caster!
        self.raycaster = raytrace.RayCaster(verbose=False)

        # Set up VTK
        self.qvtkWidget = qt.QVTKRenderWindowInteractor(self.vtkframe)
        self.vtkframe.layout().addWidget(self.qvtkWidget)
        self.pointpicker = vtkinteractorstyles.ImageAnalyser()
        self.qvtkWidget.SetInteractorStyle(self.pointpicker)
        self.renderer_cad = vtk.vtkRenderer()
        self.renderer_cad.SetBackground(0,0,0)
        self.renderer_cad.SetViewport(0.5,0,1,1)
        self.renderer_im = vtk.vtkRenderer()
        self.renderer_im.SetBackground(0,0,0)
        self.renderer_im.SetViewport(0,0,0.5,1)
        self.qvtkWidget.GetRenderWindow().AddRenderer(self.renderer_cad)
        self.qvtkWidget.GetRenderWindow().AddRenderer(self.renderer_im)
        self.vtkInteractor = self.qvtkWidget.GetRenderWindow().GetInteractor()
        self.camera = self.renderer_cad.GetActiveCamera()
        self.any_points = False
        self.pointpairs_changed = False
        self.fit_changed = False

        # Populate CAD model list
        self.model_list = machine_geometry.get_available_models()
        self.model_name.addItems(sorted(self.model_list.keys()))
        self.model_name.setCurrentIndex(-1)
        self.load_model_button.setEnabled(0)


        # Disable image transform buttons if we have no image
        self.image_settings.hide()

        #self.tabWidget.setTabEnabled(1,False)
        self.tabWidget.setTabEnabled(2,False)
        self.tabWidget.setTabEnabled(3,False)

        #self.overlay_oversampling_combobox.setCurrentIndex(2)
        #self.overlay_combobox_options = [0.25, 0.5, 1., 2., 4.]

        # Callbacks for GUI elements
        self.image_sources_list.currentIndexChanged.connect(self.build_imload_gui)
        self.enable_all_button.clicked.connect(self.mass_toggle_model)
        self.disable_all_button.clicked.connect(self.mass_toggle_model)
        self.viewlist.itemClicked.connect(self.change_cad_view)
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
        self.fitresults_name_box.currentIndexChanged.connect(self.load_calibresults)
        self.show_los_checkbox.toggled.connect(self.pointpicker.update_sightlines)
        self.hist_eq_checkbox.stateChanged.connect(self.toggle_hist_eq)
        self.overlay_checkbox.toggled.connect(self.toggle_overlay)
        self.set_viewport_to_calib.clicked.connect(self.pointpicker.set_view_to_fit)

        self.pupil_coords_label.hide()
        self.calinfo_fieldnames.hide()

        # If we have an old version of openCV, histo equilisation won't work :(
        if cv2_version < 2.4 or (cv2_version == 2.4 and cv2_micro_version < 6):
            self.hist_eq_checkbox.setEnabled(False)
            self.hist_eq_checkbox.setToolTip('Requires OpenCV 2.4.6 or newer; you have {:s}'.format(cv2.__version__))


        # Populate image sources list and tweak GUI layout for image loading.
        self.imload_inputs = []

        self.image_load_options.layout().setColumnMinimumWidth(0,100)
        for imsource in image.image_sources:
            self.image_sources_list.addItem(imsource.gui_display_name)
        self.image_sources_list.setCurrentIndex(0)

        self.impos_fieldnames.hide()
        self.sightline_fieldnames.hide()
        self.calib_info.hide()

        # Start the GUI!
        self.show()
        self.pointpicker.DoInit(self.renderer_im,self.renderer_cad,self,self.raycaster)
        self.vtkInteractor.Initialize()

        # Warn the user if we don't have any CAD models
        if self.model_list == {}:
            warn_no_models(self)


    def update_position_info(self,coords_2d,coords_3d,visible):


        cadinfo_str = self.cadmodel.get_position_info(coords_3d)

        sightline_info_string = ''

        iminfo_str = ''

        sightline_fieldnames_str = ''

        impos_fieldnames_str = ''

        for field_index in range(self.raycaster.fitresults.nfields):

            prefix = ' '

            if field_index > 0:
                prefix = prefix + '<br>'
     

            sightline_exists = False

            
            impos_fieldnames_str = impos_fieldnames_str + prefix + '[' + self.raycaster.fitresults.field_names[field_index] + ']&nbsp;'

            if np.any(np.isnan(coords_2d[field_index][0])):
                iminfo_str = iminfo_str + prefix +  'Cursor outside field of view.'
            elif not visible[field_index]:
                iminfo_str = iminfo_str + prefix + 'Cursor hidden from view.'
            else:
                iminfo_str = iminfo_str + prefix + '( {:.0f} , {:.0f} ) px'.format(coords_2d[field_index][0][0],coords_2d[field_index][0][1])
                sightline_exists = True
 

            sightline_fieldnames_str = sightline_fieldnames_str + prefix*2 + '[' + self.raycaster.fitresults.field_names[field_index] + ']&nbsp;'
            if sightline_exists:
                sightline_fieldnames_str = sightline_fieldnames_str + '<br>'
                sightline = coords_3d - self.raycaster.fitresults.get_pupilpos(field=field_index)
                sdir = sightline / np.sqrt(np.sum(sightline**2))

                sightline_info_string = sightline_info_string + prefix*2 + 'Direction X,Y,Z: ( {:.3f} , {:.3f} , {:.3f} )<br>'.format(sdir[0],sdir[1],sdir[2])
                if np.sqrt(np.sum(sightline**2)) < (self.cadmodel.max_ray_length-1e-3):
                    sightline_info_string = sightline_info_string  +'Distance to camera: {:.3f} m'.format(np.sqrt(np.sum(sightline**2)))
                else:
                    sightline_info_string = sightline_info_string + 'Sight line does not inersect CAD model.'
                    cadinfo_str = 'Sight line does not intersect CAD model.'
            else:
                sightline_info_string = sightline_info_string + prefix*2 + 'No line-of-sight to cursor'

        if self.raycaster.fitresults.nfields > 1:
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


    def change_cad_view(self,view_item,init=False):

        if self.sender() is self.viewlist:
            if view_item.isDisabled() or view_item is self.views_root_results or view_item is self.views_root_synthetic or view_item is self.views_root_model:
                return

        if self.sender() is self.viewlist or init:
 
            if view_item.parent() in self.views_results:
                view = fitting.CalibResults(str(view_item.parent().text(0)))
                subfield = view.field_names.index(str(view_item.text(0)))

                self.camera.SetPosition(view.get_pupilpos(field=subfield))
                self.camera.SetFocalPoint(view.get_pupilpos(field=subfield) + view.get_los_direction(view.image_display_shape[0]/2,view.image_display_shape[1]/2))
                self.camera.SetViewAngle(view.get_fov(field=subfield)[1])
                self.camera.SetViewUp(-1.*view.get_cam_to_lab_rotation(field=subfield)[:,1]) 
            elif view_item.parent() is self.views_root_model:

                self.cadmodel.set_default_view(str(view_item.text(0)))

                # Set to that view
                self.camera.SetViewAngle(self.cadmodel.cam_fov_default)
                self.camera.SetPosition(self.cadmodel.cam_pos_default)
                self.camera.SetFocalPoint(self.cadmodel.cam_target_default)
                self.camera.SetViewUp(0,0,1)

            elif view_item.parent() is self.views_root_results or self.views_root_synthetic:

                if view_item.parent() is self.views_root_results:
                    view = fitting.CalibResults(str(view_item.text(0)))
                else:
                    view = fitting.VirtualCalib(str(view_item.text(0)))

                if view.nfields > 1:
                    view_item.setExpanded(not view_item.isExpanded())
                    return

                self.camera.SetPosition(view.get_pupilpos())
                self.camera.SetFocalPoint(view.get_pupilpos() + view.get_los_direction(view.image_display_shape[0]/2,view.image_display_shape[1]/2))
                self.camera.SetViewAngle(view.get_fov()[1])
                self.camera.SetViewUp(-1.*view.get_cam_to_lab_rotation()[:,1])

             

        else:
            self.camera.SetPosition((self.camX.value(),self.camY.value(),self.camZ.value()))
            self.camera.SetFocalPoint((self.tarX.value(),self.tarY.value(),self.tarZ.value()))
            self.camera.SetViewAngle(self.camFOV.value())

        self.update_viewport_info(self.camera.GetPosition(),self.camera.GetFocalPoint(),self.camera.GetViewAngle())

        self.refresh_vtk()



    def populate_model_variants(self):

        model = self.model_list[str(self.model_name.currentText())]
        self.model_variant.clear()
        self.model_variant.addItems(model[1])
        self.model_variant.setCurrentIndex(model[2])
        self.load_model_button.setEnabled(1)

    def change_overlay_oversampling(self):

        if self.overlay_checkbox.isChecked():
            self.toggle_overlay(False)
            self.pointpicker.fit_overlay_actor = None
            self.toggle_overlay(True)
        else:
            self.pointpicker.fit_overlay_actor = None


    def update_checked_features(self,item):

            if self.overlay_checkbox.isChecked():
                self.overlay_checkbox.setChecked(False)
            self.pointpicker.fit_overlay_actor = None
            self.feature_tree.blockSignals(True)
            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            changed_feature = str(item.text(0))
            if changed_feature in self.group_items:
                feature = changed_feature
                changed_feature = []
                for i in range(self.group_items[feature].childCount()):
                    changed_feature.append(str(self.group_items[feature].child(i).text(0)))
                    self.group_items[feature].child(i).setCheckState(0,self.group_items[feature].checkState(0))
                    if self.group_items[feature].checkState(0) == qt.Qt.Checked:
                        self.group_items[feature].child(i).setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                    else:
                        self.group_items[feature].child(i).setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
            else:
                changed_feature = [changed_feature]
                feature = item.parent()
                if feature is not self.treeitem_machine:
                    checkstates = []
                    for i in range(feature.childCount()):
                        checkstates.append(feature.child(i).checkState(0))

                    if len(list(set(checkstates))) > 1:
                        feature.setCheckState(0,qt.Qt.PartiallyChecked)
                        feature.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                    else:
                        feature.setCheckState(0,checkstates[0])
                        if checkstates[0] == qt.Qt.Checked:
                            feature.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                        else:
                            feature.setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)

            if item.checkState(0) == qt.Qt.Checked:
                item.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                self.cadmodel.enable_features(changed_feature,self.renderer_cad)
            else:
                item.setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled)
                self.cadmodel.disable_features(changed_feature,self.renderer_cad)

            self.statusbar.showMessage('Updating bounding box tree...')
            self.raycaster.set_cadmodel(self.cadmodel)
            self.statusbar.clearMessage()
            self.app.restoreOverrideCursor()
            self.refresh_vtk()
            self.feature_tree.blockSignals(False)




    def load_model(self):
        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        model = self.model_list[str(self.model_name.currentText())]

        # Dispose of the old model
        if self.cadmodel is not None:
            
            old_machine_name = self.cadmodel.machine_name
            old_enabled_features = self.cadmodel.get_enabled_features()

            for actor in self.cadmodel.get_vtkActors():
                self.renderer_cad.RemoveActor(actor)
            
            del self.cadmodel
            self.tabWidget.setTabEnabled(2,True)
        else:
            old_machine_name = None

        # Create a new one
        exec('self.cadmodel = machine_geometry.' + model[0] + '("' + str(self.model_variant.currentText()) + '")')
        self.cadmodel.link_gui_window(self)

        if not self.cad_auto_load.isChecked():
            if self.cadmodel.machine_name == old_machine_name:
                self.cadmodel.enable_only(old_enabled_features)
            else:
                for feature in self.cadmodel.features:
                    self.cadmodel.disable_features(feature[0])


        for actor in self.cadmodel.get_vtkActors():
            self.renderer_cad.AddActor(actor)

        self.statusbar.showMessage('Generating CAD bounding box tree...')
        self.raycaster.set_cadmodel(self.cadmodel)

        self.statusbar.showMessage('Setting up CAD model...')

        # Initialise the CAD model setup GUI
        init_model_settings(self)

        # Initialise other lists of things
        init_viewports_list(self)


        # Set selected CAD view to the model's default, if the machine has been changed (i.e. changing model variant will maintain the viewport)
        self.viewlist.clearSelection()
        if self.cadmodel.machine_name != old_machine_name:
            if old_machine_name is not None:
                self.pointpicker.place_cursor_3D([0,0,0],show=False)
                self.pointpicker.clear_cursors_2D()
            for i in range(self.views_root_model.childCount()):
                if str(self.views_root_model.child(i).text(0)) == self.cadmodel.default_view_name:
                    self.viewlist.setCurrentItem(self.views_root_model.child(i))
                    self.change_cad_view(self.views_root_model.child(i),init=True)

        if self.raycaster.fitresults is not None:
            self.overlay_checkbox.setEnabled(True)
            self.set_viewport_to_calib.setEnabled(True)


        self.tabWidget.setTabEnabled(2,True)
        self.enable_all_button.setEnabled(1)
        self.disable_all_button.setEnabled(1)
        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()
        self.refresh_vtk()




    def mass_toggle_model(self):
        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        if self.sender() is self.enable_all_button:
            for i in range(self.treeitem_machine.childCount()):
               self.treeitem_machine.child(i).setCheckState(0,qt.Qt.Checked)
        elif self.sender() is self.disable_all_button:
            for i in range(self.treeitem_machine.childCount()):
                self.treeitem_machine.child(i).setCheckState(0,qt.Qt.Unchecked)
        self.app.restoreOverrideCursor()


    def update_viewport_info(self,campos,camtar,fov):

        self.camX.blockSignals(True)
        self.camY.blockSignals(True)
        self.camZ.blockSignals(True)
        self.tarX.blockSignals(True)
        self.tarY.blockSignals(True)
        self.tarZ.blockSignals(True)
        self.camFOV.blockSignals(True)

        self.camX.setValue(campos[0])
        self.camY.setValue(campos[1])
        self.camZ.setValue(campos[2])
        self.tarX.setValue(camtar[0])
        self.tarY.setValue(camtar[1])
        self.tarZ.setValue(camtar[2])
        self.camFOV.setValue(fov)

        self.camX.blockSignals(False)
        self.camY.blockSignals(False)
        self.camZ.blockSignals(False)
        self.tarX.blockSignals(False)
        self.tarY.blockSignals(False)
        self.tarZ.blockSignals(False)
        self.camFOV.blockSignals(False)




    def refresh_vtk(self,im_only=False):
        if not im_only:
            self.renderer_cad.Render()
        self.renderer_im.Render()
        self.qvtkWidget.update()


    def load_image(self,init_image=None):

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.statusbar.showMessage('Loading image...')

        # Gather up the required input arguments from the image load gui
        imload_options = []
        for option in self.imload_inputs:
            imload_options.append(option[1]())
            if qt.qt_ver == 4:
                if type(imload_options[-1]) == qt.QString:
                    imload_options[-1] = str(imload_options[-1])

        newim = self.imsource(*imload_options)

        self.image_settings.hide()
        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)
        self.pointpicker.fit_overlay_actor = None
        self.tabWidget.setTabEnabled(3,False)

        existing_im_names = paths.get_save_list('Images')
        if newim.name in existing_im_names:
            testim = image.Image(newim.name)
            if not np.all(newim.data == testim.data):
                i = 0
                new_name = newim.name
                while new_name in existing_im_names:
                    i = i + 1
                    new_name = newim.name + '({:d})'.format(i)
                newim.name = new_name

        self.image = newim

        self.image_settings.show()
        self.hist_eq_checkbox.setCheckState(qt.Qt.Unchecked)
        self.pointpicker.init_image(self.image)

        if self.raycaster.fitresults is not None:
            self.raycaster.fitresults = None
            self.impos_info.setText('Image position info will appear here.')
            self.sightline_info.setText('Sightline info will appear here.')
            self.impos_fieldnames.hide()
            self.pupil_coords_label.hide()
            self.calinfo_fieldnames.hide()
            self.sightline_fieldnames.hide()
            self.populate_fitresults_list()
            self.pointpicker.clear_cursors_2D()
        else:
            self.populate_fitresults_list()

        self.tabWidget.setTabEnabled(3,True)
        self.update_image_info_string()
        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()


    def populate_fitresults_list(self):
        res_list = []
        for res_save in paths.get_save_list('FitResults'):
            try:
                res = fitting.CalibResults(res_save)
                if np.all(res.transform.get_display_shape() == self.image.transform.get_display_shape()):
                    res_list.append(res_save)
            except:
                pass

        self.fitresults_name_box.blockSignals(True)
        self.fitresults_name_box.clear()
        self.fitresults_name_box.addItems(res_list)
        self.fitresults_name_box.setCurrentIndex(-1)
        self.fitresults_name_box.blockSignals(False)

    def load_calibresults(self):
        res = fitting.CalibResults(str(self.fitresults_name_box.currentText()))
        self.raycaster.set_calibration(res)
        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)
        self.pointpicker.fit_overlay_actor = None
        
        pupilpos_str = ''
        fieldnames_str = ''
        for field in range(res.nfields):
            pupilpos = res.get_pupilpos(field=field)
            if field > 0:
                prefix = '<br>'
            else:
                prefix = ''
            fieldnames_str = fieldnames_str + prefix + '[{:s}]&nbsp;'.format(res.field_names[field])
            pupilpos_str = pupilpos_str + prefix + '( {:.3f} , {:.3f} , {:.3f} ) m'.format(pupilpos[0],pupilpos[1],pupilpos[2])

        self.calinfo_fieldnames.setText(fieldnames_str)
        self.pupil_coords_label.setText(pupilpos_str)
        if res.nfields > 1:
            self.calinfo_fieldnames.show()
        else:
            self.calinfo_fieldnames.hide()
        self.pupil_coords_label.show()
        
        self.calib_info.show()
        if self.cadmodel is not None:
            self.set_viewport_to_calib.setEnabled(True)
            self.overlay_checkbox.setEnabled(True)

        if self.pointpicker.cursor3D is not None:
            self.pointpicker.update_2D_from_3D(self.pointpicker.cursor3D[0].GetFocalPoint())

    def update_image_info_string(self):

        if np.any(self.image.transform.get_display_shape() != list(self.image.data.shape[1::-1])):
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP) [ As Displayed ]<br>{3:d} x {4:d} pixels ({5:.1f} MP) [ Raw Data ]<br>'.format(self.image.transform.get_display_shape()[0],self.image.transform.get_display_shape()[1],np.prod(self.image.transform.get_display_shape()) / 1e6 ,self.image.data.shape[1],self.image.data.shape[0],np.prod(self.image.data.shape[:2]) / 1e6 )
        else:
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP)<br>'.format(self.image.transform.get_display_shape()[0],self.image.transform.get_display_shape()[1],np.prod(self.image.transform.get_display_shape()) / 1e6 )
        
        if len(self.image.data.shape) == 2:
            info_str = info_str + 'Monochrome'
        elif len(self.image.data.shape) == 3 and self.image.data.shape[2] == 3:
            info_str = info_str + 'RGB Colour'
        elif len(self.image.data.shape) == 3 and self.image.data.shape[2] == 3:
            info_str = info_str + 'RGB Colour'

        self.image_info.setText(info_str)


    def browse_for_file(self):

        for i,option in enumerate(self.imload_inputs):
            if self.sender() in option[0]:
                filename_filter = self.imsource.gui_inputs[i]['filter']
                target_textbox = option[0][2]


        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(0)
        filedialog.setFileMode(1)
        filedialog.setWindowTitle('Select File')
        filedialog.setNameFilter(filename_filter)
        filedialog.setLabelText(3,'Select')
        filedialog.exec_()
        if filedialog.result() == 1:
            target_textbox.setText(str(filedialog.selectedFiles()[0]))


    def transform_image(self,data):


        if self.sender() is self.im_flipud:
            if len(self.image.transform.transform_actions) > 0:
                if self.image.transform.transform_actions[-1] == 'flip_up_down':
                    del self.image.transform.transform_actions[-1]
                else:
                    self.image.transform.transform_actions.append('flip_up_down')
            else:
                self.image.transform.transform_actions.append('flip_up_down')
            
        elif self.sender() is self.im_fliplr:
            if len(self.image.transform.transform_actions) > 0:
                if self.image.transform.transform_actions[-1] == 'flip_left_right':
                    del self.image.transform.transform_actions[-1]
                else:
                    self.image.transform.transform_actions.append('flip_left_right')
            else:
                self.image.transform.transform_actions.append('flip_left_right')

        elif self.sender() is self.im_rotate_button:
            if len(self.image.transform.transform_actions) > 0:
                if 'rotate_clockwise' in self.image.transform.transform_actions[-1]:
                    current_angle = int(self.image.transform.transform_actions[-1].split('_')[2])
                    del self.image.transform.transform_actions[-1]
                    new_angle = self.im_rotate_angle.value()
                    total_angle = current_angle + new_angle
                    if total_angle > 270:
                        total_angle = total_angle - 360

                    if new_angle > 0:
                        self.image.transform.transform_actions.append('rotate_clockwise_' + str(total_angle))
                else:
                    self.image.transform.transform_actions.append('rotate_clockwise_' + str(self.im_rotate_angle.value()))
            else:
                self.image.transform.transform_actions.append('rotate_clockwise_' + str(self.im_rotate_angle.value()))

        elif self.sender() is self.im_y_stretch_button:
            sideways = False
            for action in self.image.transform.transform_actions:
                if action.lower() in ['rotate_clockwise_90','rotate_clockwise_270']:
                    sideways = not sideways
            if sideways:
                self.image.transform.pixel_aspectratio = self.image.transform.pixel_aspectratio/self.im_y_stretch_factor.value()
            else:
                self.image.transform.pixel_aspectratio = self.image.transform.pixel_aspectratio*self.im_y_stretch_factor.value()

        elif self.sender() is self.im_reset:
            self.image.transform.transform_actions = []
            self.image.transform.pixel_aspectratio = 1

        if self.raycaster.fitresults is not None:
            self.raycaster.fitresults = None
            self.impos_info.setText('Image position info will appear here.')
            self.sightline_info.setText('Sightline info will appear here.')
            self.impos_fieldnames.hide()
            self.pupil_coords_label.hide()
            self.calinfo_fieldnames.hide()
            self.sightline_fieldnames.hide()
            self.populate_fitresults_list()
            self.pointpicker.clear_cursors_2D()
        else:
            self.populate_fitresults_list()


        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)


        # Update the image and point pairs
        self.pointpicker.init_image(self.image,hold_position=True)
        self.update_image_info_string()





    def toggle_overlay(self,show):

        if show:

            if self.pointpicker.fit_overlay_actor is None:

                oversampling = 1.#self.overlay_combobox_options[self.overlay_oversampling_combobox.currentIndex()]
                self.statusbar.showMessage('Rendering wireframe overlay...')
                self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
                self.app.processEvents()
                try:
                    OverlayImage = image.from_array(render.render_cam_view(self.cadmodel,self.raycaster.fitresults,Edges=True,Transparency=True,Verbose=False,EdgeColour=(0,0,1),oversampling=oversampling,ScreenSize=self.screensize))

                    self.pointpicker.fit_overlay_actor = OverlayImage.get_vtkActor()
                    self.pointpicker.fit_overlay_actor.SetPosition(0.,0.,0.01)
                    self.pointpicker.Renderer_2D.AddActor(self.pointpicker.fit_overlay_actor)

                    self.refresh_vtk()
                        

                except MemoryError:
                    self.pointpicker.fit_overlay_actor = None
                    dialog = qt.QMessageBox(self)
                    dialog.setStandardButtons(qt.QMessageBox.Ok)
                    dialog.setWindowTitle('Calcam - Memory Error')
                    dialog.setTextFormat(qt.Qt.RichText)
                    dialog.setText('Insufficient memory to render wireframe overlay.')
                    text = 'Try using a lower resolution setting for the overlay.'
                    if sys.maxsize < 2**32:
                        text = text + ' Switching to 64-bit python is highly recommended when working with large data.'
                    dialog.setInformativeText(text)
                    dialog.setIcon(qt.QMessageBox.Warning)
                    dialog.exec_()
                    self.overlay_checkbox.setChecked(False) 
                
                except:
                    self.pointpicker.fit_overlay_actor = None
                    self.statusbar.clearMessage()
                    self.overlay_checkbox.setChecked(False) 
                    self.app.restoreOverrideCursor()
                    raise


                self.statusbar.clearMessage()
                self.app.restoreOverrideCursor()

            else:

                self.pointpicker.Renderer_2D.AddActor(self.pointpicker.fit_overlay_actor)
                self.refresh_vtk()

        else:
            
            self.pointpicker.Renderer_2D.RemoveActor(self.pointpicker.fit_overlay_actor)
            self.refresh_vtk()   




    def update_cad_status(self,message):

        if message is not None:
            self.statusbar.showMessage(message)
            self.app.processEvents()
        else:
            self.statusbar.clearMessage()
            self.app.processEvents()


    def toggle_hist_eq(self,check_state):

        # Enable / disable adaptive histogram equalisation
        if check_state == qt.Qt.Checked:
            self.image.postprocessor = image_filters.hist_eq()
        else:
            self.image.postprocessor = None

        self.pointpicker.init_image(self.image,hold_position=True)




    def build_imload_gui(self,index):

        layout = self.image_load_options.layout()
        for widgets,_ in self.imload_inputs:
            for widget in widgets:
                layout.removeWidget(widget)
                widget.close()

        #layout = qt.QGridLayout(self.image_load_options)
        self.imsource = image.image_sources[index]

        self.imload_inputs = []

        row = 0
        for option in self.imsource.gui_inputs:

            labelwidget = qt.QLabel(option['label'] + ':')
            layout.addWidget(labelwidget,row,0)

            if option['type'] == 'filename':
                button = qt.QPushButton('Browse...')
                button.clicked.connect(self.browse_for_file)
                button.setMaximumWidth(80)
                layout.addWidget(button,row+1,1)
                fname = qt.QLineEdit()
                if 'default' in option:
                    fname.setText(option['default'])
                layout.addWidget(fname,row,1)
                self.imload_inputs.append( ([labelwidget,button,fname],fname.text ) )
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
                self.imload_inputs.append( ([labelwidget,valbox],valbox.value ) )
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
                self.imload_inputs.append( ([labelwidget,valbox],valbox.value ) )
                row = row + 1
            elif option['type'] == 'string':
                ted = qt.QLineEdit()
                if 'default' in option:
                    ted.setText(option['default'])
                layout.addWidget(ted,row,1)
                self.imload_inputs.append( ([labelwidget,ted],ted.text ) )
                row = row + 1
            elif option['type'] == 'bool':
                checkbox = qt.QCheckBox()
                if 'default' in option:
                    checkbox.setChecked(option['default'])
                layout.addWidget(checkbox,row,1)
                self.imload_inputs.append( ([labelwidget,checkbox],checkbox.isChecked ) )
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
                self.imload_inputs.append( ([labelwidget,cb],cb.currentText) )
                row = row + 1

 


    def update_pixel_size(self):
        if self.pixel_size_checkbox.isChecked():
            self.image.pixel_size = self.pixel_size_box.value() / 1e6
        else:
            self.image.pixel_size = None






# Class for the window
class LauncherWindow(qt.QDialog):
 
    def __init__(self, app, parent = None):

        # GUI initialisation
        qt.QDialog.__init__(self, parent)
        # Load the Qt designer file, assumed to be in the same directory as this python file and named gui.ui.
        qt.uic.loadUi(os.path.join(paths.ui,'launcher.ui'), self)
        
        self.setWindowIcon(qt.QIcon(os.path.join(paths.calcampath,'ui','icon.png')))
        self.setWindowTitle('Calcam  v{:s}'.format(__version__))
        self.layout().setSizeConstraint(qt.QLayout.SetFixedSize)

        self.app = app
        
        immap = qt.QPixmap(os.path.join(paths.ui,'logo.png'))
        self.logolabel.setPixmap(immap)

        # Callbacks for GUI elements: connect the buttons to the functions we want to run
        self.calcam_button.clicked.connect(self.launch_calcam)
        self.alignment_calib_button.clicked.connect(self.launch_alignment_calib)
        self.cad_viewer_button.clicked.connect(self.launch_cad_viewer)
        self.view_designer_button.clicked.connect(self.launch_view_designer)
        self.userguide_button.clicked.connect(self.open_manual)
        self.image_analysis_button.clicked.connect(self.launch_image_analysis)

        # Open the window!
        self.show()

        self.devnull = open(os.devnull,'wb')

    def launch_calcam(self):
        subprocess.Popen([sys.executable,os.path.join(paths.calcampath,'gui.py'),'launch_calcam'],stdin=None, stdout=self.devnull, stderr=self.devnull)

    def launch_cad_viewer(self):
        subprocess.Popen([sys.executable,os.path.join(paths.calcampath,'gui.py'),'launch_cad_viewer'],stdin=None, stdout=self.devnull, stderr=self.devnull)

    def launch_view_designer(self):
        subprocess.Popen([sys.executable,os.path.join(paths.calcampath,'gui.py'),'launch_view_designer'],stdin=None, stdout=self.devnull, stderr=self.devnull)

    def launch_alignment_calib(self):
        subprocess.Popen([sys.executable,os.path.join(paths.calcampath,'gui.py'),'launch_alignment_calib'],stdin=None, stdout=self.devnull, stderr=self.devnull)

    def open_manual(self):
        webbrowser.open('https://euratom-software.github.io/calcam/')

    def launch_image_analysis(self):
        subprocess.Popen([sys.executable,os.path.join(paths.calcampath,'gui.py'),'launch_image_analysis'],stdin=None, stdout=self.devnull, stderr=self.devnull)

def start_gui():
    app = qt.QApplication([''])
    window = LauncherWindow(app)   
    window.exec_()



# Rotate a given point about the given axis by the given angle
def rotate_3D(vect,axis,angle):

    vect = np.array(vect)
    vect_ = np.matrix(np.zeros([3,1]))
    vect_[0,0] = vect[0]
    vect_[1,0] = vect[1]
    vect_[2,0] = vect[2]
    axis = np.array(axis)

    # Put angle in radians
    angle = angle * 3.14159 / 180.

    # Make sure the axis is normalised
    axis = axis / np.sqrt(np.sum(axis**2))

    # Make a rotation matrix!
    R = np.matrix(np.zeros([3,3]))
    R[0,0] = np.cos(angle) + axis[0]**2*(1 - np.cos(angle))
    R[0,1] = axis[0]*axis[1]*(1 - np.cos(angle)) - axis[2]*np.sin(angle)
    R[0,2] = axis[0]*axis[2]*(1 - np.cos(angle)) + axis[1]*np.sin(angle)
    R[1,0] = axis[1]*axis[0]*(1 - np.cos(angle)) + axis[2]*np.sin(angle)
    R[1,1] = np.cos(angle) + axis[1]**2*(1 - np.cos(angle))
    R[1,2] = axis[1]*axis[2]*(1 - np.cos(angle)) - axis[0]*np.sin(angle)
    R[2,0] = axis[2]*axis[0]*(1 - np.cos(angle)) - axis[1]*np.sin(angle)
    R[2,1] = axis[2]*axis[1]*(1 - np.cos(angle)) + axis[0]*np.sin(angle)
    R[2,2] = np.cos(angle) + axis[2]**2*(1 - np.cos(angle))

    return np.array( R * vect_)


def object_from_file(parent,obj_type,multiple=False):

    if obj_type.lower() == 'calibration':
        filename_filter = 'Calcam Calibration (*.pickle)'
        start_dir = paths.root

    filedialog = qt.QFileDialog(parent)
    filedialog.setAcceptMode(0)

    if multiple:
        filedialog.setFileMode(3)
        empty_ret = []
    else:
        filedialog.setFileMode(1)
        empty_ret = None

    filedialog.setWindowTitle('Open...')
    filedialog.setNameFilter(filename_filter)
    filedialog.exec_()
    if filedialog.result() == 1:
        selected_paths = filedialog.selectedFiles()
    else:
        return empty_ret


    objs = []
    for path in [str(p) for p in selected_paths]:

        if obj_type.lower() == 'calibration':
            name = os.path.split(path)[-1].replace('.pickle','')
            try:
                obj = fitting.CalibResults(name)
            except:
                obj = fitting.VirtualCalib(name)

            obj.name = name


        objs.append(obj)

    if multiple:
        return objs
    else:
        return objs[0]


def get_save_filename(parent,obj_type):

    if obj_type.lower() == 'calibration':
        filename_filter = 'Calcam Calibration (*.pickle)'
        start_dir = paths.root

    filedialog = qt.QFileDialog(parent)
    filedialog.setAcceptMode(0)
    filedialog.setFileMode(1)
    filedialog.setWindowTitle('Open...')
    filedialog.setNameFilter(filename_filter)
    filedialog.exec_()
    if filedialog.result() == 1:
        path = str(filedialog.selectedFiles()[0])
    else:
        return None

    if obj_type.lower() == 'calibration':
        name = os.path.split(path)[-1].replace('.pickle','')
        try:
            obj = fitting.CalibResults(name)
        except:
            obj = fitting.VirtualCalib(name)

    obj.name = name

    return obj


def pick_colour(parent,init_colour):

    col_init = np.array(init_colour) * 255

    dialog = qt.QColorDialog(qt.QColor(col_init[0],col_init[1],col_init[2]),parent)
    res = dialog.exec_()

    if res:
        ret_col = ( dialog.currentColor().red() / 255. , dialog.currentColor().green() / 255. , dialog.currentColor().blue() / 255.)
    else:
        ret_col = None

    del dialog

    return ret_col


class colourcycle():

    def __init__(self):

        self.colours = [(0.121,0.466,0.705),
                        (1,0.498,0.054),
                        (0.172,0.627,0.172),
                        (0.829,0.152,0.156),
                        (0.580,0.403,0.741),
                        (0.549,0.337,0.294),
                        (0.890,0.466,0.760),
                        (0.498,0.498,0.498),
                        (0.737,0.741,0.133),
                        (0.09,0.745,0.811),
                        ]

        self.extra_colours = []

        self.next_index = 0

        self.next = self.__next__

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.extra_colours) > 0:
            return self.extra_colours.pop()
        else:
            col = self.colours[self.next_index]
            self.next_index = self.next_index + 1
            if self.next_index > len(self.colours) - 1:
                self.next_index = 0 
            return col

    def queue_colour(self,colour):
        self.extra_colours.insert(0,colour)



# Custom dictionary-like storage class.
# Behaves more-or-less like a dictionary but without the requirement
# that the keys are hashable. Needed so I can do things like use
# QTreeWidgetItems as keys.
class DodgyDict():

    def __init__(self):

        self.keylist = []
        self.itemlist = []
        self.iter_index = 0
        self.next = self.__next__

    def __getitem__(self,key):
        for i,ikey in enumerate(self.keylist):
            if key == ikey:
                return self.itemlist[i]
        raise IndexError()

    def __setitem__(self,key,value):

        for i,ikey in enumerate(self.keylist):
            if key == ikey:
                self.itemlist[i] = value
                return

        self.keylist.append(key)
        self.itemlist.append(value)

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index > len(self.keys()) - 1: 
            raise StopIteration
        else:
            self.iter_index += 1
            return (self.keylist[self.iter_index-1],self.itemlist[self.iter_index-1])

    def keys(self):
        return self.keylist


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1].lower() == 'launch_calcam':
            start_calcam()
        elif sys.argv[1].lower() == 'launch_cad_viewer':
            start_cad_viewer()
        elif sys.argv[1].lower() == 'launch_view_designer':
            start_view_designer()
        elif sys.argv[1].lower() == 'launch_alignment_calib':
            start_alignment_calib()
        elif sys.argv[1].lower() == 'launch_image_analysis':
            start_image_analysis()
        else:
            start_gui()     
    else:
        start_gui()