# External imports
import sys
import os
import numpy as np
import traceback
import vtk
import cv2
import time

# Calcam imports
from . import qt_wrapper as qt
from ..cadmodel import CADModel
from ..config import CalcamConfig
from ..calibration import Calibration
from ..pointpairs import PointPairs
from ..coordtransformer import CoordTransformer
from .vtkinteractorstyles import CalcamInteractorStyle2D

guipath = os.path.split(os.path.abspath(__file__))[0]


def open_gui(window_class):
    app = qt.QApplication([])
    win = window_class(app)
    if qt.QDialog in window_class.__bases__:
        return win.exec_()
    else:
        return app.exec_()


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



class CalcamGUIWindow(qt.QMainWindow):


    def init(self, ui_filename, app, parent):

        # GUI initialisation
        qt.QMainWindow.__init__(self, parent)
        qt.uic.loadUi(os.path.join(guipath,ui_filename), self)

        self.setWindowIcon(qt.QIcon(os.path.join(guipath,'icon.png')))

        self.app = app

        self.config = CalcamConfig()

        # See how big the screen is and open the window at an appropriate size
        desktopinfo = self.app.desktop()
        available_space = desktopinfo.availableGeometry(self)

        # Open the window with same aspect ratio as the screen, and no fewer than 500px tall.
        win_height = max(500,min(780,0.75*available_space.height()))
        win_width = win_height * available_space.width() / available_space.height() 
        self.resize(win_width,win_height)

         # Let's show helpful dialog boxes if we have unhandled exceptions:
        sys.excepthook = self.show_exception_dialog

        try:
            self.action_new.setIcon( qt.QIcon(os.path.join(guipath,'new.png')) )
            self.action_open.setIcon( qt.QIcon(os.path.join(guipath,'open.png')) )
            self.action_save.setIcon( qt.QIcon(os.path.join(guipath,'save.png')) )
            self.action_save_as.setIcon( qt.QIcon(os.path.join(guipath,'saveas.png')) )
        except AttributeError:
            pass

        # -------------------- Initialise View List ------------------
        self.viewlist.clear()

        # Populate viewports list
        self.views_root_model = qt.QTreeWidgetItem(['Defined in Model'])
        self.views_root_auto = qt.QTreeWidgetItem(['Auto Cross-Sections'])
        self.views_root_results = qt.QTreeWidgetItem(['From Calibrations'])


        # Auto type views
        item = qt.QTreeWidgetItem(self.views_root_auto,['Vertical cross-section'])
        item = qt.QTreeWidgetItem(self.views_root_auto,['Horizontal cross-section'])

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

        self.views_root_auto.setHidden(True)
        # ------------------------------------------------------------


    # Handle exceptions with a dialog giving the user (hopefully) useful information about the error that occured.
    def show_exception_dialog(self,excep_type,excep_value,tb):

        UserCode = False
        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()

        # First, see if we can blame code the user has written and plugged in.
        for traceback_line in traceback.format_exception(excep_type,excep_value,tb):
            if 'dummystring' in traceback_line or 'otherdummystring' in traceback_line:
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



    def update_vtk_size(self,vtksize):

        self.vtksize = vtksize


    def save_view_to_model(self):

        if str(self.view_save_name.text()) in self.cadmodel.get_view_names():

            msg = 'A view with this name already exists in the model definition. Are you sure you want to over-write the existing one?'
            reply = qt.QMessageBox.question(self, 'Overwrite?', msg, qt.QMessageBox.Yes, qt.QMessageBox.No)

            if reply == qt.QMessageBox.No:
                return

        cam_pos = (self.camX.value(),self.camY.value(),self.camZ.value())
        target = (self.tarX.value(), self.tarY.value(), self.tarZ.value())
        fov = self.camFOV.value()
        xsection = self.interactor3d.get_xsection()
        projection = self.interactor3d.projection
        roll = self.cam_roll.value()

        try:
            self.cadmodel.add_view(str(self.view_save_name.text()),cam_pos,target,fov,xsection,roll,projection)
            self.update_model_views()

        except:
            self.update_model_views()
            raise


    def pick_colour(self,init_colour):

        col_init = np.array(init_colour) * 255

        dialog = qt.QColorDialog(qt.QColor(col_init[0],col_init[1],col_init[2]),self)
        res = dialog.exec_()

        if res:
            ret_col = ( dialog.currentColor().red() / 255. , dialog.currentColor().green() / 255. , dialog.currentColor().blue() / 255.)
        else:
            ret_col = None

        del dialog

        return ret_col

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



    def object_from_file(self,obj_type,multiple=False):

        filename_filter = self.config.filename_filters[obj_type]

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(0)

        try:
           filedialog.setDirectory(self.config.file_dirs[obj_type])
        except KeyError:
            filedialog.setDirectory(os.path.expanduser('~'))

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

        # If we have selected one or more files...
        self.config.file_dirs[obj_type] = os.path.split(str(selected_paths[0]))[0]

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        objs = []
        
        for path in [str(p) for p in selected_paths]:

            if obj_type.lower() == 'calibration':
                obj = Calibration(path)
                obj.filename = path
                
            elif obj_type.lower() == 'pointpairs':
                if path.endswith('.ccc'):
                    obj = Calibration(path).pointpairs
                    obj.history = obj.history['pointpairs']
                elif path.endswith('.csv'):
                    with open(path,'r') as ppf:
                        obj = PointPairs(ppf)
                        obj.src = 'Loaded from Calcam 1.x file "{:s}"'.format(os.path.split(path)[-1])

            objs.append(obj)
        self.app.restoreOverrideCursor()
        if multiple:
            return objs
        else:
            return objs[0]



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
        


    def get_save_filename(self,obj_type):

        filename_filter = self.config.filename_filters[obj_type]
        fext = filename_filter.split('(*')[1].split(')')[0]

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(1)

        try:
           filedialog.setDirectory(self.config.file_dirs[obj_type])
        except KeyError:
            filedialog.setDirectory(os.path.expanduser('~'))
        

        filedialog.setFileMode(0)

        filedialog.setWindowTitle('Save As...')
        filedialog.setNameFilter(filename_filter)
        filedialog.exec_()
        if filedialog.result() == 1:
            selected_path = str(filedialog.selectedFiles()[0])
            self.config.file_dirs[obj_type] = os.path.split(selected_path)[0]
            if not selected_path.endswith(fext):
                selected_path = selected_path + fext
            return selected_path
        else:
            return None



    def update_model_views(self):

        self.viewlist.selectionModel().clearSelection()
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

            if view_item.parent() is self.views_root_model:

                view = self.cadmodel.get_view( str(view_item.text(0)))

                # Set to that view
                self.camera_3d.SetPosition(view['cam_pos'])
                self.camera_3d.SetFocalPoint(view['target'])
                self.camera_3d.SetViewUp(0,0,1)
                self.interactor3d.set_xsection(view['xsection'])
                self.interactor3d.set_roll(view['roll'])
                self.interactor3d.set_projection(view['projection'])
                self.interactor3d.set_fov(view['y_fov'])

            elif view_item.parent() is self.views_root_results or view_item.parent() in self.viewport_calibs.keys():

                view,subfield = self.viewport_calibs[(view_item)]
                
                if subfield is not None:
                    self.set_view_from_calib(view,subfield)

            self.update_viewport_info(keep_selection=True)


        else:
            self.camera_3d.SetPosition((self.camX.value(),self.camY.value(),self.camZ.value()))
            self.camera_3d.SetFocalPoint((self.tarX.value(),self.tarY.value(),self.tarZ.value()))
            self.interactor3d.set_roll(-self.cam_roll.value())
            try:
                self.interactor3d.set_fov(self.camFOV.value())
            except AttributeError:
                pass

        
        self.interactor3d.update_cursor_style()
        self.interactor3d.update_clipping()

        self.refresh_3d()



    def set_view_from_calib(self,calibration,subfield):

        viewmodel = calibration.view_models[subfield]
        fov = calibration.get_fov(subview=subfield,fullchip=True)
        vtk_aspect = float(self.vtksize[1]) / float(self.vtksize[0])
        fov_aspect = float(calibration.geometry.y_pixels) / float(calibration.geometry.x_pixels)
        if vtk_aspect > fov_aspect:
            h_fov = True
            fov_angle = fov[0]
        else:
            h_fov = False
            fov_angle = fov[1]

        self.camera_3d.SetPosition(viewmodel.get_pupilpos())
        self.camera_3d.SetFocalPoint(viewmodel.get_pupilpos() + viewmodel.get_los_direction(calibration.geometry.get_display_shape()[0]/2,calibration.geometry.get_display_shape()[1]/2))
        
        self.camera_3d.SetUseHorizontalViewAngle(h_fov)
        self.camera_3d.SetViewAngle(fov_angle)
        self.camera_3d.SetUseHorizontalViewAngle(h_fov)
        
        if np.isfinite(viewmodel.get_cam_roll()):
            self.cam_roll.setValue(viewmodel.get_cam_roll())
        else:
            self.cam_roll.setValue(0)
            self.camera_3d.SetViewUp(-1.*viewmodel.get_cam_to_lab_rotation()[:,1])
        
        self.interactor3d.set_xsection(None)       

        self.update_viewport_info(keep_selection=True)
        self.interactor3d.update_cursor_style()

        self.interactor3d.update_clipping()

        self.refresh_3d()


    def colour_model_by_material(self):

            if self.cadmodel is not None:
                self.cadmodel.colour_by_material()
                self.refresh_3d()


    def set_cad_colour(self):

        selected_features = []
        for treeitem in self.feature_tree.selectedItems():
            selected_features.append(self.cad_tree_items[treeitem])

        # Note: this does not mean nothing is selected;
        # rather it means the root of the model is selected!
        if None in selected_features:
            selected_features = None

        if self.sender() is self.cad_colour_choose_button:

            picked_colour = self.pick_colour(self.cadmodel.get_colour( selected_features )[0] )

            if picked_colour is not None:

                self.cadmodel.set_colour(picked_colour,selected_features)


        elif self.sender() is self.cad_colour_reset_button:

            self.cadmodel.reset_colour(selected_features)

        self.refresh_3d()




    def load_viewport_calib(self):
        cals = self.object_from_file('calibration',multiple=True)

        for cal in cals:
            listitem = qt.QTreeWidgetItem(self.views_root_results,[cal.name])
            if cal.n_subviews > 1:
                self.viewport_calibs[(listitem)] = (cal,None)
                listitem.setExpanded(True)
                for n,fieldname in enumerate(cal.subview_names):
                    self.viewport_calibs[ (qt.QTreeWidgetItem(listitem,[fieldname])) ] = (cal,n) 
            else:
                self.viewport_calibs[(listitem)] = (cal,0)

        if len(cals) > 0:
            self.views_root_results.setHidden(False)


    def populate_model_variants(self):

        model = self.model_list[str(self.model_name.currentText())]
        self.model_variant.clear()
        self.model_variant.addItems(model[1])
        self.model_variant.setCurrentIndex(model[1].index(model[2]))
        self.load_model_button.setEnabled(1)


    def update_checked_features(self,item):

        self.cadmodel.set_features_enabled(item.checkState(0) == qt.Qt.Checked,self.cad_tree_items[item])
        self.update_feature_tree_checks()

        self.refresh_3d()

        self.on_change_cad_features()



    def on_change_cad_features(self):
        pass


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


    def load_model(self,data=None,featurelist=None):

        # Dispose of the old model
        if self.cadmodel is not None:

            self.cadmodel.remove_from_renderer(self.renderer_3d)
            self.cadmodel.unload()

            del self.cadmodel


        # Create a new one
        self.cadmodel = CADModel( str(self.model_name.currentText()) , str(self.model_variant.currentText()) , self.update_cad_status)

        self.config.default_model = (str(self.model_name.currentText()),str(self.model_variant.currentText()))

        if featurelist is not None:
            self.cadmodel.enable_only(featurelist)

        elif not self.cad_auto_load.isChecked():
            self.cadmodel.set_features_enabled(False)

        self.cadmodel.add_to_renderer(self.renderer_3d)

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


        # Make sure the light lights up the whole model without annoying shadows or falloff.
        light = self.renderer_3d.GetLights().GetItemAsObject(0)
        light.PositionalOn()
        light.SetConeAngle(180)

        # Put the camera in some reasonable starting position
        self.camera_3d.SetViewAngle(90)
        self.camera_3d.SetViewUp((0,0,1))
        self.camera_3d.SetFocalPoint(0,0,0)

        self.update_model_views()

        if self.cadmodel.initial_view is not None:
            for i in range(self.views_root_model.childCount()):
                if self.views_root_model.child(i).text(0) == self.cadmodel.initial_view:
                    self.views_root_model.child(i).setSelected(True)
                    break
        else:
            model_extent = self.cadmodel.get_extent()
            if np.abs(model_extent).max() > 0:
                self.camera_3d.SetPosition(((model_extent[5] - model_extent[4])/2,(model_extent[2]+model_extent[3])/2,(model_extent[4]+model_extent[5])/2))
            else:
                self.camera_3d.SetPosition((3.,0,0))

        self.statusbar.clearMessage()
        self.interactor3d.update_clipping()

        self.refresh_3d()

        self.app.restoreOverrideCursor()

        self.on_model_load()



    def update_viewport_info(self,keep_selection = False):

        campos = self.camera_3d.GetPosition()
        camtar = self.camera_3d.GetFocalPoint()
        if self.interactor3d.projection == 'perspective':
            fov = self.camera_3d.GetViewAngle()
            fov_suffix = u'\xb0'
            fov_max = 110
            fov_min = 1
            decimals = 1
        elif self.interactor3d.projection == 'orthographic':
            fov = self.camera_3d.GetParallelScale()*2
            fov_suffix = ' m'
            fov_max = 200
            fov_min = 0.01
            decimals = 2

        roll = -self.interactor3d.cam_roll

        self.camX.blockSignals(True)
        self.camY.blockSignals(True)
        self.camZ.blockSignals(True)
        self.tarX.blockSignals(True)
        self.tarY.blockSignals(True)
        self.tarZ.blockSignals(True)
        self.cam_roll.blockSignals(True)
        try:
            self.camFOV.blockSignals(True)
        except AttributeError:
            pass

        self.camX.setValue(campos[0])
        self.camY.setValue(campos[1])
        self.camZ.setValue(campos[2])
        self.tarX.setValue(camtar[0])
        self.tarY.setValue(camtar[1])
        self.tarZ.setValue(camtar[2])
        self.cam_roll.setValue(roll)

        try:
            self.camFOV.setSuffix(fov_suffix)
            self.camFOV.setMinimum(fov_min)
            self.camFOV.setMaximum(fov_max)
            self.camFOV.setDecimals(decimals)
            self.camFOV.setValue(fov)
        except AttributeError:
            pass

        self.camX.blockSignals(False)
        self.camY.blockSignals(False)
        self.camZ.blockSignals(False)
        self.tarX.blockSignals(False)
        self.tarY.blockSignals(False)
        self.tarZ.blockSignals(False)
        self.cam_roll.blockSignals(False)
        try:
            self.camFOV.blockSignals(False)
        except AttributeError:
            pass

        if not keep_selection:
            self.viewlist.clearSelection() 



    def update_cad_status(self,message):

        if message is not None:
            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.statusbar.showMessage(message)
            self.app.processEvents()
        else:
            self.app.restoreOverrideCursor()
            self.statusbar.clearMessage()
            self.app.processEvents()



    # Populate CAD model list
    def populate_models(self):
        self.model_list = self.config.get_cadmodels()

        self.model_name.addItems(sorted(self.model_list.keys()))

        set_model = False
        set_variant = False
        self.load_model_button.setEnabled(0)

        if self.config.default_model is not None:
            for i,mname in enumerate(sorted(self.model_list.keys())):
                if mname == self.config.default_model[0]:

                    self.model_name.setCurrentIndex(i)
                    self.populate_model_variants()
                    set_model = True
                    break

            for j, vname in enumerate(self.model_list[mname][1]):
                if self.config.default_model[1] == vname:

                    self.model_variant.setCurrentIndex(j)
                    set_variant = True
                    break
        
        if not set_model:
            self.model_name.setCurrentIndex(-1)
        if not set_variant:
            self.model_variant.setCurrentIndex(-1)


    def refresh_3d(self):

        self.renderer_3d.Render()
        self.qvtkwidget_3d.update()


    def refresh_2d(self):
        self.renderer_2d.Render()
        self.qvtkwidget_2d.update()


    def show_debug_dialog(self,thing):

        message = str(thing)

        dialog = qt.QMessageBox(self)
        dialog.setStandardButtons(qt.QMessageBox.Ok)
        dialog.setTextFormat(qt.Qt.RichText)
        dialog.setWindowTitle('Calcam - Debug Message')
        dialog.setText(message)
        dialog.setIcon(qt.QMessageBox.Information)
        dialog.exec_()


    def on_close(self):

        if self.cadmodel is not None:
            self.cadmodel.remove_from_renderer(self.renderer_3d)
            self.cadmodel.unload()

        self.config.mouse_sensitivity = self.control_sensitivity_slider.value()
        self.config.save()
        sys.excepthook = sys.__excepthook__


    def on_model_load(self):
        pass


class ChessboardDialog(qt.QDialog):

    def __init__(self, parent,modelselection=False):

        # GUI initialisation
        qt.QDialog.__init__(self, parent)
        qt.uic.loadUi(os.path.join(guipath,'chessboard_image_dialog.ui'), self)

        self.parent = parent
        try:
            self.image_transformer = self.parent.calibration.geometry
            self.subview_lookup = self.parent.calibration.subview_lookup
            self.n_fields = self.parent.calibration.n_subviews
        except AttributeError:
            self.image_transformer = CoordTransformer()
            self.subview_lookup = lambda x,y: 0
            self.n_fields = 1

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

        if int(cv2.__version__[0]) < 3:
            self.fisheye_model.setEnabled(False)
            self.fisheye_model.setToolTip('Requires OpenCV 3')

        # Set up VTK
        self.qvtkwidget = qt.QVTKRenderWindowInteractor(self.image_frame)
        self.image_frame.layout().addWidget(self.qvtkwidget)
        self.interactor = CalcamInteractorStyle2D(refresh_callback=self.refresh_vtk)
        self.qvtkwidget.SetInteractorStyle(self.interactor)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.qvtkwidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkInteractor = self.qvtkwidget.GetRenderWindow().GetInteractor()
        self.image_frame.layout().addWidget(self.qvtkwidget,1)

        self.detection_run = False

        self.images = []
        self.filenames = []
        self.corner_cursors = []
        self.results = []

        self.pointpairs_result = None

        # Start the GUI!
        self.show()
        self.interactor.init()
        self.renderer.Render()
        self.vtkInteractor.Initialize()


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
                self.src_dir = os.path.split(str(fname))[0]

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
            self.update_image_display(0)
            self.detect_chessboard_button.setEnabled(True)


    def change_image(self):
        if self.sender() is self.next_im_button:
            self.update_image_display((self.current_image + 1) % len(self.images))
        elif self.sender() is self.prev_im_button:
            self.update_image_display((self.current_image - 1) % len(self.images))



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




    def update_image_display(self,image_number=None):

        if image_number is None:
            image_number = self.current_image
        else:
            self.current_image = image_number

        for cursor in self.corner_cursors:
            self.interactor.remove_passive_cursor(cursor)
        self.corner_cursors = []
        
        self.interactor.set_image(self.images[image_number])

        if self.detection_run:
            if self.chessboard_status[image_number]:
                status_string = ' - Chessboard Detected OK'
            else:
                status_string = ' - Chessboard Detection FAILED'
        else:
            status_string = ''

        self.current_filename.setText('<html><head/><body><p align="center">{:s} [#{:d}/{:d}]{:s}</p></body></html>'.format(self.filenames[image_number],image_number+1,len(self.images),status_string))
        
        if self.chessboard_status[image_number]:
            for corner in range(self.chessboard_points_2D[image_number].shape[0]):
                self.corner_cursors.append( self.interactor.add_passive_cursor(self.chessboard_points_2D[image_number][corner,:]))


    def refresh_vtk(self):
        self.renderer.Render()
        self.qvtkwidget.update()


    def apply(self):

        # List of pointpairs objects for the chessboard point pairs
        self.results = []

        chessboard_points_3D = []

        point_spacing = self.chessboard_square_size.value() * 1e-3


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
            self.results.append( (self.images[i],PointPairs()) )

            self.results[-1][1].n_subviews = self.n_fields

            # We already have the 3D positions
            self.results[-1][1].object_points = chessboard_points_3D

            # Initialise image points
            self.results[-1][1].image_points = []

            # Image shape
            self.results[-1][1].image_shape = (self.images[i].shape[1],self.images[i].shape[0])

            # Get a neater looking reference to the chessboard corners for this image
            impoints = self.chessboard_points_2D[i]

            # Loop over chessboard points
            for point in range( np.prod(self.n_chessboard_points) ):
                self.results[-1][1].image_points.append([])

                # Populate coordinates for relevant field
                for field in range(self.n_fields):
                    if self.subview_lookup(impoints[point,0],impoints[point,1]) == field:
                        self.results[-1][1].image_points[-1].append([impoints[point,0], impoints[point,1]])
                    else:
                        self.results[-1][1].image_points[-1].append(None)

        self.filenames = self.filenames[self.chessboard_status == True]

        self.chessboard_source = '{:d} chessboard images loaded from {:s}'.format(len(self.filenames),self.src_dir)

        # And close the window.
        self.done(1)