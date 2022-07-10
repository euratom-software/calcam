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

# Standard library imports
import sys
import os
import traceback

# External module imports
import numpy as np
import vtk
import cv2

# Calcam imports
from .. import __path__ as calcampath
from .. import __version__ as calcamversion
from ..cadmodel import CADModel
from ..config import CalcamConfig
from ..calibration import Calibration,Fitter
from ..pointpairs import PointPairs
from .vtkinteractorstyles import CalcamInteractorStyle2D
from . import qt_wrapper as qt
from ..misc import ColourCycle,DodgyDict, open_file

guipath = os.path.split(os.path.abspath(__file__))[0]

class CalcamGUIWindow(qt.QMainWindow):


    def init(self, ui_filename, app, parent):

        # GUI initialisation
        qt.QMainWindow.__init__(self, parent)
        qt.uic.loadUi(os.path.join(guipath,'qt_designer_files',ui_filename), self)

        self.setWindowIcon(qt.QIcon(os.path.join(guipath,'icons','calcam.png')))

        self.app = app

        self.config = CalcamConfig()

        self.manual_exc = False

        # See how big the screen is and open the window at an appropriate size
        if qt.qt_ver < 5:
            available_space = self.app.desktop().availableGeometry(self)
        else:
            available_space = self.app.primaryScreen().availableGeometry()

        # Open the window with same aspect ratio as the screen, and no fewer than 500px tall.
        win_height = max(500,min(780,0.75*available_space.height()))
        win_width = win_height * available_space.width() / available_space.height() 
        self.resize(win_width,win_height)

         # Let's show helpful dialog boxes if we have unhandled exceptions:
        sys.excepthook = self.show_exception_dialog

        try:
            self.action_new.setIcon( qt.QIcon(os.path.join(guipath,'icons','new.png')) )
            self.action_open.setIcon( qt.QIcon(os.path.join(guipath,'icons','open.png')) )
            self.action_save.setIcon( qt.QIcon(os.path.join(guipath,'icons','save.png')) )
            self.action_save_as.setIcon( qt.QIcon(os.path.join(guipath,'icons','saveas.png')) )
            self.action_cal_info.setIcon( qt.QIcon(os.path.join(guipath,'icons','info.png')) )
        except AttributeError:
            pass

        self.unsaved_changes = False

        self.detector_window = None

        # -------------------- Initialise View List ------------------
        self.viewlist.clear()

        self.fov_enabled = True
        self.viewdir_at_cc = False

        # Populate viewports list
        self.views_root_model = qt.QTreeWidgetItem(['Defined in Model'])
        self.views_root_auto = qt.QTreeWidgetItem(['Auto Cross-Sections'])
        self.views_root_results = qt.QTreeWidgetItem(['From Calibrations'])


        # Auto type views
        qt.QTreeWidgetItem(self.views_root_auto,['Vertical cross-section'])
        qt.QTreeWidgetItem(self.views_root_auto,['Horizontal cross-section'])

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

        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()

        # Check if we can blame user-written code for what's gone wrong
        # by checking if the originating python file is inside the calcam
        # code directory or not.
        fpath = traceback.extract_tb(tb)[-1][0]
        if calcampath[0] in fpath:
            external = False
        else:
            external = True

        traceback_lines = traceback.format_exception(excep_type,excep_value,tb,limit=10)

        if external and excep_type != UserWarning:
            dialog = qt.QMessageBox(self)

            #if qt.qt_ver > 5:
            #    dialog.Save = dialog.StandardButton.Save
            #    dialog.Discard = dialog.StandardButton.Discard

            dialog.setWindowFlags(dialog.windowFlags() | qt.Qt.CustomizeWindowHint)
            dialog.setWindowFlags(dialog.windowFlags() & ~qt.Qt.WindowCloseButtonHint)
            dialog.setStandardButtons(dialog.Save | dialog.Discard)
            dialog.button(dialog.Save).setText('Save error report')
            dialog.button(dialog.Discard).setText('Do not save')
            dialog.setTextFormat(qt.Qt.RichText)
            dialog.setWindowTitle('Calcam - External Code Error')
            dialog.setText('An unhandled exception has been raised by external code. This could be because of a bug in the external code or in Calcam itself.')
            
            dialog.setInformativeText(''.join(traceback_lines) + '\nWould you like to save an error report file?')
            dialog.setIcon(qt.QMessageBox.Warning)
            dialog.exec()

            if dialog.result() == dialog.Save:

                filedialog = qt.QFileDialog(self)
                filedialog.setAcceptMode(filedialog.AcceptSave)
                filedialog.setFileMode(filedialog.AnyFile)
                filedialog.setWindowTitle('Save error report')
                filedialog.setNameFilter('Text files (*.txt)')
                filedialog.exec()
                if filedialog.result() == 1:
                    fname = str(filedialog.selectedFiles()[0])
                    if not fname.endswith('.txt'):
                        fname = fname + '.txt'

                    with open(fname,'w') as dumpfile:
                        dumpfile.write('CALCAM ERROR REPORT\n===================\n\nThis file was generated by Calcam to help report/debug an unhandled exception raised by external code.\nIf you think this could be a bug in Calcam itself, please go to:\n\n      github.com/euratom-software/calcam/issues\n\nand open an issue describing when the error happened, and attach this file.\n\n\nDIAGNOSTIC INFORMATION\n----------------------\n\n')
                        dumpfile.write('Platform:       {:s}\n'.format(sys.platform))
                        dumpfile.write('Python version: {:s}\n'.format(sys.version))
                        dumpfile.write('Calcam version: {:s}\n'.format(calcamversion))
                        dumpfile.write('VTK version:    {:s}\n'.format(vtk.vtkVersion().GetVTKVersion()))
                        dumpfile.write('OpenCV version: {:s}\n'.format(cv2.__version__))
                        dumpfile.write('PyQt version:   {:s}\n\n'.format(qt.QT_VERSION_STR))
                        for line in traceback_lines:
                            dumpfile.write(line)

                    open_file(fname)
        else:

            # I'm using user warnings for information boxes which need to be raised:
            if excep_type == UserWarning:
                dialog = qt.QMessageBox(self)
                dialog.setStandardButtons(qt.QMessageBox.Ok)
                dialog.setTextFormat(qt.Qt.RichText)
                dialog.setWindowTitle('Calcam')
                dialog.setText(str(excep_value))
                dialog.setIcon(qt.QMessageBox.Information)
                dialog.exec()

            # otherwise it's really an unexpected exception:
            else:
                dialog = qt.QMessageBox(self)
                if qt.qt_ver > 5:
                    dialog.Save = dialog.StandardButton.Save
                    dialog.Discard = dialog.StandardButton.Discard

                dialog.setWindowFlags(dialog.windowFlags() | qt.Qt.CustomizeWindowHint)
                dialog.setWindowFlags(dialog.windowFlags() & ~qt.Qt.WindowCloseButtonHint)
                dialog.setStandardButtons(dialog.Save | dialog.Discard)
                dialog.button(dialog.Save).setText('Save error report')
                dialog.button(dialog.Discard).setText('Do not save')
                dialog.setTextFormat(qt.Qt.RichText)
                dialog.setWindowTitle('Calcam - Error')
                dialog.setText('An unhandled exception has been raised; the action you were performing may have partially or completely failed. This is probably a bug in Calcam; to report it please save an error report file and report the problem at <a href="https://github.com/euratom-software/calcam/issues">here</a> and/or consider contributing a fix!')
                dialog.setInformativeText(''.join(traceback_lines) + '\nWould you like to save an error report file?')
                dialog.setIcon(qt.QMessageBox.Warning)
                dialog.exec()

                if dialog.result() == dialog.Save:

                    filedialog = qt.QFileDialog(self)
                    filedialog.setAcceptMode(filedialog.AcceptSave)
                    filedialog.setFileMode(filedialog.AnyFile)
                    filedialog.setWindowTitle('Save error report')
                    filedialog.setNameFilter('Text files (*.txt)')
                    filedialog.exec()
                    if filedialog.result() == 1:
                        fname = str(filedialog.selectedFiles()[0])
                        if not fname.endswith('.txt'):
                            fname = fname + '.txt'

                        with open(fname,'w') as dumpfile:
                            dumpfile.write('CALCAM ERROR REPORT\n===================\n\nThis file was generated by Calcam to help report/debug an unhandled exception.\nTo report the error, please go to:\n\n      github.com/euratom-software/calcam/issues\n\nand open an issue describing when the error happened, and attach this file.\n\n\nDIAGNOSTIC INFORMATION\n----------------------\n\n')
                            dumpfile.write('Platform:       {:s}\n'.format(sys.platform))
                            dumpfile.write('Python version: {:s}\n'.format(sys.version))
                            dumpfile.write('Calcam version: {:s}\n'.format(calcamversion))
                            dumpfile.write('VTK version:    {:s}\n'.format(vtk.vtkVersion().GetVTKVersion()))
                            dumpfile.write('OpenCV version: {:s}\n'.format(cv2.__version__))
                            dumpfile.write('PyQt version:   {:s}\n\n'.format(qt.QT_VERSION_STR))
                            for line in traceback_lines:
                                dumpfile.write(line)

                        open_file(fname)


    def show_calib_info(self):

        if self.manual_exc:
            if self.calibration.view_models[0] is not None:
                self.update_extrinsics()
                CalibInfoDialog(self, self.calibration)
            else:
                self.show_msgbox('No calibration information to show.')
        else:
            CalibInfoDialog(self,self.calibration)

    def update_vtk_size(self,vtksize):

        self.vtksize = vtksize



    def update_chessboard_intrinsics(self,pass_calib=True):

        if pass_calib:
            cal = self.calibration
        else:
            cal = None

        dialog = ChessboardDialog(self,modelselection=True,calibration=cal)
        dialog.exec()

        if dialog.results != []:
            chessboard_pointpairs = dialog.results
            if dialog.perspective_model.isChecked():
                fitter = Fitter('rectilinear')
            elif dialog.fisheye_model.isChecked():
                fitter = Fitter('fisheye')
            fitter.set_image_shape( dialog.results[0][0].shape[1::-1] )
            fitter.ignore_upside_down=True
            fitter.set_pointpairs(chessboard_pointpairs[0][1])
            for chessboard_im in chessboard_pointpairs[1:]:
                fitter.add_intrinsics_pointpairs(chessboard_im[1])

            self.chessboard_pointpairs = chessboard_pointpairs
            self.chessboard_fit = fitter.do_fit()
            self.chessboard_src = dialog.chessboard_source

            self.update_intrinsics()

        del dialog



    def save_view_to_model(self,show_default=False):

        if '*' in str(self.view_save_name.text()):
            raise UserWarning('Cannot save view with this name: asterisk character (*) not allowed in view names!')

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
            if len(self.cadmodel.views.keys()) == 1:
                self.cadmodel.initial_view = str(self.view_save_name.text())
            self.update_model_views(show_default=show_default)

        except:
            self.update_model_views(show_default=show_default)
            raise


    def pick_colour(self,init_colour,pick_alpha=False):

        col_init = np.array(init_colour) * 255

        if pick_alpha:
            if col_init.size < 4:
                raise ValueError('If pick_alpha = True, you must supply a 4 element RGBA initial colour!')
            alpha_init = col_init[3]
        else:
            alpha_init = 255

        if pick_alpha:
            if qt.qt_ver < 6:
                new_col = qt.QColorDialog.getColor(qt.QColor(col_init[0], col_init[1], col_init[2], alpha_init), self,'Choose Colour...',qt.QColorDialog.ColorDialogOptions(qt.QColorDialog.ShowAlphaChannel))
            else:
                new_col = qt.QColorDialog.getColor(qt.QColor(col_init[0], col_init[1], col_init[2], alpha_init), self,'Choose Colour...',qt.QColorDialog.ColorDialogOption.ShowAlphaChannel)
        else:
            new_col = qt.QColorDialog.getColor(qt.QColor(col_init[0],col_init[1],col_init[2],alpha_init),self,'Choose Colour...')

        if new_col.isValid():
            ret_col = [ new_col.red() / 255. , new_col.green() / 255. , new_col.blue() / 255.]
            if pick_alpha:
                ret_col = ret_col + [ new_col.alpha() / 255. ]
        else:
            ret_col = None

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
        for option in self.imsource.get_image_arguments:

            labelwidget = qt.QLabel(option['gui_label'] + ':')
            layout.addWidget(labelwidget,row,0)

            if option['type'] == 'filename':
                button = qt.QPushButton('Browse...')
                button.setMaximumWidth(80)
                layout.addWidget(button,row+1,1)
                fname = qt.QLineEdit()
                fname_filter = option['filter']
                button.clicked.connect(lambda : self.browse_for_file(fname_filter,fname))
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

        # Keep a list of which parameters we have filled in with assumptions,
        # because sometimes we want to know later.
        newim['assumptions'] = []

        if 'subview_mask' not in newim:
            newim['subview_mask'] = np.zeros(newim['image_data'].shape[:2],dtype=np.uint8)
            newim['assumptions'].append('subview_mask')

        if 'subview_names' not in newim:
            newim['subview_names'] = []

        if 'transform_actions' not in newim:
            newim['transform_actions'] = []
            newim['assumptions'].append('transform_actions')

        if 'pixel_size' not in newim:
            newim['pixel_size'] = None

        if 'coords' not in newim:
            newim['coords'] = 'display'
            newim['assumptions'].append('coords')
            
        if 'pixel_aspect' not in newim:
            newim['pixel_aspect'] = 1.
            newim['assumptions'].append('pixel_aspect')

        if 'image_offset' not in newim:
            newim['image_offset'] = (0.,0.)
            newim['assumptions'].append('image_offset')

        self.on_load_image(newim)
        self.statusbar.clearMessage()
        self.app.restoreOverrideCursor()


    def object_from_file(self,obj_type,multiple=False,maintain_window=True):

        filename_filter = self.config.filename_filters[obj_type]

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(filedialog.AcceptOpen)

        try:
           filedialog.setDirectory(self.config.file_dirs[obj_type])
        except KeyError:
            filedialog.setDirectory(os.path.expanduser('~'))

        if multiple:
            filedialog.setFileMode(filedialog.ExistingFiles)
            empty_ret = []
        else:
            filedialog.setFileMode(filedialog.ExistingFile)
            empty_ret = None

        filedialog.setWindowTitle('Open...')
        filedialog.setNameFilter(filename_filter)
        filedialog.exec()

        if qt.qt_ver < 6:
            accepted = filedialog.result() == 1
        else:
            accepted = filedialog.result() == filedialog.Accepted

        if accepted:
            selected_paths = filedialog.selectedFiles()
        else:
            return empty_ret

        # If we have selected one or more files...
        self.config.file_dirs[obj_type] = os.path.split(str(selected_paths[0]))[0]

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        objs = []
        
        for path in [str(p) for p in selected_paths]:

            if obj_type.lower() == 'calibration':
                try:
                    obj = Calibration(path)
                    if maintain_window:
                        obj.set_detector_window(self.detector_window)
                except Exception as e:
                    self.show_msgbox('Error while opening {:s}:<br>{:}'.format(os.path.split(path)[-1],e))
                    obj = None


            elif obj_type.lower() == 'pointpairs':
                if path.endswith('.ccc'):
                    cal = Calibration(path)

                    if maintain_window:
                        cal.set_detector_window(self.detector_window)

                    obj = cal.pointpairs
                    obj.history = cal.history['pointpairs']
                elif path.endswith('.csv'):
                    with open(path,'r') as ppf:
                        obj = PointPairs(ppf)
                        obj.src = 'Loaded from Calcam CSV point pairs file "{:s}"'.format(os.path.split(path)[-1])

            objs.append(obj)
        self.app.restoreOverrideCursor()
        if multiple:
            return objs
        else:
            return objs[0]



    def update_image_info_string(self,im_array,geometry):

        if np.any(np.array(geometry.get_display_shape()) != np.array(geometry.get_original_shape())):
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP) [ As Displayed ]<br>{3:d} x {4:d} pixels ({5:.1f} MP) [ Raw Data ]<br>'.format(geometry.get_display_shape()[0],geometry.get_display_shape()[1],np.prod(geometry.get_display_shape()) / 1e6 ,geometry.get_original_shape()[0],geometry.get_original_shape()[1],np.prod(geometry.get_original_shape()) / 1e6 )
        else:
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP)<br>'.format(geometry.get_display_shape()[0],geometry.get_display_shape()[1],np.prod(geometry.get_display_shape()) / 1e6 )
        
        if len(im_array.shape) == 2:
            info_str = info_str + 'Monochrome'
        elif len(im_array.shape) == 3:
            if np.all(im_array[:,:,0] == im_array[:,:,1]) and np.all(im_array[:,:,2] == im_array[:,:,1]):
                info_str = info_str + 'Monochrome'
            else:
                info_str = info_str + 'RGB Colour'
            if im_array.shape[2] == 4:
                info_str = info_str + ' + Transparency'

        self.image_info.setText(info_str)
        


    def get_save_filename(self,obj_type):

        filename_filter = self.config.filename_filters[obj_type]
        fext = filename_filter.split('(*')[1].split(')')[0]

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(filedialog.AcceptSave)

        try:
           filedialog.setDirectory(self.config.file_dirs[obj_type])
        except KeyError:
            filedialog.setDirectory(os.path.expanduser('~'))
        

        filedialog.setFileMode(filedialog.AnyFile)

        filedialog.setWindowTitle('Save As...')
        filedialog.setNameFilter(filename_filter)
        filedialog.exec()
        if filedialog.result() == 1:
            selected_path = str(filedialog.selectedFiles()[0])
            self.config.file_dirs[obj_type] = os.path.split(selected_path)[0]
            if not selected_path.endswith(fext):
                selected_path = selected_path + fext
            return selected_path
        else:
            return None



    def update_model_views(self,show_default=False,keep_selection=False):

        if keep_selection:
            to_select = self.viewlist.selectedItems()
            if len(to_select) == 1:
                to_select = str(to_select[0].text(0))
            else:
                to_select = None
        else:
            to_select = None

        self.viewlist.selectionModel().clearSelection()
        self.views_root_model.setText(0,self.cadmodel.machine_name)
        self.views_root_model.takeChildren()

        # Add views to list
        for view in self.cadmodel.get_view_names():
            if view == self.cadmodel.initial_view and show_default:
                item = qt.QTreeWidgetItem(self.views_root_model,[view + '*'])
            else:
                item = qt.QTreeWidgetItem(self.views_root_model,[view])
            if view == to_select:
                item.setSelected(True)


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
                if self.fov_enabled:
                    self.interactor3d.set_fov(view['y_fov'])

            elif view_item.parent() is self.views_root_results or view_item.parent() in self.viewport_calibs.keys():

                view,subfield = self.viewport_calibs[(view_item)]
                
                if subfield is not None:
                    self.set_view_from_calib(view,subfield)


            elif view_item.parent() is self.views_root_auto:

                # Work out the field of view to set based on the model extent.
                # Note: this assumes the origin is at the centre of the machine.
                # I could not assume that, but then I might end up de-centred on otherwise well bahevd models.
                model_extent = self.cadmodel.get_extent()
                z_extent = model_extent[5]-model_extent[4]
                r_extent = max(model_extent[1]-model_extent[0],model_extent[3]-model_extent[2])

                xsec_fov = 30

                if str(view_item.text(0)).lower() == 'horizontal cross-section':

                    if self.interactor3d.projection == 'perspective':
                        self.camFOV.setValue(xsec_fov)
                    else:
                        self.camFOV.setValue(r_extent)

                    if self.xsection_cursor.isEnabled():
                        self.xsection_cursor.setChecked(True)
                    else:
                        self.xsection_origin.setChecked(True)

                    self.camera_3d.SetPosition( (0.,0.,r_extent/(2*np.tan(3.14159*xsec_fov/360) )))
                    self.camera_3d.SetFocalPoint( (0.,0.001,self.camera_3d.GetPosition()[2]-1.) )
                    self.xsection_checkbox.setChecked(True)

                elif str(view_item.text(0)).lower() == 'vertical cross-section':

                    R = z_extent/(2*np.tan(3.14159*xsec_fov/360))

                    if self.interactor3d.projection == 'perspective':
                        self.camFOV.setValue(xsec_fov)
                    else:
                        self.camFOV.setValue(z_extent)

                    if self.xsection_cursor.isEnabled():
                        cursorpos = self.interactor3d.get_cursor_coords(0)
                        phi = np.arctan2(cursorpos[1],cursorpos[0])
                        phi_cam = phi - 3.14159/2.
                        self.xsection_cursor.setChecked(True)

                    else:
                        phi_cam = np.arctan2(self.camY.value(),self.camX.value())
                        self.xsection_origin.setChecked(True)

                    self.camera_3d.SetPosition(R * np.cos(phi_cam),R * np.sin(phi_cam),0.)
                    self.interactor3d.set_roll(0.)
                    self.camera_3d.SetFocalPoint( (0.,0.,0.) )

                    self.xsection_checkbox.setChecked(True)

        else:
            self.camera_3d.SetPosition((self.camX.value(),self.camY.value(),self.camZ.value()))
            self.camera_3d.SetFocalPoint((self.tarX.value(),self.tarY.value(),self.tarZ.value()))
            if self.fov_enabled:
                self.interactor3d.set_fov(self.camFOV.value())
            self.interactor3d.set_roll(self.cam_roll.value())
        
        self.interactor3d.update_cursor_style()
        self.interactor3d.update_clipping()

        self.refresh_3d()

        self.update_viewport_info(keep_selection=True)

        self.on_view_changed()



    def set_view_from_calib(self,calibration,subfield):

        if self.fov_enabled:
            fov = calibration.get_fov(subview=subfield,fullchip=True)
            vtk_aspect = float(self.vtksize[1]) / float(self.vtksize[0])
            fov_aspect = float(calibration.geometry.y_pixels) / float(calibration.geometry.x_pixels)
            if vtk_aspect > fov_aspect-1e-4:
                h_fov = True
                fov_angle = fov[0]
            else:
                h_fov = False
                fov_angle = fov[1]

            self.camera_3d.SetUseHorizontalViewAngle(h_fov)
            self.camera_3d.SetViewAngle(fov_angle)
            self.camera_3d.SetUseHorizontalViewAngle(h_fov)

        self.camera_3d.SetPosition(calibration.get_pupilpos(subview=subfield))

        if self.viewdir_at_cc:
            mat = calibration.get_cam_matrix(subview=subfield)
            self.camera_3d.SetFocalPoint(calibration.get_pupilpos(subview=subfield) + calibration.get_los_direction(mat[0,2],mat[1,2]))
        else:
            self.camera_3d.SetFocalPoint(calibration.get_pupilpos(subview=subfield) + calibration.get_los_direction(calibration.geometry.get_display_shape()[0]/2,calibration.geometry.get_display_shape()[1]/2))

        self.interactor3d.set_roll(calibration.get_cam_roll(subview=subfield,centre='subview'))

        self.update_viewport_info(keep_selection=True)

        self.interactor3d.set_xsection(None)       
        
        self.interactor3d.update_cursor_style()

        self.interactor3d.update_clipping()
  
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




    def load_viewport_calib(self,event=None,set_view=True):

        cals = self.object_from_file('calibration',multiple=True)

        for cal in cals:
            
            if cal.view_models.count(None) == len(cal.view_models):
                self.show_msgbox('The calibration file {:s} does not contain any view models so does not define a view.'.format(cal.filename))
                continue

            self.views_root_results.setHidden(False)

            listitem = qt.QTreeWidgetItem(self.views_root_results,[cal.name])
            if cal.n_subviews > 1:
                self.viewport_calibs[(listitem)] = (cal,None)
                listitem.setExpanded(True)
                for n,fieldname in enumerate(cal.subview_names):
                    if cal.view_models[n] is not None:
                        sublistitem = qt.QTreeWidgetItem(listitem,[fieldname])
                        self.viewport_calibs[ sublistitem ] = (cal,n)
                        if set_view:
                            self.viewlist.clearSelection()
                            sublistitem.setSelected(True)
                            set_view = False
            else:
                self.viewport_calibs[(listitem)] = (cal,0)
                if set_view:
                    self.viewlist.clearSelection()
                    listitem.setSelected(True)
                    set_view=False
            


    def populate_model_variants(self):

        self.model_variant.clear()
        if str(self.model_name.currentText()) != '':

            model = self.model_list[str(self.model_name.currentText())]
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


    def on_view_changed(self):
        pass


    def browse_for_file(self,name_filter,target_textbox=None):

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(filedialog.AcceptOpen)
        filedialog.setFileMode(filedialog.ExistingFile)
        filedialog.setWindowTitle('Select File')
        filedialog.setNameFilter(name_filter)
        filedialog.exec()
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
                state = [qt.Qt.Unchecked,qt.Qt.PartiallyChecked,qt.Qt.Checked][self.cadmodel.get_group_enable_state(feature)]
                qitem.setCheckState(0,state)
            except KeyError:
                if feature in enabled_features:
                    qitem.setCheckState(0,qt.Qt.Checked)
                else:
                    qitem.setCheckState(0,qt.Qt.Unchecked)

        self.feature_tree.blockSignals(False)


    def load_model(self,data=None,featurelist=None,hold_view=False):

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

        self.update_model_views()

        if not hold_view:
            # Put the camera in some reasonable starting position
            self.camera_3d.SetViewAngle(90)
            self.camera_3d.SetViewUp((0,0,1))
            self.camera_3d.SetFocalPoint(0,0,0)

            if self.cadmodel.initial_view is not None:
                for i in range(self.views_root_model.childCount()):
                    if self.views_root_model.child(i).text(0).replace('*','') == self.cadmodel.initial_view:
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

        roll = self.interactor3d.cam_roll

        self.camX.blockSignals(True)
        self.camY.blockSignals(True)
        self.camZ.blockSignals(True)
        self.tarX.blockSignals(True)
        self.tarY.blockSignals(True)
        self.tarZ.blockSignals(True)
        self.cam_roll.blockSignals(True)


        self.camX.setValue(campos[0])
        self.camY.setValue(campos[1])
        self.camZ.setValue(campos[2])
        self.tarX.setValue(camtar[0])
        self.tarY.setValue(camtar[1])
        self.tarZ.setValue(camtar[2])
        self.cam_roll.setValue(roll)

        if self.fov_enabled:
            self.camFOV.blockSignals(True)
            self.camFOV.setSuffix(fov_suffix)
            self.camFOV.setMinimum(fov_min)
            self.camFOV.setMaximum(fov_max)
            self.camFOV.setDecimals(decimals)
            self.camFOV.setValue(fov)
            self.camFOV.blockSignals(False)


        self.camX.blockSignals(False)
        self.camY.blockSignals(False)
        self.camZ.blockSignals(False)
        self.tarX.blockSignals(False)
        self.tarY.blockSignals(False)
        self.tarZ.blockSignals(False)
        self.cam_roll.blockSignals(False)

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

        if len(self.model_list) == 0:
                dialog = qt.QMessageBox(self)
                dialog.addButton('Browse...',dialog.AcceptRole)
                dialog.addButton('Create New...',dialog.YesRole)
                dialog.addButton('Do Nothing and Close',dialog.RejectRole)
                dialog.setTextFormat(qt.Qt.RichText)
                dialog.setWindowTitle('No CAD Models')
                dialog.setText('This tool uses CAD models but you seem to have no CAD models set up in calcam.')
                dialog.setInformativeText('You can either browse for a folder containing existing Calcam CAD definition (.ccm) files, or create a new CAD model definition.')
                dialog.setIcon(qt.QMessageBox.Information)
                dialog.exec()

                if dialog.result() == 2:
                    self.timer = qt.QTimer.singleShot(0,self.app.quit)

                elif dialog.result() == 0:

                    filedialog = qt.QFileDialog(self)
                    filedialog.setAcceptMode(filedialog.AcceptOpen)
                    filedialog.setFileMode(filedialog.Directory)
                    filedialog.setWindowTitle('Select CAD definition Location')
                    filedialog.exec()

                    if filedialog.result() == 1:
                        path = str(filedialog.selectedFiles()[0]).replace('/',os.path.sep)
                        self.config.cad_def_paths.append(path)

                        self.model_list = self.config.get_cadmodels()

                        if len(self.model_list) == 0:
                            dialog = qt.QMessageBox(self)
                            dialog.setStandardButtons(dialog.Ok)
                            dialog.setWindowTitle('No CAD Models')
                            dialog.setText('The selected directory does not seem to contain any CAD definition files. The application will now close.')
                            dialog.setIcon(qt.QMessageBox.Information)
                            dialog.exec()
                            self.timer = qt.QTimer.singleShot(0,self.app.quit)
                        else:                      
                            self.config.save()
                    else:
                        self.timer = qt.QTimer.singleShot(0,self.app.quit)
                        
                elif dialog.result() == 1:
                    from .launcher import launch
                    launch(['--cad_edit'])
                    self.timer = qt.QTimer.singleShot(0,self.app.quit)


        self.model_name.addItems(sorted(self.model_list.keys()))

        set_model = False
        set_variant = False
        self.load_model_button.setEnabled(0)

        if self.config.default_model is not None:
            mname = None
            for i,mname in enumerate(sorted(self.model_list.keys())):
                if mname == self.config.default_model[0]:

                    self.model_name.setCurrentIndex(i)
                    self.populate_model_variants()
                    set_model = True
                    break

            if mname in self.model_list.keys():

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


    def show_msgbox(self,main_msg,sub_msg=None):

        dialog = qt.QMessageBox(self)
        dialog.setStandardButtons(qt.QMessageBox.Ok)
        dialog.setTextFormat(qt.Qt.RichText)
        dialog.setWindowTitle('Calcam')
        dialog.setText(str(main_msg))
        if sub_msg is not None:
            dialog.setInformativeText(str(sub_msg))
        dialog.setIcon(qt.QMessageBox.Information)
        dialog.exec()



    def closeEvent(self,event):

        if self.unsaved_changes:
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Save|qt.QMessageBox.Discard|qt.QMessageBox.Cancel)
            dialog.setWindowTitle('Save changes?')
            dialog.setText('There are unsaved changes. Save before exiting?')
            dialog.setIcon(qt.QMessageBox.Information)
            choice = dialog.exec()
            if choice == qt.QMessageBox.Save:
                if self.action_save.isEnabled():
                    self.action_save.trigger()
                else:
                    self.action_save_as.trigger()
                    
            elif choice == qt.QMessageBox.Cancel:
                event.ignore()
                return

        if self.cadmodel is not None:
            self.cadmodel.remove_from_renderer(self.renderer_3d)
            self.cadmodel.unload()

        self.config.mouse_sensitivity = self.control_sensitivity_slider.value()
        self.config.save()

        self.on_close()

        sys.excepthook = sys.__excepthook__


    def on_model_load(self):
        pass

    def on_close(self):
        pass

    def update_extrinsics(self):
        # Set the calibration's extrinsics to match the current view.
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

        self.calibration.set_extrinsics(campos,upvec,camtar = camtar,src=self.extrinsics_src) 


class ChessboardDialog(qt.QDialog):

    def __init__(self, parent,modelselection=False,calibration=None):

        # GUI initialisation
        qt.QDialog.__init__(self, parent,qt.Qt.WindowTitleHint | qt.Qt.WindowCloseButtonHint)
        qt.uic.loadUi(os.path.join(guipath,'qt_designer_files','chessboard_image_dialog.ui'), self)

        self.parent = parent
        if calibration is not None:
            self.image_transformer = calibration.geometry

            if self.image_transformer.get_display_shape() != self.image_transformer.get_original_shape() or self.image_transformer.transform_actions == []:
                self.orientation_label.hide()
                self.original_coords.hide()
                self.display_coords.hide()

            self.subview_lookup = calibration.subview_lookup
            self.n_fields = calibration.n_subviews
        else:
            self.image_transformer = None
            self.subview_lookup = lambda x,y,coords: 0
            self.n_fields = 1

            self.display_coords.setChecked(True)
            self.orientation_label.hide()
            self.original_coords.hide()
            self.display_coords.hide()

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

        self.next_im_button.setEnabled(False)
        self.prev_im_button.setEnabled(False)
        self.current_filename.hide()

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
        filedialog.setAcceptMode(filedialog.AcceptOpen)
        filedialog.setFileMode(filedialog.ExistingFiles)
        filedialog.setWindowTitle('Load chessboard images')
        filedialog.setNameFilter('Image Files (*.jpg *.jpeg *.png *.bmp *.jp2 *.tiff *.tif)')
        filedialog.exec()
        if filedialog.result() == 1:
            self.parent.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.images = []
            self.filenames = []
            wrong_shape = []


            self.im_shape = None

            for n,fname in enumerate(filedialog.selectedFiles()):
                self.status_text.setText('<b>Loading image {:d} / {:d} ...'.format(n,len(filedialog.selectedFiles())))
                im = cv2.imread(str(fname))
                
                if self.im_shape is None:

                    if self.image_transformer is not None:
                        imshape = im.shape[1::-1]
                        if np.all(imshape == self.image_transformer.get_display_shape()):
                            self.display_coords.setChecked(True)
                            expected_shape = self.image_transformer.get_display_shape()
                            wrong_shape.append(False)
                            self.im_shape = im.shape[:2]
                        elif np.all(imshape == self.image_transformer.get_original_shape()):
                            self.original_coords.setChecked(True)
                            expected_shape = self.image_transformer.get_original_shape()
                            wrong_shape.append(False)
                            self.im_shape = im.shape[:2]
                        else:
                            wrong_shape.append(True)
                    else:
                        wrong_shape.append(False)
                        self.im_shape = im.shape[:2]
                    
                else:
                    wrong_shape.append(np.any(self.im_shape != im.shape[:2]))
                
                # OpenCV loads colour channels in BGR order.
                if len(im.shape) == 3:
                    im[:,:,:3] = im[:,:,3::-1]

                self.images.append(im)

                self.filenames.append(os.path.split(str(fname))[1])
                self.src_dir = os.path.split(str(fname))[0]
            self.parent.app.restoreOverrideCursor()

            self.status_text.setText('')
            if np.all(wrong_shape):
                dialog = qt.QMessageBox(self)
                dialog.setStandardButtons(qt.QMessageBox.Ok)
                dialog.setTextFormat(qt.Qt.RichText)
                dialog.setWindowTitle('Calcam - Wrong image size')
                dialog.setText("The selected chessboard pattern images are the wrong dimensions for this camera calibration and have not been loaded.")
                dialog.setInformativeText("Chessboard images of {:d} x {:d} pixels or {:d} x {:d} pixels are required for this camera image.".format(self.image_transformer.get_original_shape()[0],self.image_transformer.get_original_shape()[1],self.image_transformer.get_display_shape()[0],self.image_transformer.get_display_shape()[1]))
                dialog.setIcon(qt.QMessageBox.Warning)
                dialog.exec()
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
                dialog.exec()
                for i in range(len(self.images)-1,-1,-1):

                    if wrong_shape[i]:
                        del self.images[i]
                        del self.filenames[i]

            self.detection_run = False
            self.chessboard_status = [False for i in range(len(self.images))]
            self.update_image_display(0)
            self.detect_chessboard_button.setEnabled(True)

            self.next_im_button.setEnabled(True)
            self.prev_im_button.setEnabled(True)
            self.current_filename.show()


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
            dialog.exec()
        elif np.any(self.chessboard_status):
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Ok)
            dialog.setTextFormat(qt.Qt.RichText)
            dialog.setWindowTitle('Calcam - Chessboard Detection')
            dialog.setText("A {:d} x {:d} square chessboard pattern could not be detected in the following {:d} of {:d} images, which will therefore not be included as additional chessboard constraints:".format(self.chessboard_squares_x.value(),self.chessboard_squares_y.value(),np.count_nonzero(self.chessboard_status),len(self.images)))
            dialog.setInformativeText('<br>'.join(['[#{:d}] '.format(i+1) + self.filenames[i] for i in range(len(self.filenames)) if self.chessboard_status[i] ]))
            dialog.setIcon(qt.QMessageBox.Warning)
            dialog.exec()                

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

            if self.original_coords.isChecked():
                coords = 'original'
            else:
                coords = 'display'

            # Loop over chessboard points
            for point in range( np.prod(self.n_chessboard_points) ):
                self.results[-1][1].image_points.append([])

                # Populate coordinates for relevant field
                for field in range(self.n_fields):
                    if self.subview_lookup(impoints[point,0],impoints[point,1],coords=coords) == field:
                        self.results[-1][1].image_points[-1].append([impoints[point,0], impoints[point,1]])
                    else:
                        self.results[-1][1].image_points[-1].append(None)

            if self.original_coords.isChecked():
                self.results[-1][0] = self.image_transformer.original_to_display_image(self.results[-1][0])
                self.results[-1][1] = self.image_transformer.original_to_display_pointpairs(self.results[-1][1])


        self.filenames = [self.filenames[i] for i in range(len(self.filenames)) if self.chessboard_status[i]]

        self.chessboard_source = '{:d} chessboard images loaded from "{:s}"'.format(len(self.filenames),self.src_dir.replace('/',os.sep))

        # And close the window.
        self.done(1)


class NameInputDialog(qt.QDialog):

    def __init__(self,parent,title,question,init_text=None):

        qt.QDialog.__init__(self, parent,qt.Qt.WindowTitleHint | qt.Qt.WindowCloseButtonHint)
        layout = qt.QVBoxLayout()
        label = qt.QLabel(question)
        buttonbox = qt.QDialogButtonBox(qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel,qt.Qt.Horizontal,self)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.text_input = qt.QLineEdit()
        if init_text is not None:
            self.text_input.setText(init_text)
        layout.addWidget(label)
        layout.addWidget(self.text_input)
        layout.addWidget(buttonbox)
        self.setLayout(layout)
        self.setWindowTitle(title)


class AreYouSureDialog(qt.QDialog):

    def __init__(self,parent,title,question):

        qt.QDialog.__init__(self, parent,qt.Qt.WindowTitleHint | qt.Qt.WindowCloseButtonHint)
        layout = qt.QVBoxLayout()
        label = qt.QLabel(question)
        buttonbox = qt.QDialogButtonBox(qt.QDialogButtonBox.Yes | qt.QDialogButtonBox.No,qt.Qt.Horizontal,self)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        layout.addWidget(label)
        layout.addWidget(buttonbox)
        self.setLayout(layout)
        self.setWindowTitle(title)



class ImageMaskDialog(qt.QDialog):

    def __init__(self, parent, image,allow_subviews=True):

        # GUI initialisation
        qt.QDialog.__init__(self, parent,qt.Qt.WindowTitleHint | qt.Qt.WindowCloseButtonHint)
        qt.uic.loadUi(os.path.join(guipath,'qt_designer_files','split_field_dialog.ui'), self)

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
        self.overlay_opacity = int(2.55*self.mask_alpha_slider.value())

        # Callbacks for GUI elements
        self.method_mask.toggled.connect(self.change_method)
        self.method_points.toggled.connect(self.change_method)
        self.no_split.toggled.connect(self.change_method)
        self.method_floodfill.toggled.connect(self.change_method)
        self.mask_alpha_slider.valueChanged.connect(self.update_mask_alpha)
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self.apply)
        self.mask_browse_button.clicked.connect(self.load_mask_image)
        self.fill_reset_button.clicked.connect(lambda : self.change_method(True))
        self.save_image_button.clicked.connect(self.save_image)

        self.points_options.hide()
        self.mask_options.hide()
        self.mask_image = None
        self.points = []

        self.medfilt_pixels.lineEdit().setReadOnly(True)

        self.interactor.cursor_size = 0.05

        self.update_mask_alpha(self.mask_alpha_slider.value())

        self.allow_subviews = allow_subviews
        if not allow_subviews:
            self.method_points.hide()
            self.mask_instructions.setText('Load an image file, the same size as the camera image, with pixels containing the image filled a different colour to pixels with no image.')


        # Start the GUI!
        self.show()
        self.interactor.init()
        self.renderer.Render()
        self.vtkInteractor.Initialize()
        self.interactor.set_image(self.image)

        if self.parent.calibration.n_subviews == 1 and self.parent.calibration.subview_mask.min() == 0:
            self.no_split.setChecked(True)
        else:
            self.method_mask.setChecked(True)
            self.update_fieldmask(mask=self.parent.calibration.get_subview_mask(coords='Display'),mask_formatted=True,names=self.parent.calibration.subview_names)
            


    def refresh_2d(self):
        self.renderer.Render()
        self.qvtkwidget.update()


    def change_method(self,checkstate=None):

        if not checkstate:
            return

        if self.method_points.isChecked():
            self.update_fieldmask(mask=np.zeros(self.image.shape[:2], dtype=np.int8))
            self.points_options.show()
            self.mask_options.hide()
            self.floodfill_options.hide()
            self.mask_alpha_slider.setEnabled(True)
            self.mask_alpha_label.setEnabled(True)
            for point in self.points:
                point[1] = self.interactor.add_active_cursor(point[0])
            if len(self.points) == 2:
                self.interactor.set_cursor_focus(point[1])


        elif self.method_mask.isChecked():
            self.mask_options.show()
            self.points_options.hide()
            self.floodfill_options.hide()
            for point in self.points:
                self.interactor.remove_active_cursor(point[1])
            self.points = []
            self.mask_alpha_slider.setEnabled(True)
            self.mask_alpha_label.setEnabled(True)
            self.update_fieldmask( mask = np.zeros(self.image.shape[:2],dtype=np.int8) )

        elif self.method_floodfill.isChecked():
            for point in self.points:
                self.interactor.remove_active_cursor(point[1])
            self.points = []
            self.mask_alpha_slider.setEnabled(True)
            self.mask_alpha_label.setEnabled(True)
            self.mask_options.hide()
            self.points_options.hide()
            self.floodfill_options.show()
            self.update_fieldmask(mask=np.zeros(self.image.shape[:2], dtype=np.int8))

        else:
            self.mask_options.hide()
            self.points_options.hide()
            self.floodfill_options.hide()
            for point in self.points:
                self.interactor.remove_active_cursor(point[1])
            self.points = []
            self.mask_alpha_slider.setEnabled(False)
            self.mask_alpha_label.setEnabled(False)
            self.update_fieldmask( mask = np.zeros(self.image.shape[:2],dtype=np.int8) )

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
            self.field_checkboxes = []
            if n_fields > 1:
                for field in range(n_fields):
                    self.field_names_boxes.append(qt.QLineEdit())
                    self.field_names_boxes[-1].setStyleSheet("background: rgb({:.0f}, {:.0f}, {:.0f}); color: rgb(255,255,255); border: 1px solid;".format(colours[field][0],colours[field][1],colours[field][2]))
                    self.field_names_boxes[-1].setMaximumWidth(150)
                    self.field_names_boxes[-1].setText(names[field])
                    self.field_checkboxes.append(qt.QCheckBox('Region contains image'))
                    self.field_checkboxes[-1].setChecked(True)
                    self.field_checkboxes[-1].toggled.connect(self.update_field_enables)
                    layout.addWidget(self.field_names_boxes[-1],field,0)
                    layout.addWidget(self.field_checkboxes[-1],field,1)


    def update_field_enables(self):

        for i,cb in enumerate(self.field_checkboxes):

            if self.sender() is cb:
                if cb.isChecked():
                    self.field_names_boxes[i].setEnabled(True)
                else:
                    self.field_names_boxes[i].setEnabled(False)

        mask_image = np.zeros([self.fieldmask.shape[0], self.fieldmask.shape[1], 4], dtype=np.uint8)

        for field in range(self.fieldmask.max()+1):
            inds = np.where(self.fieldmask == field)
            if self.field_checkboxes[field].isChecked():
                mask_image[inds[0], inds[1], 0] = self.field_colours[field][0]
                mask_image[inds[0], inds[1], 1] = self.field_colours[field][1]
                mask_image[inds[0], inds[1], 2] = self.field_colours[field][2]

        mask_image[:, :, 3] = self.overlay_opacity

        self.interactor.set_overlay_image(mask_image)

        self.mask_image = mask_image


    def apply(self):

        if self.fieldmask.max() == 0:
            self.field_names = ['Image']
        else:
            final_fieldmask = np.zeros_like(self.fieldmask)
            self.field_names = []
            i = 0
            for field in range(self.fieldmask.max() + 1):
                inds = np.where(self.fieldmask == field)
                if self.field_checkboxes[field].isChecked():
                    final_fieldmask[inds] = i
                    self.field_names.append(self.field_names_boxes[field].text())
                    i += 1
                else:
                    final_fieldmask[inds] = -1

            self.fieldmask = final_fieldmask

        self.done(1)



    def load_mask_image(self):

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(filedialog.AcceptOpen)
        filedialog.setFileMode(filedialog.ExistingFile)
        filedialog.setWindowTitle('Select Mask File')
        filedialog.setNameFilter('Image Files (*.png *.bmp *.jp2 *.tiff *.tif)')
        filedialog.exec()

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

        elif self.method_floodfill.isChecked():
            newmask = np.zeros((self.image.shape[0]+2,self.image.shape[1]+2),np.uint8)
            try:
                n_channels = self.image.shape[2]
            except IndexError:
                n_channels = 1
            filtered_image = cv2.medianBlur(self.image,self.medfilt_pixels.value())
            cv2.floodFill(filtered_image,newmask,seedPoint=tuple(np.round(im_coords).astype(np.uint32)),newVal=(0,)*n_channels,loDiff=(self.fill_tolerance.value(),)*n_channels,upDiff=(self.fill_tolerance.value(),)*n_channels,flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)
            mask = -1 * ((newmask[1:-1,1:-1] == 1) | (self.fieldmask < 0))
            self.update_fieldmask(mask=mask.astype(np.int8),mask_formatted=True)
 


    def update_fieldmask(self,cursor_moved=None,updated_position=None,mask=None,names=[],mask_formatted=False):

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

            if not mask_formatted:

                if len(mask.shape) > 2:
                    mask = mask.astype(np.float32)
                    for channel in range(mask.shape[2]):
                        mask[:, :, channel] = mask[:, :, channel] + 255 * channel
                    mask = np.sum(mask, axis=2)

                lookup = list(np.unique(mask))
                n_fields = len(lookup)

                if n_fields > 50:
                    raise UserWarning('The loaded mask image contains {:d} different colours. This seems wrong.')
                newmask = np.zeros((mask.shape[0],mask.shape[1]),dtype=np.int8)
                for value in lookup:
                    newmask[mask == value] = lookup.index(value)

                self.fieldmask = newmask

            else:
                n_fields = int(mask.max() + 1)

                self.fieldmask = np.int8(mask)


        self.field_colours = []

        if n_fields > 1 or self.fieldmask.min() < 0 or self.method_floodfill.isChecked():
            
            colourcycle = ColourCycle()
            self.field_colours = [ np.uint8(np.array(next(colourcycle)) * 255) for i in range(n_fields) ]

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

            if len(names) != n_fields:

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
                            names = ['Sub FOV # 1', 'Sub FOV # 2']
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


    def save_image(self):

        config = CalcamConfig()
        filename_filter = config.filename_filters['image']
        fext = filename_filter.split('(*')[1].split(')')[0]

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(filedialog.AcceptSave)

        try:
            filedialog.setDirectory(config.file_dirs['image'])
        except KeyError:
            filedialog.setDirectory(os.path.expanduser('~'))

        filedialog.setFileMode(filedialog.AnyFile)

        filedialog.setWindowTitle('Save As...')
        filedialog.setNameFilter(filename_filter)
        filedialog.exec()
        if filedialog.result() == 1:
            selected_path = str(filedialog.selectedFiles()[0])
            config.file_dirs['image'] = os.path.split(selected_path)[0]
            if not selected_path.endswith(fext):
                selected_path = selected_path + fext

            cv2.imwrite(selected_path,self.image)


class CalibInfoDialog(qt.QDialog):

    def __init__(self,parent,calibration):

        qt.QDialog.__init__(self,parent,qt.Qt.WindowTitleHint | qt.Qt.WindowCloseButtonHint)
        
        text_browse = qt.QTextBrowser()

        self.setWindowTitle('Calibration Information.')

        self.resize(parent.size().width()/2,parent.size().height()/2)

        self.setModal(False)

        text_browse.setHtml('<pre>' + str(calibration) + '<pre>')
        text_browse.setLineWrapMode(text_browse.NoWrap)
        layout = qt.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(text_browse)

        self.show()

