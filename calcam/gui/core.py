# External imports
import sys
import os
import numpy as np
import traceback
import vtk

# Calcam imports
from . import qt_wrapper as qt
from ..cadmodel import CADModel
from ..config import CalcamConfig
from ..calibration import Calibration


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

        try:
            self.cadmodel.add_view(str(self.view_save_name.text()),cam_pos,target,fov,xsection)
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



    def object_from_file(self,obj_type,multiple=False):

        filename_filter = self.config.filename_filters[obj_type]
        start_dir = self.config.file_dirs[obj_type]

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(0)
        filedialog.setDirectory(start_dir)

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
        self.config.file_dirs[obj_type] = os.path.split(str(selected_paths[0]))[0]
        for path in [str(p) for p in selected_paths]:

            if obj_type.lower() == 'calibration':
                obj = Calibration(path)


            objs.append(obj)

        if multiple:
            return objs
        else:
            return objs[0]



    def get_save_filename(self,obj_type):

        filename_filter = self.config.filename_filters[obj_type]
        fext = filename_filter.split('(*')[1].split(')')[0]
        start_dir = self.config.file_dirs[obj_type]

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(1)
        filedialog.setDirectory(start_dir)

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
                self.camera_3d.SetViewAngle(view['y_fov'])
                self.camera_3d.SetPosition(view['cam_pos'])
                self.camera_3d.SetFocalPoint(view['target'])
                self.camera_3d.SetViewUp(0,0,1)
                self.interactor3d.set_xsection(view['xsection'])

            elif view_item.parent() is self.views_root_results or view_item.parent() in self.viewport_calibs.keys():

                view,subfield = self.viewport_calibs[(view_item)]
                
                if subfield is not None:
                    self.set_view_from_calib(view,subfield)

                return


        else:
            self.camera_3d.SetPosition((self.camX.value(),self.camY.value(),self.camZ.value()))
            self.camera_3d.SetFocalPoint((self.tarX.value(),self.tarY.value(),self.tarZ.value()))
            self.camera_3d.SetViewAngle(self.camFOV.value())

        self.update_viewport_info(keep_selection=True)

        self.interactor3d.update_clipping()

        self.refresh_3d()



    def set_view_from_calib(self,calibration,subfield):

        viewmodel = calibration.view_models[subfield]

        self.camera_3d.SetPosition(viewmodel.get_pupilpos())
        self.camera_3d.SetFocalPoint(viewmodel.get_pupilpos() + viewmodel.get_los_direction(calibration.geometry.get_display_shape()[0]/2,calibration.geometry.get_display_shape()[1]/2))
        self.camera_3d.SetViewAngle(calibration.get_fov(subview=subfield)[1])
        self.camera_3d.SetViewUp(-1.*viewmodel.get_cam_to_lab_rotation()[:,1])
        self.interactor3d.set_xsection(None)       

        self.update_viewport_info(keep_selection=True)

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

            self.cadmodel.remove_from_renderer(self.renderer_3d)
            self.cadmodel.unload()

            del self.cadmodel

            # Turn off any wall contour
            self.contour_off.setChecked(True)


        # Create a new one
        self.cadmodel = CADModel( str(self.model_name.currentText()) , str(self.model_variant.currentText()) , self.update_cad_status)

        self.config.default_model = (str(self.model_name.currentText()),str(self.model_variant.currentText()))

        if not self.cad_auto_load.isChecked():
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

        if self.on_model_load is not None:
            self.on_model_load()



    def update_viewport_info(self,keep_selection = False):

        campos = self.camera_3d.GetPosition()
        camtar = self.camera_3d.GetFocalPoint()
        fov = self.camera_3d.GetViewAngle()

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


    def debug_dialog(self,thing):

        message = str(thing)

        dialog = qt.QMessageBox(self)
        dialog.setStandardButtons(qt.QMessageBox.Ok)
        dialog.setTextFormat(qt.Qt.RichText)
        dialog.setWindowTitle('Calcam - Debug Message')
        dialog.setText(message)
        dialog.setIcon(qt.QMessageBox.Information)
        dialog.exec_()


    def on_close(self):

        self.config.save()
        sys.excepthook = sys.__excepthook__