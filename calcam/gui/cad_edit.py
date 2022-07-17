'''
* Copyright 2015-2021 European Atomic Energy Community (EURATOM)
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

import json
import copy

from .core import *
from .vtkinteractorstyles import CalcamInteractorStyle3D
from .. import render
from ..cadmodel import CADModel,ModelFeature
from ..io import ZipSaveFile
from ..misc import import_source, open_file

# CAD viewer window.
# This allows viewing of the CAD model and overlaying raycasted sight-lines, etc.
class CADEdit(CalcamGUIWindow):
 
    def __init__(self, app, parent = None,filename=None):

        CalcamGUIWindow.init(self,'cad_edit.ui',app,parent)

        # Start up with no CAD model
        self.cadmodel = None

        # Set up VTK
        self.qvtkwidget_3d = qt.QVTKRenderWindowInteractor(self.vtk_frame)
        self.vtk_frame.layout().addWidget(self.qvtkwidget_3d,0,0,1,2)
        self.interactor3d = CalcamInteractorStyle3D(refresh_callback=self.refresh_3d,viewport_callback=self.update_viewport_info,newpick_callback=self.add_cursor,cursor_move_callback=self.update_cursor_position)
        self.qvtkwidget_3d.SetInteractorStyle(self.interactor3d)
        self.renderer_3d = vtk.vtkRenderer()
        self.renderer_3d.SetBackground(0, 0, 0)
        self.qvtkwidget_3d.GetRenderWindow().AddRenderer(self.renderer_3d)
        self.camera_3d = self.renderer_3d.GetActiveCamera()


        # Callbacks for GUI elements
        self.viewlist.itemSelectionChanged.connect(self.change_cad_view)
        self.viewlist.itemSelectionChanged.connect(self.update_remove_view_button)
        self.camX.valueChanged.connect(self.change_cad_view)
        self.camY.valueChanged.connect(self.change_cad_view)
        self.camZ.valueChanged.connect(self.change_cad_view)
        self.tarX.valueChanged.connect(self.change_cad_view)
        self.tarY.valueChanged.connect(self.change_cad_view)
        self.tarZ.valueChanged.connect(self.change_cad_view)
        self.cam_roll.valueChanged.connect(self.change_cad_view)
        self.camFOV.valueChanged.connect(self.change_cad_view)
        #self.feature_tree.itemChanged.connect(self.update_feature)
        self.feature_tree.itemSelectionChanged.connect(self.update_current_feature)
        #self.cad_colour_choose_button.clicked.connect(self.set_cad_colour)
        self.save_view_button.clicked.connect(lambda: self.save_view_to_model(True))
        self.control_sensitivity_slider.valueChanged.connect(lambda x: self.interactor3d.set_control_sensitivity(x*0.01))
        self.rmb_rotate.toggled.connect(self.interactor3d.set_rmb_rotate)
        self.model_name_box.textChanged.connect(self.set_model_name)
        self.remove_view_button.clicked.connect(self.remove_view)

        self.action_open.triggered.connect(lambda: self.open())
        self.action_save.triggered.connect(self.save)
        self.action_save_as.triggered.connect(lambda: self.save(saveas=True))
        self.action_new.triggered.connect(self.reset)

        self.new_feature_button.clicked.connect(self.add_feature)
        self.rename_variant_button.clicked.connect(self.rename_variant)
        self.model_variant.currentIndexChanged.connect(self.change_variant)

        self.control_sensitivity_slider.setValue(self.config.mouse_sensitivity)
        self.remove_variant_button.clicked.connect(self.remove_variant)

        self.feature_tree.itemChanged.connect(self.on_featuretree_change)

        self.cad_colour_choose_button.clicked.connect(self.edit_feature)

        self.replace_file_button.clicked.connect(self.edit_feature)
        self.mesh_scale_box.valueChanged.connect(self.edit_feature)
        self.del_feature_button.clicked.connect(self.edit_feature)

        self.new_group_button.clicked.connect(self.add_group)

        self.show_contour_checkbox.toggled.connect(self.update_contour)
        self.load_contour_button.clicked.connect(self.load_wall_contour)
        self.remove_contour_button.clicked.connect(self.remove_wall_contour)

        self.load_formatter_button.clicked.connect(self.load_formatter)
        self.refresh_formatter_button.clicked.connect(self.refresh_formatter)
        self.remove_formatter_button.clicked.connect(self.remove_formatter)

        self.set_default_view_button.clicked.connect(self.change_default_view)

        # Add the event filter which will handle drag & drop in the feature tree
        self.feature_tree.viewport().installEventFilter(self)


        self.cursor = None

        # Start the GUI!
        self.show()

        self.interactor3d.init()
        self.views_root_auto.setHidden(True)
        self.qvtkwidget_3d.GetRenderWindow().GetInteractor().Initialize()

        if filename is not None:
            self.open(filename)
        else:
            self.reset()
            



    def on_featuretree_change(self,item):

        if item is not self.tree_root:
            old_name = self.cad_tree_items[item]
            if len(old_name.split('/')) > 1:
                new_name = '{:s}/{:s}'.format(old_name.split('/')[0],str(item.text(0)))
            else:
                new_name = str(item.text(0))

            if old_name != new_name:

                if old_name in self.cadmodel.groups:

                    glist = self.cadmodel.groups.pop(old_name)
                    self.cadmodel.groups[new_name] = glist
                    self.cad_tree_items[item] = new_name

                    for i,iitem in enumerate(glist):

                        self.cadmodel.groups[new_name][i] = '{:s}/{:s}'.format(new_name,iitem.split('/')[1])

                        fdict = self.model_features[self.cadmodel.model_variant].pop(iitem)
                        self.model_features[self.cadmodel.model_variant]['{:s}/{:s}'.format(new_name,iitem.split('/')[1])] = fdict

                        f = self.cadmodel.features.pop(iitem)
                        self.cadmodel.features['{:s}/{:s}'.format(new_name,iitem.split('/')[1])] = f

                    for treeitem in self.cad_tree_items.keys():
                        if self.cad_tree_items[treeitem] is not None and len(self.cad_tree_items[treeitem].split('/')) == 2 and old_name == self.cad_tree_items[treeitem].split('/')[0]:
                            self.cad_tree_items[treeitem] = '{:s}/{:s}'.format(new_name,self.cad_tree_items[treeitem].split('/')[1])


                else:

                    fdict = self.model_features[self.cadmodel.model_variant].pop(old_name)
                    self.model_features[self.cadmodel.model_variant][new_name] = fdict

                    f = self.cadmodel.features.pop(old_name)
                    self.cadmodel.features[new_name] = f

                    self.cad_tree_items[item] = new_name
                    
                    for group in self.cadmodel.groups:
                        if old_name in self.cadmodel.groups[group]:
                            self.cadmodel.groups[group].remove(old_name)
                            self.cadmodel.groups[group].append(new_name)

        CalcamGUIWindow.update_checked_features(self,item)
        self.update_current_feature()
        fnames = self.model_features[self.cadmodel.model_variant].keys()

        for item,fname in self.cad_tree_items:
            if fname in fnames:
                self.model_features[self.cadmodel.model_variant][fname]['default_enable'] = item.checkState(0) == qt.Qt.Checked

        self.unsaved_changes = True

    def on_close(self):
        self.qvtkwidget_3d.close()


    def change_default_view(self):
        selected_view = str(self.viewlist.selectedItems()[0].text(0)).replace('*','')
        self.cadmodel.initial_view = selected_view
        self.update_model_views(show_default=True,keep_selection=True)


    def add_group(self):

        namedialog = NameInputDialog(self,'Add new part group','Enter a name for the new part group:')
        namedialog.exec()

        if namedialog.result() == 1:

            name = str(namedialog.text_input.text())

            self.cadmodel.groups[name] = []
            self.extra_groups[self.cadmodel.model_variant].append(name)
            self.update_feature_tree()


    def load_wall_contour(self):

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(filedialog.AcceptOpen)
        filedialog.setFileMode(filedialog.ExistingFile)

        filedialog.setWindowTitle('Load wall contour...')
        filedialog.setNameFilter('Data files (*.txt *.csv *.npy)')
        filedialog.exec()

        filename = filedialog.selectedFiles()
        if len(filename) == 1:

            if filename[0].endswith('.npy'):
                contour = np.load(filename[0])
            else:
                for delimiter in ['\t',' ',',']:
                    try:
                        contour = np.loadtxt(filename[0],delimiter=delimiter)
                        break
                    except ValueError:
                        continue

            try:
                if contour.shape[1] != 2:
                    raise UserWarning('Loaded file contains a {:s} array. Expected a 2 column list of R,Z coordinates.'.format(' x '.join([str(d) for d in contour.shape])))
            except NameError:
                raise UserWarning('Could not load numeric data from the selected file.<br>Please select a file containing tab, space or comma delimited data and which is not UTF encoded.')

            self.contour_description.setText('Loaded from file: {:d} points'.format(contour.shape[0]))
            self.wall_contour = contour
            if self.cursor is not None:
                self.show_contour_checkbox.setEnabled(True)
                self.update_contour()



    def remove_wall_contour(self):

        dialog = AreYouSureDialog('Remove wall contour?','Are you sure you want to remove the wall contour?')
        dialog.exec()
        if dialog.result() == 1:
            self.show_contour_checkbox.setChecked(False)
            self.show_contour_checkbox.setEnabled(False)
            self.wall_contour = None
            self.contour_description.setText('No wall contour.')




    def edit_feature(self):

        if self.sender() is self.replace_file_button:
            mesh_path = self.browse_for_mesh()

            if mesh_path is not None:
                self.cadmodel.set_features_enabled(False,self.selected_feature)
                if self.cadmodel.def_file is not None:
                    old_path = self.model_features[self.cadmodel.model_variant][self.selected_feature]['mesh_file']
                    if self.cadmodel.def_file.get_temp_path() in old_path:
                        self.removed_mesh_files.append(old_path.replace(self.cadmodel.def_file.get_temp_path() + os.sep,''))

                self.model_features[self.cadmodel.model_variant][self.selected_feature]['mesh_file'] = mesh_path
                self.cadmodel.features[self.selected_feature].filename = mesh_path
                self.cadmodel.features[self.selected_feature].filetype = mesh_path.split('.')[-1].lower()
                self.cadmodel.features[self.selected_feature].polydata = None
                self.cadmodel.features[self.selected_feature].solid_actor = None
                self.cadmodel.features[self.selected_feature].edge_actor = None
                self.cadmodel.set_features_enabled(True,self.selected_feature)

        elif self.sender() is self.mesh_scale_box:
            self.cadmodel.set_features_enabled(False,self.selected_feature)
            self.model_features[self.cadmodel.model_variant][self.selected_feature]['mesh_scale'] = self.mesh_scale_box.value()
            self.cadmodel.features[self.selected_feature].scale = self.mesh_scale_box.value()
            self.cadmodel.features[self.selected_feature].polydata = None
            self.cadmodel.features[self.selected_feature].solid_actor = None
            self.cadmodel.features[self.selected_feature].edge_actor = None
            self.cadmodel.set_features_enabled(True,self.selected_feature)

        elif self.sender() is self.cad_colour_choose_button:

            self.set_cad_colour()
            self.model_features[self.cadmodel.model_variant][self.selected_feature]['colour'] = self.cadmodel.features[self.selected_feature].colour

        elif self.sender() is self.del_feature_button:
            dialog = AreYouSureDialog(self,'Confirm Delete Model Part','Are you sure you want to remove the model part "{:s}"?'.format(self.selected_feature.split('/')[-1]))
            dialog.exec()
            if dialog.result():

                # Make sure any dragging & dropping is sorted.
                self.update_feature_tree()

                self.cadmodel.set_features_enabled(False,self.selected_feature)
                defdict = self.model_features[self.cadmodel.model_variant].pop(self.selected_feature)
                if self.cadmodel.def_file is not None:
                    if self.cadmodel.def_file.get_temp_path() in defdict['mesh_file']:
                        self.removed_mesh_files.append(defdict['mesh_file'].replace(self.cadmodel.def_file.get_temp_path()+os.sep,''))
                del self.cadmodel.features[self.selected_feature]
                if self.current_group is not None:
                    self.cadmodel.groups[self.current_group].remove(self.selected_feature)
                self.update_feature_tree()
    
        self.update_current_feature()
        self.unsaved_changes = True
        self.refresh_3d()


    def add_variant(self,variant_name=None):

        if variant_name is None:
            dialog = NameInputDialog(self,'Add Model Variant','Enter a name for the new model variant:')

            dialog.exec()
            if dialog.result() == 1:
                variant_name = str(dialog.text_input.text())
            else:
                return

            if variant_name in self.model_features.keys():
                raise UserWarning('A model variant with this name already exists, please choose a unique name.')
            elif variant_name == '':
                return

        self.model_variant.blockSignals(True)
        self.model_variant.addItem(variant_name)
        self.model_variant.setCurrentIndex(-1)
        self.model_variant.blockSignals(False)
        self.cadmodel.variants.append(variant_name)
        self.model_features[variant_name] = {}
        self.extra_groups[variant_name] = []
        self.model_variant.setCurrentIndex(len(self.cadmodel.variants))
        
        self.unsaved_changes = True


    def remove_variant(self):

        if len(self.model_features.keys()) == 1:
            raise UserWarning('Cannot remove the only model variant!')

        dialog = AreYouSureDialog(self,'Remove model variant','Are you sure you want to remove the model variant "{:s}"?'.format(self.cadmodel.model_variant))
        dialog.exec()
        
        if dialog.result() == 1:

            variant_to_remove = self.cadmodel.model_variant
            index_to_remove = self.model_variant.currentIndex()
            if index_to_remove > 1:
                self.model_variant.setCurrentIndex(index_to_remove-1)
            else:
                self.model_variant.setCurrentIndex(index_to_remove+1)
            self.model_variant.removeItem(index_to_remove)
            self.cadmodel.variants.remove(variant_to_remove)
            del self.model_features[variant_to_remove]
            if variant_to_remove in self.cadmodel.mesh_path_roots:
                del self.cadmodel.mesh_path_roots[variant_to_remove]


    def update_current_feature(self):

        self.current_group = None

        selected_features = self.feature_tree.selectedItems()
        if len(selected_features) ==  1 and selected_features[0].checkState(0) == qt.Qt.Checked:
            self.selected_feature = self.cad_tree_items[selected_features[0]]
            self.selected_treeitem = selected_features[0]
        else:
            self.component_settings.setEnabled(False)
            self.del_feature_button.setEnabled(False)
            return

        try:
            feature = self.cadmodel.features[self.selected_feature]
            self.component_settings.setEnabled(True)
            self.del_feature_button.setEnabled(True)
            self.cad_colour_choose_button.setEnabled(True)
        except KeyError:
            if self.selected_feature in self.cadmodel.groups:
                self.current_group = self.selected_feature

            self.component_settings.setEnabled(False)
            self.del_feature_button.setEnabled(False)
            return

        if len(self.selected_feature.split('/')) > 1:
            self.current_group = self.selected_feature.split('/')[0]


        filename = feature.filename
        scale = feature.scale

        filesize = os.path.getsize(filename) / 1024.**2

        if self.cadmodel.def_file is not None:
            filename = filename.replace(self.cadmodel.def_file.get_temp_path(),'').replace('.large','')[2:]

        filename = filename + ' [{:.1f} MiB]'.format(filesize)

        self.mesh_filename.setText(filename)
        self.mesh_scale_box.blockSignals(True)
        self.mesh_scale_box.setValue(scale)
        self.mesh_scale_box.blockSignals(False)

        feature_extent = feature.get_polydata().GetBounds()

        size_str = ''
        for dim in range(3):
            if dim > 0:
                size_str = size_str + ' x '
            dim_size = feature_extent[dim*2+1] - feature_extent[dim*2]
            if dim_size < 0.01:
                size_str = size_str + '{:.0f} mm'.format(dim_size*1e3)
            elif dim_size < 1:
                size_str =  size_str + '{:.1f} cm'.format(dim_size*1e2)
            else:
                size_str =  size_str + '{:.1f} m'.format(dim_size)

        self.part_size.setText(size_str)



    def rename_variant(self):

        dialog = NameInputDialog(self,'Rename Model Variant','Enter a new name for the "{:s}" model variant:'.format(self.cadmodel.model_variant),init_text=self.cadmodel.model_variant)

        dialog.exec()
        if dialog.result() == 1:
            new_name = str(dialog.text_input.text())
        else:
            return

        if new_name in self.model_features.keys():
            raise UserWarning('A model variant with this name already exists, please choose a unique name.')
        elif new_name == '' or new_name == self.cadmodel.model_variant:
            return

        try:
            mesh_root_path = self.cadmodel.mesh_path_roots.pop(self.cadmodel.model_variant)
            self.cadmodel.mesh_path_roots[new_name] = mesh_root_path
        except KeyError:
            pass
        
        grouplist = self.extra_groups.pop(self.cadmodel.model_variant)
        self.extra_groups[new_name] = grouplist

        self.model_variant.setItemText(self.model_variant.currentIndex(),new_name)
        self.cadmodel.variants[self.model_variant.currentIndex()-1] = new_name
        self.model_features[new_name] = self.model_features[self.cadmodel.model_variant]
        del self.model_features[self.cadmodel.model_variant]
        self.cadmodel.model_variant = new_name
        
        if self.tree_root is not None:
            self.feature_tree.blockSignals(True)
            self.tree_root.setText(0,'{:s} ({:s})'.format(self.cadmodel.machine_name,self.cadmodel.model_variant))
            self.feature_tree.blockSignals(False)   

        self.unsaved_changes = True




    def change_variant(self):

        self.update_feature_tree()

        if self.model_variant.currentIndex() == 0:
            self.add_variant()
            return

        self.cadmodel.model_variant = str(self.model_variant.currentText())

        self.cadmodel.remove_from_renderer(self.renderer_3d)
        # Create the features!
        self.cadmodel.features = {}
        self.cadmodel.groups = {}

        for feature_name,feature_def in self.model_features[self.cadmodel.model_variant].items():

            # Get the feature's group, if any
            if len(feature_name.split('/')) > 1:
                group = feature_name.split('/')[0]
                if group not in self.cadmodel.groups.keys():
                    self.cadmodel.groups[group] = [feature_name]
                else:
                    self.cadmodel.groups[group].append(feature_name)

            # Actually make the feature object
            self.cadmodel.features[feature_name] = ModelFeature(self.cadmodel,feature_def,abs_path=True)

        for groupname in self.extra_groups[self.cadmodel.model_variant]:
            if groupname not in self.cadmodel.groups.keys():
                self.cadmodel.groups[groupname] = []
            else:
                self.extra_groups[self.cadmodel.model_variant].remove(groupname)

        self.update_feature_tree()

        self.cadmodel.add_to_renderer(self.renderer_3d)
        self.refresh_3d()


    def set_model_name(self,name):

        if str(name) != '':
            self.cadmodel.machine_name = str(name)
            if self.tree_root is not None:
                self.feature_tree.blockSignals(True)
                self.tree_root.setText(0,'{:s} ({:s})'.format(self.cadmodel.machine_name,self.cadmodel.model_variant))
                self.feature_tree.blockSignals(False)

            self.update_model_views(show_default=True)
            self.unsaved_changes = True


    def reset(self):

        if self.unsaved_changes:
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(qt.QMessageBox.Save|qt.QMessageBox.Discard|qt.QMessageBox.Cancel)
            dialog.setWindowTitle('Save changes?')
            dialog.setText('There are unsaved changes. Save before closing the current file?')
            dialog.setIcon(qt.QMessageBox.Information)
            choice = dialog.exec()
            if choice == qt.QMessageBox.Save:
                self.action_save.trigger()
            elif choice == qt.QMessageBox.Cancel:
                return

        # Dispose of the old model
        if self.cadmodel is not None:

            self.cadmodel.remove_from_renderer(self.renderer_3d)
            self.cadmodel.unload()

            del self.cadmodel


        self.cadmodel = CADModel(status_callback = self.update_cad_status)
        self.cadmodel.discard_changes = True
        self.cadmodel.add_to_renderer(self.renderer_3d)
        self.update_model_views(show_default=True)

        self.coord_formatter = [None,False]
        self.formatter_info.setText('Coordinate formatter: Built-in default')
        self.remove_formatter_button.setEnabled(False)
        self.refresh_formatter_button.setEnabled(False)
        self.load_formatter_button.setText('Load Custom...')

        # Make sure the light lights up the whole model without annoying shadows or falloff.
        light = self.renderer_3d.GetLights().GetItemAsObject(0)
        light.PositionalOn()
        light.SetConeAngle(180)

        self.removed_mesh_files = []

        self.selected_feature = None
        self.selected_treeitem = None
        self.tree_root = None
        self.current_group = None
        self.pending_parent_change = []

        self.model_features = {}
        self.extra_groups = {}

        if self.cursor is not None:
            self.interactor3d.remove_cursor(self.cursor)
            self.cursor = None

        self.coord_info.setText('No cursor exists. Click on the model to create one.')

        self.model_variant.blockSignals(True)
        self.model_variant.clear()
        self.model_variant.addItem('New...')
        self.model_variant.setCurrentIndex(-1)
        self.model_variant.blockSignals(False)

        self.model_name_box.setText('New Model')
        self.add_variant('Default')

        self.show_contour_checkbox.setEnabled(False)
        self.contour_description.setText('No wall contour.')
        self.show_contour_checkbox.setChecked(False)
        self.wall_contour = None
        self.contour_actor = None

        self.unsaved_changes = False


    def open(self,filename=None):

        if filename is None:

            filedialog = qt.QFileDialog(self)
            filedialog.setAcceptMode(filedialog.AcceptOpen)

            try:
                filedialog.setDirectory(self.config.cad_def_paths[0])
            except:
                filedialog.setDirectory(os.path.expanduser('~'))


            filedialog.setFileMode(filedialog.ExistingFile)

            filedialog.setWindowTitle('Open...')
            filedialog.setNameFilter('Calcam CAD model definitions (*.ccm)')
            filedialog.exec()

            if filedialog.result() == 1:
                filename = filedialog.selectedFiles()[0]
            else:
                return

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))

        self.reset()

        # Open the model
        self.cadmodel = CADModel( filename , status_callback = self.update_cad_status)

        self.cadmodel.discard_changes = True

        self.model_name_box.setText(self.cadmodel.machine_name)

        self.model_variant.blockSignals(True)
        self.model_variant.clear()
        self.model_variant.addItem('New...')
        self.model_variant.addItems(self.cadmodel.variants)
        self.model_variant.setCurrentIndex(self.cadmodel.variants.index(self.cadmodel.model_variant)+1)
        self.model_variant.blockSignals(False)
        self.cadmodel.add_to_renderer(self.renderer_3d)

        self.extra_groups = {}

        self.statusbar.showMessage('Setting up CAD model...')


        with self.cadmodel.def_file.open_file( 'model.json','r' ) as f:
            model_def = json.load(f)

        self.model_features = model_def['features']

        for variant in self.model_features.keys():
            self.extra_groups[variant] = []
            path_root = model_def['mesh_path_roots'][variant]
            for feature in self.model_features[variant].values():
                feature['mesh_file'] = os.path.join(self.cadmodel.def_file.get_temp_path(),path_root,feature['mesh_file'])


        # Make sure the light lights up the whole model without annoying shadows or falloff.
        light = self.renderer_3d.GetLights().GetItemAsObject(0)
        light.PositionalOn()
        light.SetConeAngle(180)

        # Put the camera in some reasonable starting position
        self.camera_3d.SetViewAngle(90)
        self.camera_3d.SetViewUp((0,0,1))
        self.camera_3d.SetFocalPoint(0,0,0)

        self.update_model_views(show_default = True)

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

        self.update_feature_tree()

        self.remove_view_button.setEnabled(True)
        self.set_default_view_button.setEnabled(True)

        if self.cadmodel.usermodule is None:
            self.formatter_info.setText('Coordinate formatter: Built-in default')
            self.load_formatter_button.setText('Load Custom...')
            self.refresh_formatter_button.setEnabled(False)
            self.remove_formatter_button.setEnabled(False)
            self.coord_formatter = [None,False]
        else:
            self.formatter_info.setText('Coordinate formatter: Model Specific')
            self.load_formatter_button.setText('Edit...')
            self.refresh_formatter_button.setEnabled(True)
            self.remove_formatter_button.setEnabled(True)
            self.coord_formatter = [self.cadmodel.usermodule,True]

        if self.cadmodel.wall_contour is not None:
            self.contour_description.setText('Stored in model: {:d} points'.format(self.cadmodel.wall_contour.shape[0]))
            self.wall_contour = self.cadmodel.wall_contour
            self.remove_contour_button.setEnabled(True)
        else:
            self.wall_contour = None
            self.contour_description.setText('No wall contour stored in model.')
            self.remove_contour_button.setEnabled(False)


        self.refresh_3d()

        if self.cadmodel.def_file.is_readonly():
            self.action_save.setEnabled(False)
        else:
            self.action_save.setEnabled(True)

        self.unsaved_changes = False
        self.app.restoreOverrideCursor()


    def update_remove_view_button(self):

        items = self.viewlist.selectedItems()
        if len(items) > 0:
            self.remove_view_button.setEnabled(True)
            self.set_default_view_button.setEnabled(True)
        else:
            self.remove_view_button.setEnabled(False)
            self.set_default_view_button.setEnabled(False)


    def remove_view(self):
        
        selected_view = str(self.viewlist.selectedItems()[0].text(0)).replace('*','')
        dialog = AreYouSureDialog(self,'Remove preset view','Are you sure you want to remove the view "{:s}"?'.format(selected_view))
        dialog.exec()
        if dialog.result() == 1:
            self.cadmodel.views.pop(selected_view)
            if self.cadmodel.initial_view == selected_view:
                self.cadmodel.initial_view = None

            self.update_model_views(show_default=True,keep_selection=True)
            self.unsaved_changes = True


    def refresh_formatter(self):

        self.coord_formatter[1] = False
        self.coord_info.setText('')

        try:
            codepath = self.coord_formatter[0].__path__[0]
        except AttributeError:
            codepath = self.coord_formatter[0].__file__

        try:
            self.coord_formatter[0] = import_source(codepath)
        except Exception:
            tb_info = ''.join(traceback.format_exception(*sys.exc_info(), limit=3))
            raise UserWarning('Error while re-importing coordinate formatter: ',tb_info.replace('\n','<br>'))

        self.validate_formatter(self.coord_formatter[0])
        self.coord_formatter[1] = True

        if self.cursor is not None:
            self.update_cursor_position(0,self.interactor3d.get_cursor_coords(self.cursor))




    def remove_formatter(self):

        self.cadmodel.usermodule = None
        self.coord_formatter = [None,False]
        self.formatter_info.setText('Coordinate formatter: Built-in default')
        self.remove_formatter_button.setEnabled(False)
        self.refresh_formatter_button.setEnabled(False)
        self.load_formatter_button.setText('Load Custom...')
        if self.cursor is not None:
            self.update_cursor_position(self.cursor,self.interactor3d.get_cursor_coords(self.cursor))        


    def load_formatter(self):

        # If we already have a formatter and the user clicks the button to edit:
        if self.coord_formatter[0] is not None:
            try:
                edit_path = self.coord_formatter[0].__path__[0]
            except AttributeError:
                edit_path = self.coord_formatter[0].__file__
            open_file(edit_path)

        # Or if we have no formatter loaded and we want to load one:
        else:
            filedialog = qt.QFileDialog(self)
            filedialog.setAcceptMode(filedialog.AcceptOpen)
            filedialog.setFileMode(filedialog.ExistingFile)

            filedialog.setWindowTitle('Load custom coordinate formatter...')
            filedialog.setNameFilter('Python files (*.py)')
            filedialog.exec()

            if len(filedialog.selectedFiles()) == 1:

                source_path = str(filedialog.selectedFiles()[0])

                if os.path.split(source_path)[1].lower() == '__init__.py':
                    source_path = os.path.split(source_path)[0]

                try:
                    usermodule = import_source(source_path)
                except Exception:
                    tb_info = ''.join(traceback.format_exception(*sys.exc_info(), limit=3))
                    raise UserWarning('Error while importing coordinate formatter:',tb_info.replace('\n','<br>'))

                if self.validate_formatter(usermodule):
                    self.coord_formatter = [usermodule,True]
                    try:
                        self.formatter_info.setText('Coordinate formatter: loaded from "{:s}"'.format(usermodule.__path__[0]))
                    except AttributeError:
                        self.formatter_info.setText('Coordinate formatter: loaded from "{:s}"'.format(usermodule.__file__))

                    self.remove_formatter_button.setEnabled(True)
                    self.refresh_formatter_button.setEnabled(True)
                    self.load_formatter_button.setText('Edit...')
                    if self.cursor is not None:
                        self.update_cursor_position(0,self.interactor3d.get_cursor_coords(self.cursor))


                

    def validate_formatter(self,usermodule):

        if callable(usermodule.format_coord):
            try:
                test_out = usermodule.format_coord( (0.1,0.1,0.1) )
                if type(test_out) == str:
                    return True
                else:
                    raise UserWarning('Loaded format_coord() function must return a string; the loaded function returned type "{:s}"'.format(type(test_out)))
            except Exception:
                tb_info = ''.join(traceback.format_exception(*sys.exc_info(), limit=3))
                raise UserWarning('Exception when testing custom format_coord() function:<br>{:s}'.format(tb_info.replace('\n','<br>')))
        else:
            raise UserWarning('No "format_coord()" function was found in the selected python module.')


    def add_feature(self):

        # This is run here to make sure dragging & dropping has been sorted out.
        self.update_feature_tree()

        set_view = False

        if len(self.cadmodel.features.keys()) == 0:
            set_view = True
        else:
            old_extent = self.cadmodel.get_extent()
            
        mesh_paths = self.browse_for_mesh(multiple=True)

        if len(mesh_paths) == 0:
            return

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        for mesh_path in mesh_paths:
            init_name = os.path.split(mesh_path)[-1][:-4]

            if self.current_group is not None:
                init_name = '{:s}/{:s}'.format(self.current_group,init_name)
                self.cadmodel.groups[self.current_group].append(init_name)

            feature_dict = {'mesh_file':mesh_path,'default_enable':False,'mesh_scale':self.mesh_scale_box.value(),'colour':(0.75,0.75,0.75)}

            self.model_features[self.cadmodel.model_variant][init_name] = copy.copy(feature_dict)
            self.cadmodel.features[init_name] = ModelFeature(self.cadmodel,feature_dict,abs_path=True)
            self.cadmodel.set_features_enabled(True,init_name)
            self.model_features[self.cadmodel.model_variant][init_name]['default_enable'] = True

        self.selected_feature = init_name
        self.update_feature_tree()

        if not set_view:
            new_extent = self.cadmodel.get_extent()
            if np.any(np.abs(new_extent) > np.abs(old_extent)):
                set_view = True

        if set_view:
            self.set_view_to_whole()
        else:
            self.refresh_3d()

        self.app.restoreOverrideCursor()


    def set_view_to_whole(self):

        fov = 60
        self.camFOV.setValue(fov)
        model_extent = self.cadmodel.get_extent()
        z_extent = model_extent[5]-model_extent[4]     

        R = z_extent/(2*np.tan(3.14159*fov/360))

        phi_cam = np.arctan2(self.camY.value(),self.camX.value())

        self.camera_3d.SetPosition(R * np.cos(phi_cam),R * np.sin(phi_cam),0.)
        self.interactor3d.set_roll(0.)
        self.camera_3d.SetFocalPoint( (0.,0.,0.) )

        self.update_viewport_info()
        self.interactor3d.update_clipping()
        self.refresh_3d()



    def update_feature_tree(self):

        set_selected = None

        # Make sure any drag and dropping has been applied to the model
        for feature in self.pending_parent_change:
            feature_name = self.cad_tree_items[feature]
            if self.selected_feature == feature_name:
                current_feature = True
            else:
                current_feature = False

            fo = self.cadmodel.features.pop(feature_name)
            parent = feature.parent()
            fdict = self.model_features[self.cadmodel.model_variant][feature_name]
            del self.model_features[self.cadmodel.model_variant][feature_name]

            if len(feature_name.split('/')) == 2:
                self.cadmodel.groups[feature_name.split('/')[0]].remove(feature_name)
                feature_name = feature_name.split('/')[1]

            if parent is self.tree_root:
                self.model_features[self.cadmodel.model_variant][feature_name] = fdict
                self.cadmodel.features[feature_name] = fo
                if current_feature:
                    self.selected_feature = feature_name
            else:
                self.model_features[self.cadmodel.model_variant]['{:s}/{:s}'.format(parent.text(0),feature_name)] = fdict
                self.cadmodel.features['{:s}/{:s}'.format(parent.text(0),feature_name)] = fo
                self.cadmodel.groups[parent.text(0)].append('{:s}/{:s}'.format(parent.text(0),feature_name))
                if current_feature:
                    self.selected_feature = '{:s}/{:s}'.format(parent.text(0),feature_name)


        self.pending_parent_change = []


        # -------------------------- Populate the model feature tree ------------------------------
        self.feature_tree.blockSignals(True)
        self.feature_tree.clear()

        self.cad_tree_items = DodgyDict()

        invisible_root = self.feature_tree.invisibleRootItem()
        invisible_root.setFlags(qt.Qt.ItemIsEnabled)
        self.feature_tree.addTopLevelItem(invisible_root)

        treeitem_top = qt.QTreeWidgetItem(invisible_root,['{:s} ({:s})'.format(self.cadmodel.machine_name,self.cadmodel.model_variant)])
        treeitem_top.setFlags( qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable | qt.Qt.ItemIsDropEnabled)
        
        treeitem_top.setExpanded(True)
        self.tree_root = treeitem_top

        self.cad_tree_items[treeitem_top] = None

        group_items = {}

        enabled_features = self.cadmodel.get_enabled_features()


        # We need to add the group items first, to make the tree look sensible:
        for groupname in self.cadmodel.groups.keys():
            newitem = qt.QTreeWidgetItem(treeitem_top,[groupname])
            newitem.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled | qt.Qt.ItemIsDropEnabled | qt.Qt.ItemIsEditable)
            newitem.setExpanded(True)
            self.cad_tree_items[newitem] = groupname
            group_items[groupname] = newitem

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
            newitem.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled | qt.Qt.ItemIsDragEnabled | qt.Qt.ItemIsEditable)

            self.cad_tree_items[newitem] = feature

            if feature == self.selected_feature:
                set_selected = newitem

        self.feature_tree.blockSignals(False)

        self.update_feature_tree_checks()

        if set_selected is None:
            self.component_settings.setEnabled(False)
            self.selected_feature = None
            self.selected_treeitem = None
        else:
            set_selected.setSelected(True)
            self.selected_treeitem = set_selected
        # ---------------------------------------------------------------------------------------


    # Function to handle drag & drop events in the feature tree i.e. moving a feature
    # between groups. The name of the function has to be "eventFilter" because of 
    # the way it hooks in to Qt.
    def eventFilter(self,object,event):

        if qt.qt_ver < 6:
            drop_event = qt.QEvent.Drop
        else:
            drop_event = qt.QEvent.Type.Drop

        if event.type() == drop_event:

            if self.selected_treeitem is not None and self.selected_treeitem not in self.pending_parent_change:
                self.pending_parent_change.append(self.selected_treeitem)

            self.unsaved_changes = True

        return False

        


    def add_cursor(self,coords):

        if self.interactor3d.focus_cursor is None:
            self.cursor = self.interactor3d.add_cursor(coords)
            self.interactor3d.set_cursor_focus(self.cursor)
            self.update_cursor_position(self.cursor,coords)
            



    def update_cursor_position(self,cursor_id,position):

        if self.coord_formatter[0] is not None:
            if self.coord_formatter[1]:
                self.coord_info.setText(self.coord_formatter[0].format_coord(position))
            else:
                self.coord_info.setText('Custom coordinate formatter is not functional!')
        else:
            self.coord_info.setText(self.cadmodel.format_coord(position))

        if self.wall_contour is not None:
            self.show_contour_checkbox.setEnabled(True)
            if self.show_contour_checkbox.isChecked():
                self.update_contour()


    def browse_for_mesh(self,multiple=False):

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(filedialog.AcceptOpen)
        if multiple:
            filedialog.setFileMode(filedialog.ExistingFiles)
        else:
            filedialog.setFileMode(filedialog.ExistingFile)

        filedialog.setWindowTitle('Select Mesh File...')
        filedialog.setNameFilter('Supported mesh files (*.stl *.obj)')
        filedialog.exec()

        if filedialog.result() == 1:
            mesh_paths = [str(path).replace(os.sep,'/') for path in filedialog.selectedFiles()]
            if multiple:
                return mesh_paths
            else:
                return mesh_paths[0]
        else:
            if multiple:
                return []
            else:
                return None


    def update_contour(self):

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))

        if self.contour_actor is not None:
            try:
                self.interactor3d.remove_extra_actor(self.contour_actor)
            except:
                self.renderer_3d.RemoveActor(self.contour_actor)
            self.contour_actor = None

        if self.show_contour_checkbox.isChecked():

            cursor_pos = self.interactor3d.get_cursor_coords(0)
            phi = np.arctan2(cursor_pos[1],cursor_pos[0])
            self.contour_actor = render.get_wall_contour_actor(self.wall_contour,'contour',phi)
            self.contour_actor.GetProperty().SetLineWidth(3)
            self.contour_actor.GetProperty().SetColor((1,0,0))
            self.interactor3d.add_extra_actor(self.contour_actor)

        self.refresh_3d()
        self.app.restoreOverrideCursor()


    def save(self,saveas=False):

        if self.cadmodel.initial_view is None:
            raise UserWarning('No default viewport has been set up. You must set a default view on the "Viewports" tab before saving the model definition.')

        if self.coord_formatter[0] is not None and self.coord_formatter[1] == False:
            raise UserWarning('The current user-defined 3D coordinate formatting function appears to be broken. You must fix and refresh this code or remove it (In the "Additional Information" tab) before saving.')


        model_def_dict = {}
        add_path_prompt = None

        if self.cadmodel.def_file is None or saveas:

            filedialog = qt.QFileDialog(self)
            filedialog.setAcceptMode(filedialog.AcceptSave)

            try:
                filedialog.setDirectory(self.config.cad_def_paths[0])
            except:
                filedialog.setDirectory(os.path.expanduser('~'))

            filedialog.setFileMode(filedialog.AnyFile)

            filedialog.setWindowTitle('Save As...')
            filedialog.setNameFilter('Calcam CAD model definitions (*.ccm)')
            filedialog.exec()

            if filedialog.result() == 1:
                filename = filedialog.selectedFiles()[0]
            else:
                return

            if not filename.endswith('.ccm'):
                filename = filename + '.ccm'
            defpath = os.path.split(filename)[0].replace('/',os.path.sep)
            if defpath not in self.config.cad_def_paths:
                add_path_prompt = defpath

            if self.cadmodel.def_file is not None:
                old_def_file = self.cadmodel.def_file

            self.cadmodel.def_file = ZipSaveFile(filename,'rw')

            model_def_dict['mesh_path_roots'] = {}

            self.action_save.setEnabled(True)

        else:
            model_def_dict['mesh_path_roots'] = self.cadmodel.mesh_path_roots


        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.statusbar.showMessage('Saving to file {:s}...'.format(self.cadmodel.def_file.filename))
        
        self.update_feature_tree()

        model_def_dict['features'] = copy.deepcopy(self.model_features)

        for old_mesh in self.removed_mesh_files:
            self.cadmodel.def_file.remove(old_mesh)

        for variant in self.cadmodel.variants:

            if variant not in model_def_dict['mesh_path_roots']:
                model_def_dict['mesh_path_roots'][variant] = '.large/{:s}'.format(variant.replace(' ','_').lower())

            # Existing working directory for file
            tmp_path = self.cadmodel.def_file.get_temp_path().replace(os.sep,'/')

            for feature in model_def_dict['features'][variant]:

                # Where the mesh file currently is
                current_path = self.model_features[variant][feature]['mesh_file'].replace(os.sep,'/')


                # If the file isn't already in the zip file, add it and set the mesh location
                # to the new copy.
                if tmp_path not in current_path:
                    fname = os.path.split(current_path)[1]
                    rel_mesh_path = os.path.join(model_def_dict['mesh_path_roots'][variant],fname)
                    self.cadmodel.def_file.add(current_path,rel_mesh_path,replace=True)
                    self.model_features[variant][feature]['mesh_file'] = os.path.join(tmp_path,rel_mesh_path)
                else:
                    fname = current_path.replace(os.path.join(tmp_path,model_def_dict['mesh_path_roots'][variant]).replace(os.sep,'/'),'')
                    if fname.startswith('/'):
                        fname = fname[1:]

                # Just the filename will be written to the mesh_path field.
                model_def_dict['features'][variant][feature]['mesh_file'] = fname


        model_def_dict['views'] = self.cadmodel.views
        model_def_dict['default_variant'] = self.cadmodel.model_variant
        model_def_dict['machine_name'] = str(self.model_name_box.text())
        model_def_dict['initial_view'] = self.cadmodel.initial_view


        if self.coord_formatter[0] is not None:
            try:
                codepath = self.coord_formatter[0].__path__[0]
            except AttributeError:
                codepath = self.coord_formatter[0].__file__

            if self.cadmodel.def_file.get_temp_path() not in codepath:
                self.cadmodel.def_file.add_usercode(codepath,replace=True)
        else:
            self.cadmodel.def_file.clear_usercode()

        self.removed_mesh_files = []

        with self.cadmodel.def_file.open_file( 'model.json','w' ) as f:
            json.dump(model_def_dict,f,indent=4,sort_keys=True)        

        if self.wall_contour is not None:
            np.savetxt(os.path.join(self.cadmodel.def_file.get_temp_path(),'wall_contour.txt'),self.wall_contour,fmt='%.4f')

        self.cadmodel.def_file.update()

        self.unsaved_changes = False

        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()

        if add_path_prompt:
            dialog = AreYouSureDialog(self,'Add to calcam configuration?',"Model saved.<br><br>The folder it is saved in:<br><br> {:s}<br><br>is currently not in Calcam's cad model search path,<br>so this model will not yet be detected by Calcam.<br><br>Would you like to add this path to Calcam's<br>model search path configuration now?".format(add_path_prompt))
            dialog.exec()
            if dialog.result() == 1:
                self.config.cad_def_paths.append(add_path_prompt)
                self.config.save()
