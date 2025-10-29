'''
* Copyright 2015-2025 European Atomic Energy Community (EURATOM)
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

import time

from .core import *
from .vtkinteractorstyles import CalcamInteractorStyle3D
from .. import render

opacity_params = {'volume':(1,3.45),'lines':(5,2.66),'wall_coverage':(10,1.17)}

def nextcheckstate(checkbox):
    """
    Cycles a checkbox to its next state. This is used to overload the
    usual nextCheckState() method of a tri-state qCheckBox so we get a
    tri-state checkbox which can't be set partially checked by the user.

    Parameters:
        checkbox : A QCheckBox instance on which to act.

    """
    if checkbox.checkState() == qt.Qt.Checked:
        checkbox.setCheckState(qt.Qt.Unchecked)
    else:
        checkbox.setCheckState(qt.Qt.Checked)


# CAD viewer window.
# This allows viewing of the CAD model and overlaying raycasted sight-lines, etc.
class Viewer(CalcamGUIWindow):
 
    def __init__(self, app, parent = None):

        CalcamGUIWindow.init(self,'viewer.ui',app,parent)

        # Start up with no CAD model
        self.cadmodel = None

        # Set up VTK
        self.qvtkwidget_3d = qt.QVTKRenderWindowInteractor(self.vtk_frame)
        self.vtk_frame.layout().addWidget(self.qvtkwidget_3d,0,0,1,2)
        self.interactor3d = CalcamInteractorStyle3D(refresh_callback=self.refresh_3d,viewport_callback=self.update_viewport_info,newpick_callback=self.add_cursor,cursor_move_callback=self.update_cursor_position,resize_callback=self.on_resize,save_coords_callback=self.save_cursor_coords)
        self.qvtkwidget_3d.SetInteractorStyle(self.interactor3d)
        self.renderer_3d = vtk.vtkRenderer()
        self.renderer_3d.SetBackground(0, 0, 0)
        self.qvtkwidget_3d.GetRenderWindow().AddRenderer(self.renderer_3d)
        self.camera_3d = self.renderer_3d.GetActiveCamera()

        self.action_save_coord.triggered.connect(self.save_cursor_coords)

        self.populate_models()
            

        # Callbacks for GUI elements
        self.viewlist.itemSelectionChanged.connect(self.change_cad_view)
        self.camX.valueChanged.connect(self.change_cad_view)
        self.camY.valueChanged.connect(self.change_cad_view)
        self.camZ.valueChanged.connect(self.change_cad_view)
        self.tarX.valueChanged.connect(self.change_cad_view)
        self.tarY.valueChanged.connect(self.change_cad_view)
        self.tarZ.valueChanged.connect(self.change_cad_view)
        self.cam_roll.valueChanged.connect(self.change_cad_view)
        self.camFOV.valueChanged.connect(self.change_cad_view)
        self.sightlines_list.itemChanged.connect(self.update_sightlines)
        self.lines_3d_list.itemChanged.connect(self.update_lines)
        self.load_model_button.clicked.connect(self.load_model)
        self.feature_tree.itemChanged.connect(self.update_checked_features)
        self.feature_tree.itemSelectionChanged.connect(self.update_cadtree_selection)
        self.sightline_opacity_slider.valueChanged.connect(self.update_sightlines)
        self.rendertype_edges.toggled.connect(self.toggle_wireframe)
        self.viewport_load_calib.clicked.connect(self.load_viewport_calib)
        self.sightlines_load_button.clicked.connect(self.update_sightlines)
        self.pick_sightlines_colour.clicked.connect(self.update_sightlines)
        self.sightlines_list.itemSelectionChanged.connect(self.update_selected_sightlines)
        self.sightline_type_volume.clicked.connect(self.update_sightlines)
        self.sightline_type_lines.clicked.connect(self.update_sightlines)
        self.sightline_type_wallcoverage.clicked.connect(self.update_sightlines)
        self.sightlines_legend_checkbox.toggled.connect(self.update_sightlines)
        self.render_button.clicked.connect(self.do_render)
        self.render_cam_view.toggled.connect(self.change_render_type)
        self.render_unfolded_view.toggled.connect(self.change_render_type)
        self.centre_at_cursor.toggled.connect(self.update_unfolded_render_settings)
        self.unfolded_auto_size.toggled.connect(self.update_unfolded_render_settings)
        self.render_coords_combobox.currentIndexChanged.connect(self.update_render_coords)
        self.render_load_button.clicked.connect(self.load_render_result)
        self.cad_colour_reset_button.clicked.connect(self.set_cad_colour)
        self.cad_colour_choose_button.clicked.connect(self.set_cad_colour)
        self.save_view_button.clicked.connect(self.save_view_to_model)
        self.save_colours_button.clicked.connect(self.save_cad_colours)
        self.model_name.currentIndexChanged.connect(self.populate_model_variants)
        self.load_lines_button.clicked.connect(self.update_lines)
        self.pick_lines_colour.clicked.connect(self.update_lines)
        self.lines_3d_list.itemSelectionChanged.connect(self.update_selected_lines)
        self.control_sensitivity_slider.valueChanged.connect(lambda x: self.interactor3d.set_control_sensitivity(x*0.01))
        self.rmb_rotate.toggled.connect(self.interactor3d.set_rmb_rotate)
        self.sightlines_legend = None
        self.render_calib = None
        self.line_width_box.valueChanged.connect(self.update_lines)
        self.marker_diameter_box.valueChanged.connect(self.update_lines)
        self.remove_lines_button.clicked.connect(self.update_lines)
        self.coords_legend_checkbox.toggled.connect(self.update_legend)
        self.slice_apply_button.clicked.connect(self.update_slicing)
        self.no_slicing_rb.clicked.connect(self.update_slicing_options)
        self.cakeslice_rb.clicked.connect(self.update_slicing_options)
        self.chordslice_rb.clicked.connect(self.update_slicing_options)
        self.zslice_checkbox.toggled.connect(self.update_slicing_options)

        self.lines_toroidal_angle_box.valueChanged.connect(self.update_lines)
        self.line_width_box.valueChanged.connect(self.update_lines)
        self.lines_frustrum_d0_box.valueChanged.connect(self.update_lines)
        self.lines_frustrum_angle_box.valueChanged.connect(self.update_lines)
        self.marker_diameter_box.valueChanged.connect(self.update_lines)
        self.lines_toroidal_angle_rb.clicked.connect(self.update_lines)
        self.lines_lines_rb.clicked.connect(self.update_lines)
        self.lines_frustrum_rb.clicked.connect(self.update_lines)
        self.lines_points_rb.clicked.connect(self.update_lines)
        self.lines_toroidal_angle_rb.clicked.connect(self.update_lines)
        self.lines_rz_to_3d_rb.clicked.connect(self.update_lines)
        self.lines_opacity_slider.valueChanged.connect(self.update_lines)
        self.load_mesh_file_button.clicked.connect(self.load_extra_mesh)
        self.remove_mesh_button.clicked.connect(self.remove_extra_mesh)

        self.lines_toroidal_angle_to_cursor.clicked.connect(lambda : self.lines_toroidal_angle_box.setValue(self.cursor_angles[0]))

        self.save_coords_button.clicked.connect(self.export_cursor_coords)

        self.control_sensitivity_slider.setValue(int(self.config.mouse_sensitivity))

        self.proj_perspective.toggled.connect(self.set_projection)

        self.del_coord_button.clicked.connect(self.delete_saved_coord)

        self.extra_mesh_actors = {}

        self.model_actors = {}

        self.cursor_angles = (None,None)

        self.line_actors = DodgyDict()

        self.saved_coords_item = None

        self.wall_contour_listitem = None

        self.sightlines = DodgyDict()
        self.colour_q = []
        self.model_custom_colour = None
        self.viewport_calibs = DodgyDict()

        self.colourcycle = ColourCycle()

        self.tabWidget.setTabEnabled(1,False)
        self.tabWidget.setTabEnabled(2,False)
        self.tabWidget.setTabEnabled(3,False)
        self.tabWidget.setTabEnabled(4,False)

        self.render_coords_text.setHidden(True)
        self.render_coords_combobox.setHidden(True)
        self.render_coords_combobox.setHidden(True)
        self.rendersettings_calib_label.setHidden(True)
        self.render_calib_namelabel.setHidden(True)
        self.render_load_button.setHidden(True)
        self.unfolded_render_settings.setHidden(True)

        # Overload the nextCheckState() method of these checkboxes so that I can use them
        # as tri-state from the code but the user can only toggle them between fully checked and
        # un-checked. The "proper" way to do this would be to write my own subclass of qCheckBox
        # which overloads this method and then insert that in the GUI, but since it's built in QT
        # designer I'm doing this instead, which feels a little hacky but works fine :)
        #self.enable_lines_checkbox.nextCheckState = lambda : nextcheckstate(self.enable_lines_checkbox)
        #self.enable_points_checkbox.nextCheckState = lambda: nextcheckstate(self.enable_points_checkbox)

        self.cursor_coords_table.setColumnCount(3)
        self.cursor_coords_table.setHorizontalHeaderLabels(['X','Y','Z'])

        # Start the GUI!
        self.show()

        self.interactor3d.init()
        self.views_root_auto.setHidden(False)
        self.qvtkwidget_3d.GetRenderWindow().GetInteractor().Initialize()


    def load_extra_mesh(self):

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(filedialog.AcceptOpen)
        filedialog.setFileMode(filedialog.ExistingFile)

        filedialog.setWindowTitle('Select Mesh File...')
        filedialog.setNameFilter('Supported 3D mesh files (*.stl *.obj)')
        filedialog.exec()

        if filedialog.result() == filedialog.Accepted:

            filepath = filedialog.selectedFiles()[0]

            import_settings = MeshImportDialog(self,filepath)
            result = import_settings.exec()

            if result == import_settings.Accepted:

                self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))

                if filepath.lower().endswith('stl'):
                    reader = vtk.vtkSTLReader()
                elif filepath.lower().endswith('obj'):
                    reader = vtk.vtkOBJReader()

                reader.SetFileName(filepath)
                reader.Update()

                transformer = vtk.vtkTransformPolyDataFilter()

                transform = vtk.vtkTransform()
                transform.PostMultiply()
                scale = import_settings.mesh_scale_box.value()
                if import_settings.handedness_box.currentText().lower() == 'left-handed':
                    transform.Scale(scale,scale,-scale)
                elif import_settings.handedness_box.currentText().lower() == 'right-handed':
                    transform.Scale(scale, scale, scale)

                if import_settings.meshup_box.currentText() == '+X':
                    transform.RotateY(-90)
                elif import_settings.meshup_box.currentText() =='-X':
                    transform.RotateY(90)
                elif import_settings.meshup_box.currentText() == '+Y':
                    transform.RotateX(90)
                elif import_settings.meshup_box.currentText() == '-Y':
                    transform.RotateX(-90)
                elif import_settings.meshup_box.currentText() == '-Z' and import_settings.handedness_box.currentText().lower() == 'right-handed':
                    transform.RotateX(180)
                elif import_settings.meshup_box.currentText() == '+Z' and import_settings.handedness_box.currentText().lower() == 'left-handed':
                    transform.RotateX(180)

                transform.RotateZ(import_settings.z_rotate_box.value())
                transformer.SetInputData(reader.GetOutput())
                transformer.SetTransform(transform)
                transformer.Update()

                if import_settings.handedness_box.currentText().lower() == 'left-handed':
                    reverser = vtk.vtkReverseSense()
                    reverser.ReverseNormalsOff()
                    reverser.ReverseCellsOn()
                    reverser.SetInputData(transformer.GetOutput())
                    reverser.Update()
                    polydata = reverser.GetOutput()
                    transformer.SetInputData(reverser.GetOutput())
                elif import_settings.handedness_box.currentText().lower() == 'right-handed':
                    polydata = transformer.GetOutput()

                # Remove all the lines from the PolyData. As far as I can tell for "normal" mesh files this shouldn't
                # remove anything visually important, but it avoids running in to issues with vtkFeatureEdges trying to allocate
                # way too much memory in VTK 9.1+.
                polydata.SetLines(vtk.vtkCellArray())

                mapper =  vtk.vtkPolyDataMapper()
                mapper.SetInputData( polydata )

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                treeitem_top = qt.QTreeWidgetItem([os.path.split(filepath)[-1]])
                treeitem_top.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEditable)

                self.feature_tree.addTopLevelItem(treeitem_top)
                self.cad_tree_items[treeitem_top] = actor
                treeitem_top.setCheckState(0,qt.Qt.Checked)

                self.app.restoreOverrideCursor()


    def update_checked_features(self,item):

        if isinstance(self.cad_tree_items[item],vtk.vtkActor):
            if item.checkState(0) == qt.Qt.Checked:
                self.interactor3d.add_extra_actor(self.cad_tree_items[item])
            elif item.checkState(0) == qt.Qt.Unchecked:
                self.interactor3d.remove_extra_actor(self.cad_tree_items[item])
            self.refresh_3d()
        else:
            super().update_checked_features(item)


    def set_cad_colour(self):

        selected_features = []
        extra_actors = []
        for treeitem in self.feature_tree.selectedItems():
            if isinstance(self.cad_tree_items[treeitem],vtk.vtkActor):
                extra_actors.append(self.cad_tree_items[treeitem])
            else:
                selected_features.append(self.cad_tree_items[treeitem])

        # Note: this does not mean nothing is selected;
        # rather it means the root of the model is selected!
        if None in selected_features:
            selected_features = None

        if self.sender() is self.cad_colour_choose_button:

            if len(selected_features) > 0 or selected_features is None:
                picked_colour = self.pick_colour(self.cadmodel.get_colour( selected_features )[0] )
            else:
                picked_colour = self.pick_colour(extra_actors[0].GetProperty().GetColor())

            if picked_colour is not None:
                if len(selected_features) > 0 or selected_features is None:
                    self.cadmodel.set_colour(picked_colour,selected_features)
                for actor in extra_actors:
                    actor.GetProperty().SetColor(picked_colour)


        elif self.sender() is self.cad_colour_reset_button:

            self.cadmodel.reset_colour(selected_features)

            for actor in extra_actors:
                actor.GetProperty().SetColor(1,1,1)

        self.refresh_3d()


    def update_cadtree_selection(self):

        self.remove_mesh_button.setEnabled(False)
        if len(self.feature_tree.selectedItems()) == 0:
            self.cad_colour_choose_button.setEnabled(False)
            self.cad_colour_reset_button.setEnabled(False)
        else:
            self.cad_colour_choose_button.setEnabled(True)
            self.cad_colour_reset_button.setEnabled(True)
            if len(self.feature_tree.selectedItems()) == 1 and isinstance(self.cad_tree_items[self.feature_tree.selectedItems()[0]],vtk.vtkActor):
                self.remove_mesh_button.setEnabled(True)


    def remove_extra_mesh(self):
        for item in self.feature_tree.selectedItems():
            if isinstance(self.cad_tree_items[item],vtk.vtkActor):
                self.interactor3d.remove_extra_actor(self.cad_tree_items[item])
                self.feature_tree.takeTopLevelItem(self.feature_tree.indexOfTopLevelItem(item))
                del self.cad_tree_items[item]
                self.refresh_3d()

    def update_slicing(self):
        self.update_xsection()

    def save_cursor_coords(self):

        if self.interactor3d.focus_cursor is not None:
            if self.saved_coords_item is None:
                # Add it to the lines list
                self.saved_coords_item = qt.QListWidgetItem('Saved Cursor Positions (below)')

                actor = render.CoordsActor(np.empty((0,3)),linewidth=0,markersize=1e-2)

                self.line_actors[self.saved_coords_item] = actor

                self.saved_coords_item.setFlags(self.saved_coords_item.flags() | qt.Qt.ItemIsSelectable)
                self.lines_3d_list.addItem(self.saved_coords_item)
                self.saved_coords_item.setCheckState(qt.Qt.Checked)
                self.lines_3d_list.setCurrentItem(self.saved_coords_item)

            else:
                actor = self.line_actors[self.saved_coords_item]

            coords = self.interactor3d.get_cursor_coords(0)
            newrow = actor.coords.shape[0]

            if self.saved_coords_item.checkState() == qt.Qt.Checked:
                self.saved_coords_item.setCheckState(qt.Qt.Unchecked)
                reenable=True
            else:
                reenable = False

            actor.coords = np.concatenate((actor.coords,np.array(coords)[np.newaxis,:]),0)
            self.cursor_coords_table.setRowCount(newrow+1)
            x = qt.QTableWidgetItem('{:.3f} m'.format(coords[0]))
            y = qt.QTableWidgetItem('{:.3f} m'.format(coords[1]))
            z = qt.QTableWidgetItem('{:.3f} m'.format(coords[2]))
            x.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
            y.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
            z.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
            self.cursor_coords_table.setItem(newrow,0,x)
            self.cursor_coords_table.setItem(newrow, 1,y)
            self.cursor_coords_table.setItem(newrow, 2,z)

            if reenable:
                self.saved_coords_item.setCheckState(qt.Qt.Checked)

            self.refresh_3d()

    def delete_saved_coord(self):

        if self.cursor_coords_table.currentRow() > -1:
            self.saved_coords_item.coords = np.delete(self.saved_coords_item.coords,self.cursor_coords_table.currentRow(),axis=0)
            self.cursor_coords_table.removeRow(self.cursor_coords_table.currentRow())
            self.saved_coords_item.set_markers(not self.saved_coords_item.markers)
            self.saved_coords_item.set_markers(not self.saved_coords_item.markers)
            self.refresh_3d()

    def export_cursor_coords(self):

        filename_filter = 'ASCII CSV Data (*.csv)'

        filedialog = qt.QFileDialog(self)

        filedialog.setAcceptMode(filedialog.AcceptSave)
        filedialog.setFileMode(filedialog.AnyFile)

        filedialog.setWindowTitle('Save...')
        filedialog.setNameFilter(filename_filter)
        filedialog.exec()

        if filedialog.result() == filedialog.Accepted:
            fname = str(filedialog.selectedFiles()[0])

            if not fname.endswith('.csv'):
                fname = fname + '.csv'

            np.savetxt(fname,self.line_actors[self.saved_coords_item].coords,fmt='%.4f',delimiter=',',header='X (m), Y(m), Z(m)')



    def update_lines(self,data):
        # Load or update coordinate data to display

        if self.sender() is self.load_lines_button:
            # User clicked the "Load from ASCII" button
            filename_filter = 'ASCII Data (*.txt *.csv *.dat)'
            filedialog = qt.QFileDialog(self)
            filedialog.setAcceptMode(filedialog.AcceptOpen)
            filedialog.setFileMode(filedialog.ExistingFile)
            filedialog.setWindowTitle('Open...')
            filedialog.setNameFilter(filename_filter)
            filedialog.exec()

            if filedialog.result() == filedialog.Accepted:
                fname = str(filedialog.selectedFiles()[0])

                coords = None
                for delimiter in ['\t',' ',',']:
                    try:
                        coords = np.loadtxt(fname,delimiter=delimiter)
                        lines_name = os.path.split(fname)[1].split('.')[0]

                        if len(coords.shape) == 1:
                            coords = coords[np.newaxis,:]
                    except ValueError:
                        continue

                if coords is None or len(coords.shape) != 2 or coords.shape[1] not in [2,3,4,6]:
                    # We didn't manage to load any data, or it's a shape we don't know how to interpret
                    raise UserWarning('Could not load coordinates from the file. Please ensure the file is formatted as N rows, 3 or 6 columns and is tab, space or comma delimited.')

                else:
                    # Ask the user what coordinate system the data are in
                    coords_dialog = CoordsDialog(self,coords.shape)
                    coords_dialog.exec()
                    if coords_dialog.result() == coords_dialog.Accepted:
                        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
                        # Add it to the lines list
                        listitem = qt.QListWidgetItem(lines_name)

                        self.line_actors[listitem] = render.CoordsActor(coords,coords_type=coords_dialog.coords_type,linewidth=3,markersize=0,frustrumsize=(0,0),rz_to_3d=False,closed_contour=coords_dialog.closed_contour_checkbox.isChecked())
                        listitem.setFlags(listitem.flags() | qt.Qt.ItemIsEditable | qt.Qt.ItemIsSelectable)
                        listitem.setToolTip(lines_name)
                        self.lines_3d_list.addItem(listitem)
                        listitem.setCheckState(qt.Qt.Checked)
                        self.lines_3d_list.setCurrentItem(listitem)


        elif self.sender() is self.lines_3d_list:
            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            if data.checkState() == qt.Qt.Checked:
                self.interactor3d.add_extra_actor(self.line_actors[data])
                self.lines_3d_list.setCurrentItem(data)
            else:
                self.interactor3d.remove_extra_actor(self.line_actors[data])
            self.update_selected_lines()

        elif self.sender() is self.pick_lines_colour and len(self.lines_3d_list.selectedItems()) > 0:

            current_colour = self.line_actors[self.lines_3d_list.selectedItems()[0]].colour
            picked_colour = self.pick_colour(current_colour)

            if picked_colour is not None:
                self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
                for item in self.lines_3d_list.selectedItems():
                    self.line_actors[item].colour = picked_colour + [self.lines_opacity_slider.value()/100]

        elif self.sender() is self.lines_opacity_slider and len(self.lines_3d_list.selectedItems()) > 0:

            colour = list(self.line_actors[self.lines_3d_list.selectedItems()[0]].colour)[:3]

            for item in self.lines_3d_list.selectedItems():
                self.line_actors[item].colour = colour + [self.lines_opacity_slider.value()/100]


        elif self.sender() is self.remove_lines_button:
            for item in self.lines_3d_list.selectedItems():
                self.interactor3d.remove_extra_actor(self.line_actors[item])
                self.lines_3d_list.takeItem(self.lines_3d_list.row(item))
                del self.line_actors[item]


        else:
            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            item = self.lines_3d_list.selectedItems()[0]
            actor = self.line_actors[item]
            enabled = item.checkState() == qt.Qt.Checked
            self.interactor3d.remove_extra_actor(actor)

            if self.phi_settings.isEnabled():
                if self.lines_rz_to_3d_rb.isChecked():
                    actor.rz_to_3d = True
                    self.lines_toroidal_angle_box.setEnabled(False)
                    self.lines_toroidal_angle_to_cursor.setEnabled(False)
                    self.lines_type_settings.setEnabled(False)
                else:
                    actor.rz_to_3d = False
                    actor.phi = self.lines_toroidal_angle_box.value()
                    self.lines_toroidal_angle_box.setEnabled(True)
                    self.lines_toroidal_angle_to_cursor.setEnabled(self.cursor_angles[0] is not None)
                    self.lines_type_settings.setEnabled(True)
            else:
                self.lines_type_settings.setEnabled(True)

            if self.lines_type_settings.isEnabled():
                if self.lines_lines_rb.isChecked():
                    actor.linewidth = self.line_width_box.value()
                    actor.markersize = 0
                    actor.frustrumsize = (0,0)
                elif self.lines_frustrum_rb.isChecked():
                    actor.linewidth = 0
                    actor.markersize = 0
                    actor.frustrumsize = (self.lines_frustrum_d0_box.value()/100,self.lines_frustrum_angle_box.value())
                elif self.lines_points_rb.isChecked():
                    actor.linewidth = 0
                    actor.markersize = self.marker_diameter_box.value()/100
                    actor.frustrumsize = (0,0)

            if enabled:
                self.interactor3d.add_extra_actor(actor)

        self.refresh_3d()
        self.update_legend()
        self.app.restoreOverrideCursor()


    def get_fov_opacity(self,actor_type):

        slider_value = self.sightline_opacity_slider.value() / 100
        omin,gamma = opacity_params[actor_type]
        delta_tot = 100 - omin

        range_pos = slider_value**gamma

        return (omin + range_pos*delta_tot)/100


    def update_unfolded_render_settings(self,enable):

        if self.sender() is self.centre_at_cursor:
            if enable:
                self.toroidal_centre_box.setEnabled(False)
                self.poloidal_centre_box.setEnabled(False)
                self.toroidal_centre_box.setValue(self.cursor_angles[0])
                self.poloidal_centre_box.setValue(self.cursor_angles[1])
            else:
                self.toroidal_centre_box.setEnabled(True)
                self.poloidal_centre_box.setEnabled(True)

        elif self.sender() is self.unfolded_auto_size:
            if enable:
                self.unfolded_w.setEnabled(False)
            else:
                self.unfolded_w.setEnabled(True)

    def set_fov_opacity(self,opacity,actor_type):

        omin, gamma = opacity_params[actor_type]
        delta_tot = 100 - omin

        opacity = (opacity*100 - omin)/delta_tot
        opacity = opacity**(1/gamma)

        self.sightline_opacity_slider.blockSignals(True)
        self.sightline_opacity_slider.setValue(int(opacity * 100))
        self.sightline_opacity_slider.blockSignals(False)


    def update_unfolded_render_settings(self,enable):

        if self.sender() is self.centre_at_cursor:
            if enable:
                self.toroidal_centre_box.setEnabled(False)
                self.poloidal_centre_box.setEnabled(False)
                self.toroidal_centre_box.setValue(self.cursor_angles[0])
                self.poloidal_centre_box.setValue(self.cursor_angles[1])
            else:
                self.toroidal_centre_box.setEnabled(True)
                self.poloidal_centre_box.setEnabled(True)

        elif self.sender() is self.unfolded_auto_size:
            if enable:
                self.unfolded_w.setEnabled(False)
            else:
                self.unfolded_w.setEnabled(True)

    def update_selected_lines(self):

        if len(self.lines_3d_list.selectedItems()) > 0:

            lines_actor = self.line_actors[self.lines_3d_list.selectedItems()[0]]
            enabled = self.lines_3d_list.selectedItems()[0].checkState() == qt.Qt.Checked

            if self.lines_3d_list.selectedItems()[0]  not in [self.wall_contour_listitem,self.saved_coords_item]:
                self.remove_lines_button.setEnabled(True)
            else:
                self.remove_lines_button.setEnabled(False)

            if lines_actor.coords_type == 'rz':
                self.phi_settings.setEnabled(True)
                if lines_actor.rz_to_3d:
                    self.lines_rz_to_3d_rb.setChecked(True)
                else:
                    self.lines_toroidal_angle_rb.setChecked(True)
                    self.lines_toroidal_angle_box.setValue(lines_actor.phi)
            else:
                self.phi_settings.setEnabled(False)

            if lines_actor.linewidth > 0:
                self.lines_lines_rb.setChecked(True)
                self.line_width_box.setValue(lines_actor.linewidth)

            if lines_actor.frustrumsize != (0,0):
                self.lines_frustrum_rb.setChecked(True)
                self.lines_frustrum_d0_box.setValue(lines_actor.frustrumsize[0])
                self.lines_frustrum_angle_box.setValue(lines_actor.frustrumsize[1])

            if lines_actor.markersize > 0:
                self.lines_points_rb.setChecked(True)
                self.marker_diameter_box.setValue(lines_actor.markersize*100)

            self.lines_appearance_box.setEnabled(enabled)

            if enabled:
                if lines_actor.coords.shape[0] != 2 and lines_actor.coords.shape[1] in [2, 3]:
                    self.lines_frustrum_rb.setChecked(False)
                    self.lines_frustrum_rb.setToolTip('Option only available for straight line segments')
                    self.lines_frustrum_rb.setEnabled(False)
                    self.lines_frustrum_d0_box.setEnabled(False)
                    self.lines_frustrum_angle_label.setEnabled(False)
                    self.lines_frustrum_angle_box.setEnabled(False)
                else:
                    self.lines_frustrum_rb.setEnabled(True)
                    self.lines_frustrum_rb.setToolTip('')
                    self.lines_frustrum_d0_box.setEnabled(True)
                    self.lines_frustrum_angle_label.setEnabled(True)
                    self.lines_frustrum_angle_box.setEnabled(True)

        else:
            self.lines_appearance_box.setEnabled(False)
            self.remove_lines_button.setEnabled(False)


    def on_close(self):
        self.qvtkwidget_3d.close()


    def set_projection(self):

        if self.proj_perspective.isChecked():
            self.interactor3d.set_projection('perspective')
        elif self.proj_orthographic.isChecked():
            self.interactor3d.set_projection('orthographic')

        self.update_viewport_info()
        self.refresh_3d()


    def on_view_changed(self):

        if self.interactor3d.get_projection() == 'perspective':
            self.proj_perspective.setChecked(True)
        else:
            self.proj_orthographic.setChecked(True)

        

    def on_change_cad_features(self):
        
        for key,item in self.sightlines:

            recheck = False

            if key.checkState() == qt.Qt.Checked:
                recheck = True
                key.setCheckState(qt.Qt.Unchecked)

            item[1] = None
            if recheck:
                key.setCheckState(qt.Qt.Checked)

        self.update_selected_sightlines()

 
    def toggle_wireframe(self,wireframe):
        
        if self.cadmodel is not None:

            self.cadmodel.set_wireframe( wireframe )

            self.refresh_3d()


    def add_cursor(self,coords):

        if self.interactor3d.focus_cursor is None:
            self.interactor3d.add_cursor(coords)
            self.interactor3d.set_cursor_focus(0)
            self.centre_at_cursor.setEnabled(True)
            self.update_cursor_position(0,coords)

    def load_model(self, data=None, featurelist=None, hold_view=False):

        keep_tla_features = []
        keep_tla_actors = []
        if self.feature_tree.topLevelItemCount() > 1:
            for i in range(1,self.feature_tree.topLevelItemCount()):
                keep_tla_features.append(self.feature_tree.takeTopLevelItem(i))
                keep_tla_actors.append(self.cad_tree_items[keep_tla_features[-1]])

        super().load_model(data,featurelist,hold_view)

        for key,item in zip(keep_tla_features,keep_tla_actors):
            self.feature_tree.addTopLevelItem(key)
            self.cad_tree_items[key] = item


    def on_model_load(self):

        self.centre_at_cursor.setChecked(False)

        # Enable the other tabs!
        self.tabWidget.setTabEnabled(1,True)
        self.tabWidget.setTabEnabled(2,True)
        self.tabWidget.setTabEnabled(3,True)
        self.tabWidget.setTabEnabled(4,True)

        self.cad_colour_controls.setEnabled(True)
        self.cad_colour_reset_button.setEnabled(False)
        self.cad_colour_choose_button.setEnabled(False)


        # Enable or disable contour controls as appropriate
        contour_exists = self.cadmodel.wall_contour is not None
        self.render_unfolded_view.setEnabled(contour_exists)
        self.render_unfolded_view_description.setEnabled(contour_exists)
        if self.wall_contour_listitem is not None:
            self.lines_3d_list.takeItem(self.lines_3d_list.row(self.wall_contour_listitem))
            self.interactor3d.remove_extra_actor(self.line_actors[self.wall_contour_listitem])
            del self.line_actors[self.wall_contour_listitem]
            self.wall_contour_listitem = None
        if contour_exists:
            #self.update_contour()
            self.render_unfolded_view.setToolTip('')
            self.render_unfolded_view_description.setToolTip('')
            listitem = qt.QListWidgetItem('Wall Contour - {:s} ({:s})'.format(self.cadmodel.machine_name,self.cadmodel.model_variant))
            listitem.setCheckState(qt.Qt.Unchecked)

            self.line_actors[listitem] = render.CoordsActor(self.cadmodel.wall_contour, coords_type='rz',linewidth=3,colour=(1,0,0,1))

            listitem.setFlags(listitem.flags() | qt.Qt.ItemIsSelectable)
            listitem.setToolTip('Wall contour data contained in the CAD model definition file.')
            self.lines_3d_list.addItem(listitem)
            self.wall_contour_listitem = listitem
        else:
            if self.render_unfolded_view.isChecked():
                self.render_current_view.toggle()
            self.render_unfolded_view.setToolTip('Rendering un-folded first wall view is only available for CAD models with R,Z wall contours defined.')
            self.render_unfolded_view_description.setToolTip('Rendering un-folded first wall view is only available for CAD models with R,Z wall contours defined.')

        model_extent = self.cadmodel.get_extent()
        rmax = np.max(np.abs(model_extent[:4]))
        self.chordslice_r.setValue(rmax/2)
        self.zslice_zmin.setValue(model_extent[4]-1e-3)
        self.zslice_zmax.setValue(model_extent[5]+1e-3)
        self.toggle_wireframe(self.rendertype_edges.isChecked())

    def update_selected_sightlines(self):

        self.sightlines_settings_box.setEnabled(False)

        if len(self.sightlines_list.selectedItems()) > 0:

            for sightlines in self.sightlines_list.selectedItems():

                if self.sightlines[sightlines][1] is not None:

                    self.sightlines_settings_box.setEnabled(True)

                    self.sightline_type_volume.blockSignals(True)
                    self.sightline_type_lines.blockSignals(True)
                    self.sightline_type_wallcoverage.blockSignals(True)
                    if self.sightlines[sightlines][2] == 'volume':
                        self.sightline_type_volume.setChecked(True)
                    elif self.sightlines[sightlines][2] == 'lines':
                        self.sightline_type_lines.setChecked(True)
                    elif self.sightlines[sightlines][2] == 'wall_coverage':
                        self.sightline_type_wallcoverage.setChecked(True)
                    self.sightline_type_volume.blockSignals(False)
                    self.sightline_type_lines.blockSignals(False)
                    self.sightline_type_wallcoverage.blockSignals(False)

                    self.set_fov_opacity(self.sightlines[sightlines][1].GetProperty().GetOpacity(),self.sightlines[sightlines][2])

                    break


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
            self.rendersettings_calib_label.setHidden(False)
            self.render_calib_namelabel.setHidden(False)
            self.render_load_button.setHidden(False)
            self.unfolded_render_settings.setHidden(True)
            self.render_aa.setHidden(False)
            self.render_aa_label.setHidden(False)
            self.render_resolution.setHidden(False)
            self.render_resolution_label.setHidden(False)
            self.update_render_coords()
            self.render_include_cursor.setHidden(False)
            self.render_bg_label.setHidden(False)
            self.render_bg_black.setHidden(False)
            self.render_bg_transparent.setHidden(False)

        elif self.render_current_view.isChecked():
            self.render_coords_text.setHidden(True)
            self.render_coords_combobox.setHidden(True)
            self.render_coords_combobox.setHidden(True)
            self.rendersettings_calib_label.setHidden(True)
            self.render_calib_namelabel.setHidden(True)
            self.render_load_button.setHidden(True)
            self.render_resolution.setCurrentIndex(-1)
            self.render_button.setEnabled(True)
            self.unfolded_render_settings.setHidden(True)
            self.render_aa.setHidden(False)
            self.render_aa_label.setHidden(False)
            self.render_resolution.setHidden(False)
            self.render_resolution_label.setHidden(False)
            self.interactor3d.on_resize()
            self.render_include_cursor.setHidden(False)
            self.render_bg_label.setHidden(False)
            self.render_bg_black.setHidden(False)
            self.render_bg_transparent.setHidden(False)

        elif self.render_unfolded_view.isChecked():

            self.unfolded_render_settings.setHidden(False)

            self.render_coords_text.setHidden(True)
            self.render_coords_combobox.setHidden(True)
            self.render_coords_combobox.setHidden(True)
            self.rendersettings_calib_label.setHidden(True)
            self.render_calib_namelabel.setHidden(True)
            self.render_load_button.setHidden(True)
            self.render_aa.setHidden(True)
            self.render_aa_label.setHidden(True)
            self.render_button.setEnabled(True)
            self.render_resolution.setHidden(True)
            self.render_resolution_label.setHidden(True)
            self.render_include_cursor.setHidden(True)
            self.render_bg_label.setHidden(True)
            self.render_bg_black.setHidden(True)
            self.render_bg_transparent.setHidden(True)



    def load_render_result(self):
        cal = self.object_from_file('calibration')
        if cal is not None:
            self.render_calib = cal
            self.render_calib_namelabel.setText(cal.name)
            self.render_button.setEnabled(True)
            self.update_render_coords()


    def update_render_coords(self):

        self.render_resolution.clear()
        if self.render_calib is not None:
            if self.render_coords_combobox.currentIndex() == 0:
                base_size = self.render_calib.geometry.get_display_shape()            
                self.render_resolution.addItem('{:d} x {:d} (same as camera)'.format(base_size[0],base_size[1]))
                self.render_resolution.addItem('{:d} x {:d}'.format(base_size[0]*2,base_size[1]*2))
                self.render_resolution.addItem('{:d} x {:d}'.format(base_size[0]*4,base_size[1]*4))

            elif self.render_coords_combobox.currentIndex() == 1:
                base_size = self.render_calib.geometry.get_original_shape()
                self.render_resolution.addItem('{:d} x {:d} (same as camera)'.format(base_size[0],base_size[1]))



    def on_resize(self,vtk_size):

        if self.render_current_view.isChecked():
            index = max(self.render_resolution.currentIndex(),0)
            self.render_resolution.clear()
            self.render_resolution.addItem('{:d} x {:d} (Window Size)'.format(vtk_size[0],vtk_size[1]))
            self.render_resolution.addItem('{:d} x {:d}'.format(vtk_size[0]*2,vtk_size[1]*2))
            self.render_resolution.addItem('{:d} x {:d}'.format(vtk_size[0]*4,vtk_size[1]*4))
            self.render_resolution.setCurrentIndex(index)
        self.update_vtk_size(vtk_size)


    def do_render(self):

        filename = self.get_save_filename('image')
        if filename is None:
            return

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.statusbar.showMessage('Rendering image to {:s} ...'.format(filename))


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
            #if self.interactor3d.legend is not None:
            #    temp_actors.append(self.interactor3d.legend)

            # Get rid of the cursor unless the user said not to.
            if not self.render_include_cursor.isChecked():
                if self.interactor3d.focus_cursor is not None:
                    temp_actors.append( self.interactor3d.cursors[0]['actor'] )

            for actor in temp_actors:
                self.renderer_3d.RemoveActor(actor)
            # -------------------------------------------------------------------------

            # Do the render
            im = render.render_hires(self.renderer_3d,oversampling=oversampling,aa=aa,transparency=use_transparency,legendactor=self.interactor3d.legend)

            # Add back the temporarily removed actors
            for actor in temp_actors:
                self.renderer_3d.AddActor(actor)

        # Render a calibrated camera's point of view
        elif self.render_cam_view.isChecked():

            # For render_cam_view we need to tell it what to include apart from the cad model.
            # So make a list of the other actors we need to add:
            extra_actors = []
            for listitem,sightlines in self.sightlines:
                if listitem.checkState() == qt.Qt.Checked:
                    extra_actors.append(sightlines[1])
            for listitem,lines_3d in self.line_actors:
                if listitem.checkState() == qt.Qt.Checked:
                    extra_actors.append(lines_3d)                


            coords = ['Display','Original'][ self.render_coords_combobox.currentIndex()]
            
            im = render.render_cam_view(self.cadmodel,self.render_calib,extra_actors = extra_actors,oversampling=oversampling,aa=aa,transparency=use_transparency,verbose=False,coords=coords)

        elif self.render_unfolded_view.isChecked():

            # Build a dictionary of arguments to pass to the rendering function
            args = {'cadmodel':self.cadmodel}

            args['extra_actors'] = []

            for listitem,stuff in self.sightlines:
                if listitem.checkState() == qt.Qt.Checked and stuff[2]  == 'wall_coverage':
                    args['extra_actors'].append(stuff[1])

            for listitem,actor in self.line_actors:
                if listitem.checkState() == qt.Qt.Checked:
                    args['extra_actors'].append(actor)

            args['theta_start'] = self.poloidal_centre_box.value() - 180
            args['phi_start'] = self.toroidal_centre_box.value() - 180

            if self.unfolded_manual_size.isChecked():
                args['w'] = self.unfolded_w.value()

            render_dialog = RenderUnfoldedDialog(self,args)
            im = render_dialog.render()
            if render_dialog.canceled:
                self.app.restoreOverrideCursor()
                self.statusbar.clearMessage()
                return
            render_dialog.accept()

        # Save the image!
        im[:,:,:3] = im[:,:,2::-1]
        result = cv2.imwrite(filename,im)

        self.renderer_3d.Render()

        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()

        if not result:
            dialog = qt.QMessageBox(self)
            dialog.setWindowFlags(dialog.windowFlags() | qt.Qt.CustomizeWindowHint)
            dialog.setWindowFlags(dialog.windowFlags() & ~qt.Qt.WindowCloseButtonHint)
            dialog.setStandardButtons(dialog.Save | dialog.Discard)
            dialog.button(dialog.Save).setText('Save As...')
            dialog.button(dialog.Discard).setText('Cancel')
            dialog.setTextFormat(qt.Qt.RichText)
            dialog.setWindowTitle('Error saving image')
            dialog.setText('Could not write to file {:s}.'.format(filename))
            dialog.setInformativeText('Click "Save As..." to select another location / filename and try again, or "Cancel" to give up.')
            dialog.setIcon(qt.QMessageBox.Warning)
            dialog.exec()

            if dialog.result() == dialog.Save:
                self.save_image(im)


        


    def update_sightlines(self,data):

        if self.sender() is self.sightlines_load_button:

            cals = self.object_from_file('calibration',multiple=True)

            for cal in cals:

                any_result = False
                for subview in range(cal.n_subviews):
                    if cal.view_models[subview] is not None:
                        any_result = True
                        # Add it to the sight lines list
                        cal_name = cal.name
                        if cal.n_subviews > 1:
                            cal_name = cal_name + ' [{:s}]'.format(cal.subview_names[subview])
                        listitem = qt.QListWidgetItem(cal_name)
                        listitem.setFlags(listitem.flags() | qt.Qt.ItemIsEditable | qt.Qt.ItemIsSelectable)
                        listitem.setToolTip(cal_name)
                        self.sightlines_list.addItem(listitem)
                        if self.sightline_type_volume.isChecked():
                            actor_type = 'volume'
                        elif self.sightline_type_lines.isChecked():
                            actor_type = 'lines'
                        elif self.sightline_type_wallcoverage.isChecked():
                            actor_type = 'wall_coverage'
                        if cal.n_subviews > 1:
                            sv = subview
                        else:
                            sv = None
                        self.sightlines[listitem] = [cal,None,actor_type,sv,next(self.colourcycle)]
                        listitem.setCheckState(qt.Qt.Checked)
                        self.sightlines_list.setCurrentItem(listitem)

                if not any_result:
                    self.show_msgbox('The calibration file {:s} does not contain any calibration results and will not be loaded.'.format(cal.filename))

        elif self.sender() is self.sightlines_list:

            if data.checkState() == qt.Qt.Checked:
                
                n = 0
                for item in self.sightlines.keys():
                    if item.checkState() == qt.Qt.Checked:
                        n = n + 1

                if n > 1:
                    self.sightlines_legend_checkbox.setChecked(True)

                if self.sightlines[data][1] is None:
                    self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
                    self.statusbar.showMessage('Ray casting camera sight lines...')
                    self.app.processEvents()
                    if self.sightlines[data][2] == 'wall_coverage':
                        actor = render.get_wall_coverage_actor(self.sightlines[data][0],self.cadmodel,subview=self.sightlines[data][3],clearance=1e-2,resolution=256)
                    else:
                        actor = render.get_fov_actor(self.cadmodel,self.sightlines[data][0],self.sightlines[data][2],subview=self.sightlines[data][3])
                    self.statusbar.clearMessage()
                    actor.GetProperty().SetColor(self.sightlines[data][4])
                    actor.GetProperty().SetOpacity( self.get_fov_opacity(self.sightlines[data][2]) )
                    self.sightlines[data][1] = actor
                    self.interactor3d.add_extra_actor(actor)
                    self.app.restoreOverrideCursor()

                else:

                    self.interactor3d.add_extra_actor(self.sightlines[data][1])

                self.update_selected_sightlines()

            else:
                self.interactor3d.remove_extra_actor(self.sightlines[data][1])

                n = 0
                for item in self.sightlines.keys():
                    if item.checkState() == qt.Qt.Checked:
                        n = n + 1

                if n < 2:
                    self.sightlines_legend_checkbox.setChecked(False)

        elif self.sender() is self.sightline_opacity_slider:

            for item in self.sightlines_list.selectedItems():
                self.sightlines[item][1].GetProperty().SetOpacity( self.get_fov_opacity(self.sightlines[item][2]) )


        elif self.sender() is self.pick_sightlines_colour and len(self.sightlines_list.selectedItems()) > 0:

            picked_colour = self.pick_colour(self.sightlines[self.sightlines_list.selectedItems()[0]][1].GetProperty().GetColor())
            if picked_colour is not None:
                for item in self.sightlines_list.selectedItems():
                    self.sightlines[item][1].GetProperty().SetColor( picked_colour )

        elif self.sender() in [self.sightline_type_volume,self.sightline_type_lines,self.sightline_type_wallcoverage]:

            for item in self.sightlines_list.selectedItems():
                item.setCheckState(qt.Qt.Unchecked)
                self.sightlines[item][1] = None
                if self.sightline_type_volume.isChecked():
                    self.sightlines[item][2] = 'volume'
                elif self.sightline_type_lines.isChecked():
                    self.sightlines[item][2] = 'lines'
                elif self.sightline_type_wallcoverage.isChecked():
                    self.sightlines[item][2] = 'wall_coverage'

                item.setCheckState(qt.Qt.Checked)

            
        self.update_legend()

        self.refresh_3d()


    def update_legend(self):

        legend_items = []

        if self.sightlines_legend_checkbox.isChecked():

            for item in self.sightlines.keys():
                if item.checkState() == qt.Qt.Checked and self.sightlines[item][1] is not None:
                    legend_items.append((str(item.text()), self.sightlines[item][1].GetProperty().GetColor()))


        if self.coords_legend_checkbox.isChecked():

            for item in self.line_actors.keys():
                if item.checkState() == qt.Qt.Checked:
                    legend_items.append((str(item.text()), self.line_actors[item].colour[:3]))

        self.interactor3d.set_legend(legend_items)
        self.refresh_3d()


    def update_cursor_position(self,cursor_id,position):
        info = 'Cursor location: ' + self.cadmodel.format_coord(position).replace('\n',' | ')
        self.statusbar.showMessage(info)


        phi = np.arctan2(position[1], position[0])
        if phi < 0.:
            phi = phi + 2 * 3.14159
        phi = phi / 3.14159 * 180

        if self.cadmodel.wall_contour is not None:
            r_min = self.cadmodel.wall_contour[:,0].min()
            r_max = self.cadmodel.wall_contour[:,0].max()
            z_min = self.cadmodel.wall_contour[:,1].min()
            z_max = self.cadmodel.wall_contour[:,1].max()
            r_mid = (r_max + r_min) / 2
            z_mid = (z_max + z_min) / 2

            dr = np.sqrt(position[0]**2 + position[1]**2) - r_mid
            dz = position[2] - z_mid

            theta = 180*np.arctan2(dz,dr)/np.pi

            self.cursor_angles = (phi,theta)

            if self.centre_at_cursor.isChecked():
                self.toroidal_centre_box.setValue(phi)
                self.poloidal_centre_box.setValue(theta)
        else:
            self.cursor_angles = (phi,None)

        self.lines_toroidal_angle_to_cursor.setEnabled(True)


class CoordsDialog(qt.QDialog):

    def __init__(self, parent,coords_shape):

        # GUI initialisation
        qt.QDialog.__init__(self, parent)
        qt.uic.loadUi(os.path.join(guipath,'qt_designer_files','line_coords.ui'), self)

        self.parent = parent

        if coords_shape[1] == 6:
            self.lines_label.setText('Importing coordinates for {:d} 3D lines from file.'.format(coords_shape[0]))
            self.closed_contour_checkbox.hide()
            self.coord_types = ['xyz','rzd','rzr']
            coord_type_options = ['X (m) , Y (m) , Z (m)','R (m) , Z (m) , Phi (degrees)','R (m) , Z (m) , Phi (radians)']
        elif coords_shape[1] == 3:
            self.lines_label.setText('Importing a sequence of {:d} 3D points from file.'.format(coords_shape[0]))
            self.closed_contour_checkbox.hide()
            self.coord_types = ['xyz','rzd','rzr']
            coord_type_options = ['X (m) , Y (m) , Z (m)','R (m) , Z (m) , Phi (degrees)','R (m) , Z (m) , Phi (radians)']
        elif coords_shape[1] == 4:
            self.lines_label.setText('Importing coordinates for {:d} 2D lines from file.'.format(coords_shape[0]))
            self.closed_contour_checkbox.show()
            self.coord_types = ['rz']
            coord_type_options = ['R (m) , Z (m)']
        elif coords_shape[1] == 2:
            self.lines_label.setText('Importing a sequence of {:d} 2D points from file.'.format(coords_shape[0]))
            self.closed_contour_checkbox.show()
            self.coord_types = ['rz']
            coord_type_options = ['R (m) , Z (m)']

        self.line_coords_combobox.currentIndexChanged.connect(self.change_coords)
        self.line_coords_combobox.addItems(coord_type_options)
        self.line_coords_combobox.setCurrentIndex(0)
        if len(coord_type_options) < 2:
            self.line_coords_combobox.setEnabled(False)

    def change_coords(self,index):
        self.coords_type = self.coord_types[index]


class MeshImportDialog(qt.QDialog):

    def __init__(self, parent,filename):

        # GUI initialisation
        qt.QDialog.__init__(self, parent)
        qt.uic.loadUi(os.path.join(guipath,'qt_designer_files','mesh_import_settings.ui'), self)

        self.parent = parent
        self.filename.setText(os.path.split(filename)[-1])


class RenderUnfoldedDialog(qt.QDialog):

    def __init__(self, parent,render_args):

        # GUI initialisation
        qt.QDialog.__init__(self, parent,qt.Qt.WindowTitleHint | qt.Qt.WindowCloseButtonHint)
        qt.uic.loadUi(os.path.join(guipath,'qt_designer_files','rendering_progress.ui'), self)

        self.parent = parent
        self.canceled = False
        self.args = render_args

        self.args['progress_callback'] = self.update
        self.args['cancel'] = self.cancel_check_callback

        self.time_remaining_text.hide()

        self.previous_remaining_prediction = None

        self.progress_ref = None

    def render(self):
        self.show()
        return render.render_unfolded_wall(**self.args)

    def update(self,progress):

        self.parent.app.processEvents()
        self.repaint()
        try:
            self.progressbar.setValue(int(progress*100))
            if self.progress_ref is None:
                self.progress_ref = (time.time(),progress)
            else:
                dt = time.time() - self.progress_ref[0]
                dp = progress - self.progress_ref[1]
                t_remaining = (1-progress) * dt/dp

                if self.previous_remaining_prediction is None:
                    self.previous_remaining_prediction = (time.time(),t_remaining)
                    return
                else:
                    if np.abs( t_remaining + (time.time() - self.previous_remaining_prediction[0]) - self.previous_remaining_prediction[1]) / self.previous_remaining_prediction[1] > 0.2:
                        self.progress_ref = (time.time(), progress)
                        self.previous_remaining_prediction = None
                        self.time_remaining_text.hide()
                        return

                if t_remaining < 60:
                    t_string = '< 1 minute'
                elif t_remaining > 60**2:
                    min_remaining = np.round(t_remaining/60)
                    t_string = '{:.0f}h {:.0f}min'.format(np.floor(min_remaining/60),min_remaining % 60)
                else:
                    t_string = '{:.0f} minutes'.format(t_remaining/60)
                self.time_remaining_text.show()
                self.time_remaining_text.setText('Estimated time remaining: {:s}'.format(t_string))

        except TypeError:
            self.label.setText(progress)

    def cancel_check_callback(self):
        return self.canceled

    def reject(self):
        self.canceled = True
        super().reject()
