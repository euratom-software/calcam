'''
* Copyright 2015-2020 European Atomic Energy Community (EURATOM)
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
        self.interactor3d = CalcamInteractorStyle3D(refresh_callback=self.refresh_3d,viewport_callback=self.update_viewport_info,newpick_callback=self.add_cursor,cursor_move_callback=self.update_cursor_position,resize_callback=self.on_resize)
        self.qvtkwidget_3d.SetInteractorStyle(self.interactor3d)
        self.renderer_3d = vtk.vtkRenderer()
        self.renderer_3d.SetBackground(0, 0, 0)
        self.qvtkwidget_3d.GetRenderWindow().AddRenderer(self.renderer_3d)
        self.camera_3d = self.renderer_3d.GetActiveCamera()


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
        self.xsection_checkbox.toggled.connect(self.update_xsection)
        self.xsection_origin.toggled.connect(self.update_xsection)
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
        self.contour_off.clicked.connect(self.update_contour)
        self.contour_2d.clicked.connect(self.update_contour)
        self.contour_3d.clicked.connect(self.update_contour)
        self.load_lines_button.clicked.connect(self.update_lines)
        self.pick_lines_colour.clicked.connect(self.update_lines)
        self.lines_3d_list.itemSelectionChanged.connect(self.update_selected_lines)
        self.control_sensitivity_slider.valueChanged.connect(lambda x: self.interactor3d.set_control_sensitivity(x*0.01))
        self.rmb_rotate.toggled.connect(self.interactor3d.set_rmb_rotate)
        self.sightlines_legend = None
        self.render_calib = None
        self.enable_lines_checkbox.toggled.connect(self.update_lines)
        self.enable_points_checkbox.toggled.connect(self.update_lines)
        self.line_width_box.valueChanged.connect(self.update_lines)
        self.marker_diameter_box.valueChanged.connect(self.update_lines)
        self.remove_lines_button.clicked.connect(self.update_lines)
        self.coords_legend_checkbox.toggled.connect(self.update_legend)

        self.control_sensitivity_slider.setValue(self.config.mouse_sensitivity)

        self.proj_perspective.toggled.connect(self.set_projection)

        self.model_actors = {}

        self.cursor_angles = None

        self.line_actors = DodgyDict()

        self.contour_actor = None

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
        self.enable_lines_checkbox.nextCheckState = lambda : nextcheckstate(self.enable_lines_checkbox)
        self.enable_points_checkbox.nextCheckState = lambda: nextcheckstate(self.enable_points_checkbox)

        # Start the GUI!
        self.show()

        self.interactor3d.init()
        self.views_root_auto.setHidden(False)
        self.qvtkwidget_3d.GetRenderWindow().GetInteractor().Initialize()


    # Load arbitrary 3D lines from file to display
    def update_lines(self,data):

        if self.sender() is self.load_lines_button:

            filename_filter = 'ASCII Data (*.txt *.csv *.dat)'

            filedialog = qt.QFileDialog(self)

            filedialog.setAcceptMode(filedialog.AcceptOpen)
            filedialog.setFileMode(filedialog.ExistingFile)

            filedialog.setWindowTitle('Open...')
            filedialog.setNameFilter(filename_filter)
            filedialog.exec()

            if qt.qt_ver < 6:
                accepted = filedialog.result() == 1
            else:
                accepted = filedialog.result() == filedialog.Accepted

            if accepted:
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

                if coords is None or len(coords.shape) != 2 or coords.shape[1] not in [3,6]:
                    raise UserWarning('Could not load coordinates from the file. Please ensure the file is formatted as N rows, 3 or 6 columns and is tab, space or comma delimited.')

                else:
                    coords_dialog = CoordsDialog(self,coords.shape)
                    coords_dialog.exec()
                    if coords_dialog.result() == 1:
                        
                        # If the coordinates are in R,Z,phi, convert them to cartesian.
                        if coords_dialog.line_coords_combobox.currentIndex() == 1:
                            x = coords[:,0] * np.cos(coords[:,2])
                            y = coords[:,0] * np.sin(coords[:,2])
                            coords[:,2] = coords[:,1]
                            coords[:,0] = x
                            coords[:,1] = y

                            if coords.shape[1] == 6:
                                x = coords[:,3] * np.cos(coords[:,5])
                                y = coords[:,3] * np.sin(coords[:,5])
                                coords[:,5] = coords[:,4]
                                coords[:,3] = x
                                coords[:,4] = y


                        # Add it to the lines list
                        listitem = qt.QListWidgetItem(lines_name)

                        self.line_actors[listitem] = render.CoordsActor(coords,coords_dialog.show_lines.isChecked(),coords_dialog.show_points.isChecked())

                        listitem.setFlags(listitem.flags() | qt.Qt.ItemIsEditable | qt.Qt.ItemIsSelectable)
                        listitem.setToolTip(lines_name)
                        self.lines_3d_list.addItem(listitem)
                        listitem.setCheckState(qt.Qt.Checked)
                        self.lines_3d_list.setCurrentItem(listitem)


        elif self.sender() is self.lines_3d_list:

            if data.checkState() == qt.Qt.Checked:
                self.interactor3d.add_extra_actor(self.line_actors[data])
            else:
                self.interactor3d.remove_extra_actor(self.line_actors[data])

            self.refresh_3d()


        elif self.sender() is self.pick_lines_colour and len(self.lines_3d_list.selectedItems()) > 0:

            current_colour = self.line_actors[self.lines_3d_list.selectedItems()[0]].colour
            picked_colour = self.pick_colour(current_colour)

            if picked_colour is not None:
                for item in self.lines_3d_list.selectedItems():
                    self.line_actors[item].set_colour(picked_colour)

        elif self.sender() is self.enable_lines_checkbox:
            for item in self.lines_3d_list.selectedItems():
                self.line_actors[item].set_lines(data)
            if data:
                self.line_width_box.setEnabled(True)
            else:
                self.line_width_box.setEnabled(False)
            self.refresh_3d()

        elif self.sender() is self.enable_points_checkbox:
            for item in self.lines_3d_list.selectedItems():
                self.line_actors[item].set_markers(data)
            if data:
                self.marker_diameter_box.setEnabled(True)
            else:
                self.marker_diameter_box.setEnabled(False)
            self.refresh_3d()

        elif self.sender() is self.line_width_box:
            for item in self.lines_3d_list.selectedItems():
                self.line_actors[item].linewidth = self.line_width_box.value()
                if self.line_actors[item].lines:
                    self.line_actors[item].set_lines(False)
                    self.line_actors[item].set_lines(True)
            self.refresh_3d()

        elif self.sender() is self.marker_diameter_box:
            for item in self.lines_3d_list.selectedItems():
                self.line_actors[item].markersize = self.marker_diameter_box.value() / 100
            if self.line_actors[item].markers:
                self.line_actors[item].set_markers(False)
                self.line_actors[item].set_markers(True)
            self.refresh_3d()

        elif self.sender() is self.remove_lines_button:
            for item in self.lines_3d_list.selectedItems():
                self.interactor3d.remove_extra_actor(self.line_actors[item])
                self.lines_3d_list.takeItem(self.lines_3d_list.row(item))
                del self.line_actors[item]
            self.refresh_3d()

        self.update_legend()

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
        self.sightline_opacity_slider.setValue(opacity * 100)
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

            lines = 0
            markers = 0
            linewidth = []
            markersize = []
            for item in self.lines_3d_list.selectedItems():
                if self.line_actors[item].lines:
                    lines += 1
                if self.line_actors[item].markers:
                    markers += 1
                linewidth.append(self.line_actors[item].linewidth)
                markersize.append(self.line_actors[item].markersize*100)

            linewidth = int(np.round(np.mean(linewidth)))
            markersize = np.mean(markersize)

            if lines == len(self.lines_3d_list.selectedItems()):

                linecheckstate = qt.Qt.Checked
            elif lines == 0:
                linecheckstate = qt.Qt.Unchecked
            elif lines < len(self.lines_3d_list.selectedItems()):
                linecheckstate = qt.Qt.PartiallyChecked

            if markers == len(self.lines_3d_list.selectedItems()):
                markercheckstate = qt.Qt.Checked
            elif markers == 0:
                markercheckstate = qt.Qt.Unchecked
            elif markers < len(self.lines_3d_list.selectedItems()):
                markercheckstate = qt.Qt.PartiallyChecked

            self.line_width_box.blockSignals(True)
            self.line_width_box.setValue(linewidth)
            self.line_width_box.blockSignals(False)

            self.marker_diameter_box.blockSignals(True)
            self.marker_diameter_box.setValue(markersize)
            self.marker_diameter_box.blockSignals(False)

            self.enable_lines_checkbox.blockSignals(True)
            self.enable_lines_checkbox.setCheckState(linecheckstate)
            self.enable_lines_checkbox.blockSignals(False)

            if linecheckstate == qt.Qt.Unchecked:
                self.line_width_box.setEnabled(False)
            else:
                self.line_width_box.setEnabled(True)

            self.enable_points_checkbox.blockSignals(True)
            self.enable_points_checkbox.setCheckState(markercheckstate)
            self.enable_points_checkbox.blockSignals(False)

            if markercheckstate == qt.Qt.Unchecked:
                self.marker_diameter_box.setEnabled(False)
            else:
                self.marker_diameter_box.setEnabled(True)

            self.lines_appearance_box.setEnabled(True)
            self.remove_lines_button.setEnabled(True)

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

        self.xsection_checkbox.blockSignals(True)
        self.xsection_cursor.blockSignals(True)
        self.xsection_origin.blockSignals(True)

        if self.interactor3d.get_xsection() is not None:
            self.xsection_checkbox.setChecked(True)
            if np.all( np.array(self.interactor3d.get_xsection()) == 0):
                self.xsection_origin.setChecked(True)
            else:
                self.add_cursor(self.interactor3d.get_xsection())
                self.interactor3d.set_cursor_coords(0,self.interactor3d.get_xsection())
                self.xsection_cursor.setChecked(True)
        else:
            self.xsection_checkbox.setChecked(False)

        self.xsection_checkbox.blockSignals(False)
        self.xsection_cursor.blockSignals(False)
        self.xsection_origin.blockSignals(False)

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


    def update_xsection(self):

        if self.xsection_checkbox.isChecked():
            if self.xsection_cursor.isChecked():
                self.interactor3d.set_xsection(self.interactor3d.get_cursor_coords(0))
            else:
                self.interactor3d.set_xsection((0,0,0))
        else:
            self.interactor3d.set_xsection(None)

        self.interactor3d.update_clipping()
        self.refresh_3d()


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
        self.contour_off.setEnabled(contour_exists)
        self.contour_2d.setEnabled(contour_exists & (self.interactor3d.focus_cursor is not None) )
        self.contour_3d.setEnabled(contour_exists)
        self.render_unfolded_view.setEnabled(contour_exists)
        self.render_unfolded_view_description.setEnabled(contour_exists)
        if contour_exists:
            self.update_contour()
            self.render_unfolded_view.setToolTip('')
            self.render_unfolded_view_description.setToolTip('')
        else:
            if not self.contour_off.isChecked():
                self.contour_off.toggle()
                self.update_contour()
            if self.render_unfolded_view.isChecked():
                self.render_current_view.toggle()
            self.render_unfolded_view.setToolTip('Rendering un-folded first wall view is only available for CAD models with R,Z wall contours defined.')
            self.render_unfolded_view_description.setToolTip('Rendering un-folded first wall view is only available for CAD models with R,Z wall contours defined.')


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
            


    def update_contour(self):

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))

        if self.contour_actor is not None:
            self.interactor3d.remove_extra_actor(self.contour_actor)
            self.contour_actor = None

        if self.contour_2d.isChecked():
            cursor_pos = self.interactor3d.get_cursor_coords(0)
            phi = np.arctan2(cursor_pos[1],cursor_pos[0])
            self.contour_actor = render.get_wall_contour_actor(self.cadmodel.wall_contour,'contour',phi)
            self.contour_actor.GetProperty().SetLineWidth(3)
            self.contour_actor.GetProperty().SetColor((1,0,0))
            self.interactor3d.add_extra_actor(self.contour_actor)

        elif self.contour_3d.isChecked():
            self.contour_actor = render.get_wall_contour_actor(self.cadmodel.wall_contour,'surface')
            self.contour_actor.GetProperty().SetColor((1,0,0))
            self.interactor3d.add_extra_actor(self.contour_actor)

        self.refresh_3d()
        self.app.restoreOverrideCursor()




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

            if self.contour_actor is not None:
                extra_actors.append(self.contour_actor)

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
        cv2.imwrite(filename,im)

        self.renderer_3d.Render()

        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()


        


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
                        actor = render.get_wall_coverage_actor(self.sightlines[data][0],self.cadmodel,resolution=256,subview=self.sightlines[data][3])
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
                    legend_items.append((str(item.text()), self.line_actors[item].colour))

        self.interactor3d.set_legend(legend_items)
        self.refresh_3d()


    def update_cursor_position(self,cursor_id,position):
        info = 'Cursor location: ' + self.cadmodel.format_coord(position).replace('\n',' | ')
        self.statusbar.showMessage(info)

        self.contour_2d.setEnabled(self.cadmodel.wall_contour is not None)
        
        if self.contour_2d.isChecked():
            self.update_contour()

        self.xsection_cursor.setEnabled(True)
        if self.xsection_checkbox.isChecked():
            self.interactor3d.set_xsection(self.interactor3d.get_cursor_coords(0))

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



class CoordsDialog(qt.QDialog):

    def __init__(self, parent,coords_shape):

        # GUI initialisation
        qt.QDialog.__init__(self, parent)
        qt.uic.loadUi(os.path.join(guipath,'qt_designer_files','line_coords.ui'), self)

        self.parent = parent

        if coords_shape[1] == 6:
            self.lines_label.setText('Importing coordinates for {:d} 3D lines from file.'.format(coords_shape[0]))
        elif coords_shape[1] == 3:
            self.lines_label.setText('Importing a sequence of {:d} points from file.'.format(coords_shape[0]))

        if coords_shape[0] == 1:
            self.show_lines.setChecked(False)
            self.show_lines.setEnabled(False)
            self.show_points.setChecked(True)



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
            self.progressbar.setValue(progress*100)
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
