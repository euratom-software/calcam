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

import cv2

from .core import *
from .vtkinteractorstyles import CalcamInteractorStyle3D
from .. import render

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

        self.control_sensitivity_slider.setValue(self.config.mouse_sensitivity)

        self.proj_perspective.toggled.connect(self.set_projection)

        self.model_actors = {}

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
        self.render_calib_description.setHidden(True)
        self.rendersettings_calib_label.setHidden(True)
        self.render_calib_namelabel.setHidden(True)
        self.render_load_button.setHidden(True)
        self.render_current_description.setHidden(False)


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
            filedialog.setAcceptMode(0)
            filedialog.setFileMode(1)


            filedialog.setWindowTitle('Open...')
            filedialog.setNameFilter(filename_filter)
            filedialog.exec_()
            if filedialog.result() == 1:
                fname = str(filedialog.selectedFiles()[0])

                coords = None
                for delimiter in ['\t',' ',',']:
                    try:
                        coords = np.loadtxt(fname,delimiter=delimiter)
                        lines_name = os.path.split(fname)[1].split('.')[0]
                    except ValueError:
                        continue

                if coords is None:
                    raise UserWarning('Could not load coordinates from the file. Please ensure the file is formatted as N rows, 3 or 6 columns and is tab, space or comma delimited.')

                elif coords.shape[1] in [3,6]:

                    coords_dialog = CoordsDialog(self,coords.shape)
                    coords_dialog.exec_()
                    if coords_dialog.result() == 1:

                        if coords_dialog.line_coords_combobox.currentIndex() == 1:

                            x = coords[:,0] * np.cos(coords[:,2])
                            y = coords[:,0] * np.cos(coords[:,2])
                            coords[:,2] = coords[:,1]
                            coords[:,0] = x
                            coords[:,1] = y

                            if coords.shape[1] == 6:
                                x = coords[:,3] * np.cos(coords[:,5])
                                y = coords[:,3] * np.cos(coords[:,5])
                                coords[:,5] = coords[:,4]
                                coords[:,3] = x
                                coords[:,4] = y


                        # Add it to the lines list
                        listitem = qt.QListWidgetItem(lines_name)
                        self.line_actors[listitem] = render.get_lines_actor(coords)

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

            picked_colour = self.pick_colour(self.line_actors[self.lines_3d_list.selectedItems()[0]].GetProperty().GetColor())
            if picked_colour is not None:
                for item in self.lines_3d_list.selectedItems():
                    self.line_actors[item].GetProperty().SetColor( picked_colour )


    def update_selected_lines(self):

        if len(self.lines_3d_list.selectedItems()) > 0:

            self.lines_appearance_box.setEnabled(True)

        else:
            self.lines_appearance_box.setEnabled(False)


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
            self.colourcycle.queue_colour(item[1].GetProperty().GetColor())

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
            self.update_cursor_position(0,coords)

    def on_load_model(self):
        
        # Turn off any wall contour
        self.contour_off.setChecked(True)

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




    def update_selected_sightlines(self):

        self.sightlines_settings_box.setEnabled(False)

        if len(self.sightlines_list.selectedItems()) > 0:

            for sightlines in self.sightlines_list.selectedItems():

                if self.sightlines[sightlines][1] is not None:

                    self.sightlines_settings_box.setEnabled(True)

                    self.sightline_type_volume.blockSignals(True)
                    if self.sightlines[sightlines][2] == 'volume':
                        self.sightline_type_volume.setChecked(True)
                    else:
                        self.sightline_type_lines.setChecked(True)
                    self.sightline_type_volume.blockSignals(False)

                    self.sightline_opacity_slider.blockSignals(True)
                    self.sightline_opacity_slider.setValue(100*np.log(self.sightlines[sightlines][1].GetProperty().GetOpacity()*100.)/np.log(100))

                    self.sightline_opacity_slider.blockSignals(False)

                    break
            


    def update_contour(self):

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))

        if self.contour_actor is not None:
            try:
                self.interactor3d.remove_extra_actor(self.contour_actor)
            except:
                self.renderer_3d.RemoveActor(self.contour_actor)
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
            self.renderer_3d.AddActor(self.contour_actor)

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
            self.interactor3d.on_resize()


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
        self.statusbar.showMessage('Render image to {:s} ...'.format(filename))


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
                temp_actors.append(self.interactor3d.legend)

            # Get rid of the cursor unless the user said not to.
            if not self.render_include_cursor.isChecked():
                if self.interactor3d.focus_cursor is not None:
                    temp_actors.append( self.interactor3d.cursors[0]['actor'] )

            for actor in temp_actors:
                self.renderer_3d.RemoveActor(actor)
            # -------------------------------------------------------------------------

            # Do the render
            im = render.render_hires(self.renderer_3d,oversampling=oversampling,aa=aa,transparency=use_transparency)

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

            im = render.render_cam_view(self.cadmodel,self.render_calib,extra_actors = extra_actors,oversampling=oversampling,aa=aa,transparency=use_transparency,verbose=False)
        

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

                if None in cal.view_models:
                    self.show_msgbox('The calibration file {:s} is missing one or more view models, so does not define sight-lines to view and will not be loaded.'.format(cal.filename))
                    
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
                
                n = 0
                for item in self.sightlines.keys():
                    if item.checkState() == qt.Qt.Checked:
                        n = n + 1

                if n > 1:
                    self.sightlines_legend_checkbox.setChecked(True)

                if self.sightlines[data][1] is None:
                    self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
                    self.statusbar.showMessage('Ray casting camera sight lines...')
                    actor = render.get_fov_actor(self.cadmodel,self.sightlines[data][0],self.sightlines[data][2])
                    self.statusbar.clearMessage()
                    actor.GetProperty().SetColor(next(self.colourcycle))
                    actor.GetProperty().SetOpacity(100.**(self.sightline_opacity_slider.value()/100.)/100.)
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
                self.sightlines[item][1].GetProperty().SetOpacity( 100.**(data/100.)/100. )

        elif self.sender() is self.pick_sightlines_colour and len(self.sightlines_list.selectedItems()) > 0:

            picked_colour = self.pick_colour(self.sightlines[self.sightlines_list.selectedItems()[0]][1].GetProperty().GetColor())
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

            self.interactor3d.set_legend(legend_items)  

        else:
            self.interactor3d.set_legend([])

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





class CoordsDialog(qt.QDialog):

    def __init__(self, parent,coords_shape):

        # GUI initialisation
        qt.QDialog.__init__(self, parent)
        qt.uic.loadUi(os.path.join(guipath,'qt_designer_files','line_coords.ui'), self)

        self.parent = parent

        if coords_shape[1] == 6:
            self.lines_label.setText('Importing coordinates for {:d} 3D lines from file.'.format(coords_shape[0]))
        elif coords_shape[1] == 3:
            self.lines_label.setText('Importing 3D line containing {:d} points from file.'.format(coords_shape[0]))