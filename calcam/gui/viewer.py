import cv2

from .core import *
from .vtkinteractorstyles import CalcamInteractorStyle3D
from .. import render

# CAD viewer window.
# This allows viewing of the CAD model and overlaying raycasted sight-lines, etc.
class ViewerWindow(CalcamGUIWindow):
 
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
        self.camFOV.valueChanged.connect(self.change_cad_view)
        self.sightlines_list.itemChanged.connect(self.update_sightlines)
        self.lines_3d_list.itemChanged.connect(self.update_lines)
        self.load_model_button.clicked.connect(self.load_model)
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
        self.model_name.currentIndexChanged.connect(self.populate_model_variants)
        self.contour_off.clicked.connect(self.update_contour)
        self.contour_2d.clicked.connect(self.update_contour)
        self.contour_3d.clicked.connect(self.update_contour)
        self.load_lines_button.clicked.connect(self.update_lines)
        self.pick_lines_colour.clicked.connect(self.update_lines)
        self.lines_3d_list.itemSelectionChanged.connect(self.update_selected_lines)
        
        self.sightlines_legend = None
        self.render_calib = None

        self.model_actors = {}

        self.line_actors = DodgyDict()

        self.contour_actor = None

        self.sightlines = DodgyDict()
        self.colour_q = []
        self.model_custom_colour = None
        self.viewport_calibs = DodgyDict()

        self.colourcycle = colourcycle()

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

        # Warn the user if we don't have any CAD models
        if self.model_list == {}:
            warn_no_models(self)


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
                fname = filedialog.selectedFiles()[0]

                coords = None
                for delimiter in ['\t',' ',',']:
                    try:
                        coords = np.loadtxt(fname,delimiter=delimiter)
                        lines_name = os.path.split(fname)[1].split('.')[0]
                    except:
                        continue

                if coords is None:
                    raise UserWarning('Could not load coordinates from the file. Please ensure the file is formatted as N rows, 3 or 6 columns and is tab, space or comma delimited.')

                elif coords.shape[1] in [3,6]:

                    coords_dialog = CoordsDialog(self,coords.shape)
                    coords_dialog.exec()
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
                self.renderer_3d.AddActor(self.line_actors[data])
            else:
                self.renderer_3d.RemoveActor(self.line_actors[data])

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
                self.camera_3d.SetViewAngle(view['y_fov'])
                self.camera_3d.SetPosition(view['cam_pos'])
                self.camera_3d.SetFocalPoint(view['target'])
                self.camera_3d.SetViewUp(0,0,1)
                self.interactor3d.set_xsection(view['xsection'])

            elif view_item.parent() is self.views_root_results or view_item.parent() in self.viewport_calibs.keys():

                view,subfield = self.viewport_calibs[(view_item)]
                if subfield is None:
                    return

                self.camera_3d.SetPosition(view.get_pupilpos(subview=subfield))
                self.camera_3d.SetFocalPoint(view.get_pupilpos(subview=subfield) + view.get_los_direction(view.geometry.get_display_shape()[0]/2,view.geometry.get_display_shape()[1]/2))
                self.camera_3d.SetViewAngle(view.get_fov(subview=subfield)[1])
                self.camera_3d.SetViewUp(-1.*view.get_cam_to_lab_rotation(subview=subfield)[:,1])
                self.interactor3d.set_xsection(None)               

            elif view_item.parent() is self.views_root_auto and self.interactor3d.focus_cursor is not None:

                cursorpos = self.interactor3d.get_cursor_coords(0)

                if str(view_item.text(0)).lower() == 'horizontal cross-section thru cursor':
                    self.camera_3d.SetViewUp(0,1,0)
                    self.camera_3d.SetPosition( (0.,0.,max(self.camZ.value(),cursorpos[2]+1.)) )
                    self.camera_3d.SetFocalPoint( (0.,0.,cursorpos[2]-1.) )
                    self.xsection_checkbox.setChecked(True)

                elif str(view_item.text(0)).lower() == 'vertical cross-section thru cursor':
                    self.camera_3d.SetViewUp(0,0,1)
                    R_cursor = np.sqrt( cursorpos[1]**2 + cursorpos[0]**2 )
                    phi = np.arctan2(cursorpos[1],cursorpos[0])
                    phi_cam = phi - 3.14159/2.
                    R_cam = np.sqrt( self.camX.value()**2 + self.camY.value()**2 )
                    self.camera_3d.SetPosition( (max(R_cam,R_cursor + 1) * np.cos(phi_cam), max(R_cam,R_cursor + 1) * np.sin(phi_cam), 0.) )
                    self.camera_3d.SetFocalPoint( (0.,0.,0.) )
                    self.xsection_checkbox.setChecked(True)


        else:
            self.camera_3d.SetPosition((self.camX.value(),self.camY.value(),self.camZ.value()))
            self.camera_3d.SetFocalPoint((self.tarX.value(),self.tarY.value(),self.tarZ.value()))
            self.camera_3d.SetViewAngle(self.camFOV.value())

        self.update_viewport_info(keep_selection=True)

        self.interactor3d.update_clipping()

        self.refresh_3d()
        

    def on_change_cad_view(self):
        
        for key,item in self.sightlines:
            recheck = False
            if key.checkState() == qt.Qt.Checked:
                recheck = True
                key.setCheckState(qt.Qt.Unchecked)
                self.colourcycle.queue_colour(item[1].GetProperty().GetColor())
            item[1] = None
            if recheck:
                key.setCheckState(qt.Qt.Checked)

 
    def toggle_wireframe(self,wireframe):
        
        if self.cadmodel is not None:

            self.cadmodel.set_wireframe( wireframe )

            self.refresh_3d()


    def add_cursor(self,coords):

        if self.interactor3d.focus_cursor is None:
            self.interactor3d.add_cursor(coords)
            self.interactor3d.set_cursor_focus(0)
            self.update_cursor_position(coords)

    def on_load_model(self):
        
        # Turn off any wall contour
        self.contour_off.setChecked(True)

    def toggle_cursor_xsection(self,onoff):

        if onoff:
            self.interactor3d.set_xsection(self.interactor3d.get_cursor_coords(0))
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


    def update_contour(self):

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))

        if self.contour_actor is not None:
            self.renderer_3d.RemoveActor(self.contour_actor)
            self.contour_actor = None

        if self.contour_2d.isChecked():

            cursor_pos = self.interactor3d.get_cursor_coords(0)
            phi = np.arctan2(cursor_pos[1],cursor_pos[0])
            self.contour_actor = render.get_wall_contour_actor(self.cadmodel.wall_contour,'contour',phi)
            self.contour_actor.GetProperty().SetLineWidth(3)
            self.contour_actor.GetProperty().SetColor((1,0,0))
            self.renderer_3d.AddActor(self.contour_actor)

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
                    self.renderer_3d.AddActor(actor)
                    self.app.restoreOverrideCursor()

                else:

                    self.renderer_3d.AddActor(self.sightlines[data][1])

            else:
                self.renderer_3d.RemoveActor(self.sightlines[data][1])

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




    def update_cursor_position(self,position):
        info = 'Cursor location: ' + self.cadmodel.format_coord(position).replace('\n',' | ')
        self.statusbar.showMessage(info)

        self.contour_2d.setEnabled(self.cadmodel.wall_contour is not None)
        
        if self.contour_2d.isChecked():
            self.update_contour()

        self.xsection_checkbox.setEnabled(True)
        for i in range(self.views_root_auto.childCount()):
            self.views_root_auto.child(i).setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
        if self.xsection_checkbox.isChecked():
            self.interactor3d.set_xsection(self.interactor3d.get_cursor_coords(0))




    def closeEvent(self,event):

        if self.cadmodel is not None:
            self.cadmodel.remove_from_renderer(self.renderer_3d)
            self.cadmodel.unload()

        self.on_close()



class CoordsDialog(qt.QDialog):

    def __init__(self, parent,coords_shape):

        # GUI initialisation
        qt.QDialog.__init__(self, parent)
        qt.uic.loadUi(os.path.join(guipath,'line_coords.ui'), self)

        self.parent = parent

        if coords_shape[1] == 6:
            self.lines_label.setText('Importing coordinates for {:d} 3D lines from file.'.format(coords_shape[0]))
        elif coords_shape[1] == 3:
            self.lines_label.setText('Importing 3D line containing {:d} points from file.'.format(coords_shape[0]))