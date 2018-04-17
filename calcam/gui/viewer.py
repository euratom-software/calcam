import cv2

from .core import *
from .vtkinteractorstyles import CalcamInterActorStyle3D
from .. import render

# CAD viewer window.
# This allows viewing of the CAD model and overlaying raycasted sight-lines, etc.
class ViewerWindow(CalcamGUIWindow):
 
    def __init__(self, app, parent = None):

        CalcamGUIWindow.init(self,'viewer.ui',app,parent)

        # Start up with no CAD model
        self.cadmodel = None

        # Set up VTK
        self.qvtkWidget = qt.QVTKRenderWindowInteractor(self.vtk_frame)
        self.vtk_frame.layout().addWidget(self.qvtkWidget,0,0,1,2)
        self.interactor3d = CalcamInterActorStyle3D(refresh_callback=self.refresh_vtk,viewport_callback=self.update_viewport_info,newpick_callback=self.add_cursor,cursor_move_callback=self.update_cursor_position,resize_callback=self.on_resize)
        self.qvtkWidget.SetInteractorStyle(self.interactor3d)
        self.renderer_3d = vtk.vtkRenderer()
        self.renderer_3d.SetBackground(0, 0, 0)
        self.qvtkWidget.GetRenderWindow().AddRenderer(self.renderer_3d)
        self.vtkInteractor = self.qvtkWidget.GetRenderWindow().GetInteractor()
        self.camera = self.renderer_3d.GetActiveCamera()


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

        self.sightlines_legend = None
        self.render_calib = None

        self.model_actors = {}

        self.contour_actor = None

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


        # Start the GUI!
        self.show()

        self.interactor3d.init()
        self.views_root_auto.setHidden(False)
        self.vtkInteractor.Initialize()

        # Warn the user if we don't have any CAD models
        if self.model_list == {}:
            warn_no_models(self)


 
    def toggle_wireframe(self,wireframe):
        
        if self.cadmodel is not None:

            self.cadmodel.set_wireframe( wireframe )

            self.refresh_vtk()


    def add_cursor(self,coords):

        if self.interactor3d.focus_cursor is None:
            self.interactor3d.add_cursor(coords)
            self.update_cursor_position(coords)


    def toggle_cursor_xsection(self,onoff):

        if onoff:
            self.interactor3d.set_xsection(self.interactor3d.get_cursor_coords(0))
        else:
            self.interactor3d.set_xsection(None)

        self.interactor3d.update_clipping()
        self.refresh_vtk()


    def on_model_load(self):

        # Enable the other tabs!
        self.tabWidget.setTabEnabled(1,True)
        self.tabWidget.setTabEnabled(2,True)
        self.tabWidget.setTabEnabled(3,True)

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

        self.refresh_vtk()
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
            self.interactor3d.OnWindowSizeAdjust()


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
                
                if len(self.interactor3d.sightline_actors) == 1:
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
                    self.interactor3d.sightline_actors.append(actor)
                else:
                    self.renderer_3d.AddActor(self.sightlines[data][1])
                    self.interactor3d.sightline_actors.append(self.sightlines[data][1])

            else:
                self.renderer_3d.RemoveActor(self.sightlines[data][1])
                self.interactor3d.sightline_actors.remove(self.sightlines[data][1])
                if len(self.interactor3d.sightline_actors) < 2:
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

        self.qvtkWidget.update()




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

        self.config.save()
        if self.cadmodel is not None:
            self.cadmodel.remove_from_renderer(self.renderer_3d)
            self.cadmodel.unload()

        # If we're exiting, put python'e exception handling back to normal.
        sys.excepthook = sys.__excepthook__