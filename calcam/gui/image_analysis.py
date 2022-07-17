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


import copy

import matplotlib.cm

from .core import *
from .vtkinteractorstyles import CalcamInteractorStyle2D, CalcamInteractorStyle3D
from ..render import render_cam_view,get_wall_coverage_actor,render_hires
from ..coordtransformer import CoordTransformer
from ..raycast import raycast_sightlines
from ..image_enhancement import enhance_image, scale_to_8bit
from ..movement import manual_movement
from ..calibration import NoSubviews

type_description = {'alignment': 'Manual Alignment', 'fit':'Point pair fitting','virtual':'Virtual'}

cmaps = []
for cmap in [('Grayscale','gray'),('Inferno','inferno'),('Magma','magma'),('Plasma','plasma'),('Viridis','viridis'),('JET','jet'),('HSV','hsv')]:
    try:
        cmaps.append((cmap[0],matplotlib.cm.get_cmap(cmap[1])))
    except Exception:
        continue

# Main calcam window class for actually creating calibrations.
class ImageAnalyser(CalcamGUIWindow):

    def __init__(self, app, parent = None):

        # GUI initialisation
        CalcamGUIWindow.init(self,'image_analyser.ui',app,parent)

        # Start up with no CAD model
        self.cadmodel = None
        self.calibration = None

        # Set up VTK
        self.qvtkwidget_3d = qt.QVTKRenderWindowInteractor(self.vtkframe_3d)
        self.vtkframe_3d.layout().addWidget(self.qvtkwidget_3d)
        self.interactor3d = CalcamInteractorStyle3D(refresh_callback=self.refresh_3d,viewport_callback=self.update_viewport_info,resize_callback=self.update_vtk_size,newpick_callback=self.update_from_3d,cursor_move_callback=lambda cid,coords: self.update_from_3d(coords))
        self.interactor3d.allow_focus_change = False
        self.qvtkwidget_3d.SetInteractorStyle(self.interactor3d)
        self.renderer_3d = vtk.vtkRenderer()
        self.renderer_3d.SetBackground(0, 0, 0)
        self.qvtkwidget_3d.GetRenderWindow().AddRenderer(self.renderer_3d)
        self.camera_3d = self.renderer_3d.GetActiveCamera()

        self.qvtkwidget_2d = qt.QVTKRenderWindowInteractor(self.vtkframe_2d)
        self.vtkframe_2d.layout().addWidget(self.qvtkwidget_2d)
        self.interactor2d = CalcamInteractorStyle2D(refresh_callback=self.refresh_2d,newpick_callback=lambda x: self.update_from_2d([x]),cursor_move_callback=lambda cid,coords: self.update_from_2d(coords))
        self.interactor2d.allow_focus_change = False
        self.qvtkwidget_2d.SetInteractorStyle(self.interactor2d)
        self.renderer_2d = vtk.vtkRenderer()
        self.renderer_2d.SetBackground(0, 0, 0)
        self.qvtkwidget_2d.GetRenderWindow().AddRenderer(self.renderer_2d)
        self.camera_2d = self.renderer_2d.GetActiveCamera()

        self.image_geometry = None
        self.image_assumptions = []
        self.image_load_coords = None
        self.mov_correction = None
        self.im_projection = None

        self.coords_3d = None

        self.populate_models()



        # Disable image transform buttons if we have no image
        self.image_settings.hide()

        self.tabWidget.setTabEnabled(2,False)
        self.tabWidget.setTabEnabled(3,False)

        self.cmap_dropdown.addItems([cmap[0] for cmap in cmaps])
        self.cmap_dropdown.setCurrentIndex(0)

        # Callbacks for GUI elements
        self.image_sources_list.currentIndexChanged.connect(self.build_imload_gui)
        self.viewlist.itemSelectionChanged.connect(self.change_cad_view)
        self.camX.valueChanged.connect(self.change_cad_view)
        self.camY.valueChanged.connect(self.change_cad_view)
        self.camZ.valueChanged.connect(self.change_cad_view)
        self.tarX.valueChanged.connect(self.change_cad_view)
        self.tarY.valueChanged.connect(self.change_cad_view)
        self.tarZ.valueChanged.connect(self.change_cad_view)
        self.cam_roll.valueChanged.connect(self.change_cad_view)
        self.camFOV.valueChanged.connect(self.change_cad_view)
        self.load_model_button.clicked.connect(self.load_model)
        self.model_name.currentIndexChanged.connect(self.populate_model_variants)
        self.feature_tree.itemChanged.connect(self.update_checked_features)
        self.load_image_button.clicked.connect(self.load_image)
        self.load_calib_button.clicked.connect(self.load_calib)
        self.reset_view_button.clicked.connect(lambda: self.set_view_from_calib(self.calibration,0))
        self.cursor_closeup_button.clicked.connect(self.set_view_to_cursor)
        self.cal_props_button.clicked.connect(self.show_calib_info)
        self.viewport_load_calib.clicked.connect(self.load_viewport_calib)
        self.cmap_checkbox.toggled.connect(self.update_cmap)
        self.cmap_dropdown.currentIndexChanged.connect(self.update_cmap)
        self.cmap_min.valueChanged.connect(self.update_cmap)
        self.cmap_max.valueChanged.connect(self.update_cmap)
        self.movement_correction_button.clicked.connect(self.update_movement_correction)
        self.im_mapping_checkbox.toggled.connect(self.update_mapped_image)
        self.mapped_im_opacity_slider.valueChanged.connect(self.update_mapped_image)
        self.render_button.clicked.connect(self.save_image)
        self.render_image_checkbox.toggled.connect(self.update_render_resolution)
        self.render_cad_checkbox.toggled.connect(self.update_render_resolution)
        self.overlay_opacity_slider.valueChanged.connect(self.change_overlay_colour)
        self.overlay_type_box.currentIndexChanged.connect(self.update_overlay)
        self.overlay_colour_button.clicked.connect(self.change_overlay_colour)

        self.sightline_checkbox.toggled.connect(lambda: self.update_from_3d(self.coords_3d))

        self.overlay_checkbox.toggled.connect(self.update_overlay)

        self.enhance_checkbox.stateChanged.connect(self.toggle_enhancement)

        self.image = None
        self.cmap_options.hide()
        self.viewport_calibs = DodgyDict()

        self.mapped_im_opacity_slider.hide()
        self.mapped_im_opacity_label.hide()

        self.control_sensitivity_slider.valueChanged.connect(lambda x: self.interactor3d.set_control_sensitivity(x*0.01))
        self.rmb_rotate.toggled.connect(self.interactor3d.set_rmb_rotate)
        self.interactor3d.set_control_sensitivity(self.control_sensitivity_slider.value()*0.01)


        # Populate image sources list and tweak GUI layout for image loading.
        self.imload_inputs = {}
        self.image_load_options.layout().setColumnMinimumWidth(0,100)

        self.image_sources = self.config.get_image_sources()
        index = -1
        for i,imsource in enumerate(self.image_sources):
            self.image_sources_list.addItem(imsource.display_name)
            if imsource.display_name == self.config.default_image_source:
                index = i

        self.image_sources_list.setCurrentIndex(index)

        self.cursor_ids = {'3d':None,'2d':{'visible':None,'hidden':None}}
        self.sightline_actors = []
        self.overlay_colour = [0,0,1,0.5]

        self.overlay_settings.hide()

        # Start the GUI!
        self.show()
        self.interactor2d.init()
        self.interactor3d.init()
        self.qvtkwidget_3d.GetRenderWindow().GetInteractor().Initialize()
        self.qvtkwidget_2d.GetRenderWindow().GetInteractor().Initialize()
        self.interactor3d.on_resize()


    def update_render_resolution(self):

        w = self.vtksize[0]
        h = self.vtksize[1]

        if not self.render_image_checkbox.isChecked() and not self.render_cad_checkbox.isChecked():
            self.render_button.setEnabled(False)
            self.render_resolution.setEnabled(False)
        else:
            self.render_button.setEnabled(True)
            self.render_resolution.setEnabled(True)

        if self.render_image_checkbox.isChecked():
            self.render_resolution.setCurrentIndex(0)
            self.render_resolution.setEnabled(False)
        else:
            self.render_resolution.setEnabled(True)

        if self.render_image_checkbox.isChecked() and self.render_cad_checkbox.isChecked():
            w = w * 2

        index = max(self.render_resolution.currentIndex(), 0)
        self.render_resolution.clear()
        self.render_resolution.addItem('{:d} x {:d} (Window Size)'.format(w, h))
        self.render_resolution.addItem('{:d} x {:d}'.format(w * 2, h * 2))
        self.render_resolution.addItem('{:d} x {:d}'.format(w * 4, h * 4))
        self.render_resolution.setCurrentIndex(index)



    def update_vtk_size(self,vtksize):

        self.vtksize = vtksize
        self.update_render_resolution()


    def save_image(self):

        filename = self.get_save_filename('image')
        if filename is None:
            return

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.statusbar.showMessage('Rendering image to {:s} ...'.format(filename))

        c3d = None
        c2d = []
        if not self.render_cursor_checkbox.isChecked():
            if self.interactor3d.focus_cursor is not None:
                c3d = self.interactor3d.cursors[self.interactor3d.focus_cursor]['actor']
            if self.interactor2d.focus_cursor is not None:
                for actor in self.interactor2d.active_cursors[self.interactor2d.focus_cursor]['actors']:
                    c2d.append(actor)

            self.renderer_3d.RemoveActor(c3d)
            for actor in c2d:
                self.renderer_2d.RemoveActor(actor)

        oversampling = 2 ** (self.render_resolution.currentIndex())

        h = self.vtksize[1] * oversampling
        im = np.zeros((h,0,3),dtype=np.uint8)

        if self.render_image_checkbox.isChecked():
            im = np.hstack((im,render_hires(self.renderer_2d,oversampling=oversampling)))
        if self.render_cad_checkbox.isChecked():
            im = np.hstack((im,render_hires(self.renderer_3d,oversampling=oversampling)))

        im[:,:,:3] = im[:,:,2::-1]
        cv2.imwrite(filename,im)

        if c3d is not None:
            self.renderer_3d.AddActor(c3d)
        for actor in c2d:
            self.renderer_2d.AddActor(actor)

        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()

    def update_movement_correction(self):

        if self.calibration is not None:
            transformer = self.calibration.geometry
        else:
            transformer = self.image_geometry

        self.mov_correction = manual_movement(self.calibration,transformer.original_to_display_image(self.image),self.mov_correction,parent_window=self)

        if self.coords_3d is not None:
            self.update_from_3d(self.coords_3d)

        if self.overlay_checkbox.isChecked():
            self.update_overlay()

        self.update_mapped_image()


    def update_from_3d(self,coords_3d):

        # Sight line drawing style
        visible_linewidth = 3
        visible_colour = (0,0.8,0)
        invisible_linewidth = 1
        invisible_colour = (0.8,0,0)

        if self.calibration is not None and self.image is not None and coords_3d is not None:

            self.cursor_closeup_button.setEnabled(True)

            # Clear anything which already exists
            if self.cursor_ids['3d'] is not None:
                self.interactor3d.remove_cursor(self.cursor_ids['3d'])
                self.cursor_ids['3d'] = None

            if self.cursor_ids['2d']['visible'] is not None:
                self.interactor2d.remove_active_cursor(self.cursor_ids['2d']['visible'])
                self.cursor_ids['2d']['visible'] = None

            if self.cursor_ids['2d']['hidden'] is not None:
                self.interactor2d.remove_active_cursor(self.cursor_ids['2d']['hidden'])
                self.cursor_ids['2d']['hidden'] = None

            for actor in self.sightline_actors:
                self.interactor3d.remove_extra_actor(actor)
            self.sightline_actors = []


            intersection_coords = None

            visible = [False] * self.calibration.n_subviews

            # Find where the cursor(s) is/are in 2D.
            image_pos_nocheck = self.calibration.project_points([coords_3d],coords='original')

            image_pos = self.calibration.project_points([coords_3d],check_occlusion_with=self.cadmodel,occlusion_tol=1e-2,coords='original')

            for i in range(len(image_pos)):
                image_pos[i][0][:] = self.calibration.geometry.original_to_display_coords(*image_pos[i][0])
                image_pos_nocheck[i][0][:] = self.calibration.geometry.original_to_display_coords(*image_pos_nocheck[i][0])

                if np.any(np.isnan(image_pos_nocheck[i])):
                    visible[i] = False
                    continue

                raydata = raycast_sightlines(self.calibration,self.cadmodel,image_pos_nocheck[i][0,0],image_pos_nocheck[i][0,1],verbose=False,force_subview=i)
                visible[i] =True

                if self.mov_correction is not None:
                    image_pos_nocheck[i][0][:] = self.mov_correction.ref_to_moved_coords(*image_pos_nocheck[i][0])

                if np.any(np.isnan(image_pos[i])):
                    visible[i] = False
                    intersection_coords = raydata.ray_end_coords

                if visible[i]:

                    if self.cursor_ids['2d']['visible'] is None:
                        self.cursor_ids['2d']['visible'] = self.interactor2d.add_active_cursor(image_pos_nocheck[i][0,:])
                    else:
                        self.interactor2d.add_active_cursor(image_pos_nocheck[i][0,:],add_to=self.cursor_ids['2d']['visible'])

                    if self.sightline_checkbox.isChecked():
                        linesource = vtk.vtkLineSource()
                        pupilpos = self.calibration.get_pupilpos(subview=i)
                        linesource.SetPoint1(pupilpos[0],pupilpos[1],pupilpos[2])
                        linesource.SetPoint2(coords_3d)
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(linesource.GetOutputPort())
                        actor = vtk.vtkActor()
                        actor.SetMapper(mapper)
                        actor.GetProperty().SetLineWidth(visible_linewidth)
                        actor.GetProperty().SetColor(visible_colour)
                        self.sightline_actors.append(actor)
                        self.interactor3d.add_extra_actor(actor)

                        model_extent = self.cadmodel.get_extent()
                        model_size = model_extent[1::2] - model_extent[::2]
                        max_ray_length = model_size.max() * 4
                        sldir = coords_3d - pupilpos
                        sldir = sldir / np.sqrt(np.sum(sldir**2))
                        rayend = pupilpos + sldir*max_ray_length

                        linesource = vtk.vtkLineSource()
                        linesource.SetPoint1(coords_3d)
                        linesource.SetPoint2(rayend)
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(linesource.GetOutputPort())
                        actor = vtk.vtkActor()
                        actor.SetMapper(mapper)
                        actor.GetProperty().SetLineWidth(invisible_linewidth)
                        actor.GetProperty().SetColor(invisible_colour)
                        self.sightline_actors.append(actor)
                        self.interactor3d.add_extra_actor(actor)



                else:
                    if self.cursor_ids['2d']['hidden'] is None:
                        self.cursor_ids['2d']['hidden'] = self.interactor2d.add_active_cursor(image_pos_nocheck[i][0,:])
                    else:
                        self.interactor2d.add_active_cursor(image_pos_nocheck[i][0,:],add_to=self.cursor_ids['2d']['hidden'])

                    if self.sightline_checkbox.isChecked():
                        linesource = vtk.vtkLineSource()
                        pupilpos = self.calibration.get_pupilpos(subview=i)
                        linesource.SetPoint1(*pupilpos)
                        linesource.SetPoint2(*intersection_coords)
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(linesource.GetOutputPort())
                        actor = vtk.vtkActor()
                        actor.SetMapper(mapper)
                        actor.GetProperty().SetLineWidth(invisible_linewidth)
                        actor.GetProperty().SetColor(visible_colour)
                        self.sightline_actors.append(actor)
                        self.interactor3d.add_extra_actor(actor)

                        linesource = vtk.vtkLineSource()
                        linesource.SetPoint1(*intersection_coords)
                        linesource.SetPoint2(*coords_3d)
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(linesource.GetOutputPort())
                        actor = vtk.vtkActor()
                        actor.SetMapper(mapper)
                        actor.GetProperty().SetLineWidth(invisible_linewidth)
                        actor.GetProperty().SetColor(invisible_colour)
                        self.sightline_actors.append(actor)
                        self.interactor3d.add_extra_actor(actor)

            self.cursor_ids['3d'] = self.interactor3d.add_cursor(coords_3d)

            if self.cursor_ids['2d']['visible'] is not None:
                self.interactor2d.set_cursor_focus(self.cursor_ids['2d']['visible'])
                self.interactor3d.set_cursor_focus(self.cursor_ids['3d'])


            self.coords_2d = image_pos_nocheck
            self.coords_3d = coords_3d

            self.update_position_info(self.coords_2d,self.coords_3d,visible)




    def update_from_2d(self,coords_2d):

        if self.calibration is not None and self.image is not None and self.cadmodel is not None:

            for i,coords in enumerate(coords_2d):
                if coords is not None:
                    if self.mov_correction is not None:
                        coords = self.mov_correction.moved_to_ref_coords(*coords)

                    if self.calibration.subview_lookup(*coords) == -1:
                        raise UserWarning('The clicked position is outside the calibrated field of view.')
                    elif np.any(coords != self.coords_2d[i]):
                        raydata = raycast_sightlines(self.calibration,self.cadmodel,coords[0],coords[1],coords='Display',verbose=False)
                        self.update_from_3d(raydata.ray_end_coords[0,:])
                        return


    def on_close(self):
        self.qvtkwidget_3d.close()
        self.qvtkwidget_2d.close()


    def update_mapped_image(self):

        if self.sender() is self.mapped_im_opacity_slider:
            if self.im_projection is not None:
                self.im_projection.GetProperty().SetOpacity(self.mapped_im_opacity_slider.value()/100)
            self.refresh_3d()
            return

        if self.im_projection is not None:
            self.renderer_3d.RemoveActor(self.im_projection)

        if self.im_mapping_checkbox.isChecked():
            self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            self.statusbar.showMessage('Mapping image to CAD surface...')
            self.app.processEvents()

            if self.mov_correction is not None:
                im = self.mov_correction.warp_moved_to_ref(self.interactor2d.get_image())[0]
            else:
                im = self.interactor2d.get_image()

            if self.im_projection is not None:
                try:
                    self.im_projection.update_image(im)
                except ValueError:
                    actor = get_wall_coverage_actor(self.calibration, cadmodel=self.cadmodel, image=im,imagecoords='Display')
                    actor.GetProperty().SetOpacity(self.mapped_im_opacity_slider.value() / 100)
                    self.im_projection = actor
            else:
                actor = get_wall_coverage_actor(self.calibration,cadmodel=self.cadmodel,image=im,imagecoords='Display')
                actor.GetProperty().SetOpacity(self.mapped_im_opacity_slider.value() / 100)
                self.im_projection = actor

            self.renderer_3d.AddActor(self.im_projection)
            self.mapped_im_opacity_slider.show()
            self.mapped_im_opacity_label.show()
            self.app.restoreOverrideCursor()
            self.statusbar.clearMessage()

        else:
            self.mapped_im_opacity_slider.hide()
            self.mapped_im_opacity_label.hide()

        self.refresh_3d()

    def load_calib(self):

        opened_calib = self.object_from_file('calibration')

        # If we get no calibration from the above call it means the user has cancelled
        if opened_calib is None:
            return

        if None in opened_calib.view_models:
            raise UserWarning('The selected calibration file does not contain a full set of calibration parameters. Only calibration files containing all calibration parameters can be used.')

        self.mov_correction = None

        if self.image is not None:
            if self.image_load_coords.lower() == 'display':

                if self.image_geometry.get_display_shape() != opened_calib.geometry.get_display_shape() or (self.image_geometry.offset != opened_calib.geometry.offset and 'image_offset' not in self.image_assumptions):
                    osize = opened_calib.geometry.display_to_original_shape(self.image_geometry.get_display_shape())
                    try:
                        opened_calib.set_detector_window((self.image_geometry.offset[0],self.image_geometry.offset[1],osize[0],osize[1]))
                    except NoSubviews:
                        raise UserWarning('The detector area according to the image metadata ({:d}x{:d} pixels at offset {:d}x{:d}) does not contain any image according to the calibration! Perhaps the image detector offset is set incorrectly?'.format(osize[0],osize[1],self.image_geometry.offset[0],self.image_geometry.offset[1]))
                if self.image_geometry.transform_actions == []:
                    self.image = opened_calib.geometry.display_to_original_image(self.image)

            else:
                if self.image_geometry.get_original_shape() != opened_calib.geometry.get_original_shape() or (self.image_geometry.offset != opened_calib.geometry.offset and 'image_offset' not in self.image_assumptions):
                    osize = self.image_geometry.get_original_shape()
                    try:
                        opened_calib.set_detector_window((self.image_geometry.offset[0],self.image_geometry.offset[1],osize[0],osize[1]))
                    except NoSubviews:
                        raise UserWarning('The detector area according to the image metadata ({:d}x{:d} pixels at offset {:d}x{:d}) does not contain any image according to the calibration! Perhaps the image detector offset is set incorrectly?'.format(osize[0],osize[1],self.image_geometry.offset[0],self.image_geometry.offset[1]))

            if opened_calib.image is not None:
                self.movement_correction_button.setEnabled(True)
            else:
                self.movement_correction_button.setEnabled(False)

            if self.cadmodel is not None:
                self.im_mapping_checkbox.setEnabled(True)

        self.calibration = opened_calib
        self.image_geometry = self.calibration.geometry
        self.calib_name.setText(os.path.split(self.calibration.filename)[1].replace('.ccc',''))
        self.cal_props_button.setEnabled(True)

        self.overlay_checkbox.setEnabled(True)
        self.im_mapping_checkbox.setEnabled(True)
        self.reset_view_button.setEnabled(True)
        self.overlay = None
        self.im_projection = None

        if self.image is not None:
            if self.enhance_checkbox.isChecked():
                self.interactor2d.set_image(enhance_image(self.image_geometry.original_to_display_image(self.image)))
            elif self.cmap_checkbox.isChecked():
                self.update_cmap()
            else:
                self.interactor2d.set_image(self.image_geometry.original_to_display_image(self.image))

        self.interactor2d.set_subview_lookup(self.calibration.n_subviews,self.calibration.subview_lookup)

        if opened_calib.cad_config is not None:
            cconfig = opened_calib.cad_config
            if self.cadmodel is not None and self.cadmodel.machine_name == cconfig['model_name'] and self.cadmodel.model_variant == cconfig['model_variant']:
                keep_model = True
            else:
                keep_model = False

            if keep_model:
                try:
                    self.cadmodel.enable_only(cconfig['enabled_features'])
                except Exception as e:
                    if 'Unknown feature' not in str(e):
                        raise
                self.update_feature_tree_checks()
            else:
                cconfig = opened_calib.cad_config
                load_model = True
                try:
                    name_index = sorted(self.model_list.keys()).index(cconfig['model_name'])
                    self.model_name.setCurrentIndex(name_index)
                    variant_index = self.model_list[ cconfig['model_name'] ][1].index(cconfig['model_variant'])
                    self.model_variant.setCurrentIndex(variant_index)
                except ValueError:
                    self.model_name.setCurrentIndex(-1)
                    load_model=False


                if load_model:
                    try:
                        self.load_model(featurelist=cconfig['enabled_features'])
                    except Exception as e:
                        self.cadmodel = None
                        if 'Unknown feature' not in str(e):
                            raise

                # I'm not sure why I have to call this twice to get it to work properly.
                # On the first call it almost works, but not quite accurately. Then
                # on the second call onwards it's fine. Should be investigated further.
                self.set_view_from_calib(self.calibration,0)
                self.set_view_from_calib(self.calibration,0)

        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)
            self.overlay_checkbox.setChecked(True)

        if self.cursor_ids['3d'] is not None:
            self.update_from_3d(self.coords_3d)
        else:
            self.coords_2d = [None] * self.calibration.n_subviews



    def unload_calib(self):

        if self.calibration is not None:
            self.calibration = None
            self.calib_name.setText('No Calibration Loaded.')
            self.overlay_checkbox.setChecked(False)
            self.overlay_checkbox.setEnabled(False)
            self.im_mapping_checkbox.setChecked(False)
            self.im_mapping_checkbox.setEnabled(False)
            self.reset_view_button.setEnabled(False)
            self.overlay = None
            self.im_projection = None
            self.cal_props_button.setEnabled(False)

            if self.cursor_ids['2d']['visible'] is not None:
                self.interactor2d.remove_active_cursor(self.cursor_ids['2d']['visible'])
                self.cursor_ids['2d']['visible'] = None

            if self.cursor_ids['2d']['hidden'] is not None:
                self.interactor2d.remove_active_cursor(self.cursor_ids['2d']['hidden'])
                self.cursor_ids['2d']['hidden'] = None


    def set_view_to_cursor(self):
        cursorpos = self.interactor3d.get_cursor_coords(self.cursor_ids['3d'])

        pupilpos = self.calibration.get_pupilpos(subview=0)
        campos_vect = (pupilpos - cursorpos)
        campos_vect = campos_vect / np.sqrt(np.sum(campos_vect**2))

        self.camera_3d.SetFocalPoint(cursorpos)
        self.camera_3d.SetViewAngle(60)

        self.camera_3d.SetPosition(cursorpos + 0.3*campos_vect)

        self.update_viewport_info()
        self.cam_roll.setValue(0)
        self.refresh_3d()


    def on_model_load(self):
        # Enable the other tabs!
        self.tabWidget.setTabEnabled(2,True)
        #self.tabWidget.setTabEnabled(2,True)
        #self.tabWidget.setTabEnabled(3,True)
        if self.image is not None:
            self.tabWidget.setTabEnabled(3,True)

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.statusbar.showMessage('Generating model octree...')
        self.cadmodel.get_cell_locator()
        self.statusbar.clearMessage()
        self.app.restoreOverrideCursor()
        self.overlay = None
        if self.image is not None and self.calibration is not None:
            self.im_mapping_checkbox.setEnabled(True)
        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)
            self.overlay_checkbox.setChecked(True)



    def on_load_image(self,newim):

        image = scale_to_8bit(newim['image_data'])

        newim_geometry = CoordTransformer(offset=newim['image_offset'],paspect=newim['pixel_aspect'])
        newim_geometry.set_transform_actions(newim['transform_actions'])
        newim_geometry.set_image_shape(newim['image_data'].shape[1],newim['image_data'].shape[0],coords=newim['coords'])

        reset_overlay = False

        if self.calibration is not None:

            if newim['coords'].lower() == 'original':
                size_mismatch = newim_geometry.get_original_shape() != self.calibration.geometry.get_original_shape()
            else:
                size_mismatch = newim_geometry.get_display_shape() != self.calibration.geometry.get_display_shape()

            if 'image_offset' not in newim['assumptions']:
                size_mismatch = size_mismatch | (newim_geometry.offset[0] != self.calibration.geometry.offset[0]) | (newim_geometry.offset[1] != self.calibration.geometry.offset[1])

            if size_mismatch:

                if 'coords' in newim['assumptions']:

                    if newim_geometry.get_display_shape() == self.calibration.geometry.get_original_shape():
                        newim_geometry = self.calibration.geometry
                        newim['coords'] = 'original'

                new_geom = copy.deepcopy(self.calibration.geometry)
                new_geom.set_image_shape(image.shape[1],image.shape[0],coords=newim['coords'])
                new_geom.set_offset(*newim_geometry.offset)
                osize = new_geom.get_original_shape()

                self.calibration.geometry.set_pixel_aspect(new_geom.pixel_aspectratio,relative_to='original')
                try:
                    self.calibration.set_detector_window((newim_geometry.offset[0], newim_geometry.offset[1], osize[0], osize[1]))
                except NoSubviews:
                    raise UserWarning('The detector area according to the image metadata ({:d}x{:d} pixels at offset {:d}x{:d}) does not contain any image according to the calibration! Perhaps the image detector offset is set incorrectly?'.format(osize[0], osize[1], self.image_geometry.offset[0], self.image_geometry.offset[1]))


                self.overlay = None
                if self.overlay_checkbox.isChecked():
                    self.overlay_checkbox.setChecked(False)
                    reset_overlay = True

            self.image_geometry = self.calibration.geometry

            if self.calibration.image is not None:
                self.movement_correction_button.setEnabled(True)

            if self.cadmodel is not None:
                self.im_mapping_checkbox.setEnabled(True)

        else:
            self.image_geometry = newim_geometry

        if newim['coords'].lower() == 'display':
            self.image = self.image_geometry.display_to_original_image(image)
        else:
            self.image = image

        self.image_assumptions = newim['assumptions']
        self.image_load_coords = newim['coords']


        if self.calibration is not None:
            self.interactor2d.set_image(self.image_geometry.original_to_display_image(self.image),n_subviews=self.calibration.n_subviews,subview_lookup=self.calibration.subview_lookup)
        else:
            self.interactor2d.set_image(self.image_geometry.original_to_display_image(self.image))

        self.overlay_checkbox.setChecked(reset_overlay)
        self.image_settings.show()

        if self.enhance_checkbox.isChecked():
            self.enhance_checkbox.setChecked(False)
            self.enhance_checkbox.setChecked(True)

        try:
            coords3d = self.interactor3d.get_cursor_coords(self.cursor_ids['3d'])
            self.update_from_3d(coords3d)
        except KeyError as e:
            pass

        self.update_image_info_string(self.image,self.image_geometry)
        self.app.restoreOverrideCursor()
        self.statusbar.clearMessage()

        if self.cadmodel is not None:
            self.tabWidget.setTabEnabled(3,True)

        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)
            self.overlay_checkbox.setChecked(True)

        if self.cmap_checkbox.isChecked():
            self.update_cmap()


    def update_image_info_string(self, im_array, geometry):

        if np.any(np.array(geometry.get_display_shape()) != np.array(geometry.get_original_shape())):
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP) [ As Displayed ]<br>{3:d} x {4:d} pixels ({5:.1f} MP) [ Raw Data ]<br>'.format(
                geometry.get_display_shape()[0], geometry.get_display_shape()[1],
                np.prod(geometry.get_display_shape()) / 1e6, geometry.get_original_shape()[0],
                geometry.get_original_shape()[1], np.prod(geometry.get_original_shape()) / 1e6)
        else:
            info_str = '{0:d} x {1:d} pixels ({2:.1f} MP)<br>'.format(geometry.get_display_shape()[0],
                                                                      geometry.get_display_shape()[1],
                                                                      np.prod(geometry.get_display_shape()) / 1e6)

        if len(im_array.shape) == 2:
            info_str = info_str + 'Monochrome'
            self.cmap_checkbox.setEnabled(True)
            self.cmap_min.blockSignals(True)
            self.cmap_max.blockSignals(True)
            self.cmap_min.setMinimum(im_array.min())
            self.cmap_min.setMaximum(im_array.max())
            self.cmap_min.setValue(im_array.min())
            self.cmap_max.setMinimum(im_array.min())
            self.cmap_max.setMaximum(im_array.max())
            self.cmap_max.setValue(im_array.max())
            self.cmap_min.blockSignals(False)
            self.cmap_max.blockSignals(False)
            self.cmap_checkbox.setToolTip('')
        elif len(im_array.shape) == 3:
            if np.all(im_array[:, :, 0] == im_array[:, :, 1]) and np.all(im_array[:, :, 2] == im_array[:, :, 1]):
                info_str = info_str + 'Monochrome'
                self.cmap_min.blockSignals(True)
                self.cmap_max.blockSignals(True)
                self.cmap_checkbox.setEnabled(True)
                self.cmap_checkbox.setEnabled(True)
                self.cmap_min.setMinimum(im_array.min())
                self.cmap_min.setMaximum(im_array.max())
                self.cmap_min.setValue(im_array.min())
                self.cmap_max.setMinimum(im_array.min())
                self.cmap_max.setMaximum(im_array.max())
                self.cmap_max.setValue(im_array.max())
                self.cmap_min.blockSignals(False)
                self.cmap_max.blockSignals(False)
                self.cmap_checkbox.setToolTip('')
            else:
                info_str = info_str + 'RGB Colour'
                self.cmap_checkbox.setChecked(False)
                self.cmap_checkbox.setEnabled(False)
                self.cmap_checkbox.setToolTip('Colour mapping can only be applied to monochrome images.')
            if im_array.shape[2] == 4:
                info_str = info_str + ' + Transparency'

        self.image_info.setText(info_str)

    def update_cmap(self,data=None):

        if self.sender() is self.cmap_checkbox:
            if self.cmap_checkbox.isChecked():
                self.enhance_checkbox.setChecked(False)
                self.cmap_options.show()
            else:
                self.cmap_options.hide()
                self.toggle_enhancement()
                return

        self.cmap_min.blockSignals(True)
        self.cmap_max.blockSignals(True)
        self.cmap_min.setMaximum(self.cmap_max.value() - 1)
        self.cmap_max.setMinimum(self.cmap_min.value() + 1)
        self.cmap_min.blockSignals(False)
        self.cmap_max.blockSignals(False)

        image = self.image.astype(np.float32)
        if len(image.shape) > 2:
            image = image[:,:,0]

        cmin = self.cmap_min.value()
        cmax = self.cmap_max.value()

        cmap = cmaps[self.cmap_dropdown.currentIndex()][1]

        im = (image - cmin) / (cmax - cmin)
        im[im > 1] = 1
        im[im < 0] = 0
        cmapped_im = cmap(im)[:,:,:3]
        cmapped_im = (255 * cmapped_im).astype(np.uint8)

        if self.calibration is not None:
            transformer = self.calibration.geometry
        else:
            transformer = self.image_geometry

        self.interactor2d.set_image(transformer.original_to_display_image(cmapped_im), hold_position=True)
        if self.calibration is not None:
            self.interactor2d.set_subview_lookup(self.calibration.n_subviews,self.calibration.subview_lookup)

        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)
            self.overlay_checkbox.setChecked(True)

        self.update_mapped_image()


    def update_overlay(self,data=None):

        # Clear the existing overlay image to force it to re-render if the user has changed between solid / wireframe
        if self.sender() is self.overlay_type_box:
            self.overlay = None

        # Show or hide extra controls depending on what overlays are enabled
        self.overlay_settings.setVisible(self.overlay_checkbox.isChecked())

        if self.overlay_checkbox.isChecked():

            if self.overlay is None:
                self.overlay = self.render_overlay_image(self.overlay_type_box.currentIndex() == 0)

            # Apply desired colour
            im = (self.overlay * np.tile(np.array(self.overlay_colour)[np.newaxis, np.newaxis, :], self.overlay.shape[:2] + (1,))).astype(np.uint8)

            if self.mov_correction is not None:
                self.interactor2d.set_overlay_image(self.mov_correction.warp_ref_to_moved(im)[0])
            else:
                self.interactor2d.set_overlay_image(im)


        else:
            self.interactor2d.set_overlay_image(None)

        self.refresh_2d()


    def change_overlay_colour(self):

        if self.sender() is self.overlay_opacity_slider:
            new_colour = self.overlay_colour[:3]

        else:
            old_colour = self.overlay_colour
            new_colour = self.pick_colour(init_colour=old_colour)

        if new_colour is not None:

            new_colour = new_colour + [self.overlay_opacity_slider.value() / 100]
            self.overlay_colour = new_colour

            self.update_overlay()


    def render_overlay_image(self,wireframe):

        oversampling = int(np.ceil(min(1000/np.array(self.calibration.geometry.get_display_shape()))))

        self.statusbar.showMessage('Rendering CAD image overlay...')
        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.app.processEvents()

        orig_colours = self.cadmodel.get_colour()

        self.cadmodel.set_wireframe(wireframe)
        self.cadmodel.set_colour((1, 1, 1))
        image = render_cam_view(self.cadmodel, self.calibration, transparency=True, verbose=False, aa=2,oversampling=oversampling)
        self.cadmodel.set_colour(orig_colours)
        self.cadmodel.set_wireframe(False)

        self.statusbar.clearMessage()
        self.app.restoreOverrideCursor()

        return image



    def toggle_enhancement(self):

        # Enable / disable adaptive histogram equalisation
        if self.enhance_checkbox.isChecked():
            self.cmap_checkbox.setChecked(False)
            image = enhance_image(self.image)
        else:
            image = self.image

        if self.calibration is not None:
            transformer = self.calibration.geometry
        else:
            transformer = self.image_geometry

        self.interactor2d.set_image(transformer.original_to_display_image(image),hold_position=True)
        if self.calibration is not None:
            self.interactor2d.set_subview_lookup(self.calibration.n_subviews,self.calibration.subview_lookup)

        self.update_mapped_image()


    def on_change_cad_features(self):
        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.statusbar.showMessage('Updating model octree...')
        self.cadmodel.get_cell_locator()
        self.statusbar.clearMessage()
        self.app.restoreOverrideCursor()
        if self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(False)
        self.overlay = None
        self.im_mapping_checkbox.setChecked(False)
        self.im_projection = None


    def update_position_info(self,coords_2d,coords_3d,visible):

        model_extent = self.cadmodel.get_extent()
        model_size = model_extent[1::2] - model_extent[::2]
        max_ray_length = model_size.max() * 4

        cadinfo_str = self.cadmodel.format_coord(coords_3d)

        sightline_info_string = ''

        iminfo_str = ''

        sightline_fieldnames_str = ''

        impos_fieldnames_str = ''

        self.cursor_closeup_button.setEnabled(False)

        for field_index in range(self.calibration.n_subviews):

            if field_index > 0:
                impos_fieldnames_str = impos_fieldnames_str + '<br><br>'
                sightline_fieldnames_str = sightline_fieldnames_str + '<br><br>'
                iminfo_str = iminfo_str + '<br><br>'
                sightline_info_string = sightline_info_string + '<br><br>'

            sightline_exists = False


            impos_fieldnames_str = impos_fieldnames_str + '[{:s}]&nbsp;'.format( self.calibration.subview_names[field_index] )

            if np.any(np.isnan(coords_2d[field_index][0])):
                iminfo_str = iminfo_str + 'Cursor outside field of view.'
            elif not visible[field_index]:
                iminfo_str = iminfo_str +  'Cursor hidden from view.'
            else:
                iminfo_str = iminfo_str + 'X,Y : ( {:.0f} , {:.0f} ) px'.format(coords_2d[field_index][0][0],coords_2d[field_index][0][1])
                sightline_exists = True
                self.cursor_closeup_button.setEnabled(True)


            sightline_fieldnames_str = sightline_fieldnames_str + '[{:s}]&nbsp;'.format( self.calibration.subview_names[field_index] )
            if sightline_exists:
                sightline_fieldnames_str = sightline_fieldnames_str + '<br><br>'
                pupilpos = self.calibration.get_pupilpos(subview=field_index)
                sightline = coords_3d - pupilpos
                sdir = sightline / np.sqrt(np.sum(sightline**2))


                sightline_info_string = sightline_info_string + 'Origin X,Y,Z : ( {:.3f} , {:.3f} , {:.3f} )<br>'.format(pupilpos[0],pupilpos[1],pupilpos[2])
                sightline_info_string = sightline_info_string + 'Direction X,Y,Z : ( {:.3f} , {:.3f} , {:.3f} )<br>'.format(sdir[0],sdir[1],sdir[2])
                if np.sqrt(np.sum(sightline**2)) < (max_ray_length-1e-3):
                    sightline_info_string = sightline_info_string  +'Distance from camera to cursor: {:.3f} m'.format(np.sqrt(np.sum(sightline**2)))
                else:
                    sightline_info_string = sightline_info_string + 'Sight line does not inersect CAD model.'
                    cadinfo_str = 'Sight line does not intersect CAD model.'
            else:

                sightline_info_string = sightline_info_string + 'No line-of-sight to cursor'

        if self.calibration.n_subviews > 1:
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