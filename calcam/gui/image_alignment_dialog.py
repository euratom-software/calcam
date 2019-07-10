'''
* Copyright 2015-2019 European Atomic Energy Community (EURATOM)
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
import os

import cv2
import vtk
import numpy as np

from .vtkinteractorstyles import CalcamInteractorStyle2D
from . import qt_wrapper as qt
from .. import movement

guipath = os.path.split(os.path.abspath(__file__))[0]


class ImageAlignDialog(qt.QDialog):

    def __init__(self, parent,ref_image,new_image,app=None):

        # GUI initialisation

        if parent is None:
            if app is None:
                raise ValueError('Either parent or app must be provided!')
            else:
                self.app = app
        else:
            self.app = parent.app

        qt.QDialog.__init__(self, parent,qt.Qt.WindowTitleHint | qt.Qt.WindowCloseButtonHint)
        qt.uic.loadUi(os.path.join(guipath,'qt_designer_files','image_alignment_dialog.ui'), self)

        self.qvtkwidget_ref = qt.QVTKRenderWindowInteractor(self.ref_frame)
        self.ref_frame.layout().addWidget(self.qvtkwidget_ref)
        self.interactor_ref = CalcamInteractorStyle2D(refresh_callback=self.refresh,newpick_callback = lambda x: self.new_point('ref',x),focus_changed_callback=lambda x: self.change_point_focus('ref',x))
        self.qvtkwidget_ref.SetInteractorStyle(self.interactor_ref)
        self.renderer_ref = vtk.vtkRenderer()
        self.renderer_ref.SetBackground(0, 0, 0)
        self.qvtkwidget_ref.GetRenderWindow().AddRenderer(self.renderer_ref)
        self.camera_ref = self.renderer_ref.GetActiveCamera()

        self.qvtkwidget_new = qt.QVTKRenderWindowInteractor(self.new_frame)
        self.new_frame.layout().addWidget(self.qvtkwidget_new)
        self.interactor_new = CalcamInteractorStyle2D(refresh_callback=self.refresh,newpick_callback = lambda x: self.new_point('new',x),focus_changed_callback=lambda x: self.change_point_focus('new',x))
        self.qvtkwidget_new.SetInteractorStyle(self.interactor_new)
        self.renderer_new = vtk.vtkRenderer()
        self.renderer_new.SetBackground(0, 0, 0)
        self.qvtkwidget_new.GetRenderWindow().AddRenderer(self.renderer_new)
        self.camera_new = self.renderer_new.GetActiveCamera()

        self.fitted_points_checkbox.hide()

        # GUI callbacks
        self.image_enhancement_checkbox.toggled.connect(self.toggle_enhancement)
        self.update_homography_button.clicked.connect(self.get_transform)
        #self.fitted_points_checkbox.toggled.connect(self.toggle_fitted_points)
        self.overlay_button.pressed.connect(lambda: self.toggle_overlay(True))
        self.overlay_button.released.connect(lambda: self.toggle_overlay(False))

        # State initialisation
        self.buttonbox.button(qt.QDialogButtonBox.Ok).setEnabled(False)
        self.point_pairings = []
        self.overlay_cursors = []
        self.selected_pointpair = None
        self.transform_params.hide()
        self.transform = None


        # Start the GUI!
        self.show()
        self.interactor_ref.init()
        self.interactor_new.init()
        self.qvtkwidget_ref.GetRenderWindow().GetInteractor().Initialize()
        self.qvtkwidget_new.GetRenderWindow().GetInteractor().Initialize()

        self.interactor_ref.link_with(self.interactor_new)
        self.interactor_new.link_with(self.interactor_ref)

        self.ref_image = ref_image
        self.ref_image_enhanced = None
        self.new_image = new_image
        self.new_image_enhanced = None


        self.interactor_ref.set_image(ref_image)
        self.interactor_new.set_image(new_image)
        self.refresh()

        self.show()



    def refresh(self):
        self.renderer_ref.Render()
        self.qvtkwidget_ref.update()
        self.renderer_new.Render()
        self.qvtkwidget_new.update()


    def new_point(self,image,position):

        if image == 'ref':
            index = self.interactor_ref.add_active_cursor(position)
            if self.selected_pointpair is not None and self.point_pairings[self.selected_pointpair][0] is None:
                self.point_pairings[self.selected_pointpair][0] = index
            else:
                if self.selected_pointpair is not None and self.point_pairings[self.selected_pointpair][1] is None:
                    self.interactor_ref.remove_active_cursor(self.point_pairings[self.selected_pointpair][0])
                    self.point_pairings.remove(self.point_pairings[self.selected_pointpair])
                self.point_pairings.append([index,None])
                self.selected_pointpair = len(self.point_pairings) - 1

        elif image == 'new':

            index = self.interactor_new.add_active_cursor(position)

            if self.selected_pointpair is not None and self.point_pairings[self.selected_pointpair][1] is None:
                self.point_pairings[self.selected_pointpair][1] = index
            else:
                if self.selected_pointpair is not None and self.point_pairings[self.selected_pointpair][0] is None:
                    self.interactor_new.remove_active_cursor(self.point_pairings[self.selected_pointpair][1])
                    self.point_pairings.remove(self.point_pairings[self.selected_pointpair])
                self.point_pairings.append([None, index])
                self.selected_pointpair = len(self.point_pairings) - 1

        self.interactor_ref.set_cursor_focus(self.point_pairings[self.selected_pointpair][0])
        self.interactor_new.set_cursor_focus(self.point_pairings[self.selected_pointpair][1])

        if self.count_pointpairs() > 2:
            self.update_homography_button.setEnabled(True)
        else:
            self.update_homography_button.setEnabled(False)

        self.refresh()



    def change_point_focus(self,sender,id):

        if sender == 'ref':
            test_ind = 0
        elif sender == 'new':
            test_ind = 1

        for pp_ind,pointpair in enumerate(self.point_pairings):
            if pointpair[test_ind] == id:
                self.selected_pointpair = pp_ind
                self.interactor_ref.set_cursor_focus(pointpair[0])
                self.interactor_new.set_cursor_focus(pointpair[1])

        self.refresh()


    def toggle_enhancement(self,onoff):

        if onoff:
            if self.ref_image_enhanced is None:
                self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
                self.ref_image_enhanced = movement.preprocess_image(self.ref_image,maintain_scale=True)
                self.new_image_enhanced = movement.preprocess_image(self.new_image, maintain_scale=True)
                self.app.restoreOverrideCursor()

            self.interactor_new.set_image(self.new_image_enhanced,hold_position=True)
            self.interactor_ref.set_image(self.ref_image_enhanced,hold_position=True)
        else:
            self.interactor_new.set_image(self.new_image,hold_position=True)
            self.interactor_ref.set_image(self.ref_image,hold_position=True)

        self.refresh()


    def get_transform(self):

        self.fitted_points_checkbox.setChecked(False)
        ref_points = np.zeros((self.count_pointpairs(),2))
        new_points = np.zeros((self.count_pointpairs(),2))

        for i,pp in enumerate(self.point_pairings):
            if i < ref_points.shape[0]:
                ref_points[i,:] = self.interactor_ref.get_cursor_coords(pp[0])[0]
                new_points[i,:] = self.interactor_new.get_cursor_coords(pp[1])[0]

        self.transform = np.matrix(cv2.estimateRigidTransform(new_points, ref_points, fullAffine=False))

        err = []
        for pp in self.point_pairings:
            if pp[1] is not None:
                oldpt = np.matrix(np.concatenate((self.interactor_new.get_cursor_coords(pp[1])[0], np.ones(1)))).T
                refpt = np.matrix(np.concatenate((self.interactor_ref.get_cursor_coords(pp[0])[0], np.ones(1)))).T
                fitted_pt = self.transform * oldpt
                err.append(refpt[0] - fitted_pt[0])
                err.append(refpt[1] - fitted_pt[1])
        rms_err = np.sqrt(np.sum(np.array(err)**2))

        desc = ' ( {:.1f} , {:.1f} ) pixels<br>'.format(self.transform[0, 2], self.transform[1, 2])
        desc = desc + ' {:.2f} degrees<br>'.format(180 * np.arctan2(self.transform[1, 0], self.transform[0, 0]) / 3.14159)
        desc = desc + ' {:.2f}<br>'.format(np.sqrt(self.transform[1, 0] ** 2 + self.transform[0, 0] ** 2))
        desc = desc + ' {:.2f} px'.format(rms_err)
        self.transform_params.setText(desc)
        self.transform_params.show()

        self.fitted_points_checkbox.setEnabled(True)
        self.overlay_button.setEnabled(True)
        self.buttonbox.button(qt.QDialogButtonBox.Ok).setEnabled(True)
        self.fitted_points_checkbox.setChecked(True)



    def count_pointpairs(self):

        if self.selected_pointpair is not None and None in self.point_pairings[self.selected_pointpair]:
            return len(self.point_pairings) - 1
        else:
            return len(self.point_pairings)


    def toggle_fitted_points(self,onoff):

        for cursor_id in self.overlay_cursors:
            self.interactor_ref.remove_passive_cursor(cursor_id)
        self.overlay_cursors = []

        if onoff:
            for pp in self.point_pairings:
                if pp[1] is not None:
                    oldpt = np.matrix(np.concatenate((self.interactor_new.get_cursor_coords(pp[1])[0],np.ones(1)))).T
                    fitted_pt = self.transform * oldpt
                    self.overlay_cursors.append( self.interactor_ref.add_passive_cursor([fitted_pt[0],fitted_pt[1]]) )


    def toggle_overlay(self,onoff):

        if onoff:
            if self.image_enhancement_checkbox.isChecked():
                orig_im = self.new_image_enhanced
            else:
                orig_im = self.new_image

            if self.transform is not None:

                overlay_image = movement.warp_image(orig_im, self.transform)[0]

            else:

                overlay_image = orig_im

            self.interactor_ref.set_overlay_image(overlay_image)
            self.interactor_new.set_overlay_image(overlay_image)

        else:
            self.interactor_ref.set_overlay_image(None)
            self.interactor_new.set_overlay_image(None)

        self.refresh()
