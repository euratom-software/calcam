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
from .. import movement, misc
from ..image_enhancement import enhance_image
from ..config import CalcamConfig

guipath = os.path.split(os.path.abspath(__file__))[0]


class ImageAlignDialog(qt.QDialog):

    def __init__(self,app,parent,ref_image,new_image,correction=None):

        # GUI initialisation
        if parent is None:
            if app is None:
                raise ValueError('Either parent or app must be provided!')
            else:
                self.app = app
        else:
            self.app = parent.app

        qt.QDialog.__init__(self, parent,qt.Qt.WindowTitleHint | qt.Qt.WindowCloseButtonHint)
        qt.uic.loadUi(os.path.join(guipath,'qt_designer_files','movement_corr_dialog.ui'), self)
        self.setWindowIcon(qt.QIcon(os.path.join(guipath, 'icons', 'calcam.png')))

        # See how big the screen is and open the window at an appropriate size
        if qt.qt_ver < 5:
            available_space = self.app.desktop().availableGeometry(self)
        else:
            available_space = self.app.primaryScreen().availableGeometry()

        # Open the window with same aspect ratio as the screen, and no fewer than 500px tall.
        win_height = min(780,0.75*available_space.height())
        win_width = win_height * available_space.width() / available_space.height()
        self.resize(win_width,win_height)

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


        self.interactor_new.cursor_size = 0.06
        self.interactor_ref.cursor_size = 0.06

        # GUI callbacks
        self.image_enhancement_checkbox.toggled.connect(self.toggle_enhancement)
        self.update_homography_button.clicked.connect(self.get_transform)
        self.fitted_points_checkbox.toggled.connect(self.toggle_fitted_points)
        self.overlay_button.pressed.connect(lambda: self.toggle_overlay(True))
        self.overlay_button.released.connect(lambda: self.toggle_overlay(False))
        self.clear_button.clicked.connect(self.remove_all_points)
        self.auto_points_button.clicked.connect(self.find_points)
        self.clear_transform_button.clicked.connect(lambda : self.set_correction(None))
        self.save_button.clicked.connect(self.save)
        self.load_button.clicked.connect(self.open)

        # State initialisation
        self.point_pairings = []
        self.overlay_cursors = [[],[]]
        self.selected_pointpair = None
        self.transform_params.hide()
        self.transform = None


        # Start the GUI!
        self.show()
        self.interactor_ref.init()
        self.interactor_new.init()
        self.qvtkwidget_ref.GetRenderWindow().GetInteractor().Initialize()
        self.qvtkwidget_new.GetRenderWindow().GetInteractor().Initialize()

        self.ref_image = ref_image
        self.ref_image_enhanced = None
        self.new_image = new_image
        self.new_image_enhanced = None

        self.interactor_ref.set_image(ref_image)
        self.interactor_new.set_image(new_image)
        self.refresh()

        self.interactor_ref.link_with(self.interactor_new)
        self.interactor_new.link_with(self.interactor_ref)

        sc = qt.QShortcut(qt.QKeySequence(qt.QKeySequence.Delete),self)
        sc.setContext(qt.Qt.WindowShortcut)
        sc.activated.connect(self.remove_current_point)

        self.config = CalcamConfig()

        self.show()

        if correction is not None:
            self.set_correction(correction,include_points=True)


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

    def find_points(self):

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        self.remove_all_points()

        ref_points,new_points = movement.find_pointpairs(self.ref_image,self.new_image)

        self.app.restoreOverrideCursor()

        if ref_points.shape[0] > 0:

            for npair in range(ref_points.shape[0]):
                self.new_point('ref',ref_points[npair,:])
                self.new_point('new',new_points[npair,:])

        else:
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(dialog.Ok)
            dialog.setTextFormat(qt.Qt.RichText)
            dialog.setWindowTitle('Could not auto-detect points')
            dialog.setText('Could not auto-detect corresponding points in these two images.')
            dialog.setInformativeText('You will have to identify points manually')
            dialog.setIcon(qt.QMessageBox.Information)
            dialog.exec()




    def remove_current_point(self):

        if self.selected_pointpair is not None:
            if self.point_pairings[self.selected_pointpair][1] is not None:
                self.interactor_new.remove_active_cursor(self.point_pairings[self.selected_pointpair][1])
            if self.point_pairings[self.selected_pointpair][0] is not None:
                self.interactor_ref.remove_active_cursor(self.point_pairings[self.selected_pointpair][0])
            self.point_pairings.remove(self.point_pairings[self.selected_pointpair])

            if len(self.point_pairings) > 0:
                self.selected_pointpair = len(self.point_pairings) - 1
                self.interactor_ref.set_cursor_focus(self.point_pairings[self.selected_pointpair][0])
                self.interactor_new.set_cursor_focus(self.point_pairings[self.selected_pointpair][1])
            else:
                self.selected_pointpair = None

            if len(self.point_pairings) < 3:
                self.update_homography_button.setEnabled(False)

            self.set_correction(None)


    def remove_all_points(self):
        while self.selected_pointpair is not None:
            self.remove_current_point()

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
                self.ref_image_enhanced = enhance_image(self.ref_image)
                self.new_image_enhanced = enhance_image(self.new_image)
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

        m = np.matrix(cv2.cv2.estimateAffinePartial2D(new_points, ref_points)[0])

        if m[0,0] is None:
            self.transform = None
            self.fitted_points_checkbox.setEnabled(False)
            self.transform_params.hide()
            dialog = qt.QMessageBox(self)
            dialog.setStandardButtons(dialog.Ok)
            dialog.setTextFormat(qt.Qt.RichText)
            dialog.setWindowTitle('Could not determine transform')
            dialog.setText('Could not determine the image transformation. Try adjusting the points and trying again.')
            dialog.setIcon(qt.QMessageBox.Warning)
            dialog.exec()
            return
        else:
            correction = movement.MovementCorrection(m,self.ref_image.shape[:2],ref_points,new_points,'Created using GUI tool by {:s} on {:s} at {:s}'.format(misc.username,misc.hostname,misc.get_formatted_time()))
            self.set_correction(correction)




    def set_correction(self,correction,include_points=False):

        if correction is None:

            if include_points:
                self.remove_all_points()

            self.transform = None
            self.transform_params.hide()
            self.fitted_points_checkbox.setChecked(False)
            self.fitted_points_checkbox.setEnabled(False)
            self.clear_transform_button.setEnabled(False)
            self.save_button.setEnabled(False)


        else:
            if include_points:
                self.remove_all_points()
                for i in range(correction.ref_points.shape[0]):
                    self.new_point('ref',correction.ref_points[i,:])
                    self.new_point('new',correction.moved_points[i, :])

            self.transform = correction

            ddscore = self.transform.get_ddscore(self.ref_image,self.new_image)

            desc = ' ( {:.1f} , {:.1f} ) pixels<br>'.format(*self.transform.translation)
            desc = desc + ' {:.2f}\u00b0<br>'.format(self.transform.rotation)
            desc = desc + ' {:.3f}<br>'.format(self.transform.scale)
            if ddscore >= 0:
                desc = desc + ' <font color="#006400">{:.2f}</font>'.format(ddscore)
            else:
                desc = desc + ' <font color="#ff0000">{:.2f}</font>'.format(ddscore)

            self.transform_params.setText(desc)
            self.transform_params.show()

            self.fitted_points_checkbox.setEnabled(True)
            self.fitted_points_checkbox.setChecked(True)
            self.clear_transform_button.setEnabled(True)
            self.save_button.setEnabled(True)


    def count_pointpairs(self):

        if self.selected_pointpair is not None and None in self.point_pairings[self.selected_pointpair]:
            return len(self.point_pairings) - 1
        else:
            return len(self.point_pairings)


    def toggle_fitted_points(self,onoff):

        for cursor_id in self.overlay_cursors[0]:
            self.interactor_ref.remove_passive_cursor(cursor_id)
        for cursor_id in self.overlay_cursors[1]:
            self.interactor_new.remove_passive_cursor(cursor_id)
        self.overlay_cursors = [[],[]]

        if onoff:
            for pp in self.point_pairings:
                if pp[1] is not None:
                    oldpt = self.interactor_new.get_cursor_coords(pp[1])[0]
                    fitted_pt = self.transform.moved_to_ref_coords(oldpt[0],oldpt[1])
                    self.overlay_cursors[0].append(self.interactor_ref.add_passive_cursor([fitted_pt[0],fitted_pt[1]]))
                    self.overlay_cursors[1].append(self.interactor_new.add_passive_cursor([fitted_pt[0], fitted_pt[1]]))


    def toggle_overlay(self,onoff):

        if onoff:
            if self.image_enhancement_checkbox.isChecked():
                orig_im = self.new_image_enhanced
            else:
                orig_im = self.new_image

            if self.transform is not None:

                overlay_image =self.transform.warp_moved_to_ref(orig_im)[0]

            else:

                overlay_image = orig_im

            self.interactor_ref.set_overlay_image(overlay_image)
            self.interactor_new.set_overlay_image(overlay_image)

        else:
            self.interactor_ref.set_overlay_image(None)
            self.interactor_new.set_overlay_image(None)

        self.refresh()


    def save(self):

        filename_filter = self.config.filename_filters['movement']
        fext = filename_filter.split('(*')[1].split(')')[0]

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(filedialog.AcceptSave)

        try:
            filedialog.setDirectory(self.config.file_dirs['movement'])
        except KeyError:
            filedialog.setDirectory(os.path.expanduser('~'))


        filedialog.setFileMode(filedialog.AnyFile)

        filedialog.setWindowTitle('Save As...')
        filedialog.setNameFilter(filename_filter)
        filedialog.exec()

        if filedialog.result() == 1:
            selected_path = str(filedialog.selectedFiles()[0])
            self.config.file_dirs['movement'] = os.path.split(selected_path)[0]
            self.config.save()
            if not selected_path.endswith(fext):
                selected_path = selected_path + fext

            self.transform.save(selected_path)


    def open(self):

        filename_filter = self.config.filename_filters['movement']

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(filedialog.AcceptOpen)

        try:
            filedialog.setDirectory(self.config.file_dirs['movement'])
        except KeyError:
            filedialog.setDirectory(os.path.expanduser('~'))

        filedialog.setFileMode(filedialog.ExistingFile)

        filedialog.setWindowTitle('Open...')
        filedialog.setNameFilter(filename_filter)

        filedialog.exec()


        if filedialog.result() == 1:
            selected_file = filedialog.selectedFiles()[0]
            self.config.file_dirs['movement'] = os.path.split(selected_file)[0]
            self.config.save()

            try:
                correction = movement.MovementCorrection.load(selected_file)
            except Exception as e:
                return

            self.set_correction(correction,include_points=True)