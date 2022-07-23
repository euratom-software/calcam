'''
* Copyright 2015-2022 European Atomic Energy Community (EURATOM)
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
import sys
import cv2

from .. import __version__, vtk
from .core import guipath
from ..misc import DodgyDict, open_file
from ..config import CalcamConfig
from . import qt_wrapper as qt
from .launcher import launch

if qt.qt_ver < 6:
    red = qt.Qt.red
else:
    red = qt.QColor('red')

class Settings(qt.QMainWindow):


    def __init__(self, app, parent = None):

        # GUI initialisation
        qt.QMainWindow.__init__(self, parent)
        qt.uic.loadUi(os.path.join(guipath,'qt_designer_files','settings.ui'), self)

        self.setWindowIcon(qt.QIcon(os.path.join(guipath,'icons','calcam.png')))

        self.cad_path_list.itemChanged.connect(self.change_cad_paths)
        self.add_modelpath_button.clicked.connect(self.add_cadpath)
        self.cad_path_list.itemSelectionChanged.connect(self.update_cadpath_selection)
        self.imsource_path_list.itemChanged.connect(self.change_imsource_paths)
        self.imsource_path_list.itemSelectionChanged.connect(self.update_imsource_path_selection)
        self.imsource_list.itemSelectionChanged.connect(self.update_imsource_selection)
        self.model_list.itemSelectionChanged.connect(self.update_model_selection)
        self.add_impath_button.clicked.connect(self.add_imsource_path)
        self.rem_modelpath_button.clicked.connect(self.remove_cad_path)
        self.rem_impath_button.clicked.connect(self.remove_imsource_path)
        self.edit_model_button.clicked.connect(self.open_model_edit)
        self.new_model_button.clicked.connect(self.open_model_edit)
        self.edit_imsource_button.clicked.connect(self.edit_imsource)

        self.refresh_timer = qt.QTimer()
        self.refresh_timer.setInterval(7000)
        self.refresh_timer.timeout.connect(self.update)
        self.refresh_timer.start()

        if vtk is not None:
            vtk_str = vtk.vtkVersion().GetVTKVersion()
        else:
            vtk_str = 'NO VTK!'

        env_str_left = '<pre>Platform:       {:s}\nPython version: {:s}\nCalcam version: {:s}</pre>'.format(sys.platform,'.'.join([str(num) for num in sys.version_info[:3]]),__version__)
        env_str_right = '<pre>VTK version:    {:s}\nOpenCV version: {:s}\nPyQt version:   {:s}\nQtOpenGL:       {:s}</pre>'.format(vtk_str,cv2.__version__,qt.QT_VERSION_STR,'OK' if qt.qt_opengl else 'UNAVAILABLE')

        self.env_info_left.setText(env_str_left)
        self.env_info_right.setText(env_str_right)

        self.app = app

        self.update()

        self.show()


    def edit_imsource(self):
        path = self.imsource_paths[self.imsource_list.selectedItems()[0]]
        open_file(path)


    def remove_cad_path(self):
        path = str(self.cad_path_list.selectedItems()[0].text())
        self.config.cad_def_paths.remove(path)
        self.config.save()
        self.update()


    def update_model_selection(self):
        self.edit_model_button.setEnabled(True)


    def remove_imsource_path(self):

        path =  str(self.imsource_path_list.selectedItems()[0].text())
        self.config.image_source_paths.remove(path)
        self.config.save()
        self.update()


    def change_cad_paths(self,item):

        index = self.cad_path_list.row(item)
        self.config.cad_def_paths[index] = str(item.text())
        self.config.save()
        self.update()

    def update_cadpath_selection(self):
        if len(self.cad_path_list.selectedItems()) > 0:
            self.rem_modelpath_button.setEnabled(True)
        else:
            self.rem_modelpath_button.setEnabled(False)

    def change_imsource_paths(self,item):
        index = self.imsource_path_list.row(item)
        self.config.image_source_paths[index] = str(item.text())
        self.config.save()
        self.update()


    def add_cadpath(self):

        path = self.browse_for_folder()
        if path is not None:
            self.config.cad_def_paths.append(path)
            self.config.save()
            self.update()

    def open_model_edit(self):

        if self.sender() is self.edit_model_button:
            model_name = str(self.model_list.selectedItems()[0].text())
            model_info =  self.config.get_cadmodels()
            for model in model_info.keys():
                if model_name == model:
                    launch(['--cad_edit',model_info[model][0]])
                    return
        else:
            launch(['--cad_edit'])

    def add_imsource_path(self):

        path = self.browse_for_folder()
        if path is not None:
            self.config.image_source_paths.append(path)
            self.config.save()
            self.update()


    def update_imsource_path_selection(self):
        if len(self.imsource_path_list.selectedItems()) > 0:
            self.rem_impath_button.setEnabled(True)       
        else:
            self.rem_impath_button.setEnabled(False)

    def update_imsource_selection(self):

        if len(self.imsource_list.selectedItems()) > 0:
            if self.imsource_paths[self.imsource_list.selectedItems()[0]] is not None:
                self.edit_imsource_button.setEnabled(True)
            else:
                self.edit_imsource_button.setEnabled(False)
        else:
            self.edit_imsource_button.setEnabled(False)

    def update(self):

        self.config = CalcamConfig()

        # Populate lists
        try:
            to_select = str(self.cad_path_list.selectedItems()[0].text())
        except IndexError:
            to_select = None

        self.cad_path_list.clear()

        for path in sorted(self.config.cad_def_paths):

            listitem = qt.QListWidgetItem(path)
            listitem.setFlags(listitem.flags() | qt.Qt.ItemIsEditable | qt.Qt.ItemIsSelectable)

            if not os.path.isdir(path):
                listitem.setForeground(red)
                listitem.setToolTip('Path does not exist or cannot be accessed.')
            
            self.cad_path_list.addItem(listitem)
            if path == to_select:
                listitem.setSelected(True)

        try:
            to_select = str(self.model_list.selectedItems()[0].text())
        except IndexError:
            to_select = None
        self.model_list.clear()
        self.edit_model_button.setEnabled(False)
        for model in sorted(self.config.get_cadmodels().keys()):
            listitem = qt.QListWidgetItem(model)
            listitem.setToolTip(self.config.get_cadmodels()[model][0])
            self.model_list.addItem(listitem)
            if model == to_select:
                listitem.setSelected(True)

        try:
            to_select = str(self.imsource_path_list.selectedItems()[0].text())
        except IndexError:
            to_select = None

        self.imsource_path_list.clear()

        for path in sorted(self.config.image_source_paths):

            listitem = qt.QListWidgetItem(path)
            listitem.setFlags(listitem.flags() | qt.Qt.ItemIsEditable | qt.Qt.ItemIsSelectable)

            if not os.path.isdir(path):
                listitem.setForeground(red)
                listitem.setToolTip('Path does not exist or cannot be accessed.')
                
            self.imsource_path_list.addItem(listitem)
            if path == to_select:
                listitem.setSelected(True)  

        try:
            to_select = str(self.imsource_list.selectedItems()[0].text())
        except IndexError:
            to_select = None   

        self.imsource_list.clear()
        self.imsource_paths = DodgyDict()

        for imsource in sorted(self.config.get_image_sources(meta_only=True)):
            listitem = qt.QListWidgetItem(imsource[0])
            self.imsource_paths[listitem] = imsource[1]
            if imsource[2]:
                listitem.setForeground(red)
                listitem.setToolTip(imsource[2])
            elif imsource[1]:
                listitem.setToolTip(imsource[1])

            self.imsource_list.addItem(listitem)
            if imsource[0] == to_select:
                listitem.setSelected(True)


    def browse_for_folder(self):

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(filedialog.AcceptOpen)
        filedialog.setFileMode(filedialog.Directory)
        filedialog.setWindowTitle('Select Directory')
        filedialog.exec()
        if filedialog.result() == 1:
            path = str(filedialog.selectedFiles()[0])
            return path.replace('/',os.path.sep)
        else:
            return None