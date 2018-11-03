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

import os
import sys

from .core import guipath
from ..config import CalcamConfig
from . import qt_wrapper as qt
from .launcher import launch

class Settings(qt.QMainWindow):


    def __init__(self, app, parent = None):

        # GUI initialisation
        qt.QMainWindow.__init__(self, parent)
        qt.uic.loadUi(os.path.join(guipath,'settings.ui'), self)

        self.setWindowIcon(qt.QIcon(os.path.join(guipath,'icon.png')))

        self.cad_path_list.itemChanged.connect(self.change_cad_paths)
        self.add_modelpath_button.clicked.connect(self.add_cadpath)
        self.cad_path_list.itemSelectionChanged.connect(self.update_cadpath_selection)
        self.imsource_path_list.itemChanged.connect(self.change_imsource_paths)
        self.imsource_path_list.itemSelectionChanged.connect(self.update_imsource_selection)
        self.model_list.itemSelectionChanged.connect(self.update_model_selection)
        self.add_impath_button.clicked.connect(self.add_imsource_path)
        self.rem_modelpath_button.clicked.connect(self.remove_cad_path)
        self.rem_impath_button.clicked.connect(self.remove_imsource_path)
        self.edit_model_button.clicked.connect(self.open_model_edit)
        self.new_model_button.clicked.connect(self.open_model_edit)

        self.refresh_timer = qt.QTimer()
        self.refresh_timer.setInterval(5000)
        self.refresh_timer.timeout.connect(self.update)
        self.refresh_timer.start()


        self.app = app

        self.update()

        self.show()


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


    def update_imsource_selection(self):
        if len(self.imsource_path_list.selectedItems()) > 0:
            self.rem_impath_button.setEnabled(True)       
        else:
            self.rem_impath_button.setEnabled(False)


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
                listitem.setForeground(qt.Qt.red)
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
                listitem.setForeground(qt.Qt.red)
                listitem.setToolTip('Path does not exist or cannot be accessed.')
                
            self.imsource_path_list.addItem(listitem)
            if path == to_select:
                listitem.setSelected(True)  

        try:
            to_select = str(self.imsource_list.selectedItems()[0].text())
        except IndexError:
            to_select = None   

        self.imsource_list.clear()

        for imsource in sorted(self.config.get_image_sources(meta_only=True)):
            listitem = qt.QListWidgetItem(imsource[0])
            if imsource[1]:
                listitem.setForeground(qt.Qt.red)
                listitem.setToolTip(imsource[1])

            self.imsource_list.addItem(listitem)
            if imsource[0] == to_select:
                listitem.setSelected(True)


    def browse_for_folder(self):

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(0)
        filedialog.setFileMode(2)
        filedialog.setWindowTitle('Select Directory')
        filedialog.exec_()
        if filedialog.result() == 1:
            path = str(filedialog.selectedFiles()[0])
            return path.replace('/',os.path.sep)
        else:
            return None