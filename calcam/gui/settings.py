import os

from .core import guipath
from ..config import CalcamConfig
from . import qt_wrapper as qt

class SettingsWindow(qt.QMainWindow):


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
        self.add_impath_button.clicked.connect(self.add_imsource_path)
        self.rem_modelpath_button.clicked.connect(self.remove_cad_path)
        self.rem_impath_button.clicked.connect(self.remove_imsource_path)

        self.app = app

        self.update()

        self.show()


    def remove_cad_path(self):

        path =  str(self.cad_path_list.selectedItems()[0].text())
        self.config.cad_def_paths.remove(path)
        self.config.save()
        self.update()


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

        self.cad_path_list.clear()
        self.imsource_path_list.clear()
        self.model_list.clear()
        self.imsource_list.clear()

        self.config = CalcamConfig()

        # Populate lists
        for path in self.config.cad_def_paths:

            listitem = qt.QListWidgetItem(path)
            listitem.setFlags(listitem.flags() | qt.Qt.ItemIsEditable | qt.Qt.ItemIsSelectable)

            if not os.path.isdir(path):
                listitem.setForeground(qt.Qt.red)
                listitem.setToolTip('Path does not exist or cannot be accessed.')
            
            self.cad_path_list.addItem(listitem)

        for model in self.config.get_cadmodels().keys():
            listitem = qt.QListWidgetItem(model)
            self.model_list.addItem(listitem)

        for path in self.config.image_source_paths:

            listitem = qt.QListWidgetItem(path)
            listitem.setFlags(listitem.flags() | qt.Qt.ItemIsEditable | qt.Qt.ItemIsSelectable)

            if not os.path.isdir(path):
                listitem.setForeground(qt.Qt.red)
                listitem.setToolTip('Path does not exist or cannot be accessed.')
                

            self.imsource_path_list.addItem(listitem)

        for imsource in self.config.get_imsource_list():
            listitem = qt.QListWidgetItem(imsource[0])
            if not imsource[1]:
                listitem.setForeground(qt.Qt.red)
                listitem.setToolTip(imsource[2])

            self.imsource_list.addItem(listitem)


    def browse_for_folder(self):

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(0)
        filedialog.setFileMode(2)
        filedialog.setWindowTitle('Select Directory')
        filedialog.exec_()
        if filedialog.result() == 1:
            path = filedialog.selectedFiles()[0]
            return path.replace('/',os.path.sep)
        else:
            return None