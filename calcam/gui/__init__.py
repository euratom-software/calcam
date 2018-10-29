from .viewer import ViewerWindow
from .launcher import LauncherWindow
from .virtual_calib import VirtualCalibrationWindow
from .fitting_calib import FittingCalibrationWindow
from .alignment_calib import AlignmentCalibWindow
from .image_analysis import ImageAnalyserWindow
from .settings import SettingsWindow
from .cad_edit import CADEditorWindow

from . import qt_wrapper as qt

def open_window(window_class,*args):
    app = qt.QApplication([])
    win = window_class(app,None,*args)
    if qt.QDialog in window_class.__bases__:
        return win.exec_()
    else:
        return app.exec_()

def start_gui():
    open_window(LauncherWindow)