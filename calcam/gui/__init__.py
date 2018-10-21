from .viewer import ViewerWindow
from .launcher import LauncherWindow
from .virtual_calib import VirtualCalibrationWindow
from .fitting_calib import FittingCalibrationWindow
from .alignment_calib import AlignmentCalibWindow
from .image_analysis import ImageAnalyserWindow
from .settings import SettingsWindow

from . import qt_wrapper as qt

def open_window(window_class):
    app = qt.QApplication([])
    win = window_class(app)
    if qt.QDialog in window_class.__bases__:
        return win.exec_()
    else:
        return app.exec_()

def start_gui():
    open_window(LauncherWindow)