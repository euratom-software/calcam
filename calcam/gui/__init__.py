from .core import open_gui
from .viewer import ViewerWindow
from .launcher import LauncherWindow
from .virtual_calib import VirtualCalibrationWindow
from .fitting_calib import FittingCalibrationWindow

def start_gui():
	open_gui(LauncherWindow)