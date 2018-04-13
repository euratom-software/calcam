from .core import open_gui
from .viewer import ViewerWindow
from .launcher import LauncherWindow 

def start_gui():
	open_gui(LauncherWindow)