from .core import *
import webbrowser
from multiprocessing import Process
from .. import __version__

from .viewer import ViewerWindow
from .fitting_calib import FittingCalibrationWindow
from .virtual_calib import VirtualCalibrationWindow

# Class for the window
class LauncherWindow(qt.QDialog):
 
    def __init__(self, app, parent = None):


        # GUI initialisation
        qt.QDialog.__init__(self, parent)
        # Load the Qt designer file, assumed to be in the same directory as this python file and named gui.ui.
        qt.uic.loadUi(os.path.join(guipath,'launcher.ui'), self)
        
        self.setWindowIcon(qt.QIcon(os.path.join(guipath,'icon.png')))
        self.setWindowTitle('Calcam  v{:s}'.format(__version__))
        self.layout().setSizeConstraint(qt.QLayout.SetFixedSize)

        self.app = app
        
        immap = qt.QPixmap(os.path.join(guipath,'logo.png'))
        self.logolabel.setPixmap(immap)

        # Callbacks for GUI elements: connect the buttons to the functions we want to run
        self.calcam_button.clicked.connect(self.launch_calcam)
        #self.alignment_calib_button.clicked.connect(self.launch_alignment_calib)
        self.cad_viewer_button.clicked.connect(self.launch_viewer)
        self.view_designer_button.clicked.connect(self.launch_virtual_calib_edit)
        self.userguide_button.clicked.connect(self.open_manual)
        #self.image_analysis_button.clicked.connect(self.launch_image_analysis)

        # Open the window!
        self.show()

    def launch_calcam(self):
        Process(target=open_gui,args=[FittingCalibrationWindow]).start()

    def launch_viewer(self):
        Process(target=open_gui,args=[ViewerWindow]).start()

    def launch_virtual_calib_edit(self):
        Process(target=open_gui,args=[VirtualCalibrationWindow]).start()

    def launch_alignment_calib(self):
        Process(target=start_alignment_calib).start()

    def open_manual(self):
        webbrowser.open('https://euratom-software.github.io/calcam/')

    def launch_image_analysis(self):
        Process(target=start_image_analysis).start()