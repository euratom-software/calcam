from .core import *
import webbrowser
import subprocess
from .. import __version__
import sys
import os

def launch(args):
    subprocess.Popen([sys.executable,os.path.join( os.path.split(__file__)[0],'launch_script.py' )] + args,stdin=None, stdout=open(os.devnull,'wb'), stderr=open(os.devnull,'wb'))

# Class for the window
class LauncherWindow(qt.QDialog):
 
    def __init__(self, app, parent = None):


        # GUI initialisation
        qt.QDialog.__init__(self, parent, qt.Qt.WindowTitleHint | qt.Qt.WindowCloseButtonHint)
        # Load the Qt designer file, assumed to be in the same directory as this python file and named gui.ui.
        qt.uic.loadUi(os.path.join(guipath,'launcher.ui'), self)
        
        self.setWindowIcon(qt.QIcon(os.path.join(guipath,'icon.png')))
        self.setWindowTitle('Calcam  v{:s}'.format(__version__))
        self.layout().setSizeConstraint(qt.QLayout.SetFixedSize)

        self.app = app
        
        immap = qt.QPixmap(os.path.join(guipath,'logo.png'))
        self.logolabel.setPixmap(immap)

        # Callbacks for GUI elements: connect the buttons to the functions we want to run
        self.calcam_button.clicked.connect(lambda : launch(['--fitting_calib']))
        self.alignment_calib_button.clicked.connect(lambda : launch(['--alignment_calib']))
        self.cad_viewer_button.clicked.connect(lambda : launch(['--viewer']))
        self.view_designer_button.clicked.connect(lambda : launch(['--virtual_calib']))
        self.userguide_button.clicked.connect(self.open_manual)
        self.image_analysis_button.clicked.connect(lambda : launch(['--image_analyser']))
        self.settings_button.clicked.connect(lambda : launch(['--settings']))

        # Open the window!
        self.show()



    def open_manual(self):
        webbrowser.open('https://euratom-software.github.io/calcam/')