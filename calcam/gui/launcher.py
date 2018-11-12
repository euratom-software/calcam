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
import subprocess

import sys
import os
try:
  import urllib.request as request
except:
  import urllib2 as request
import json

from .. import __version__
from .core import qt, guipath


def launch(args):
    subprocess.Popen([sys.executable,os.path.join( os.path.split(__file__)[0],'launch_script.py' )] + args,stdin=None, stdout=open(os.devnull,'wb'), stderr=open(os.devnull,'wb'))


# Generate a rich text string prompting the user to go to the released webpage
# if the latest release of calcam has a higher version number than the current one.
# Because this isn't an important feature, if anything goes wrong in this function we
# just act as if there is no update available.
def update_prompt_string():

  try:

    response = request.urlopen('https://api.github.com/repos/euratom-software/calcam/releases',timeout=0.5)
    data = json.loads(response.read().decode('utf-8'))
    latest_release_string = data[0]['tag_name']

    latest_release_split = latest_release_string.replace('v','').split('.')[::-1]
    current_release_split = __version__.split('.')[::-1]

    current_ver = 0
    latest_ver = 0

    for i in range(3):
      try:
          current_ver = current_ver + int(current_release_split[i]) * 100**i
      except:
          pass
      try:
          latest_ver = latest_ver + int(latest_release_split[i]) * 100**i
      except:
          pass

    if latest_ver > current_ver:
        return '<b>A newer version of Calcam ({:s}) is available; see the <a href=https://github.com/euratom-software/calcam/releases>releases page</a> for details.</b>'.format(latest_release_string)
    else:
        return None
  except:
      return None


# Class for the window
class Launcher(qt.QDialog):
 
    def __init__(self, app, parent = None):

        # GUI initialisation
        qt.QDialog.__init__(self, parent, qt.Qt.WindowTitleHint | qt.Qt.WindowCloseButtonHint)
        # Load the Qt designer file, assumed to be in the same directory as this python file and named gui.ui.
        qt.uic.loadUi(os.path.join(guipath,'qt_designer_files','launcher.ui'), self)

        self.setWindowIcon(qt.QIcon(os.path.join(guipath,'icons','calcam.png')))
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
        self.image_analysis_button.clicked.connect(lambda : launch(['--image_analyser']))
        self.settings_button.clicked.connect(lambda : launch(['--settings']))

        # Check if there is a newer Calcam release than the current one.
        update_string = update_prompt_string()
        if update_string is None:
          self.update_label.hide()
        else:
          self.update_label.setText(update_string)

        # Open the window!
        self.show()