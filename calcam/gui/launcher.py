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
import multiprocessing
import sys
import os
import json
import time
try:
  import urllib.request as request
except:
  import urllib2 as request
  

from .. import __version__
from .core import qt, guipath


def launch(args):
    subprocess.Popen([sys.executable,os.path.join( os.path.split(__file__)[0],'launch_script.py' )] + args,stdin=None)


# Generate a rich text string prompting the user to go to the github page
# if the latest release of calcam has a higher version number than the current one.
# Because this isn't an important feature, if anything goes wrong in this function we
# just act as if there is no update available.
def update_prompt_string(queue):

  try:

    # Get full version strings of current version and latest release tag from GitHub
    response = request.urlopen('https://api.github.com/repos/euratom-software/calcam/tags',timeout=0.5)
    data = json.loads(response.read().decode('utf-8'))
    latest_version_str = data[0]['name'].replace('v','')
    latest_ver_base10 = sum([(10**(i*3))*int(num) for i,num in enumerate(reversed(latest_version_str.split('.')))])

    # Strip any pre-release or dev identifier from the current version nuber.
    this_ver_base10 = 0
    for i,substring in enumerate(reversed(__version__.split('.'))):
        number = None
        while number is None and len(substring) > 0:
            try:
                number = int(substring)
            except ValueError:
                substring = substring[:-1]
        if number is not None:
            this_ver_base10 = this_ver_base10 + (10**(i*3))*number

    # If a newer version is available, return a string to tell the user that.
    updatestring = None
    if latest_ver_base10 > this_ver_base10:
        updatestring = '<b>Calcam {:} now available! Find out what changed <a href=https://github.com/euratom-software/calcam/blob/{:s}/CHANGELOG.txt>here</a>; update with pip or download the <a href={:s}>source zip</a>; or read more about updating <a href=https://euratom-software.github.io/calcam/html/intro_install_setup.html#updating>here</a>!</b>'.format(data[0]['name'],data[0]['commit']['sha'],data[0]['zipball_url'])

    queue.put(updatestring)

  except Exception as e:
      print(e)
      queue.put(None)


# Class for the window
class Launcher(qt.QDialog):
 
    def __init__(self, app, parent = None):

        # GUI initialisation
        qt.QDialog.__init__(self, parent, qt.Qt.WindowTitleHint | qt.Qt.WindowCloseButtonHint)
        # Load the Qt designer file, assumed to be in the same directory as this python file and named gui.ui.
        qt.uic.loadUi(os.path.join(guipath,'qt_designer_files','launcher.ui'), self)

        self.setWindowIcon(qt.QIcon(os.path.join(guipath,'icons','calcam.png')))
        self.setWindowTitle('Calcam  v{:s}'.format(__version__))
        if qt.qt_ver < 6:
            self.layout().setSizeConstraint(qt.QLayout.SetFixedSize)
        else:
            self.layout().setSizeConstraint(qt.QLayout.SizeConstraint.SetFixedSize)

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
        # Annoyingly, urllib2 can cause Python to segfault when using SSL under some
        # quite specific circumstances, so to be robust against that I resort 
        # to the ridiculous over-complication doing this in a separate process. Grumble.
        q = multiprocessing.Queue()
        checkprocess = multiprocessing.Process(target=update_prompt_string,args=(q,))
        checkprocess.start()
        while checkprocess.is_alive() and q.empty():
            time.sleep(0.05)
        
        if not q.empty():
            update_string = q.get()
        else:
            update_string = None

        if update_string is None:
          self.update_label.hide()
        else:
          self.update_label.setText(update_string)

        # Open the window!
        self.show()