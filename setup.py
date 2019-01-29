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

'''
Calcam Setup script.
'''

from setuptools import setup,find_packages
import sys
import os
import subprocess
import glob


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
       From http://code.activestate.com/recipes/577058/

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        try:
            choice = raw_input().lower()
        except NameError:
            choice = input().lower()

        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")



# Manually check dependencies which cannot be (or cannot 
# be guaranteed to be) serviced by PyPI / setuptools.
# ---------------------------------------------------------------------------------
if 'install' in sys.argv or 'develop' in sys.argv:
   pyqt_ = False
   vtk_ = False
   vtk_qt = False
   opencv = False

   try:
     from PyQt5 import QtCore
     pyqt_= 5
   except:
     from PyQt4 import QtCore
     pyqt_= 4
   finally:
     pass

   try:
     import vtk
     vtk_ = vtk.vtkVersion().GetVTKMajorVersion()
     if pyqt_:
         try:
            if pyqt_ == 4:
               from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
            else:
               from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
            vtk_qt = True
         except:
            pass
   except:
      pass


   try:
      import cv2
      opencv = True
   except:
      pass


   if not opencv:
      print('\nERROR: The OpenCV Python module ("cv2") does not seem to be installed or is not working properly. This module is required for Calcam to work; please install it first.\n')
      exit()

   if not pyqt_ or not vtk_ or not vtk_qt or not opencv or vtk_ < 6:

      badlist = ''
      if not pyqt_:
         badlist = badlist + 'PyQt4 or PyQt5\n'
      if not vtk_:
         badlist = badlist + 'VTK (v6.0+)\n'
      elif vtk_ < 6:
         badlist = badlist + 'VTK v6.0+ : you have v{:s}'.format(vtk.vtkVersion.GetVTKVersion())
      if not vtk_qt:
         badlist = badlist + 'PyQt extensions for VTK (VTK must be built with these enabled)\n'

      msg = """
            WARNING: One or more importtant dependencies do(es) not appear to be satisfied.
            A minimal feature set of Calcam may still work, however major parts of the module
            may not work at all or stably. The missing dependencies are:

            """ + badlist + '\nDo you still want to proceed with the installation?' 

      if not query_yes_no(msg,default='no'):
         exit()
# ---------------------------------------------------------------------------------


# Actually do the install
setup(
      name='Calcam',
      version='2.2.0.a1',
      url='https://euratom-software.github.io/calcam/',
      license='European Union Public License 1.1',
      author='Scott Silburn et.al.',
      install_requires=['numpy','scipy'],
      packages=find_packages(),
      package_data={'calcam':['gui/icons/*','gui/qt_designer_files/*.ui','gui/logo.png','builtin_image_sources/*.py']},
      entry_points={ 'gui_scripts': [ 'calcam = calcam:start_gui'] },
      zip_safe=False
      )


# Post-install stuff
# ---------------------------------------------------------------------------------
if len(sys.argv) > 1 and 'install' in sys.argv or 'develop' in sys.argv:

   # If on Windows, tell the user where the application executable has been created.
   if sys.platform == 'win32':
      import sysconfig
      import os
      script_dir = sysconfig.get_path('scripts')
      print('\n****\nPath to Calcam GUI launcher:\n"{:s}"\n****\n'.format(os.path.join(script_dir,'calcam.exe')))

   # If the user has Calcam 1.x files, prompt the user to convert their old files to Calcam 2 format.
   if os.path.isdir( os.path.join(os.path.expanduser('~'),'calcam') ):
      if query_yes_no('\nInstallaction complete.\nIt appears you have a data directory from Calcam 1.x and might be upgrading to Calcam 2.\nWould you like to run the Calcam 1 -> Calcam 2 file conversion tool now?'):
          subprocess.Popen([sys.executable,os.path.join( os.path.split(__file__)[0],'calcam1_file_converter','convert_files.py' )])
      print(' ')
