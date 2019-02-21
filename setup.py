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
import sysconfig



if 'install' in sys.argv or 'develop' in sys.argv:

    # Organise dependencies
    dependencies = ['scipy','matplotlib']

    if sys.version_info[0] > 2:
        # Sensible codepath for Python 3

        dependencies.append('vtk','opencv-python')

        # Since we can use PyQt 4 or 5, check if the user already
        # has one of these. If so, use the one they already have
        # so as to not unnecesserily clutter their environment.
        try:
            from PyQt5 import QtCore
            dependencies.append('PyQt5')
        except:
            try:
                from PyQt4 import QtCore
                dependencies.append('PyQt4')
            except:
                dependencies.append('PyQt5')


    else:
        # Still supporting Python 2! (whhhyyyy???)
        # In this case we can't rely on getting VTK or PyQt from PyPi.
        # So we just manually check them and warn the user if they're missing.
        # Also for some reason setuptools doesn't find OpenCV but pip does, so
        # we launch a pip install subprocess to do OpenCV.
        try:
            import cv2
        except:
            try:
                import subprocess
                subprocess.call([sys.executable, '-m', 'pip', 'install','opencv-python'])
            except:
                print('Arrgh!')
                exit()
        try:
            import vtk
            vtk = True
        except:
            vtk = False

        try:
            from PyQt5 import QtCore
            pyqt = True
        except:
            try:
                from PyQt4 import QtCore
                pyqt = True
            except:
                pyqt = False


        if not pyqt or not vtk or vtk < 6:

            badlist = ''
            if not pyqt:
                badlist = badlist + 'PyQt4 or PyQt5\n'
            if not vtk:
                badlist = badlist + 'VTK (v6.0+)\n'
            elif vtk < 6:
                badlist = badlist + 'VTK v6.0+ : you have v{:s}'.format(vtk.vtkVersion.GetVTKVersion())

            msg = '\n\nWARNING: One or more important dependencies do not appear to be satisfied.\n' \
                  'Installation will continue and at least some of the Calcam API for working \n' \
                  'with calibration results should work, however the calcam GUI module and some \n' \
                  'API features will not work until you manually install the following python modules:\n\n' \
                  + badlist + '\nPress any key to continue installation...'

            raw_input(msg)



# Actually do the requested setup actions
setup(
      name='Calcam',
      version='2.2rc1',
      url='https://euratom-software.github.io/calcam/',
      license='European Union Public License 1.1',
      author='Scott Silburn et.al.',
      install_requires=dependencies,
      packages=find_packages(),
      package_data={'calcam':['gui/icons/*','gui/qt_designer_files/*.ui','gui/logo.png','builtin_image_sources/*.py']},
      entry_points={ 'gui_scripts': [ 'calcam = calcam:start_gui'] },
      zip_safe=False
      )


# Post-install stuff: tell the user where the application executable has been created.
if 'install' in sys.argv or 'develop' in sys.argv:

    if sys.platform == 'win32':
        script_name = 'calcam.exe'
    else:
        script_name = 'calcam'

    script_dir = sysconfig.get_path('scripts')
    print('\n****\nPath to Calcam GUI launcher: \n"{:s}"\n****\n'.format(os.path.join(script_dir,script_name)))