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

import sys
import os
import subprocess

from setuptools import setup,find_packages, Distribution 
from setuptools.command.install import install
from setuptools.command.develop import develop



def pip_install(pkg_name):
    '''
    A small utility function to call pip externally to install a given package.
    Sadly, this seems to be needed because setuptools.setup cannot be trusted to 
    locate some dependencies, or their correct versions, and pip works more reliably.
    '''    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',pkg_name])
        return True
    except:
        return False


if 'install' in sys.argv or 'develop' in sys.argv:

    # Organise dependencies, in a sadly manual way (I can't find a suitably
    # portable way to do it more automatically)

    # Hard dependencies
    for prettyname,pkgname,importname in [ ('SciPy','scipy','scipy') ,('MatPlobLib','matplotlib','matplotlib'),('OpenCV (a.k.a opencv-python a.k.a cv2)','opencv-python','cv2')]:
        try:
            __import__(importname)
        except:
            if not pip_install(pkgname):
                print('Could not install hard dependency {:s}. Please install it before installing Calcam.'.format(prettyname))
                exit()
    
    # Softer dependencies
    warning_list = []
    try:
        import vtk
        if vtk.vtkVersion.GetVTKMajorVersion() < 6:
            warning_list.append('VTK 6.0+ (you have {:}'.format(vtk.vtkVersion.GetVTKVersion()))
    except:
        if not pip_install('vtk'):
            warning_list.append('VTK 6.0+')
            
    try:
        from PyQt5 import QtCore
    except:
        try:
            from PyQt4 import QtCore
        except:    
            if not pip_install('PyQt5'):
                warning_list.append('PyQt4 or PyQt5')
   

    if len(warning_list) > 0:

        msg = '\n\nWARNING: One or more important dependencies do not appear to be satisfied.\n' \
              'Installation will continue and at least some of the Calcam API for working \n' \
              'with calibration results should work, however the calcam GUI module and some \n' \
              'API features will not work until you manually install the following python modules:\n\n' \
              + '\n'.join(warning_list) + '\n\nPress any key to continue installation...'

        raw_input(msg)



# Actually do the requested setup actions
s = setup(
          name='Calcam',
          version='2.2rc1',
          url='https://euratom-software.github.io/calcam/',
          license='European Union Public License 1.1',
          author='Scott Silburn et.al.',
          packages=find_packages(),
          package_data={'calcam':['gui/icons/*','gui/qt_designer_files/*.ui','gui/logo.png','builtin_image_sources/*.py']},
          entry_points={ 'gui_scripts': [ 'calcam = calcam:start_gui'] },
          zip_safe=False
         )


# Post-install stuff: reassure the user that things went well (assuming they did)
# and give some useful info.
if 'install' in sys.argv or 'develop' in sys.argv:

    if 'install' in sys.argv:
        script_dir = s.command_obj['install'].install_scripts
    elif 'develop' in sys.argv:
        script_dir = s.command_obj['develop'].script_dir

    try:
        env_path = os.environ['PATH']
    except KeyError:
        env_path = None
    
    extra_msg = '\nIt can be imported within python with "import calcam"'
    if env_path is not None:
        if script_dir not in env_path:
            extra_msg = extra_msg + '\n\nNOTE: The path containing the Calcam GUI launch script:\n\n{:s}\n\nis not in your PATH environment variable; consider\nadding it to enable launching the Calcam GUI directly!'.format(script_dir)
        else:
            extra_msg = extra_msg + '\nThe Calcam GUI can be started by typing "calcam" at a terminal.'
     
    else:
        extra_msg = extra_msg + '\nLocation of Calcam GUI launcher:\n\n{:s}'.format(script_dir)
    
    
                                      
    print('\n*****************************\nCalcam installation complete.{:s}\n*****************************\n'.format(extra_msg))