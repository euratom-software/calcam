'''
* Copyright 2015-2022 European Atomic Energy Community (EURATOM)
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
Calcam Setup script, uses pip and setuptools.

This uses the standard setuptools install_requires to specify the "easy" dependencies of SciPy and MatPlotLib,
but deals with OpenCV, VTK and PyQt using special custom logic and separate pip processes. This is because
(1) It seems to work across more platforms & environments, and (2) It allows implementing logic of fallback
dependencies, which as far as I can tell are not possible using the usual python packaging methods.

To make sure the logic in this script gets run at install even with pip, I have to deliberately break wheel building.
Frankly, this is all a horrible hack in terms of how python package installation is supposed to work, but after trying things
on several platforms, this actually makes it work smoothest on the widest range of platforms. So I'm prioritising creating an easy
experience for the user over good practise for Python packaging, and I can only hope that one day before pip stops supporting direct
installation altogether, the official tools provide a way to do this in a more "proper" way.
'''

import sys
import os
import subprocess

if 'bdist_wheel' in sys.argv:
    raise Exception("Calcam currently cannot be distributed as wheels due to having important install-time logic in setup.py.")

# Read version from the version file.
with open(os.path.join(os.path.split(os.path.abspath(__file__))[0],'calcam','__version__'),'r') as ver_file:
    version = ver_file.readline().rstrip()

# Read the readme.
with open(os.path.join(os.path.split(os.path.abspath(__file__))[0],'README.md'),'r',encoding='utf-8') as readme_file:
    readme = readme_file.read()

print('\n***************************************************************')
print('                Calcam v{:s} Setup Script'.format(version))
print('***************************************************************\n')


# Check if we have pip and setuptools and if not try to give the user a friendly message.
try:
    import pip
except Exception:
    print('\nThe calcam installation process requires pip, but module pip could not be imported. Please install pip and try again (see https://pip.pypa.io/en/stable/installing/).')
    exit()

try:
    from setuptools import setup,find_packages
except Exception:
    print('\nThe calcam installation process requires Python setuptools, but module setuptools could not be imported. Please install setuptools and try again.')
    exit()


def test_dependency(test_cmd):
    """
    Check if a module is importable, using an external python process.
    The reason for doing this in a separate process is because otherwise
    if we install a module in a separate process then try to import it
    on a subsequent check in this process,there be segfaults. So to avoid
    segfaults, use this.
    """
    try:
        subprocess.check_call([sys.executable, '-c',test_cmd],stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def pip_install(pkg_name):
    """
    A small utility function to call pip in a new process to install a given package.
    It seems setuptools cannot be trusted to locate some dependencies, or their correct versions,
    and pip works more reliably. Also I pass through the "--user" option from this setup.py but that's all.
    """
    if '--user' in [arg.lower() for arg in sys.argv]:
        extra_opts = ['--user']
    else:
        extra_opts = []

    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',pkg_name] + extra_opts,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False



if 'install' in sys.argv or 'develop' in sys.argv:

    print('Dealing with more difficult dependencies\nbefore handing over to setuptools...\n')

    if test_dependency('from cv2 import __version__'):
        print('OpenCV-Python installation found.')
    else:
        print('No OpenCV-Python installation found - trying to install opencv-python-headless using pip...')
        pip_install('opencv-python-headless')
        if test_dependency('from cv2 import __version__'):
           print('   OK!')
        else:
            print('\nCould not install opencv-python-headless using pip. Please install OpenCV for python (cv2) manually then try installing Calcam again.')
            exit()


    # Check if any supported version of PyQt is already present
    pyqt = test_dependency('from PyQt6.QtCore import *')
    if not pyqt:
        pyqt = test_dependency('from PyQt5.QtCore import *')
    if not pyqt:
        pyqt = test_dependency('from PyQt4.QtCore import *')

    if pyqt:
        print('Compatible PyQt installation found.')
    else:
        print('No compatible PyQt installation found.')

    # If not, try to install one with pip. Any will do.
    if not pyqt:
        print('   Trying to install PyQt6 using pip...')
        pip_install('PyQt6')
        pyqt = test_dependency('from PyQt6.QtCore import *')
        if pyqt: print('      OK!')
    if not pyqt:
        print('      Failed.\n   Trying PyQt5 instead...')
        pip_install('PyQt5')
        pyqt = test_dependency('from PyQt5.QtCore import *')
        if pyqt: print('      OK!')
    if not pyqt:
        print('      Failed.\n   Trying PyQt4 instead...')
        pip_install('PyQt4')
        pyqt = test_dependency('from PyQt4.QtCore import *')
        if pyqt:
            print('      OK!')
        else:
            print('      Failed.')

    # VTK
    if test_dependency('from vtk import vtkVersion;ver=vtkVersion();v=ver.GetVTKMajorVersion()*100+ver.GetVTKMinorVersion();assert v>600 & v<901'):
        print('Compatible VTK installation found.')
        vtk = True
    else:
        print('No compatible VTK installation found - trying to install with pip...')
        pip_install('vtk>=6,<9.1')
        if test_dependency('from vtk import vtkVersion;ver=vtkVersion();v=ver.GetVTKMajorVersion()*100+ver.GetVTKMinorVersion();assert v>600 & v<901'):
            print('   OK!\n')
            vtk = True
        else:
            print('   Failed.\n')
            vtk = False

    # Make sure a file exists at calcam/gui/__executable_path__ before calling setup() - this makes sure it gets marked as part of the package
    # so allows clean uninnstallation, but without having to keep a blank copy of the file in the git repo which causes git confusion.
    open(os.path.join(os.path.split(__file__)[0],'calcam','gui','__executable_path__'), 'a').close()

    print('\nNow handing over to setuptools...\n')

# Actually do the requested setup actions
s = setup(
        name='calcam',
        version=version,
        license='European Union Public License 1.1',
        author='Scott Silburn et. al.',
        author_email='scott.silburn@ukaea.uk',
        description='Spatial calibration tools for science & engineering camera systems.',
        long_description=readme,
        long_description_content_type='text/markdown',
        classifiers=[
                    "Development Status :: 5 - Production/Stable",
                    "Operating System :: Microsoft :: Windows",
                    "Operating System :: MacOS :: MacOS X",
                    "Operating System :: POSIX :: Linux",
                    "Environment :: Console",
                    "Environment :: X11 Applications :: Qt",
                    "Intended Audience :: Science/Research",
                    "Intended Audience :: Developers",
                    "License :: OSI Approved :: European Union Public Licence 1.1 (EUPL 1.1)",
                    "Natural Language :: English",
                    "Programming Language :: Python :: 3",
                    "Topic :: Scientific/Engineering :: Physics",
                    "Topic :: Scientific/Engineering :: Visualization",
                    "Topic :: Scientific/Engineering :: Image Processing"
                    ],
        project_urls={
                    'Documentation': 'https://euratom-software.github.io/calcam',
                    'Source': 'https://github.com/euratom-software/calcam/',
                    'Issue Tracker': 'https://github.com/euratom-software/calcam/issues',
                    'Zenodo':'https://doi.org/10.5281/zenodo.1478554'
                    },
        license_files = ('LICENCE.txt',),
        packages=find_packages(),
        package_data={'calcam':['gui/icons/*','gui/qt_designer_files/*.ui','gui/logo.png','builtin_image_sources/*.py','__version__','gui/__executable_path__']},
        entry_points={ 'gui_scripts': [ 'calcam = calcam:start_gui'] },
        zip_safe=False,
        install_requires=['scipy','matplotlib']
        )


if 'install' in sys.argv or 'develop' in sys.argv:
    # Find out where setup() put the the GUI launcher script / executable and do 2 things with it:
    # (1) Print a message telling the user about it,
    # (2) Write it down in a file in the code path so that the user can easily look it up later.
    if 'install' in sys.argv:
        script_dir = s.command_obj['install'].install_scripts
        code_dir = s.command_obj['install'].install_lib
    elif 'develop' in sys.argv:
        script_dir = s.command_obj['develop'].script_dir
        code_dir = os.path.split(__file__)[0]

    executable_path = os.path.join(script_dir.replace('/',os.path.sep),'calcam')
    if sys.platform == 'win32':
        executable_path = executable_path + '.exe'

    extra_msg = '\n\nThe GUI can be launched using the executable:\n{:s}'.format(executable_path)

    try:
        with open(os.path.join(code_dir,'calcam','gui','__executable_path__'),'w') as f:
            f.write(executable_path)
    except Exception:
        pass

    # Check if the executable path is included in the PATH environment variable which makes the GUI easier to run
    try:
        env_path = os.environ['PATH']
    except KeyError:
        env_path = ''

    if script_dir in env_path:
        extra_msg = extra_msg + '\nor just by typing "calcam" at a terminal.'

    print('\n***************************************************************\n\nCalcam installation complete.\n\nIt can be imported as a Python module with "import calcam"{:s}\n\n***************************************************************\n'.format(extra_msg))

else:
    print('\n***************************************************************\n')