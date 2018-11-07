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



# Setuptools install for Calcam
setup(
      name='Calcam',
      version='2.0.0-beta1',
      url='https://github.com/euratom-software/calcam',
      license='European Union Public License 1.1',
      packages=find_packages(),
      package_data={'calcam':['gui/*.ui','gui/*.png','image_sources/*.py']},
      entry_points={ 'gui_scripts': [ 'calcam = calcam.gui:start_gui'] },
      zip_safe=False
      )


# Used for migration tool choice below.
# From http://code.activestate.com/recipes/577058/
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

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



if len(sys.argv) > 1:

    if 'install' in sys.argv or 'develop' in sys.argv:

        if sys.platform == 'win32' and :
            import sysconfig
            import os
            script_dir = sysconfig.get_path('scripts')
            print('\n****\nPath to Calcam GUI launcher:\n"{:s}"\n****\n'.format(os.path.join(script_dir,'calcam.exe')))

        if os.path.isdir( os.path.join(os.path.expanduser('~'),'calcam') ):

            if query_yes_no('It appears you have a data directory from Calcam 1.x and might be upgrading to Calcam 2.\nWould you like to run the file migration tool now?'):
                subprocess.Popen([sys.executable,os.path.join( os.path.split(__file__)[0],'calcam1_file_converter','convert_files.py' )])