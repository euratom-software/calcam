'''
* Copyright 2015-2017 European Atomic Energy Community (EURATOM)
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


"""
CAD model definition logistics

If you came here looking for CAD model definitions themselves, I'm afraid your code is in another castle.

This file imports the CAD model definitions from user definitions at ~/calcam/UserCode/machine_geometry/
and provides a few housekeeping functions to do with CAD model definition files.

Written by Scott Silburn
"""

import paths
import shutil
import os
import sys
import inspect
import filecmp
from copy import deepcopy as copy
import traceback

# First of all, check if the user has an "old stype" machine_geometry file, and if so, move it and tell them what's going on.
if os.path.isfile(os.path.join(paths.code,'_machine_geometry.py')):
    shutil.move(os.path.join(paths.code,'_machine_geometry.py'),os.path.join(paths.machine_geometry,'machine_geometry_moved.py'))
    print('[Calcam Import] The way CAD definitions are organised has changed. Your existing definitions have been moved to ' + os.path.join(paths.machine_geometry,'machine_geometry_moved.py') + '. It is recommended to split up model definitions one-per-file in that directory, for tidyness.')

example_file = os.path.join(paths.calcampath,'usercode_examples','machine_geometry.py_')
user_files = [fname for fname in os.listdir(paths.machine_geometry) if fname.endswith('.py')]

# See if the user already has a CAD model definition example file, and if it's up to date. If not, create it.
# If the definitions might have changed, warn the user.
if 'Example.py' in user_files:
    is_current_version = filecmp.cmp(os.path.join(paths.machine_geometry,'Example.py'),example_file,shallow=False)
    if not is_current_version:
        shutil.copy2(example_file,os.path.join(paths.machine_geometry,'Example.py'))
        print('[Calcam Import] The latest CAD model definition example is different from your user copy. Your existing copy has been updated. If you get CAD model related errors, you may need to check and edit the CAD definition files in ' + paths.machine_geometry )
    user_files.remove('Example.py')
else:
    shutil.copy2(example_file,os.path.join(paths.machine_geometry,'Example.py'))
    print('[Calcam Import] Created CAD model definition example in ' + os.path.join(paths.machine_geometry,'Example.py'))

if 'machine_geometry.py' in user_files:
    os.rename(os.path.join(paths.machine_geometry,'machine_geometry.py'),os.path.join(paths.machine_geometry,'machine_geometry_moved.py'))
    print('[Calcam Import] ' + os.path.join(paths.machine_geometry,'machine_geometry.py') + ' renamed to machine_geometry_moved.py.')
    user_files.remove('machine_geometry.py')
    user_files.append('machine_geometry_moved.py')
	
# Go through all the python files which aren't examples, and import the CAD definitions
for def_filename in user_files:
    try:
        exec('import ' + def_filename[:-3] + ' as CADDef')
        classes = inspect.getmembers(CADDef, inspect.isclass)
        for iclass in classes:
            if inspect.getmodule(iclass[1]) is CADDef:
                exec('from {:s} import {:s}'.format(def_filename[:-3],iclass[0]) )
        del CADDef
    except Exception as e:
        estack = traceback.extract_tb(sys.exc_info()[2])
        lineno = None
        for einf in estack:
            if def_filename in einf[0]:
                lineno = einf[1]
        if lineno is not None:
            print('[Calcam Import] CAD definition file {:s} not imported due to exception at line {:d}: {:s}'.format(def_filename,lineno,e))
        else:
            print('[Calcam Import] CAD definition file {:s} not imported due to exception: {:s}'.format(def_filename,e))
	
	
# Get a list of available CAD models and their available variants
def get_available_models():

    models = {}
	
    # List of classes in this module
    model_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    # Ignoring the parent CADModel class, create a test instance of each model to grab its name and variants
    for modelclass in model_classes:
        if modelclass[0] != 'CADModel':
            test_instance = modelclass[1]()
            models[copy(test_instance.machine_name)] = ( modelclass[0] , copy(test_instance.model_variants) , test_instance.model_variants.index(test_instance.model_variant) )

    return models
