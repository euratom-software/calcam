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


import os
import json
import sys
import glob
import traceback
import multiprocessing
import warnings

from .io import ZipSaveFile
from .misc import import_source, unload_source


# Number of CPUs to use for multiprocessing enabled
# calculations. Always initialised to (n_cpus - 1)
# and any changes by the user only apply to that session.
n_cpus = max(1,multiprocessing.cpu_count()-1)

# Path where the "built in" image source code lives; this is always in the calcam source directory.
builtin_imsource_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],'builtin_image_sources')

# File where calcam stores the user's configuration
user_cfg_path = os.path.expanduser('~/.calcam_config')

# If the user doesn't have their own configuration, look for a default config file in the calcam install directory.
# This can be used e.g. if installing calcam on a multi-user system where you want to provide a default config for all users.
# But there is not one included by default with Calcam!
default_cfg_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],'site_defaults.cfg')

# Filename filters for different types of file
filename_filters = {'calibration':'Calcam Calibration (*.ccc)','image':'PNG Image (*.png)','pointpairs':'Calcam Point Pairs (*.ccc *.csv)','movement':'Calcam Affine Transform (*.cmc)'}

class CalcamConfig():
    '''
    Class to represent the persistent calcam settings.
    
    It is convenient to use a class for this because whenever this is
    instantiated somewhere in Calcam, we always get the most recent configuration
    from disk.
    '''
    def __init__(self):

        # Configuration fields and their default values
        self.fields = {'image_source_paths': [],
                       'cad_def_paths': [],
                       'file_dirs': {},
                       'default_model': None,
                       'default_image_source':'Image File',
                       'mouse_sensitivity':75,
                       'main_overlay_colour':(0,0,1.,0.6),
                       'second_overlay_colour':(1.,0,0,0.6)
                       }

        # Filename filters (which should never need to change so are defined above)
        self.filename_filters = filename_filters

        # Try to load the user's own config
        try:
            with open(user_cfg_path,'r') as f:
                user_dict = json.load(f)
        except IOError:
            user_dict = {}

        # Try to load the default config
        try:
            with open(default_cfg_path, 'r') as f:
                defaults_dict = json.load(f)
        except IOError:
            defaults_dict = {}

        # For each field, first try to get the user's own value,
        # then the one from the defaults file, then fall back to the version defined above.
        for key in self.fields:
            if key in user_dict:
                setattr(self,key,user_dict[key])
            elif key in defaults_dict:
                setattr(self,key,defaults_dict[key])
            else:
                setattr(self,key,self.fields[key])

        self.save()


    def save(self):
        """
        Save the configuration to disk in the user's home directory.
        """
        save_dict = { key : getattr(self,key) for key in self.fields}

        try:
            with open(user_cfg_path,'w') as f:
                json.dump(save_dict,f,indent=4)
        except Exception as e:
            warnings.warn('Could not save user calcam configuration to file "{:s}": {:}'.format(user_cfg_path,e))



    def get_cadmodels(self):
        """
        Get the list of available CAD models.

        Returns:
            A dictionary where the keys are human-readable names of the CAD model and the velues are a list of: \
            [Filename of the CAD model, List of features in that CAD model, Default variant name]
        """
        cadmodels = {}

        for path in self.cad_def_paths:
            filelist = glob.glob(os.path.join(path,'*.ccm'))

            for fname in filelist:

                try:
                    with ZipSaveFile(fname,'rs') as f:
                        with f.open_file('model.json','r') as j: 
                               caddef = json.load(j)
                except:
                    continue

                if caddef['machine_name'] not in cadmodels:
                    key = caddef['machine_name']
                else:
                    
                    existing_model = cadmodels.pop(caddef['machine_name'])
                    existing_key = '{:s} [{:s}/{:s}]'.format(caddef['machine_name'], existing_model[0].split(os.sep)[-2],os.path.split(existing_model[0])[-1] )
                    cadmodels[existing_key] = existing_model

                    key = '{:s} [{:s}/{:s}]'.format(caddef['machine_name'], fname.split(os.sep)[-2],os.path.split(fname)[1] )

                cadmodels[key] = [fname,[str(x) for x in caddef['features'].keys()],caddef['default_variant']]

        return cadmodels


    def get_image_sources(self,meta_only=False):
        """
        Get a list of available image source providers.
        """
        image_sources = []
        displaynames = []
        meta = []

        for path in [builtin_imsource_path] + self.image_source_paths:

            filelist = glob.glob(os.path.join(path,'*'))

            trylist = []
            for f in filelist:
                if (os.path.isdir(f) and os.path.isfile(os.path.join(f,'__init__.py'))) or f.endswith('.py'):
                    trylist.append(f)

            for fname in trylist:
                if fname.endswith('__init__.py'):
                    tidy_name = os.sep.join(fname.split(os.sep)[-3:-1])
                    fname = os.path.split(fname)[0]
                else:
                    tidy_name = os.sep.join(fname.split(os.sep)[-2:])

                try:
                    # Import the module, check it has the right attributes and add its info to the metadata table
                    usermodule = import_source(fname)

                    try:
                        if not callable(usermodule.get_image_function):
                            raise ImportError()
                    except Exception:
                        meta.append([tidy_name, fname, 'Not a valid image source definition:\nDoes not contain required function "get_image_function(..)"'])
                        continue

                    try:
                        if type(usermodule.get_image_arguments) is not list:
                            raise ImportError
                    except Exception:
                        meta.append([tidy_name, fname, 'Not a valid image source definition:\nDoes not contain required list attribute "get_image_arguments"'])
                        continue

                    try:
                        if type(usermodule.display_name) is not str:
                            raise ImportError
                    except Exception:
                        meta.append([tidy_name, fname, 'Not a valid image source definition:\nDoes not contain required string attribute  "display_name"'])
                        continue


                    # Make sure the imported modules have unique display names
                    if usermodule.display_name in displaynames:
                        old_ind = displaynames.index(usermodule.display_name)
                        for i,metadata in enumerate(meta):
                            if metadata[0] == usermodule.display_name:
                                old_meta_ind = i
                                break

                        usermodule.display_name = usermodule.display_name + ' [{:s}]'.format(tidy_name)
                        other_path = image_sources[old_ind].__file__
                        if other_path.endswith('__init__.py'):
                            tidyname = os.sep.join(other_path.split(os.sep)[-3:-1])
                        else:
                            tidyname = os.sep.join(other_path.split(os.sep)[-2:])
                        image_sources[old_ind].display_name = image_sources[old_ind].display_name + ' [{:s}]'.format(tidyname)
                        meta[old_meta_ind][0] = image_sources[old_ind].display_name

                    displaynames.append(usermodule.display_name)
                    image_sources.append(usermodule)
                    meta.append([usermodule.display_name,fname,None])
                except Exception:
                    # If it won't import or doesn't have the right attributes, show this in the metadata
                    tb_info = ''.join(traceback.format_exception(*sys.exc_info(), limit=-1))
                    meta.append([tidy_name,fname,'Cannot be imported:\n{:s}'.format(tb_info)])
                    continue

                # Built-in image sources get special metadata
                if path == builtin_imsource_path:
                    meta[-1][1] = None


        if meta_only:
            for module_meta in meta:
                if module_meta[-1] is None and module_meta[1] is not None:
                    unload_source(module_meta[1])
            return meta
        else:
            return image_sources
