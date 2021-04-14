'''
* Copyright 2015-2021 European Atomic Energy Community (EURATOM)
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

from .io import ZipSaveFile
from .misc import import_source, unload_source


# Number of CPUs to use for multiprocessing enabled
# calculations. Always initialised to (n_cpus - 1)
# and any changes by the user only apply to that session.
n_cpus = max(1,multiprocessing.cpu_count()-1)

# Path where the "built in" image source code lives; this is always in the calcam source directory.
builtin_imsource_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],'builtin_image_sources')


class CalcamConfig():
    '''
    Class to represent the persistent calcam settings.
    
    It is convenient to use a class for this because whenever this is
    instantiated somewhere in Calcam, we always get the most recent configuration
    from disk. It also makes saving 
    '''
    def __init__(self,cfg_file= os.path.expanduser('~/.calcam_config'),allow_create=True):

        self.filename = cfg_file
        self.filename_filters = {'calibration':'Calcam Calibration (*.ccc)','image':'PNG Image (*.png)','pointpairs':'Calcam Point Pairs (*.ccc *.csv)'}
        
        try:
            self.load()
        except:
            if not allow_create:
                raise

            self.file_dirs = {}
            try:
                self.cad_def_paths
            except:
                self.cad_def_paths = []

            try:
                self.image_source_paths
            except:
                self.image_source_paths = []
            
            self.default_model = None
            self.default_image_source = 'Image File'
            self.mouse_sensitivity = 75
            self.main_overlay_colour = (0,0,1.,0.6)
            self.second_overlay_colour = (1.,0,0,0.6)
            self.save()




    def load(self):

        with open(self.filename,'r') as f:
            load_dict = json.load(f)

        self.image_source_paths = load_dict['image_source_paths']
        self.cad_def_paths = load_dict['cad_def_paths']
        self.file_dirs =     load_dict['file_dirs']
        self.default_model = load_dict['default_model']
        self.default_image_source = load_dict['default_im_source']
        self.mouse_sensitivity = load_dict['mouse_sensitivity']
        self.main_overlay_colour = load_dict['main_overlay_colour']
        self.second_overlay_colour = load_dict['second_overlay_colour']


    def save(self):

        save_dict = {
                        'file_dirs'     : self.file_dirs,
                        'default_model' : self.default_model,
                        'cad_def_paths'    : self.cad_def_paths,
                        'image_source_paths':self.image_source_paths,
                        'default_im_source':self.default_image_source,
                        'mouse_sensitivity':self.mouse_sensitivity,
                        'main_overlay_colour':self.main_overlay_colour,
                        'second_overlay_colour':self.second_overlay_colour,
                    }

        with open(self.filename,'w') as f:
            json.dump(save_dict,f,indent=4)



    def get_cadmodels(self):

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
