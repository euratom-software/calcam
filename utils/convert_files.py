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
Converts Calcam 1 files to equivalent Calcam 2 versions.

Work in progress.
'''



import sys
import os
import json
import traceback
import inspect
import tempfile
import shutil
import pickle
import numpy as np
import socket
import getpass
import time

from calcam.io import ZipSaveFile
from calcam.calibration import Calibration, ViewModel
from calcam.coordtransformer import CoordTransformer
from calcam.pointpairs import PointPairs

homedir = os.path.join( os.path.expanduser('~'), 'Documents')
newpath_root = os.path.join(homedir,'Calcam 2')



def convert_cadmodel(old_class,output_path):

    # Parse the old definition and generate the new JSON structure
    model = old_class()

    print('\nConverting model definition: "{:s}"...'.format(model.machine_name))

    new_definition = {'machine_name':model.machine_name}

    views = {}
    for view in model.views:
        views[view[0]] = {'cam_pos':view[1],'target':view[2],'y_fov':view[3],'xsection':None}

    new_definition['views'] = views

    new_definition['initial_view'] = model.default_view_name

    new_definition['default_variant'] = model.model_variant

    root_paths = {}
    features = {}

    mesh_folders_to_add = []

    for variant in model.model_variants:

        varmodel = old_class(variant)

        root_paths[variant] = os.path.join('.large',variant)
        mesh_folders_to_add.append( (varmodel.filepath.rstrip('\\/') , os.path.join('.large',variant) ) )

        if not os.access( varmodel.filepath.rstrip('\\/'), os.R_OK):
            raise IOError('Cannot read mesh file path {:s}'.format(varmodel.filepath.rstrip('\\/')))

        featuredict = {}


        for feature in varmodel.features:

            if feature[3] == None:
                fname = feature[0]
            else:
                fname = '{:s}/{:s}'.format(feature[3],feature[0])

            featuredict[fname] = {'mesh_file':feature[1], 'colour': varmodel.materials[feature[2]][1] if varmodel.colourbymaterial else (0.75,0.75,0.75), 'default_enable' : feature[4], 'mesh_scale':varmodel.units }


        features[variant] = featuredict


    new_definition['features'] = features
    new_definition['mesh_path_roots'] = root_paths


    # Now make the new definition file!
    with ZipSaveFile( os.path.join(output_path, '{:s}.ccm'.format(new_definition['machine_name'])) ,'w' ) as newfile:

        with newfile.open_file('model.json','w') as mf:

            json.dump(new_definition,mf,indent=4)


        for src_path,dst_path in mesh_folders_to_add:
            newfile.add(src_path,dst_path)

    print('Calcam 2 file saved to: {:s}'.format(os.path.join(output_path, '{:s}.ccm'.format(new_definition['machine_name']))))


def convert_cadmodels(new_path= os.path.join(newpath_root,'CAD Models')):


    tmpdir = tempfile.mkdtemp()
    with open( os.path.join(tmpdir,'cadmodel.py'),'w') as pyfile:
        pyfile.write('class CADModel():\n    def init_cadmodel(self):\n        pass\n    def set_default_view(self,view):\n        self.default_view_name = view')
    sys.path.insert(0,tmpdir)

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    homedir = os.path.expanduser('~')
    root = os.path.join(homedir,'calcam')
    if os.path.isfile(os.path.join(root,'.altpath.txt')):
        f = open(os.path.join(root,'.altpath.txt'),'r')
        root = f.readline().rstrip()
        f.close()
    search_path = os.path.join(root,'UserCode','machine_geometry')

    print('\nLooking for Calcam v1.x model definitions in: {:s}'.format(search_path))

    user_files = [fname for fname in os.listdir(search_path) if fname.endswith('.py')]

    sys.path.insert(0, os.path.split(__file__) )

    # Go through all the python files which aren't examples, and import the CAD definitions
    for def_filename in user_files:
        if def_filename.endswith('Example.py'):
            continue

        try:
            CADDef = __import__(def_filename[:-3])
            modelclasses = inspect.getmembers(CADDef, inspect.isclass)

            for modelclass in modelclasses:
                if modelclass[0] != 'CADModel':
                    convert_cadmodel(modelclass[1],new_path)

            del CADDef


        except Exception as e:
            raise
            estack = traceback.extract_tb(sys.exc_info()[2])
            lineno = None
            for einf in estack:
                if def_filename in einf[0]:
                    lineno = einf[1]
            if lineno is not None:
                print('Old CAD definition file {:s} not converted due to exception at line {:d}: {}'.format(def_filename,lineno,e))
            else:
                print('Old CAD definition file {:s} not converted due to exception: {}'.format(def_filename,e))

    sys.path.remove(tmpdir)
    shutil.rmtree(tmpdir)





def load_calib(path):

    cal = Calibration()

    SaveFile = open(path,'rb')

    try:
        save = pickle.load(SaveFile)
    except:
        SaveFile.seek(0)
        save = pickle.load(SaveFile,encoding='latin1')

    cal.n_subviews = int(save['nfields'])
    cal.subview_mask = save['field_mask']

    if 'type' in save:
	    if save['type'] != 'fit':
	    	print('This is a manual alignment calibration; not supported yet! Sorry.')
	    	return None

    print(save.keys())
    cal.pointpairs = PointPairs()
    if 'PointPairs' in save:
        cal.pointpairs = load_pointpairs(save['PointPairs'][0])
        cal.image_points.append(pp['imagepoints'])
        cal.object_points.append(pp['objectpoints'])
        cal.image = load_image(pp['image'])
    elif 'objectpoints' in save:
        if len(save['objectpoints']) > 0:
            cal.pointpairs.object_points = save['objectpoints'][0]
            cal.pointpairs.image_points = save['imagepoints'][0]
            cal.pointpairs.n_subviews = len(save['imagepoints'][0][0])
            # Intrinsics point pairs
            for i in range(1,len(save['objectpoints'])):
                cal.intrinsics_constraints.append([None,PointPairs()])
                cal.intrinsics_constraints[-1][1].object_points = save['objectpoints'][i]
                cal.intrinsics_constraints[-1][1].image_points = save['imagepoints'][i]


    if 'type' in save:
        cal.calib_type = save['type']
    else:
        cal.calib_type = 'fit'


    if 'field_names' in save:
        cal.subview_names = save['field_names']
    else:
        if cal.n_subviews == 1:
            cal.subview_names = ['Image']
        else:
            cal.subview_names = []
            for nview in range(cal.n_subviews):
                cal.subview_names.append('Sub-view {:d}'.format(nview+1))


    fitoptions = save['fitoptions']
    
    if fitoptions is None:
        fitoptions = [None] * cal.n_subviews

    if len(fitoptions) == 0:
        fitoptions = [[]]

    elif type(fitoptions[0]) != list:
        fitoptions = [fitoptions] * cal.n_subviews

    for nview in range(cal.n_subviews):

        if 'model' in save:
            fit = FieldFit(save['model'][nview],save['fitparams'][nview],from_save=True)
        else:
            fit = FieldFit('perspective',save['fitparams'][nview],from_save=True)

        coeff_dict = {
                        'model':fit.model,
                        'reprojection_error':fit.rms_error,
                        'fx':fit.cam_matrix[0,0],
                        'fy':fit.cam_matrix[1,1],
                        'cx':fit.cam_matrix[0,2],
                        'cy':fit.cam_matrix[1,2],
                        'dist_coeffs':list(np.squeeze(fit.kc)),
                        'rvec':list(np.squeeze(np.array(fit.rvec).astype(np.float64))),
                        'tvec':list(np.squeeze(np.array(fit.tvec).astype(np.float64))),
                        'fit_options':fitoptions[nview]
                    }

        cal.view_models.append(ViewModel.from_dict(coeff_dict))

    try:
        save['transform_pixel_aspect'] = float(save['transform_pixel_aspect'])
    except TypeError:
        pass

    cal.geometry = CoordTransformer(save['transform_actions'],save['transform_pixels'][0],save['transform_pixels'][1],save['transform_pixel_aspect'])

    SaveFile.close()

    return cal


def convert_calibs(new_path= os.path.join(newpath_root,'Calibrations')):

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    homedir = os.path.expanduser('~')
    root = os.path.join(homedir,'calcam')
    if os.path.isfile(os.path.join(root,'.altpath.txt')):
        f = open(os.path.join(root,'.altpath.txt'),'r')
        root = f.readline().rstrip()
        f.close()
    search_path = os.path.join(root,'FitResults')

    print('\nLooking for Calcam v1.x calibrations in: {:s}'.format(search_path))

    calib_files = [fn for fn in os.listdir(search_path) if fn.endswith('.pickle')]

    hostname = socket.gethostname()
    uname = getpass.getuser()

    for file in calib_files:
        new_filename = os.path.split(file)[-1][:-7]
        print('-> Converting calibration "{:s}"...'.format(new_filename))
        try:
            cal = load_calib(os.path.join(search_path,file))
            if cal is not None:
	            cal.history = [(int(time.time()),uname,hostname,'Converted from Calcam 1.x calibration "{:s}"'.format(new_filename))]
	            cal.save( os.path.join(new_path,'{:s}.ccc'.format(new_filename)) )
	            print(' -> Calcam 2 version saved to {:s}\n'.format(os.path.join(new_path,'{:s}.ccc'.format(new_filename))))
        except:
            print(' -> Error, calibration not converted.\n')
            raise



# Small class for storing the actual fit output parameters for a sub-field.
# Instances of this are used inside the CalibResults class.
class FieldFit:

    def __init__(self,fit_model,FitParams,from_save=False):

        self.model = fit_model

        # RMS reprojection error
        self.rms_error = FitParams[0]
    
        # Camera matrix
        self.cam_matrix = FitParams[1].copy()
    
        # Distortion coefficients array
        self.kc = FitParams[2].copy()

        # Extrinsics: rotation and translation vectors
        self.rvec = FitParams[3][0].copy()
        self.tvec = FitParams[4][0].copy()

        
        if fit_model == 'perspective':
            # Split distortion coeffs in to individual properties
            self.k1 = self.kc[0][0]
            self.k2 = self.kc[0][1]
            self.k3 = 0
            self.p1 = self.kc[0][2]
            self.p2 = self.kc[0][3]
            if len(FitParams[2]) == 5:
                    self.k3 = self.kc[0][4]
            elif len(FitParams[2]) == 8:
                raise Exceotion('Calcam does not (yet) support rational OpenCV model!')
        
        elif fit_model == 'fisheye':
            # Annoyingly, fisheye calibration returns its extrinsics in a different
            # format than the perspective fitting. *sigh*. Also the format that comes
            # back from OpenCV depends in some interesting way on versions of things,
            # so we need this try...except here. Grumble.
            if not from_save:
                try:
                    self.rvec[0] = self.rvec[0].T
                    self.tvec[0] = self.tvec[0].T
                except ValueError:
                    self.rvec = self.rvec[0].T
                    self.tvec = self.tvec[0].T
                                        
            self.k1 = self.kc[0]
            self.k2 = self.kc[1]
            self.k3 = self.kc[2]
            self.k4 = self.kc[3]



if __name__ == '__main__':

    convert_calibs()