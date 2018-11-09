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
import datetime

from calcam.io import ZipSaveFile
from calcam.calibration import Calibration, ViewModel
from calcam.coordtransformer import CoordTransformer
from calcam.pointpairs import PointPairs
from scipy.io.netcdf import netcdf_file
from calcam.gui import qt_wrapper as qt
from calcam.config import CalcamConfig


calcam1_root = os.path.join(os.path.expanduser('~'),'calcam')
if os.path.isfile(os.path.join(calcam1_root,'.altpath.txt')):
    f = open(os.path.join(calcam1_root,'.altpath.txt'),'r')
    calcam1_root = f.readline().rstrip()
    f.close()

# This stuff is used for keeping track of calibration history
_user = getpass.getuser()
_host = socket.gethostname()

def _get_formatted_time(timestamp=None):
    
    if timestamp is None:
        when = datetime.datetime.now()
    else:
        when = datetime.datetime.fromtimestamp(timestamp)
    
    return when.strftime('%H:%M on %Y-%m-%d')


def convert_cadmodel(old_class,output_path,status_callback=None):

    # Parse the old definition and generate the new JSON structure
    model = old_class()

    new_definition = {'machine_name':model.machine_name}

    views = {}
    for view in model.views:
        views[view[0]] = {'cam_pos':view[1],'target':view[2],'y_fov':view[3],'xsection':None,'roll':0,'projection':'perspective'}

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

            if status_callback is not None:
                status_callback()

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





def load_calib(path,match_images=True,virtual=False):


    SaveFile = open(path,'rb')

    try:
        save = pickle.load(SaveFile)
    except:
        SaveFile.seek(0)
        save = pickle.load(SaveFile,encoding='latin1')

    calname = os.path.split(path)[-1][:-4]

    if 'type' in save and not virtual:
        if save['type'] == 'fit':
            caltype = 'fit'
        elif save['type'] == 'manual_alignment':
            caltype = 'alignment'
            if save['fitparams'][0][0] > 0:
                int_type = 'calibration'
            else:
                int_type = 'pinhole'
    elif virtual:
        caltype = 'virtual'
        match_images=False
        if save['fitparams'][0][0] > 0:
            int_type = 'calibration'
        else:
            int_type = 'pinhole'
    else:
        caltype = 'fit'

    cal = Calibration(cal_type = caltype)

    if cal._type in ['alignment','virtual']:
        cal.intrinsics_type = int_type
        cal.history['intrinsics'] = 'Converted from Calcam 1.x calibration "{:s}" by {:s} on {:s} at {:s}'.format(calname,_user,_host,_get_formatted_time())
        cal.history['extrinsics'] = cal.history['intrinsics']

    try:
        save['transform_pixel_aspect'] = float(save['transform_pixel_aspect'])
    except TypeError:
        pass

    if caltype != 'virtual':
        cal.geometry = CoordTransformer(save['transform_actions'],save['transform_pixels'][0],save['transform_pixels'][1],save['transform_pixel_aspect'])
    else:
        cal.geometry = CoordTransformer(save['transform_actions'],save['image_display_shape'][0],save['image_display_shape'][1],save['transform_pixel_aspect'])


    cal.n_subviews = int(save['nfields'])
    if cal._type != 'virtual':
        cal.subview_mask = cal.geometry.display_to_original_image(save['field_mask'])
    else:
        cal.subview_mask = np.zeros( (save['image_display_shape'][1],save['image_display_shape'][0]),dtype=np.uint8)


    if caltype == 'fit':
        cal.history['pointpairs'] = ['Converted from Calcam 1.x calibration "{:s}" by {:s} on {:s} at {:s}'.format(calname,_user,_host,_get_formatted_time()),None]
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
                    cal.history['intrinsics_constraints'] = 'Converted from Calcam 1.x calibration "{:s}" by {:s} on {:s} at {:s}'.format(calname,_user,_host,_get_formatted_time())

    if match_images:
        matched_ims = []
        imfiles = os.listdir(os.path.join(calcam1_root,'Images'))
        for imname in [name[:-3] for name in imfiles if name.endswith('.nc')]:
            if imname in calname:
                matched_ims.append(imname)
        if len(matched_ims) == 1:
            im = load_image(matched_ims[0])
            imshape = im['image_data'].shape[1::-1]
            if np.all(imshape == cal.geometry.get_original_shape()):
                cal.image = im['image_data']
                cal.history['image'] = 'Converted from Calcam 1.x image file "{:s}" by {:s} on {:s} at {:s}'.format(matched_ims[0],_user,_host,_get_formatted_time())



    if 'field_names' in save:
        cal.subview_names = save['field_names']
    else:
        if cal.n_subviews == 1:
            cal.subview_names = ['Image']
        else:
            cal.subview_names = []
            for nview in range(cal.n_subviews):
                cal.subview_names.append('Sub-view {:d}'.format(nview+1))

    if 'fitoptions' in save:
        fitoptions = save['fitoptions']
    else:
        fitoptions = []
    cal.history['fit'] = 'Converted from calcam 1.x calibration "{:s}" by {:s} on {:s} at {:s}'.format(calname,_user,_host,_get_formatted_time())
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


    SaveFile.close()

    return cal


def load_image(loadname):

    imret = {}

    f = netcdf_file(os.path.join(calcam1_root,'Images',loadname + '.nc'), 'r',mmap=False)

    # This is for dealing with "old" format save files.
    # If things are saved as 64 bit ints (why did I ever think that was sensible???)
    # let's convert things down to 8 bit int like a sensible person.
    # Also attempt some sort of recovery of any transparency info.

    if f.variables['image_data'].data.dtype == '>i4':
        data = f.variables['image_data'].data.astype('uint64')
        alpha = None
        if len(data.shape) == 3:
            if data.shape[2] == 4:
                scale_factor = 255. / data[:,:,3].max()
                alpha = np.int8(data[:,:,3] * scale_factor)

        scale_factor = 255. / data.max()
        imret['image_data'] = np.uint8(data * scale_factor)
        imret['subview_mask'] = f.variables['fieldmask'].data.astype('int8')
        clim = f.variables['calcam_clim'].data.astype('uint64')
        imret['pixel_aspect'] = f.variables['pixel_aspect_ratio'].data.astype('float32')

    else:
        # If the data is already saved as 8 bit ints, we can carry on happily :)
        imret['image_data'] = f.variables['image_data'].data.astype('uint8')
        try:
            alpha = f.variables['alpha_data'].data.astype('uint8')
        except KeyError:
            alpha = None
        imret['subview_mask'] = f.variables['fieldmask'].data.astype('uint8')
        clim = f.variables['calcam_clim'].data.astype('uint8')
        imret['pixel_aspect'] = f.variables['pixel_aspect_ratio'].data.astype('float32')

    try:
        imret['pixel_size'] = f.variables['pixel_size'].data.astype('float32')
    except KeyError:
        imret['pixel_size'] = None

    n_fields = imret['subview_mask'].max() + 1

    try:
        fns = f.field_names
        if type(fns) is bytes:
            fns = fns.decode('utf-8')
        imret['subview_names'] = fns.split(',')
    except AttributeError:
        if n_fields == 1:
            imret['subview_names'] = ['Image']
        else:
            imret['subview_names'] = []
            for field in range(n_fields):
                imret['subview_names'].append('Sub FOV # {:d}'.format(field+1))


    f.close()

    return imret


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


class MigrationToolWindow(qt.QMainWindow):


    def __init__(self, app, parent=None):

        # GUI initialisation
        qt.QMainWindow.__init__(self, parent)
        qt.uic.loadUi(os.path.join(os.path.split(__file__)[0],'convert_files.ui'), self)

        self.app = app

        n_calibs = len( [name for name in os.listdir(os.path.join(calcam1_root,'FitResults')) if name.endswith('.pickle')] )
        n_virtual_calibs = len( [name for name in os.listdir(os.path.join(calcam1_root,'VirtualCameras')) if name.endswith('.pickle')] )

        self.calib_count = n_calibs + n_virtual_calibs

        self.n_cadmodels = len( [name for name in os.listdir(os.path.join(calcam1_root,'UserCode','machine_geometry')) if name.endswith('.py') and 'Example' not in name] )

        if self.n_cadmodels == 0 and self.calib_count == 0:
            print('Did not find any Calcam 1.x files to convert. Exiting.')
            sys.exit()

        newpath_root = os.path.join(os.path.expanduser('~'),'Documents','Calcam 2')

        self.calib_output_dir.setText(os.path.join(newpath_root,'Calibrations'))
        self.virtual_output_dir.setText(os.path.join(newpath_root,'Virtual Calibrations'))
        self.cad_output_dir.setText(os.path.join(newpath_root,'CAD Models'))

        self.calib_browse_button.clicked.connect(self.change_cal_dir)
        self.virtual_browse_button.clicked.connect(self.change_vcal_dir)
        self.cad_output_browse.clicked.connect(self.change_cad_dir)

        self.cal_convert_button.clicked.connect(self.convert_calibs)
        self.cad_convert_button.clicked.connect(self.convert_cadmodels)

        self.calib_info.setText('{:d} calibrations and {:d} virtual calibrations in {:s} can be converted to Calcam 2 format. Select the output directories for the converted calibrations below:'.format(n_calibs,n_virtual_calibs,calcam1_root))
        self.cad_info.setText('{:d} machine models in {:s} can be converted to Calcam 2 format. Select the desired output directory below.'.format(self.n_cadmodels,os.path.join(calcam1_root,'UserCode','machine_geometry')))

        if self.calib_count == 0:
            self.cal_groupbox.setEnabled(False)

        if self.n_cadmodels == 0:
            self.cad_groupbox.setEnabled(False)

        self.show()


    def change_cal_dir(self):

        newpath = self.browse_for_folder(str(self.calib_output_dir.text()))

        if newpath is not None:
            self.calib_output_dir.setText(newpath)


    def change_vcal_dir(self):

        newpath = self.browse_for_folder(str(self.virtual_output_dir.text()))

        if newpath is not None:
            self.virtual_output_dir.setText(newpath)

    def change_cad_dir(self):

        newpath = self.browse_for_folder(str(self.cad_output_dir.text()))

        if newpath is not None:
            self.cad_output_dir.setText(newpath)


    def convert_calibs(self):

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))

        new_path = str(self.calib_output_dir.text())

        if not os.path.exists(new_path):
            os.makedirs(new_path)


        self.progressbar.setMaximum(self.calib_count)

        search_path = os.path.join(calcam1_root,'FitResults')

        calib_files = [fn for fn in os.listdir(search_path) if fn.endswith('.pickle')]

        self.cal_groupbox.setEnabled(False)
        self.cad_groupbox.setEnabled(False)

        done = 0
        for file in calib_files:
            new_filename = os.path.split(file)[-1][:-7]
            self.progresstext.setText('Converting calibration "{:s}"...'.format(new_filename))
            self.progressbar.setValue(done)
            self.app.processEvents()
            try:
                cal = load_calib(os.path.join(search_path,file),match_images = self.match_images_checkbox.isChecked())
                if cal is not None:
                    cal.save( os.path.join(new_path,'{:s}.ccc'.format(new_filename)) )
            except:
                print(' -> Error, calibration not converted.\n')
            done += 1

        new_path = str(self.virtual_output_dir.text())

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        search_path = os.path.join(calcam1_root,'VirtualCameras')

        calib_files = [fn for fn in os.listdir(search_path) if fn.endswith('.pickle')]

        for file in calib_files:
            new_filename = os.path.split(file)[-1][:-7]
            self.progresstext.setText('Converting virtual calibration "{:s}"...'.format(new_filename))
            self.progressbar.setValue(done)
            self.app.processEvents()
            try:
                cal = load_calib(os.path.join(search_path,file),virtual=True)
                if cal is not None:
                    cal.save( os.path.join(new_path,'{:s}.ccc'.format(new_filename)) )
            except:
                print(' -> Error, calibration not converted.\n')
            done += 1

        self.cal_groupbox.setEnabled(True)
        self.cad_groupbox.setEnabled(True)
        self.progresstext.setText('Idle.')
        self.progressbar.setValue(0)
        self.app.restoreOverrideCursor()


    def convert_cadmodels(self):

        self.app.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))

        new_path = str(self.cad_output_dir.text())

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

        user_files = [fname for fname in os.listdir(search_path) if fname.endswith('.py')]

        sys.path.insert(0, search_path )

        self.cal_groupbox.setEnabled(False)
        self.cad_groupbox.setEnabled(False)
        self.progressbar.setMaximum(self.n_cadmodels)

        done = 0
        # Go through all the python files which aren't examples, and import the CAD definitions
        for def_filename in user_files:
            if def_filename.endswith('Example.py'):
                continue
            self.progresstext.setText('Converting CAD model definition: {:s}...'.format(def_filename))
            self.progressbar.setValue(done)
            self.app.processEvents()

            try:
                CADDef = __import__(def_filename[:-3])
                modelclasses = inspect.getmembers(CADDef, inspect.isclass)

                for modelclass in modelclasses:
                    if modelclass[0] != 'CADModel':
                        convert_cadmodel(modelclass[1],new_path,status_callback = self.app.processEvents)

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
            done += 1

        sys.path.remove(tmpdir)
        shutil.rmtree(tmpdir)

        self.cal_groupbox.setEnabled(True)
        self.cad_groupbox.setEnabled(True)
        self.progresstext.setText('Idle.')
        self.progressbar.setValue(0)

        config = CalcamConfig()

        if new_path not in config.cad_def_paths and self.add_cadpath.isChecked():
            config.cad_def_paths.append(new_path)
            config.save()

        self.app.restoreOverrideCursor()


    def browse_for_folder(self,start_dir=None):

        filedialog = qt.QFileDialog(self)
        filedialog.setAcceptMode(0)
        filedialog.setFileMode(2)
        if start_dir is not None:
            filedialog.setDirectory(start_dir)
        filedialog.setWindowTitle('Select Directory')
        filedialog.exec_()
        if filedialog.result() == 1:
            path = str(filedialog.selectedFiles()[0])
            return path.replace('/',os.path.sep)
        else:
            return None


if __name__ == '__main__':

    app = qt.QApplication([])
    win = MigrationToolWindow(app,None)
    sys.exit(app.exec_())
