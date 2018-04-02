'''
Converts Calcam 1 files to equivalent Calcam 2 versions.

Work in progress.
'''



from calcam.io import ZipSaveFile
import sys
import os
import json
import traceback
import inspect
import tempfile
import shutil

homedir = os.path.expanduser('~/Documents')
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
            exec('import ' + def_filename[:-3] + ' as CADDef')
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


if __name__ == '__main__':

    convert_cadmodels()