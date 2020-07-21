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


import vtk
import numpy as np
import json
import os
import atexit
from .config import CalcamConfig
from .io import ZipSaveFile



# A little function to use for status printing if no
# user callback is specified.
def print_status(status):
    if status is not None:
        print(status)



class CADModel():
    '''
    Class for representing a CAD model in Calcam.

    Parameters:
        model_name (str)        : Either the machine name of a CAD model in Calcam's model search path, \
                                  or filename of a .ccm file to load the model from.
        model_variant (str)     : Name of the model variant to load. If not specified, the default variant \
                                  specified in the CAD model is loaded.
        status_callback (func)  : Function to call with status messages. If given, this function is called \
                                  with a string describing the status. If set to None, no status updates are \
                                  issued.
    '''
    def __init__(self,model_name=None,model_variant=None,status_callback=print_status):


        if model_name is not None:
            # -------------------------------Loading model definition-------------------------------------
            
            if not os.path.isfile(model_name):
                # Check whether we know what model definition file to use
                model_defs = CalcamConfig().get_cadmodels()
                if model_name not in model_defs.keys():
                    raise ValueError('Unknown machine model "{:s}". Available models are: {:s}.'.format(model_name,', '.join(model_defs.keys())))
                else:
                    definition_filename = model_defs[model_name][0]

                # If not specified, choose whatever model variant is specified in the metadata
                if model_variant is None:
                    self.model_variant = model_defs[model_name][2]
                else:
                    self.model_variant = model_variant

            
            else:
                definition_filename = model_name
                self.model_variant = model_variant
            

            if status_callback is not None:
                status_callback('Extracting CAD model...')


            # Open the definition file (ZIP file)
            try:
                self.def_file = ZipSaveFile(definition_filename,'rw')
            except:
                self.def_file = ZipSaveFile(definition_filename,'r')

            if status_callback is not None:
                status_callback(None)


            # Load the model definition and grab some properties from it
            with self.def_file.open_file( 'model.json','r' ) as f:
                model_def = json.load(f)

            self.model_def = model_def
            self.machine_name = model_def['machine_name']
            self.views = model_def['views']
            self.initial_view = model_def['initial_view']
            self.linewidth = 1
            self.mesh_path_roots = model_def['mesh_path_roots']

            if self.model_variant is None:
                self.model_variant = model_def['default_variant']

            self.variants = [str(x) for x in model_def['features'].keys()]

            # Validate the model variant input
            if self.model_variant not in self.variants:
                raise ValueError('Unknown model variant for {:s}: {:s}.'.format(self.machine_name,self.model_variant))

            # Check if the mesh files are from the CAD definition file itself, in which case
            # we need to point the mesh loader in the direction of out temporary extracted path
            if model_def['mesh_path_roots'][self.model_variant].startswith('.large'):
                self.mesh_path_root = os.path.join(self.def_file.get_temp_path(),model_def['mesh_path_roots'][self.model_variant])
            else:
                self.mesh_path_root = model_def['mesh_path_roots'][self.model_variant]


            # Load the wall contour, if present
            if 'wall_contour.txt' in self.def_file.list_contents():
                with self.def_file.open_file('wall_contour.txt','r') as cf:
                    self.wall_contour = np.loadtxt(cf)
            else:
                self.wall_contour = None

            
            # See if we have a user-written coordinate formatter, and if
            # we do, load it over the standard format_coord method
            self.usermodule = None
            usermodule = self.def_file.get_usercode()
            if usermodule is not None:
                if callable(usermodule.format_coord):
                    # Check the user function returns a string as expected, and use
                    # it only if it does.
                    try:
                        test_out = usermodule.format_coord( (0.1,0.1,0.1) )
                    except Exception as e:
                        self.def_file.close()
                        raise

                    if type(test_out) == str or type(test_out) == unicode:
                        self.usermodule = usermodule
                    else:
                        self.def_file.close()
                        raise Exception('CAD model user function format_coord() did not return a string as required.')


            # Create the features!
            self.features = {}
            self.groups = {}

            for feature_name,feature_def in model_def['features'][self.model_variant].items():

                # Get the feature's group, if any
                if len(feature_name.split('/')) > 1:
                    group = feature_name.split('/')[0]
                    if group not in self.groups.keys():
                        self.groups[group] = [feature_name]
                    else:
                        self.groups[group].append(feature_name)

                # Actually make the feature object
                self.features[feature_name] = ModelFeature(self,feature_def)
            
            # ----------------------------------------------------------------------------------------------

        else:
            self.variants = []
            self.model_def = {}
            self.groups = {}
            self.features = {}
            self.usermodule = None
            self.wall_contour = None
            self.def_file = None
            self.mesh_path_root = ''
            self.machine_name = ''
            self.model_variant = ''
            self.views = {}
            self.initial_view = None
            self.mesh_path_roots = {}

        self.renderers = []
        self.flat_shading = False
        self.edges = False
        self.cell_locator = None
        self.discard_changes = False

        self.set_status_callback(status_callback)
        atexit.register(self.unload)


    def set_status_callback(self,status_callback):
        '''
        Set the status callback function.

        The given function will be called with a string when the CAD model
        object does something, and is called with None when the operatio is
        finished. If set to None, the object will provide no output when it is
        busy doing things.

        Parameters:

            status_callback (fun): Status callback function.
        '''
        self.status_callback = status_callback


    def get_status_callback(self):
        '''
        Get the current callback function

        Returns:

            func or NoneType : Current status callback function, if present.
        '''
        return self.status_callback




    def add_to_renderer(self,renderer):
        '''
        Add the CAD model to a VTK renderer.

        Parameters:

            renderer (vtk.vtkRenderer) : Renderer to add the model to.
        '''
        if renderer in self.renderers:
            return

        else:
            for feature in self.features.values():
                actors = feature.get_vtk_actors()
                for actor in actors:
                    renderer.AddActor(actor)

            self.renderers.append(renderer)



    def remove_from_renderer(self,renderer):
        '''
        Remove the CAD model from the given VTK renderer.

        Parameters:

            renderer (vtk.vtkRenderer) : Renderer to remove the model from.
        '''
        if renderer not in self.renderers:
            return

        else:
            for feature in self.features.values():
                actors = feature.get_vtk_actors()
                for actor in actors:
                    renderer.RemoveActor(actor)

            self.renderers.remove(renderer)



    def set_features_enabled(self,enable,features=None):
        '''
        Enable or disable parts of the CAD model.

        Parameters:

            enable (bool)                   : Whether to set the relevant features \
                                              as enabeled (True) or disabled (False).
            features (str or list of str)   : Name(s) of the feature(s) and/or group(s) \
                                              of features of  to enable or disable. \
                                              If not specified, applies to all features \
                                              in the model.
        '''
        if features is None:
            features = self.features.keys()
        elif type(features) is not list:
            features = [features]


        for requested in features:

            if requested in self.groups.keys():
                for fname in self.groups[requested]:
                    self.features[fname].set_enabled(enable)
            elif requested in self.features.keys():
                self.features[requested].set_enabled(enable)
            else:
                raise ValueError('Unknown feature "{:s}"!'.format(requested))

        self.cell_locator = None



    def enable_only(self,features):
        '''
        Disable all model parts except those specified.

        Parameters:

            features (str or list of str) : Name(s) of the feature(s) and/or \
                                            group(s) to have enabled.
        '''
        if type(features) is not list:
            features = [features]

        self.set_features_enabled(False)
        for requested in features:
            if requested in self.groups.keys():
                for fname in self.groups[requested]:
                    self.features[fname].set_enabled(True)
            elif requested in self.features.keys():
                self.features[requested].set_enabled(True)
            else:
                raise ValueError('Unknown feature "{:s}"!'.format(requested))

        self.cell_locator = None



    def get_enabled_features(self):
        '''
        Get a list of the currently enabled features.

        Returns:

            list of str : Names of the currently enabled features.
        '''
        flist = []
        for fname,fobj in self.features.items():
            if fobj.enabled:
                flist.append(fname)

        return sorted(flist)



    def get_group_enable_state(self,group=None):
        '''
        Check the enable status of a named group of features.

        Parameters:

            group (str) : Name of the group to check. If not given, \
                          the entire model is taken to be the group.

        Returns:

            int         : 0 if no features in the group are anebled; \
                          1 if some features in the group are enabled; \
                          2 if all features in the group are enabled.
        '''
        if group is None:
            flist = self.features.keys()
        else:
            flist = self.groups[group]

        enable_state = 0
        for fname in flist:
            enable_state = enable_state + self.features[fname].enabled

        if enable_state == len(flist):
            enable_state = 2
        elif enable_state > 0:
            enable_state = 1

        return enable_state




    def format_coord(self,coords):
        '''
        Return a pretty string giving information about a specified 3D position.

        Parameters:

            coords (array-like) : 3 element array-like specifying a point in 3D \
                                  as X,Y,Z in metres.

        Returns:

            str                 : String containing information about the given point.
        '''

        if self.usermodule is not None:
            return self.usermodule.format_coord(coords)

        else:

            phi = np.arctan2(coords[1],coords[0])
            if phi < 0.:
                phi = phi + 2*3.14159
            phi = phi / 3.14159 * 180
            
            formatted_coord = 'X,Y,Z: ( {:.3f} m , {:.3f} m , {:.3f} m )'.format(coords[0],coords[1],coords[2])
            formatted_coord = formatted_coord + u'\nR,Z,\u03d5: ( {:.3f} m , {:.3f}m , {:.1f}\xb0 )'.format(np.sqrt(coords[0]**2 + coords[1]**2),coords[2],phi)

            return  formatted_coord
    



    def set_flat_shading(self,flat_shading):
        '''
        Set flat shading (no lighting applied to model rendering)
        enabled or disabled.

        Parameters:

            flat_shading (bool) : Whether to enable or disable flat shading.
        '''
        self.flat_shading = flat_shading
        if flat_shading != self.flat_shading:

            # Just running through each feature like this will force it to
            # update the colour & lighting settings
            for feature in self.features.values():
                feature.get_vtk_actors()




    def reset_colour(self,features=None):
        '''
        Reset the colour of part or all of the CAD model to the default(s).

        Parameters:

            features (list of str) : List of features for which to reset the colours. \
                                     If not given, all features will have their colours reset.
        '''
        if features is None:
            features = self.get_feature_list()

        for feature in features:
            if feature in self.groups.keys():
                for fname in self.groups[feature]:
                    self.features[fname].set_colour(self.features[fname].default_colour)
            elif feature in self.features.keys():
                self.features[feature].set_colour(self.features[feature].default_colour)
            else:
                raise ValueError('Unknown feature "{:s}"!'.format(requested))



    def set_colour(self,colour,features=None):
        '''
        Set the colour of part or all of the CAD model.

        Parameters:

            colour (tuple)          : 3-element tuple specifying a colour in (R,G,B) \
                                      where the values are in the range 0 to 1.
            features (list of str)  : List of names of the features to set this colour. \
                                      If not specified, applies to the whole model.
        '''
        if features is None:
            features = self.get_feature_list()

        try:
            0. + colour[0]
            colour = [colour] * len(features)
        except:
            if len(colour) != len(features):
                raise ValueError('The same number of colours and features must be provided!')

        for i,requested in enumerate(features):

            if requested in self.groups.keys():
                for fname in self.groups[requested]:
                    self.features[fname].set_colour(colour[i])
            elif requested in self.features.keys():
                self.features[requested].set_colour(colour[i])
            else:
                raise ValueError('Unknown feature "{:s}"!'.format(requested))



    def get_colour(self,features = None):
        '''
        Get the current colour of part or all of the CAD model.

        Parameters:

            features (list of str)  : List of names of the features to get the colour for. \
                                      If not specified, all feature colours are returned.

        Returns:

            List                    : List of 3 element tuples specifying the colours (R,G,B) \
                                      of the given features, where R, G and B range from 0 to 1.
        '''
        clist = []
        if features is None:
            features = self.get_feature_list()

        for feature in features:
            if feature in self.groups.keys():
                for fname in self.groups[feature]:
                    clist.append( self.features[fname].colour )
            elif feature in self.features.keys():
                clist.append( self.features[feature].colour )
            else:
                raise ValueError('Unknown feature "{:s}"!'.format(feature))
            
        return clist


    def get_linewidth(self,features=None):
        '''
        Get the line width used for rendering the model as wireframe.

        Parameters:

            features (list of str)  : List of names of the features to get the line width for. \
                                      If not specified, all feature line widths are returned.

        Returns:

            list of float           : List of the line widths.
        '''
        wlist = []
        if features is None:
            features = self.get_feature_list()

        for feature in features:
            if feature in self.groups.keys():
                for fname in self.groups[feature]:
                    wlist.append( self.features[fname].linewidth )
            elif feature in self.features.keys():
                wlist.append( self.features[feature].linewidth )
            else:
                raise ValueError('Unknown feature "{:s}"!'.format(feature))
            
        return wlist       



    def set_linewidth(self,linewidth,features=None):
        '''
        Set the line width used when rendering the CAD model as wireframe.

        Parameters:

            linewidth (float)       : Line width.
            features (list of str)  : List of names of the features to set the line \
                                      width for. If not specified, applies to the whole model.
        '''
        if features is None:
            features = self.get_feature_list()

        try:
            0 + linewidth
            linewidth = [linewidth] * len(features)
        except:
            if len(linewidth) != len(features):
                raise ValueError('The same number of line widths and features must be provided!')

        for i,requested in enumerate(features):

            if requested in self.groups.keys():
                for fname in self.groups[requested]:
                    self.features[fname].set_linewidth(linewidth[i])
            elif requested in self.features.keys():
                self.features[requested].set_linewidth(linewidth[i])
            else:
                raise ValueError('Unknown feature "{:s}"!'.format(requested))




    def get_cell_locator(self):
        '''
        Get a vtkCellLocator object used for ray casting.

        Returns:

            vtk.vtkCellLocator : VTK cell locator.
        '''

        # Don't return anything if we have no enabled geometry
        if len(self.get_enabled_features()) == 0:
            return None

        if self.cell_locator is None:

            appender = vtk.vtkAppendPolyData()

            for fname in self.get_enabled_features():
                appender.AddInputData(self.features[fname].get_polydata())
        
            appender.Update()

            self.cell_locator = vtk.vtkCellLocator()
            self.cell_locator.SetTolerance(1e-6)
            self.cell_locator.SetDataSet(appender.GetOutput())
            self.cell_locator.BuildLocator()

        return self.cell_locator



    def set_wireframe(self,wireframe):
        '''
        Enable or disable rendering the model as wireframe.

        Parameters:

            wireframe (bool) : Whether to render as wireframe.
        '''
        enable_features = self.get_enabled_features()

        for feature in enable_features:
            self.features[feature].set_enabled(False)

        self.edges = wireframe

        for feature in enable_features:
            self.features[feature].set_enabled(True)



    def get_feature_list(self):
        '''
        Get a list of the names of all features
        which constitute the model.

        Returns:

            list of str: List of feature names.
        '''
        return sorted(self.features.keys())



    def get_extent(self):
        '''
        Get the extent of the model in 3D space.

        Returns:

            np.array : 6 element array specifying the extent of the model in metres: \
                       [x_min, x_max, y_min, y_max, z_min, z_max]
        '''
        model_extent = np.zeros(6)

        for fname in self.get_enabled_features():
            feature_extent = self.features[fname].get_polydata().GetBounds()
            model_extent[::2] = np.minimum(model_extent[::2],feature_extent[::2])
            model_extent[1::2] = np.maximum(model_extent[1::2],feature_extent[1::2])

        return model_extent



    def get_view_names(self):
        '''
        Get a list of the views configured in the model.

        Returns:
            list of str: Names of the views configured in the model.
        '''
        return sorted(self.views.keys())



    def get_view(self,view_name):
        '''
        Get a dictionary describing one of the 
        views configured in the model.

        Parameters:

            view_name (str) : Name of the view to retrieve.

        Returns:

            dict            : Dictionary describing the view geometry.
        '''
        return self.views[view_name.replace('*','')]



    def __str__(self):
        '''
        Make print() do something useful for CAD model objects.

        Returns:

            str : String specifying the model machine name and model file path.
        '''
        return 'Calcam CAD model: "{:s}" / "{:s}" from {:s}'.format(self.machine_name,self.model_variant,self.definition_filename)



    def add_view(self,viewname,campos,camtar,fov,xsection,roll,projection):
        '''
        Add a specified camera view to the model's pre-defined views.

        Parameters:
    
            viewname (str)          : A name for the added view.
            campos (array-like)     : 3-element array-like specifying the camera position (X,Y,Z) in metres.
            camtar (array-like)     : 3-element array-like specifying a 3D point at which the camera is pointing.
            fov (float)             : Vertical field-of-view of the camera.
            xsection (array-like)   : A 3D point through which the model will be cross-sectioned. If set to None, \
                                      the model is not cross-sectioned.
            roll (float)            : Camera roll in degrees. This is the angle between the model's +Z direction \
                                      and the camera view up direction. Positie values indicate an anti-clockwise \
                                      roll of the camera.
            projection (str)        : Either ``perspective`` or ``orthographic``, what camera projection to use.
        '''
        self.views[viewname] = {'cam_pos':campos,'target':camtar,'y_fov':fov,'xsection':xsection,'roll':roll,'projection':projection}
        self.model_def['views'] = self.views
        if not self.discard_changes:
            self.update_definition_file()


    def set_default_colour(self,colour,features=None):
        '''
        Set the default colour of some or all of the model.

        Parameters:

            colour (tuple)          : 3-element tuple specifying an (R,G,B) colour where \
                                      R, G and B range from 0 to 1.
            features (list of str)  : List of feature names to which to apply the colour. \
                                      If not given, applies to the entire model.
        '''
        if features is None:
            features = self.get_feature_list()

        try:
            0. + colour[0]
            colour = [colour] * len(features)
        except:
            if len(colour) != len(features):
                raise ValueError('The same number of colours and features must be provided!')

        for i,requested in enumerate(features):

            if requested in self.groups.keys():
                for fname in self.groups[requested]:
                    self.features[fname].default_colour = colour[i]
                    self.model_def['features'][self.model_variant][fname]['colour'] = colour[i]
            elif requested in self.features.keys():
                self.features[requested].default_colour = colour[i]
                self.model_def['features'][self.model_variant][requested]['colour'] = colour[i]
            else:
                raise ValueError('Unknown feature "{:s}"!'.format(requested))

        self.update_definition_file()


    def unload(self):
        '''
        Unloads the CAD model object.
        '''
        if self.status_callback is not None:
            self.status_callback('Closing model definition {:s}/{:s}...'.format(self.machine_name,self.model_variant))

        if self.def_file is not None:
            temp_dir = self.def_file.get_temp_path()
            try:
                self.def_file.close(discard_changes=self.discard_changes)
            except:
                if self.status_callback is not None:
                    self.status_callback('WARNING: CAD model definition {:s}/{:s} could not be closed cleanly. There may be temporary files left in {:s}.'.format(self.machine_name,self.model_variant,temp_dir))
                #raise UserWarning('CAD model definition file could not be closed cleanly. There may be temporary files left in {:s}'.format(temp_dir))

        if self.status_callback is not None:
            self.status_callback(None)


    def update_definition_file(self):
        '''
        Update the CAD definition on disk with any
        changes to views or colours.
        '''
        try:

            with self.def_file.open_file( 'model.json','w' ) as f:
                json.dump(self.model_def,f,indent=4,sort_keys=True)
        
        except Exception as e:
            raise UserWarning('Cannot write changesto the model dfinition file ({:s}). The changes will only persist until this CAD model instance is unloaded.'.format(str(e)))        


# Class to represent a single CAD model feature.
# Does various grunt work and keeps the code nice and modular.
class ModelFeature():

    # Initialise with the parent CAD mdel and a dictionary
    # defining the feature
    def __init__(self,parent,definition_dict,abs_path=False):

        self.parent = parent

        if abs_path:
            self.filename = definition_dict['mesh_file']
        else:
            self.filename = os.path.join(self.parent.mesh_path_root,definition_dict['mesh_file'])

        if not os.path.isfile(self.filename):
            raise IOError('CAD mesh file {:s} not found.'.format(self.filename))

        self.filetype = self.filename.split('.')[-1].lower()

        self.enabled = definition_dict['default_enable']

        self.scale = definition_dict['mesh_scale']

        self.polydata = None
        self.solid_actor = None
        self.edge_actor = None

        self.default_colour = definition_dict['colour']
        self.colour = self.default_colour
        self.linewidth = 1

    # Get a vtkPolyData object for this
    def get_polydata(self):

        if not self.enabled:
            return None

        if self.polydata is None:

            if self.parent.status_callback is not None:
                self.parent.status_callback('Loading mesh file: {:s}...'.format(os.path.split(self.filename)[1]))

            if self.filetype == 'stl':
                reader = vtk.vtkSTLReader()
            elif self.filetype == 'obj':
                reader = vtk.vtkOBJReader()

            reader.SetFileName(self.filename)
            reader.Update()

            scaler = vtk.vtkTransformPolyDataFilter()

            scale_transform = vtk.vtkTransform()
            scale_transform.Scale(self.scale,self.scale,self.scale)


            scaler.SetInputData(reader.GetOutput())
            scaler.SetTransform(scale_transform)
            scaler.Update()

            self.polydata = scaler.GetOutput()

            if self.parent.status_callback is not None:
                self.parent.status_callback(None)

        return self.polydata


    # Enable or disable the feature
    def set_enabled(self,enable):

        if enable and not self.enabled:

            self.enabled = True

            for renderer in self.parent.renderers:
                for actor in self.get_vtk_actors():
                    renderer.AddActor(actor)

        elif self.enabled and not enable:
            for renderer in self.parent.renderers:
                for actor in self.get_vtk_actors():
                    renderer.RemoveActor(actor)

            self.enabled = False       


    # Get vtkActor object(s) for this feature
    def get_vtk_actors(self):

        if not self.enabled:

            return []
        
        else:
            
            if self.solid_actor is None:

                mapper =  vtk.vtkPolyDataMapper()
                mapper.SetInputData( self.get_polydata() )

                self.solid_actor = vtk.vtkActor()
                self.solid_actor.SetMapper(mapper)


            # Make the edge actor if it doesn't already exist and is needed
            if self.parent.edges and self.edge_actor is None:

                if self.parent.status_callback is not None:
                    self.parent.status_callback('Detecting mesh edges...')

                edge_finder = vtk.vtkFeatureEdges()

                edge_finder.SetInputData( self.get_polydata() )

                edge_finder.ManifoldEdgesOff()
                edge_finder.BoundaryEdgesOff()
                edge_finder.NonManifoldEdgesOff()
                edge_finder.SetFeatureAngle(20)
                edge_finder.ColoringOff()
                edge_finder.Update()

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(edge_finder.GetOutputPort())

                self.edge_actor = vtk.vtkActor()
                self.edge_actor.SetMapper(mapper)
                
                self.edge_actor.GetProperty().SetLineWidth(self.linewidth)
            
                if self.parent.status_callback is not None:
                    self.parent.status_callback(None)

            # Make sure the colour and lighing are set appropriately
            if self.parent.edges:
                self.solid_actor.GetProperty().SetColor((0,0,0))
                self.edge_actor.GetProperty().SetColor(self.colour)
            else:
                self.solid_actor.GetProperty().SetColor(self.colour)

                if self.parent.flat_shading:
                   self.solid_actor.GetProperty().LightingOff()

            
            if self.parent.edges:
                return [self.solid_actor,self.edge_actor]
            else:
                return [self.solid_actor]


    def set_linewidth(self,linewidth):
        self.linewidth = linewidth
        if self.edge_actor is not None:
            self.edge_actor.GetProperty().SetLineWidth(linewidth)


    # Set the colour of the feature
    def set_colour(self,colour):

        self.colour = colour
        if self.parent.edges:
            if self.edge_actor is not None:
                self.edge_actor.GetProperty().SetColor(colour)
        else:
            if self.solid_actor is not None:
                self.solid_actor.GetProperty().SetColor(colour)