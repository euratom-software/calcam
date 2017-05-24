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
CAD model class for CalCam

This class is for storing CAD models, and provides methods to work with CAD data 
and interface with other parts of CalCam. 

Actual CAD model definitions, i.e. for specific machines, are defined by sub-classing
this in the file ~/calcam/UserCode/_machine_geometry.py; this file is just back-end stuff.

At some point I need to add support for file types other than STL.

Written by Scott Silburn
"""

import vtk
import os
import sys
import numpy as np

class CADModel:

    """
    Initialise self.features, which is a list containing information about the CAD model features.
    List indices are:
    0 = Feature name (str)
    1 = CAD Model filename (str)
    2 = Material index (int)
    3 = Enabled (bool)
    4 = Colour (RGB tuple)
    5 = vtkPolyData
    6 = vtkPolyDataMapper
    7 = vtkActor
    8 = Group name
    """
    def init_cadmodel(self):

        # By default, we want lighting to work
        self.flatshading = False

        self.colourbyfeature = False

        self.obb_tree = None
        
        self.edge_actor = [None,[]]

        self.gui_window = None

        for Feature in self.features:

            # Check if the files exist
            if not os.path.isfile(self.filepath + Feature[1]):
                raise Exception('Cannot find CAD file ' + str(self.filepath + Feature[1]))

            # Extract group name and default enable status
            if len(Feature) == 4:
                enable = True
                group = None
            else:
                enable = Feature[4]
                group = Feature[3]
                del Feature[3]
                del Feature[3]

            # Enable feature, if this is the default case
            Feature.append(enable)

            # Feature colour
            if self.colourbymaterial:
                Feature.append(self.Materials[Feature[2]])
            else:
                Feature.append(self.default_colour)

            # This is where the vtk objects will go
            Feature.extend([None,None,None,group])


    # Make an OBBTree object to do ray casting with this CAD model
    def get_obb_tree(self):

        if self.obb_tree is None:
            self.obb_tree = vtk.vtkOBBTree()
            self.obb_tree.SetTolerance(1e-6)

            self.obb_tree.SetDataSet(self.get_combined_geometry())
            self.obb_tree.BuildLocator()


        return self.obb_tree

    # Load CAD files from disk
    def load(self,features=None):

        ScaleTransform =  vtk.vtkTransform()
        ScaleTransform.Scale(self.units,self.units,self.units)

        # If no specific feature is specified, load them all!
        if features is None:
            for Feature in self.features:
                # If no vtkPolyData for this feature, load the CAD file!
                if Feature[5] is None and Feature[3] == True:
                    if self.gui_window is not None:
                        self.gui_window.update_cad_status('Loading CAD data file ' + str(Feature[1]) + '...')
                    PolyData = load_cad(self.filepath + Feature[1])
                    Feature[5] = vtk.vtkTransformPolyDataFilter()
                    if vtk.vtkVersion.GetVTKMajorVersion() < 6:
                            Feature[5].SetInput(PolyData)
                    else:
                            Feature[5].SetInputData(PolyData)
                    Feature[5].SetTransform(ScaleTransform)
                    Feature[5].Update()
                    if self.gui_window is not None:
                        self.gui_window.update_cad_status(None)
                    

        else:
            if type(features) == str:
                features = [features]
            for i in range(len(features)):
                features[i] = str.lower(features[i])

                for Feature in self.features:
                    if str.lower(Feature[0]) in features:
                        if self.gui_window is not None:
                            self.gui_window.update_cad_status('Loading CAD data file ' + str(Feature[1]) + '...')
                        PolyData = load_cad(self.filepath + Feature[1])
                        Feature[5] = vtk.vtkTransformPolyDataFilter()
                        if vtk.vtkVersion.GetVTKMajorVersion() < 6:
                                Feature[5].SetInput(PolyData)
                        else:
                                Feature[5].SetInputData(PolyData)
                        Feature[5].SetTransform(ScaleTransform)
                        Feature[5].Update()
                        if self.gui_window is not None:
                            self.gui_window.update_cad_status(None)


    # Enable features
    def enable_features(self,features,renderer=None):

        if type(features) == str:
            features = [features]
        for i in range(len(features)):
            features[i] = str.lower(features[i])

            for Feature in self.features:
                if str.lower(Feature[0]) in features:
                    Feature[3] = True
                    if renderer is not None:
                        renderer.AddActor(self.get_vtkActors(Feature[0]))


    def enable_only(self,features,renderer=None):
        
        if type(features) == str:
            features = [features]
        for i in range(len(features)):
            features[i] = str.lower(features[i])

            for Feature in self.features:
                if str.lower(Feature[0]) in features:
                    self.enable_features(Feature[0],renderer)
                else:
                    self.disable_features(Feature[0],renderer)


    # Disable features
    def disable_features(self,features,renderer=None):
        if type(features) == str:
            features = [features]
        for i in range(len(features)):
            features[i] = str.lower(features[i])

            for Feature in self.features:
                if str.lower(Feature[0]) in features:
                    Feature[3] = False
                    if renderer is not None:
                        renderer.RemoveActor(self.get_vtkActors(Feature[0]))

    # Get list of vtkActors
    def get_vtkActors(self,features=None):

        Actors = []

        # If no specific features are given, do them all!
        if features is None:
            
            for Feature in self.features:
                # If the feature is enabled...
                if Feature[3] == True:
                    # If there's already an actor, just return that
                    if Feature[7] is not None:
                        Actors.append(Feature[7])
                    else:   
                        # Load the CAD file if it isn't already loaded
                        if Feature[5] is None:
                            self.load(Feature[0])

                        Feature[6] = vtk.vtkPolyDataMapper()
                        if vtk.VTK_MAJOR_VERSION <= 5:
                            Feature[6].SetInput(Feature[5].GetOutput())
                        else:
                            Feature[6].SetInputData(Feature[5].GetOutput())

                        Feature[7] = vtk.vtkActor()
                        Feature[7].SetMapper(Feature[6])
                        Feature[7].GetProperty().SetColor(Feature[4])

                        if self.flatshading:
                            Feature[7].GetProperty().LightingOff()

                        Actors.append(Feature[7])
        else:

            if type(features) == str:
                features = [features]

            for req_feature in features:

                for Feature in self.features:
                    if str.lower(req_feature) == str.lower(Feature[0]):
                        # If there's already an actor, just return that
                        if Feature[7] is not None:
                            Actors.append(Feature[7])
                        else:   
                            # Load the CAD file if it isn't already loaded
                            if Feature[5] is None:
                                self.load(Feature[0])

                            Feature[6] = vtk.vtkPolyDataMapper()
                            if vtk.VTK_MAJOR_VERSION <= 5:
                                Feature[6].SetInput(Feature[5].GetOutput())
                            else:
                                Feature[6].SetInputData(Feature[5].GetOutput())
                            Feature[7] = vtk.vtkActor()
                            Feature[7].SetMapper(Feature[6])
                            Feature[7].GetProperty().SetColor(Feature[4])

                            if self.flatshading:
                                Feature[7].GetProperty().LightingOff()

                            Actors.append(Feature[7])

            if len(Actors) == 1:
                Actors = Actors[0]

        return Actors


    # Get a single vtkPolyData object containing all loaded features
    def get_combined_geometry(self):

        appender = vtk.vtkAppendPolyData()

        for Feature in self.features:
                if Feature[3] == True:
                    if Feature[5] is None:
                        self.load(Feature[0])
                    if vtk.vtkVersion.GetVTKMajorVersion() < 6:
                        appender.AddInput(Feature[5].GetOutput())
                    else:
                        appender.AddInputData(Feature[5].GetOutput())
        
        appender.Update()

        return appender.GetOutput()


    # Default model views: these approximately match each JET camera view.
    def set_default_view(self,ViewName):

        foundview = False
        for View in self.views:
            if str.lower(ViewName) == str.lower(View[0]):
                foundview = True
                self.cam_pos_default = View[1]
                self.cam_target_default = View[2]
                self.cam_fov_default = View[3]
                self.default_view_name = View[0]
        if foundview == False:
            raise ValueError('Specified CAD view name "' + ViewName + '" not recognised!')


    # Set the colour of a component or the whole model
    def set_colour(self,Colour,features=None):

        if features is None:
            for Feature in self.features:
                Feature[4] = Colour
                if Feature[7] is not None:
                    Feature[7].GetProperty().SetColor(Colour)
        else:
            if type(features) == str:
                features = [features]

            if type(Colour[0]) is not tuple and type(Colour[0]) is not list :
                Colour = [Colour] * len(features)

            if len(Colour) != len(features):
                raise Exception('Different number of colours and feature names provided!')

            for i,req_feature in enumerate(features):
                for Feature in self.features:
                    if str.lower(req_feature) == str.lower(Feature[0]):
                        Feature[4] = Colour[i]
                        if Feature[7] is not None:
                            Feature[7].GetProperty().SetColor(Colour[i])

    # Set the colour of a component or the whole model
    def get_colour(self,features=None):

        cols_out = []

        if type(features) == str:
            features = [features]

        for req_feature in features:
            for Feature in self.features:
                if str.lower(req_feature) == str.lower(Feature[0]):
                    cols_out.append(Feature[4])

        if len(cols_out) == 1:
            return cols_out[0]
        else:
            return cols_out


    # Enable model colouring by material
    def colour_by_material(self,colourbymaterial):
        
        if colourbymaterial != self.colourbymaterial:
            if colourbymaterial:

                if self.colourbyfeature:
                    self.colour_by_feature(False)

                for Feature in self.features:
                    self.set_colour(self.materials[Feature[2]][1],features=Feature[0])

                self.colourbymaterial = True

            else:
                for Feature in self.features:
                    self.set_colour(self.default_colour,features=Feature[0])
                self.colourbymaterial = False


    def flat_shading(self,flatshading):

        if flatshading != self.flatshading:
            if flatshading:
                self.flatshading = True
                for Feature in self.features:
                    if Feature[7] is not None:
                        Feature[7].GetProperty().LightingOff()
            else:
                self.flatshading = False
                for Feature in self.features:
                    if Feature[7] is not None:
                        Feature[7].GetProperty().LightingOn()


    def colour_by_feature(self,colourbyfeature):

        if colourbyfeature != self.colourbyfeature:

            if colourbyfeature:
                if self.colourbymaterial:
                    self.colour_by_material(False)

                for featurenum,Feature in enumerate(self.features):
                    colour = np.unravel_index(featurenum + 1,[256,256,256])
                    R = colour[0]/255.
                    G = colour[1]/255.
                    B = colour[2]/255.
                    self.set_colour((R,G,B),features=Feature[0])
            else:

                for Feature in self.features:
                    self.set_colour(self.default_colour,features=Feature[0])


    # Default for getting some info to print
    # Just print the position.
    def get_position_info(self,coords):
        phi = np.arctan2(coords[1],coords[0])
        if phi < 0.:
            phi = phi + 2*3.14159
        phi = phi / 3.14159 * 180
        return 'X,Y,Z: ( {:.3f} m , {:.3f} m , {:.3f} m )'.format(coords[0],coords[1],coords[2]) + u'\nR,Z,\u03d5: ( {:.3f} m , {:.3f}m , {:.1f}\xb0 )'.format(np.sqrt(coords[0]**2 + coords[1]**2),coords[2],phi)
        
    
    
        
    def get_detected_edges_actor(self,exclude_features=[],include_only=[]):

        # Sort out any requested feature inclusion / exclusion
        full_featurelist = self.get_enabled_features()
        features = self.get_enabled_features()
        if len(exclude_features) > 0 and len(include_only) > 0:
            raise Exception('You can specify which features to exclude, or an include only list, but not both!')
        elif len(exclude_features) > 0:
            for rmfreature in exclude_features:
                features.remove(rmfeature)
        elif len(include_only) > 0:
            features_to_remove = []
            for feature in features:
                if feature.lower() not in include_only:
                    features_to_remove.append(feature)
            for rmfeature in features_to_remove:
                features.remove(rmfeature)

        if self.edge_actor[0] == None or features != self.edge_actor[1]:
            
            edgeFinder = vtk.vtkFeatureEdges()

            self.enable_only(features)
            if vtk.VTK_MAJOR_VERSION <= 5:
                edgeFinder.SetInput(self.get_combined_geometry())
            else:
                edgeFinder.SetInputData(self.get_combined_geometry())

            self.enable_only(full_featurelist)
            edgeFinder.ManifoldEdgesOff()
            edgeFinder.BoundaryEdgesOff()
            edgeFinder.NonManifoldEdgesOff()
            #edgeFinder.FeatureEdgesOff()
            edgeFinder.SetFeatureAngle(20)
            edgeFinder.ColoringOff()
            edgeFinder.Update()

            edgeMapper = vtk.vtkPolyDataMapper()
            edgeMapper.SetInputConnection(edgeFinder.GetOutputPort())

            self.edge_actor[0] = vtk.vtkActor()
            self.edge_actor[0].SetMapper(edgeMapper)
            
            self.edge_actor[1] = features        
            
        return self.edge_actor[0]
		
		
    def get_enabled_features(self):
	
        featurelist = []		
		
        for Feature in self.features:
            # If the feature is enabled...
            if Feature[3] == True:
                featurelist.append(Feature[0])
        
        return featurelist
		
		
    def link_gui_window(self,gui_window):

        self.gui_window = gui_window


def load_cad(fname):
    # Return a vtkPolyData object containing a mesh from a given filename
    if fname[-3:].lower() == 'stl':
        reader = vtk.vtkSTLReader()
    elif fname[-3:].lower() == 'obj':
        reader = vtk.vtkOBJReader()

    reader.SetFileName(fname)
    try:
        reader.Update()
    except:
        return None

    polydata = reader.GetOutput()

    return polydata
