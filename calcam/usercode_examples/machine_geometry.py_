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
This is a (ficticious) example of how to define a CAD model for use in Calcam.
CAD models are defined by writing a python class for each machine.

Model definitions should be placed in ~/calcam/UserCode/machine_geometry ,
with any filename except 'Example.py' (which is ignored by the code)
"""

# We have to import the Calcam CAD model class.
from cadmodel import CADModel

# Start of the CAD model definition - we define a subclass of CADModel.
# The class name is up to the user, and not used in the CalCam GUI, only if you want to use this model programatically.
# e.g. in this example we would create an instance of this model using "model = calcam.machine_geometry.mr_fusion()"
class mr_fusion(CADModel):

    # The __init__ function is the only one required to define the cad model. 
    # It must take an optional keyword argument "model_variant" which has a working default value.
    # In here we set all the attributes we need to define the model.
    def __init__(self,model_variant='Medium'):

        # Machine name, this is used in the Calcam GUI and save files.
        self.machine_name = "Mr Fusion"

        # Names for the possible variants of the model e.g. different detail levels, variations of machine configuration, etc.
        # These are used in the calcam GUI.
        self.model_variants = ['1955 Model','1985 Model','2015 Model']

        # Check the model_variant input argument is one of the ones we just listed.
        # You shouldn't need to edit this. It's only up here because, for exmple, later on the list of CAD filenames etc
        # could depend on model variant if you wanted, so it's nice to validate it beforehand.
        if model_variant not in self.model_variants:
            raise ValueError('Unknown ' + self.machine_name + ' model variant "' + model_variant + '"')
        self.model_variant = model_variant

        # Default colour to render the model in the format (R,G,B), where RGB go from 0 to 1.
        self.default_colour = (0.8,0.8,0.8)

        # Whether to use, by default, different colours for different materials
        self.colourbymaterial = False

        # This is the maximum possible sight line length, in metres, used when casting sight lines.
        # It must be long enough to cover the entire span of the machine.
        self.max_ray_length = 1.2
        
        # Path to the CAD files. Here we imagine having a folder structure organised by model variant name.
        # The easiest way to put paths under Windows is to replace the usual backslashes with forward slashes.
        # It doesn't matter whether this has the end slash or not.
        self.filepath = "/home/docbrown/mr_fusion_cad/" + model_variant
        
        # Here we specify names of all the materials the machine is made of,
        # and what colour they should be if we turn on colour by material.
        # (again colours are in (R,G,B) from 0 to 1.). We'll need to index
        # in to this list later.
        self.materials = [
            ("White Plastic", (0.9,0.9,0.9) ),      # White plastic will be index 0
            ("N-BK7" , (0.5,0.5,0.5) ),             # N-BK7 will be index 1
            ("Black Plastic" , (0.05,0.05,0.05) ),  # Black plastic will be index 2
        ]

        # Here we specify the actual CAD file names and what is in each one.
        # This is set up in such a way that models split in to different parts and in to groups of parts can be handled nicely.
        # Each component's list goes like [str "Feature name", str "CAD filename" , int Material (index in to materials list), str Group (Name of sub-group), bool Enabled by default]
        self.features = [
                            ["Main Lid" , "lid.stl" , 0 , 'Top part' , True],             # The main lid is made of white plastic, and is visible by default.
                            ["Viewing Window" , "window.stl" , 1 , 'Top part' , True],    # Viewing window made of N-BK7. Both the viewing window and main lid are part of a group called 'Top part'
                            ["Bottom Port" , "bottom_bit.stl" , 2 , None , True],         # The bottom part is made of black plastic, and is not part of any sub-group of components.
        ]

        # Here we define preset viewports for viewing the model.
        # E.g. Certain diagnostic ports, roughly known camera position etc.
        # At least 1 must be defined as a default starting view for the model.
        # Each view's tuple goes like (Name, Camera position in metres [x,y,z], Camera Viewing Target in metres [x,y,z], vertical FOV in degrees.)
        # All positions are in metres and field of view is degrees
        self.views = [
            ('Viewing Window', [0.3,0.25,0.15] , [0,0,-0.5] , 50),
        ]


        # Specify which of the above views should be used as the default.
        self.set_default_view('Viewing Window')


        # Units in which the CAD model is saved: real distance, in metres, corresponding to one unit of distance in the CAD model.
        # A value of 1.0 corresponds to the CAD model saved in metres, 0.001 means the CAD model is saved in millimetres,
        # 2.54e-2 would mean the CAD model was saved in inches (you monster!)
        self.units = 1.0


        # This call is required to run all the back end stuff to configure the CAD model based on these attributes.
        self.init_cadmodel()



    # Defining your own get_position_info() is entirely optional, you can delete this whole method if you want.
    # This is a function which takes a 3D position in space on the CAD model in the form of a 3 element
    # array or list [x,y,z], in metres, and returns a string with information about that position.
    # This string is displayed in the points tab of Calcam when you place a cursor on the CAD model.
    # If this is not present in this file, the default behaviour is to just show the coordinates X,Y,Z,R in metres.
    def get_position_info(self,coords):
        
        # Since Mr Fusion is a small device, let's say we want to display the coordinates in
        # centimeters instead of the default metres.
        coords = coords * 100
        out_str = 'X = ' + '{:.3f}'.format(coords[0]) + 'cm, Y = ' + '{:.3f}'.format(coords[1]) + 'cm, Z = ' + '{:.3f}'.format(coords[2]) + 'cm.'

        # Maybe we want the message to change if we're at a particular Z position
        if abs(coords[2]) => 88:
            out_str = 'Great Scott! \n' + out_str
        
        return out_str

  