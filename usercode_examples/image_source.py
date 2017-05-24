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
This is a working example of how to define an image source for Calcam.
Image sources are defined as a python class.

Image source definitions should be placed in ~/calcam/UserCode/image_sources/ ,
with any filename except 'Example.py' (which is ignored by the code)
"""

# We have to import the Calcam image source parent class and the function to
# create calcam image objects from image data.
from image import ImageSource
from image import from_array as image_from_array


# Whatever else you might want to import, in this example I use matplotlib and numpy.
import matplotlib.cm
import numpy as np


# The class name is up the user and is not used in the GUI, but  will be  
# how the image source is called within calcam programatically.
# e.g. for this example, to get a calcam image object from this source,
# the call would be MyImage = calcam.image.colour_gradient(...)
class colour_gradient(ImageSource):

    # In the __init__ function we tell the calcam GUI how to present this option
    # Nothing in here effects how the image source works when called programatically.
    def __init__(self):

        # Name displayed in the GUI for this image source
        self.gui_display_name = 'Colour gradient'

        # Next we specify the user inputs which are expected by the main get_image() function.
        # The format is a list of dictionaries, where each dictionary concerns one input.
        # The list must be in the order that the corresponding arguments are accepted by get_image().
        # This is also the order they will apear in the GUI.
        # Dictionary fields are:
        #   'label' - a string, what to call the input parameter in the Calcam GUI
        #   'type' - a string, what type of input it is. This can be 'string' , 'float' , 'int', 'filename', 'bool' or 'choice'
        #   'limits' - a two element list [lower, upper] - for int or float inputs, the allowed numerical range (if not specified, will be 0 - 64)
        #   'choices' - for 'choice' type inputs, a list of strings for the user to choose from.
        #   'default' - default value for that input. This is optional.
        #	'decimals' - for float input types, specifies how many decimal places.
        self.gui_inputs = [
                            {'label': 'Colourmap name' , 'type': 'choice' , 'choices':  ['brg','gray','hot','flag','jet'] , 'default': 'jet'} ,
                            {'label': 'Width' , 'type': 'int' , 'limits':  [0, 4096] , 'default': 640} ,
                            {'label': 'Height' , 'type': 'int' , 'limits':  [0, 4096] , 'default': 480} ,
                            {'label': 'Invert' , 'type': 'bool' , 'default': False} ,
                            {'label': 'Image Name' , 'type': 'string'} ,
                          ]

        # This call to the parent class is always required to do any back-end setup stuff.
        ImageSource.__init__(self)


    # Now we define the actual function for getting an image. This is
    # what will be called when doing MyImage = calcam.image.your_class_name()
    # Input arguments must be in the same order as in the above GUI input definitions.
    def get_image(self,cmap_name='jet',w=640,h=480,invert=False,im_name=''):


        # Error handling: if you want to show the user an error message(e.g. bad input parameters,
        # can't find a file, etc) - raise a UserWarning and Calcam will show a friendly dialog box.
        # In this example, we require the user to set their own image name, so we check if they have provided one.
        if im_name == '':
            raise UserWarning('You must specify an image name to generate an example image.')

        # Some code to generate a colour gradient image with the specified dimensions and colourmap.
        cmap  = matplotlib.cm.get_cmap(cmap_name)
        yvals = np.linspace(0,255,h)
        colours = cmap(yvals.astype(int)) * 255

        if invert:
            colours = 255 - colours

        im = np.zeros([h,w,3],dtype=np.uint8)
        im[:,:,0] = np.tile(colours[:,0],[w,1]).T
        im[:,:,1] = np.tile(colours[:,1],[w,1]).T
        im[:,:,2] = np.tile(colours[:,2],[w,1]).T


        # The return of get_image() must be the retults of image_from_array() called on our image data. 
        # Inputs for image_from_array():
        #   im - a 2D (for monochrome) or 3D (for RGB or RGBA) numpy array. The dtype and data range can be basically anything you want.
        # Optional keyword input arguments if you want to provide extra information about the image:
        #   image_name - A name for the image. This is highly recommended, if not set it will just default to 'New Image'
        #   pixel_aspectratio - pixel aspect ratio (height/width) for cameras with non-square pixels. Defaults to 1 which should be fine for almost all images.
        #   transform_actions - list of actions to transform the image from original to display coordinates, if known. See documentation for format.
        #   pixel_size - pixel size in microns, if known.
        return image_from_array(im,image_name=im_name)