=============
Image Sources
=============

Custom image soruces are a way to plug in custon code for loading images to Calcam, and having it integrate nicely in to the Calcam GUI. A custom image source takes the form of a python module or package. The module or package is required to have the following attributes at its top level:

* ``get_image_function`` : A function, or referece to a function, which actually loads the image data.
* ``display_name`` : A string specifying a user-friendly name for the image source. This will be used to denote the image source in the Calcam GUI.
* ``get_image_arguments`` : A list of dictionaries documenting the required inputs to the image loading function.


Return format of image loading function
----------------------------------------

The return type of the image loading dunction must be a Python dictionary. The following keys are mandatory:

============= ============== ================================================
Key           Data Type      Description
============= ============== ================================================
image_data    NumPy ndarray  Array containing the image data. 
------------- -------------- ------------------------------------------------
source        String         Description of where the image was loaded from.
============= ============== ================================================

The dictionary can also contain the following optional image metadata:

=================== ================== ============================================================
Key                 Data Type          Description
=================== ================== ============================================================
subview_mask        NumPy ndarray      | For images with multiple sub-views: array of
                                       | integers the same shape as the image
                                       | specifying which sub-view each image pixel belongs to.
------------------- ------------------ ------------------------------------------------------------
subview_names       List of strings    | For images with multiple sub-views: list of
                                       | strings containing user-friendly names for each sub-view
------------------- ------------------ ------------------------------------------------------------
transform_actions   List of strings    | List of strings specifying the geometrical transforms
                                       | required to convert the image from original to display
                                       | coordinates. Actions are performed from the beginning 
                                       | to the end of the list. Valid actions are:
                                       | flip_up_down
                                       | flip_left_right
                                       | rotate_clockwise_90
                                       | rotate_clockwise_180
                                       | rotate_clockwise_270
------------------- ------------------ ------------------------------------------------------------
pixel_size          float              | Physical height of the camera pixels, in metres
------------------- ------------------ ------------------------------------------------------------
coords              String             | Either Display or Original: whether the image is 
                                       | being returned in display or original orientation. 
                                       | If not provided, Display is assumed.
------------------- ------------------ ------------------------------------------------------------
pixel_aspect        float              | For cameras with non-square pixels, the pixel 
                                       | aspect ratio defined as pixel height/pixel width.
------------------- ------------------ ------------------------------------------------------------
image_offset        tuple              | For cameras where only a sub-region of the full detector
                                       | is read out or recorded, this specifies the (x,y) pixel
                                       | coordinates of the top-left pixel of the image, relative to
                                       | the full detector.
=================== ================== ============================================================


Structure of get_image_arguments
--------------------------------
The top-level module attribute ``get_image_arguments`` must be a list of dictionaries describing the required input parameters to the image loading function. Each dictionary in the list corresponds to one input arguument. All argument dictionaries must contain the following keys:

========= ========= ======================================================================
Key       Data Type Description
========= ========= ======================================================================
arg_name  String    Name of the argument as written in the function definition. 
--------- --------- ----------------------------------------------------------------------
gui_label String    User-friendly argument name which will be displayed in the GUI.
--------- --------- ----------------------------------------------------------------------
type      String    String specifying the type of input argument. See description below.
========= ========= ======================================================================

Additional keys are required depending on the type of variable. 

Input argument type: filename
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If the inpit variable type is ``filename``, the user will be presented with a text box and "Browse..." button to browse for the file. Additional keys in the dictionary describing the parameter are:

========== ========= ======================================================================
Key        Data Type Description
========== ========= ======================================================================
**filter**  String    | File name filter to be used in the file browsing window. 
                      | Format: "File type description (\*.ext1, \*.ext2)"
---------- --------- ----------------------------------------------------------------------
default     String    | Default value.
========== ========= ======================================================================

Input argument type: float
^^^^^^^^^^^^^^^^^^^^^^^^^^
If the inpit variable type is ``float``, the user will be presented with a box in which to input the value. Additional keys in the dictionary describing the parameter are:

========= ========== ======================================================================
Key       Data Type  Description
========= ========== ======================================================================
limits    array-like | 2 element array-like specifying the lower and upper limits 
                     | of acceptable values for the argument.
--------- ---------- ----------------------------------------------------------------------
default   float      | Default value. 
--------- ---------- ----------------------------------------------------------------------
decimals  int        | Number of decimal places to be shown in the user input box.
========= ========== ======================================================================

Input argument type: int
^^^^^^^^^^^^^^^^^^^^^^^^^^
If the inpit variable type is ``int``, the user will be presented with a box in which to input the value. Additional keys in the dictionary describing the parameter are:

========= ========== ======================================================================
Key       Data Type  Description
========= ========== ======================================================================
limits    array-like | 2 element array-like specifying the lower and upper limits 
                     | of acceptable values for the argument.
--------- ---------- ----------------------------------------------------------------------
default   int        | Default value. 
========= ========== ======================================================================

Input argument type: string
^^^^^^^^^^^^^^^^^^^^^^^^^^^
If the inpit variable type is ``string``, the user will be presented with a single line box in which which to input the string. Additional keys in the dictionary describing the parameter are:

========= ========== ======================================================================
Key       Data Type  Description
========= ========== ======================================================================
default   string     | Default value.
========= ========== ======================================================================

Input argument type: bool
^^^^^^^^^^^^^^^^^^^^^^^^^^
If the inpit variable type is ``bool``, the user will be presented with a checkbox to set the value true or false. Additional keys in the dictionary describing the parameter are:

========= ========== ======================================================================
Key       Data Type  Description
========= ========== ======================================================================
default   bool       | Default value. 
========= ========== ======================================================================

Input argument type: choice
^^^^^^^^^^^^^^^^^^^^^^^^^^^
If the inpit variable type is ``choice``, the user will be presented with a dropdown box from whih they can choose a number of strings. The value passed to the image loading function is whatever string the user selects. Additional keys in the dictionary describing the parameter are:

=========== =========== ======================================================================
Key         Data Type   Description
=========== =========== ======================================================================
**choices** list of str | List of strings from which the user should choose.
----------- ----------- ----------------------------------------------------------------------
default     string      | Default value. 
=========== =========== ======================================================================


Example
-------

Below is a minimal example: the build-in Calcam image source for loading images from standard format image files.

.. code-block:: python

   '''
   Built-in Calcam image source for loading images from an image file.

   Loads images using OpenCV.
   '''

   import cv2
   import os

   # The function which loads the image
   def get_image(filename):
      
       # Get the image data from a file
       dat = cv2.imread(filename)
       if dat is None:
           raise UserWarning('Could not read specified image file "' + filename + '"')

       # If it's a colour image, note OpenCV loads images in BGR.
       # Here we change the channel order to RGB.
       if len(dat.shape) == 3:

           if dat.shape[2] == 3:
               dat[:,:,:3] = dat[:,:,2::-1]

               # If R, G and B channels are all the same, just return a monochrome image
               if (dat[:,:,0] == dat[:,:,1]).all() and (dat[:,:,0] == dat[:,:,2]).all():
                   dat = dat[:,:,0] 


       # Minimal return dictionary.
       return_dict = {
                       'image_data': dat,
                       'source': 'Loaded from image file {:s}'.format(os.path.split(filename)[-1])
                       }

       return return_dict


   # Display name
   display_name = 'Image File'

   # Point calcam to the get_image function
   get_image_function = get_image

   # Description of input parameters to get_image
   get_image_arguments =  [
                               {
                               'arg_name':'filename',
                               'gui_label': 'File Name' , 
                               'type': 'filename' , 
                               'filter':'Image Files (*.png *.jpg *.jpeg *.bmp *.jp2 *.tiff *.tif)' 
                               },
                           ]

Adding to Calcam
----------------
Once written, custom image sources are added to Calcam using the :doc:`gui_settings` interface.