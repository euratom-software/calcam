'''
* Copyright 2015-2020 European Atomic Energy Community (EURATOM)
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

import cv2
from calcam.calibration import Calibration

# A function which actually gets the image.
# This can take as many arguments as you want.
# It must return a dictionary, see below
def get_image(filename):
	
    # Get the image data from a file
    try:
        cal = Calibration(filename)
    except IOError:
        raise UserWarning('Cannot read specified file "{:s}"'.format(filename))

    if cal.image is None:
        raise UserWarning('This calibration file does not contain an image!')


    return_dict = {
                    'image_data': cal.get_image(coords='original'),
                    'transform_actions': cal.geometry.get_transform_actions(),
                    'subview_mask':cal.get_subview_mask(coords='original'),
                    'source': cal.history['image'],
                    'subview_names' : cal.subview_names,
                    'coords': 'original',
                    'pixel_aspect':cal.geometry.pixel_aspectratio,
                    'pixel_size':cal.pixel_size,
                    'image_offset':cal.geometry.offset
                    }


    return return_dict


display_name = 'Calcam Calibration'

get_image_function =  get_image
                
get_image_arguments = [
						{
                        'arg_name':'filename',
                        'gui_label': 'File Name' ,
                        'type': 'filename' ,
                        'filter':'Calcam Calibration (*.ccc)' 
                        },
					  ]
