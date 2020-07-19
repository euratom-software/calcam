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

import cv2
import os

# A function which actually gets the image.
# This can take as many arguments as you want.
# It must return a dictionary, see below
def get_image(filename,coords,offset_x,offset_y):

    # Get the image data from a file
    dat = cv2.imread(filename)
    if dat is None:
        raise UserWarning('Could not read specified image file "' + filename + '"')

    # If it's a colour image, fix the colour channel order (openCV loads BGRA, for some reason)
    if len(dat.shape) == 3:

        # If we have a colour image swap the G and B channels, since OpenCV loads them in the wrong order.
        if dat.shape[2] == 3:
          
            dat[:,:,:3] = dat[:,:,2::-1]

            # If R, G and B channels are all the same, just return a monochrome image
            if (dat[:,:,0] == dat[:,:,1]).all() and (dat[:,:,0] == dat[:,:,2]).all():
                dat = dat[:,:,0] 


    return_dict = {
                    'image_data': dat,
                    'source': 'Loaded from image file {:s}'.format(os.path.split(filename)[-1]),
                    'coords':coords,
                    'image_offset':(offset_x,offset_y)
                    }

    return return_dict


display_name = 'Image File'
get_image_function = get_image
get_image_arguments =  [
                        {
                            'arg_name': 'filename',
                            'gui_label': 'File Name',
                            'type': 'filename',
                            'filter': 'Image Files (*.png *.jpg *.jpeg *.bmp *.jp2 *.tiff *.tif)'
                        },
                        {
                            'arg_name': 'coords',
                            'gui_label': 'Image Orientation',
                            'type': 'choice',
                            'choices': ['Display','Original'],
                            'default': 'Display'
                        },
                        {
                            'arg_name': 'offset_x',
                            'gui_label': 'Detector X Offset',
                            'type': 'int',
                            'limits':[0,1e4]
                        },
                        {
                            'arg_name': 'offset_y',
                            'gui_label': 'Detector Y Offset',
                            'type': 'int',
                            'limits': [0, 1e4]
                        },
                        ]