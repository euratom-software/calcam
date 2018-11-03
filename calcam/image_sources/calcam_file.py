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
                    'pixel_size':cal.pixel_size
                    }


    return return_dict


display_name = 'Calcam Calibration'

get_image_function =  get_image,
                
get_image_arguments = [
						{
                        'arg_name':'filename',
                        'gui_label': 'File Name' ,
                        'type': 'filename' ,
                        'filter':'Calcam Calibration (*.ccc)' 
                        },
					  ]
