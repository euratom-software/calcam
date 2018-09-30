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

    image_data = cal.geometry.display_to_original_image(cal.image)
    transform_actions = cal.geometry.transform_actions
    subview_mask = cal.geometry.display_to_original_image(cal.subview_mask)
    subview_names = cal.subview_names


    return_dict = {
                    'image_data': image_data,
                    'transform_actions': transform_actions,
                    'subview_mask':subview_mask,
                    'from': 'calcam calibration file {:s}'.format(filename),
                    'subview_names' : subview_names
                    }

    if cal.pixel_size is not None:
        return_dict['pixel_size'] = cal.pixel_size

    return return_dict


image_source = {
				'display_name':'Calcam Calibration',

				'get_image_function': get_image,
                
				'get_image_arguments': [
										{'arg_name':'filename','gui_label': 'File Name' , 'type': 'filename' , 'filter':'Calcam Calibration (*.ccc)' },
										]
				}