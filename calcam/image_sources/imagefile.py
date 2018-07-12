import cv2

# A function which actually gets the image.
# This can take as many arguments as you want.
# It must return a dictionary, see below
def get_image(filename):
	
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
                    'from': 'image file {:s}'.format(filename)
                    }

    return return_dict


image_source = {
				'display_name':'Image File',

				'get_image_function':get_image,
                
				'get_image_arguments': [
										{'arg_name':'filename','gui_label': 'File Name' , 'type': 'filename' , 'filter':'Image Files (*.png *.jpg *.jpeg *.bmp *.jp2 *.tiff *.tif)' },
										]
				}