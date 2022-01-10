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
import numpy as np
import copy


class CoordTransformer:
    """
    Class to handle coordinate transformations between 'display' and 'original' image coordinates.
    """
    
    def __init__(self,transform_actions=[],orig_x=None,orig_y=None,paspect=1.,offset=(0,0)):

        self.set_transform_actions(transform_actions)
        self.x_pixels = orig_x
        self.y_pixels = orig_y
        self.offset = offset
        self.pixel_aspectratio = paspect



    def set_transform_actions(self,transform_actions):
        """
        Set the actions to perform on an image to change it from original to display coordinates.

        Parameters:

            transform_actions (list of str) : List of strings specifying image transform actions, in the order \\
                                              they should be performed. Allowed strings are: 'flip_up_down', \\
                                              'flip_left_right','rotate_clockwise_90','rotate_clockwise_180', \\
                                              or 'rotate_clockwise_270'
        """
        self.transform_actions = []
        for action in transform_actions:
            if action.lower() in ['flip_up_down','flip_left_right','rotate_clockwise_90','rotate_clockwise_180','rotate_clockwise_270']:
                self.transform_actions.append(action)
            elif action in ['','rotate_clockwise_0']:
                pass            
            else:
                raise Exception('Unknown transformation action "' + action + '"')
                


    def get_transform_actions(self):
        """
        Returns a list of strings specifying the actions to perform on an image to change it from original to display coordinates,
        in the order they are performed.
        """
        return copy.copy(self.transform_actions)

    def get_image_shape(self,coords):
        """
        A convenience function to return the image shape depending on a string input for the coords.

        Params:
            coords (str) : 'Original' or 'Display' , whether to return the original or display coords

        Returns:
            Tuple with (width, height) image shape in pixels
        """
        if coords.lower() == 'original':
            return self.get_original_shape()
        elif coords.lower() == 'display':
            return self.get_display_shape()


    def set_image_shape(self,w,h,coords='Original'):
        """
        Set the shape of the image to be transformed.

        Parameters:
            w (int)      : Image witdh in pixels
            h (int)      : Image height in pixels
            coords (str) : 'Original' or 'Display', whether the specified with and height are the display or original shape
        """
        if coords.lower() == 'display':

            shape = list(self.display_to_original_shape([w,h]))

            if self._is_sideways():
                self.pixel_aspectratio = w/shape[1]
            else:
                self.pixel_aspectratio = h/shape[1]

        else:
            shape = [w, h]
            self.pixel_aspectratio = np.round(h * self.pixel_aspectratio) / shape[1]

        self.x_pixels = int(shape[0])
        self.y_pixels = int(shape[1])



    def _is_sideways(self):
        """
        Returns true if the image is rotated by 90 or 270 degrees compared to the
        raw sensor image, otherwise returns false
        """
        sideways = False
        for action in self.transform_actions:
            if action.lower() in ['rotate_clockwise_90', 'rotate_clockwise_270']:
                sideways = not sideways

        return sideways


    def set_offset(self,x_offset,y_offset):
        """
        Specify the offset of the image top-left from the detector top-left (for CMOS sensors with smaller
        redout windows.

        Parameters:

            x_offset (int) : Horizontal offset in pixels
            y_offset (int) : Vertical offset in pixels
        """
        self.offset = (x_offset,y_offset)



    def add_transform_action(self,transform_action):
        """
        Specify an additional action to change the image from original to display coordinates.
        The added action will be performed last.

        Parameters:

            transform_action (str) : String specifying the transform action (for allowed values see `set_transform_actions()`
        """
        transform_action = transform_action.lower()

        if transform_action not in ['flip_up_down','flip_left_right','rotate_clockwise_90','rotate_clockwise_180','rotate_clockwise_270']:
            raise ValueError('Unknown transform action {:s}'.format(transform_action))

        if len(self.transform_actions) == 0:
            self.transform_actions = [transform_action]
        else:
            if transform_action == 'flip_up_down' and self.transform_actions[-1] == 'flip_up_down':
                del self.transform_actions[-1]
            elif transform_action == 'flip_left_right' and self.transform_actions[-1] == 'flip_left_right':
                del self.transform_actions[-1]
            elif 'rotate_clockwise' in transform_action and 'rotate_clockwise' in self.transform_actions[-1]:
                current_angle = int(self.transform_actions[-1].split('_')[2])
                del self.transform_actions[-1]
                new_angle = int(transform_action.split('_')[2])
                total_angle = current_angle + new_angle
                if total_angle > 270:
                    total_angle = total_angle - 360
                if new_angle > 0:
                    self.transform_actions.append('rotate_clockwise_{:d}'.format(total_angle))
            else:
                self.transform_actions.append(transform_action)




    def set_pixel_aspect(self,pixel_aspect,relative_to='display',absolute=True):
        """
        Set the pixel aspect ratio of the camera (pixel height / pixel width) for cameras with non-square pixels.

        Parameters:

            pixel_aspect (float) : Ratio of pixel height / pixel width.

            relative_to (str)    : 'Display' or 'Original', whether the height/width is specified with the image \
                                   in its original or display coordinates

            absolute (bool)      : Whether the specified ratio is specified in absolute terms (default) \
                                   or relative to the currently set pixel aspect ratio (i.e. only has an effect \
                                   if the aspect ratio is already set != 1).
        """
        if absolute:
            ref_aspect = 1.
        else:
            ref_aspect = float(self.pixel_aspectratio)

        if relative_to.lower() == 'original':
            self.pixel_aspectratio = pixel_aspect * ref_aspect
        else:
            
            if self._is_sideways():
                self.pixel_aspectratio = ref_aspect/pixel_aspect
            else:
                self.pixel_aspectratio = ref_aspect*pixel_aspect



    def original_to_display_shape(self,shape):
        """
        Based on the transform actions and pixel aspect ratio, get the display image shape for a
        given original image shape.

        Parameters:

           shape (sequence) : 2-element sequence specifying the original image (width,height)

        Returns:

           Tuple : 2-element tuple specifying the displayed image (width,height).
        """
        shape = np.array(shape,dtype=np.float32)
        shape[1] = shape[1] * self.pixel_aspectratio

        if self._is_sideways():
            shape = shape[::-1]

        return tuple(np.round(shape).astype(int))



    def display_to_original_shape(self,shape):
        """
        Based on the transform actions and pixel aspect ratio, get the original image shape for a
        given displayed image shape.

        Parameters:

           shape (sequence) : 2-element sequence specifying the displayed image (width,height)

        Returns:

           Tuple : 2-element tuple specifying the original image (width,height).
        """
        shape = np.array(shape,dtype=np.float32)

        if self._is_sideways():
            shape = shape[::-1]

        shape[1] = shape[1] / self.pixel_aspectratio

        return tuple(np.round(shape).astype(int))




    def original_to_display_image(self,image,interpolation='nearest'):
        """
        Transform an image from original to display orientation.

        Parameters:

            image (np.ndarray)  : Array containing the image in original orientation.

            interpolation (str) : Interpolation method, allowed strings are 'nearest' or 'cubic'.


        Returns:

            np.ndarray : Array containing the image in display orientation.
        """
        if interpolation.lower() == 'nearest':
            interp_method = cv2.INTER_NEAREST
        elif interpolation.lower() == 'cubic':
            interp_method = cv2.INTER_CUBIC
        else:
            raise ValueError('Interpolation method must be "nearest" or "cubic".')

        expected_size = np.array(self.get_original_shape())
        im_size = np.array(image.shape[1::-1])
        ratio = expected_size / im_size

        if np.abs(ratio[0]-ratio[1]) < 1e-5:
            binning = ratio[0]
        else:
            raise Exception('Expected (multiple of) {:d}x{:d} pixel image, got {:d}x{:d}!'.format(expected_size[0],expected_size[1],image.shape[1],image.shape[0]))

        data_out = image.copy()


        for action in self.transform_actions:
            if action.lower() == 'flip_up_down':
                data_out = np.flipud(data_out)
            elif action.lower() == 'flip_left_right':
                data_out = np.fliplr(data_out)
            elif action.lower() == 'rotate_clockwise_90':
                data_out = np.rot90(data_out,k=3)
            elif action.lower() == 'rotate_clockwise_180':
                data_out = np.rot90(data_out,k=2)
            elif action.lower() == 'rotate_clockwise_270':
                data_out = np.rot90(data_out,k=1)

        out_shape = self.get_display_shape()

        if data_out.shape[0] != int(out_shape[1]/binning) or data_out.shape[1] != int(out_shape[0]/binning):
            data_out = cv2.resize(data_out,(int(out_shape[0]/binning),int(out_shape[1]/binning)),interpolation=interp_method)

        return data_out



    def display_to_original_image(self,image,interpolation='nearest'):
        """
        Transform an image from display to original orientation.

        Parameters:

            image (np.ndarray)  : Array containing the image in display orientation.

            interpolation (str) : Interpolation method, allowed strings are 'nearest' or 'cubic'.


        Returns:

            np.ndarray : Array containing the image in original orientation.
        """
        if interpolation.lower() == 'nearest':
            interp_method = cv2.INTER_NEAREST
        elif interpolation.lower() == 'cubic':
            interp_method = cv2.INTER_CUBIC
        else:
            raise ValueError('Interpolation method must be "nearest" or "cubic".')
            

        expected_size = np.array(self.get_display_shape())
        im_size = np.array(image.shape[1::-1])
        ratio = expected_size / im_size

        if np.abs(ratio[0]-ratio[1]) < 1e-5:
            binning = ratio[0]
        else:
            raise Exception('Expected (multiple of) {:d}x{:d} pixel image, got {:d}x{:d}!'.format(expected_size[0],expected_size[1],image.shape[1],image.shape[0]))

        data_out = image.copy()

        for action in reversed(self.transform_actions):
            if action.lower() == 'flip_up_down':
                data_out = np.flipud(data_out)
            elif action.lower() == 'flip_left_right':
                data_out = np.fliplr(data_out)
            elif action.lower() == 'rotate_clockwise_90':
                data_out = np.rot90(data_out,k=1)
            elif action.lower() == 'rotate_clockwise_180':
                data_out = np.rot90(data_out,k=2)
            elif action.lower() == 'rotate_clockwise_270':
                data_out = np.rot90(data_out,k=3)

        out_shape = self.get_original_shape()

        if data_out.shape[0] != int(out_shape[1]/binning) or data_out.shape[1] != int(out_shape[0]/binning):
            data_out = cv2.resize(data_out,(int(out_shape[0]/binning),int(out_shape[1]/binning)),interpolation=interp_method)

        return data_out



    def original_to_display_coords(self,x,y):
        """
        Given pixel coordinates in original coordinates, translate these to display coordinates.

        Parameters:

            x (float or np.ndarray) : Original x pixel coordinates. Can be a single value or array. \\
                                      x and y must be the same shape.

            y (float or np.ndarray) : Original y pixel coordinates. Can be a single value or array. \\
                                      x and y must be the same shape.


        Returns:

            x : Display x coordinates. Single value or array with the same shape as input x.

            y : Display y coordinates. Single value or array with the same shape as input y.
        """

        # Let's not overwrite the input arrays, just in case
        x_out = np.array(x)
        y_out = np.array(y) * self.pixel_aspectratio

        current_pixels = [self.x_pixels,int(self.y_pixels*self.pixel_aspectratio)]

        for action in self.transform_actions:
            if action.lower() == 'flip_up_down':
                y_out = (current_pixels[1]-1) - y_out
            elif action.lower() == 'flip_left_right':
                x_out = (current_pixels[0]-1) - x_out
            elif action.lower() == 'rotate_clockwise_90':
                # Temporary values...
                yt = y_out.copy()
                y_out = x_out.copy()

                x_out = (current_pixels[1]-1) - yt
                current_pixels = list(reversed(current_pixels))

            elif action.lower() == 'rotate_clockwise_180':

                y_out = (current_pixels[1]-1) - y_out
                x_out = (current_pixels[0]-1) - x_out
            elif action.lower() == 'rotate_clockwise_270':
                # Temporary values...
                yt = y_out.copy()

                y_out = (current_pixels[0]-1) - x_out
                x_out = yt
                current_pixels = list(reversed(current_pixels))

        if len(x_out.shape) == 0:
            x_out = float(x_out)
            y_out = float(y_out)

        return x_out,y_out



    def display_to_original_coords(self,x,y):
        """
        Given pixel coordinates in display coordinates, translate these to original coordinates.

        Parameters:

            x (float or np.ndarray) : Display x pixel coordinates. Can be a single value or array. \\
                                      x and y must be the same shape.

            y (float or np.ndarray) : Display y pixel coordinates. Can be a single value or array. \\
                                      x and y must be the same shape.


        Returns:

            x : Original x coordinates. Single value or array with the same shape as input x.

            y : Original y coordinates. Single value or array with the same shape as input y.
        """

        # Let's not overwrite the input arrays, just in case
        x_out = np.array(x)
        y_out = np.array(y)

        current_pixels = self.get_display_shape()

        for action in reversed(self.transform_actions):
            if action.lower() == 'flip_up_down':
                y_out = (current_pixels[1]-1) - y_out
            elif action.lower() == 'flip_left_right':
                x_out = (current_pixels[0]-1) - x_out
            elif action.lower() == 'rotate_clockwise_90':
                # Temporary values...
                yt = y_out.copy()

                y_out = (current_pixels[0]-1) - x_out
                x_out = yt
                current_pixels = list(reversed(current_pixels))
            elif action.lower() == 'rotate_clockwise_180':
                # Temporary values...
                y_out = (current_pixels[1]-1) - y_out
                x_out = (current_pixels[0]-1) - x_out
            elif action.lower() == 'rotate_clockwise_270':
                # Temporary values...
                yt = y_out.copy()
                y_out = x_out.copy()

                x_out = (current_pixels[1]-1) - yt
                current_pixels = list(reversed(current_pixels))

        y_out = y_out / self.pixel_aspectratio

        if len(x_out.shape) == 0:
            x_out = float(x_out)
            y_out = float(y_out)

        return x_out,y_out




    def get_display_shape(self):
        """
        Get the shape of the image in display orientation.

        Returns:

            Tuple specifying the (width,height) of the image in display orientation
        """

        display_shape = [self.x_pixels,int(np.round(self.y_pixels*self.pixel_aspectratio))]

        for action in self.transform_actions:
            if action.lower() in ['rotate_clockwise_90','rotate_clockwise_270']:
                display_shape.reverse()

        return tuple(display_shape)



    def get_original_shape(self):
        """
        Get the shape of the image in original orientation (as it comes from the camera).

        Returns:

            Tuple specifying the (width,height) of the image in original orientation
        """
        return (self.x_pixels,self.y_pixels)



    def display_to_original_pointpairs(self,pointpairs):
        """
        Given a set of point pairs in display coordinates, transform them to original coordinates.

        Parameters:

            pointpairs (calcam.PointPairs) : Point pairs object with the point pairs in display coordinates.

        Returns:

            calcam.PointPairs object with the point pairs in original coordinates.
        """
        if pointpairs is None:
            return None

        pp_out = copy.copy(pointpairs)

        for ipoint in range(pp_out.get_n_pointpairs()):
            for iview in range(pp_out.n_subviews):
                if pp_out.image_points[ipoint][iview] is not None:
                    pp_out.image_points[ipoint][iview] = tuple(self.display_to_original_coords(*pp_out.image_points[ipoint][iview]))

        return pp_out



    def original_to_display_pointpairs(self,pointpairs):
        """
        Given a set of point pairs in original coordinates, transform them to display coordinates.

        Parameters:

            pointpairs (calcam.PointPairs) : Point pairs object with the point pairs in original coordinates.

        Returns:

            calcam.PointPairs object with the point pairs in display coordinates.
        """
        if pointpairs is None:
            return None

        pp_out = copy.copy(pointpairs)

        for ipoint in range(pp_out.get_n_pointpairs()):
            for iview in range(pp_out.n_subviews):
                if pp_out.image_points[ipoint][iview] is not None:
                    pp_out.image_points[ipoint][iview] = tuple(self.original_to_display_coords(*pp_out.image_points[ipoint][iview]))

        return pp_out