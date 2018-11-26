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


"""
Image coordinate transformer class for CalCam

This class keeps track of the relationship between 'original'
image pixel coordinates (i.e. straight from the camera) and 'display'
(flipped, rotated etc to appear the "right way round") coordinates.

Methods are provided for transforming between original and display coordinates.

Written by Scott Silburn
2015-05-17
"""

import cv2
import numpy as np
import copy

class CoordTransformer:
    
    def __init__(self,transform_actions=[],orig_x=None,orig_y=None,paspect=1.):


        self.set_transform_actions(transform_actions)
        self.x_pixels = orig_x
        self.y_pixels = orig_y

        self.pixel_aspectratio = paspect


    # Define the actions to perform on an image to change it from original to display coordinates.
    # Input: transform_actions: a list of strings, specifying image transform actions, in the order
    # they should be performed. See line 37 for allowed strings.
    def set_transform_actions(self,transform_actions):

        self.transform_actions = []
        for action in transform_actions:
            if action.lower() in ['flip_up_down','flip_left_right','rotate_clockwise_90','rotate_clockwise_180','rotate_clockwise_270']:
                self.transform_actions.append(action)
            elif action in ['','rotate_clockwise_0']:
                pass            
            else:
                raise Exception('Unknown transformation action "' + action + '"')
                

    def get_transform_actions(self):
        return copy.copy(self.transform_actions)


    def set_image_shape(self,w,h,coords='Original'):
        
        shape = [w,h]
        
        if coords.lower() == 'display':
            for action in self.transform_actions:
                if action.lower() in ['rotate_clockwise_90','rotate_clockwise_270']:
                    shape = list(reversed(shape))
            shape[1] = np.round(shape[1] / self.pixel_aspectratio)
            
        self.x_pixels = int(shape[0])
        self.y_pixels = int(shape[1])
            
            


    def add_transform_action(self,transform_action):

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

        if absolute:
            ref_aspect = 1.
        else:
            ref_aspect = float(self.pixel_aspectratio)

        if relative_to.lower() == 'original':
            self.pixel_aspectratio = pixel_aspect * ref_aspect
        else:
            sideways = False
            for action in self.transform_actions:
                if action.lower() in ['rotate_clockwise_90','rotate_clockwise_270']:
                    sideways = not sideways
            
            if sideways:
                self.pixel_aspectratio = ref_aspect/pixel_aspect
            else:
                self.pixel_aspectratio = ref_aspect*pixel_aspect
                



    # Given an array containing an image in original coordinates, returns an array containing the image in display coordinates.
    # Inputs:   image - numpy ndarray containing the image in original coordinates.
    #           skip_resize - 
    #           binning - 
    # Returns: data_out - numpy ndarray containing the image in display coordinates.
    def original_to_display_image(self,image,interpolation='nearest'):

        if interpolation.lower() == 'nearest':
            interp_method = cv2.INTER_NEAREST
        elif interpolation.lower() == 'cubic':
            interp_method = cv2.INTER_CUBIC
        else:
            raise ValueError('Interpolation method must be "nearest" or "cubic".')

        expected_size = np.array(self.get_original_shape())
        im_size = np.array(image.shape[1::-1])
        ratio = expected_size / im_size
        binning = 1
        if np.any(expected_size != im_size):
            if not np.any(np.mod(expected_size,im_size)) and np.abs(ratio[0]-ratio[1]) < 1e-5:
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
        data_out = cv2.resize(data_out,(int(out_shape[0]/binning),int(out_shape[1]/binning)),interpolation=interp_method)

        return data_out


    # Given an array containing an image in display coordinates, returns an array containing the image in original coordinates.
    # Inputs:   image - numpy ndarray containing the image in display coordinates.
    #           skip_resize - 
    #           binning - 
    # Returns: data_out - numpy ndarray containing the image in original coordinates.
    def display_to_original_image(self,image,interpolation='nearest'):

        if interpolation.lower() == 'nearest':
            interp_method = cv2.INTER_NEAREST
        elif interpolation.lower() == 'cubic':
            interp_method = cv2.INTER_CUBIC
        else:
            raise ValueError('Interpolation method must be "nearest" or "cubic".')
            

        expected_size = np.array(self.get_display_shape())
        im_size = np.array(image.shape[1::-1])
        ratio = expected_size / im_size
        binning = 1
        if np.any(expected_size != im_size):
            if not np.any(np.mod(expected_size,im_size)) and np.abs(ratio[0]-ratio[1]) < 1e-5:
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
                

        data_out = cv2.resize(data_out,(self.x_pixels//binning,self.y_pixels//binning),interpolation=interp_method)

        return data_out


    # Given pixel coordinates in original coordinates, translate these to display coordinates.
    # Inputs:   x,y - array-like objects containing the x and y pixel coordinates in the original image
    # Outputs: x_out, y_out: numpy arrays, the same size and shape as the input x and y, giving the corresponding
    #           coordinates in the 'display' format image.
    def original_to_display_coords(self,x,y):


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

        return x_out,y_out


    # Given pixel coordinates in display coordinates, translate these to original coordinates.
    # Inputs:   x,y - array-like objects containing the x and y pixel coordinates in the display format image
    # Outputs: x_out, y_out: numpy arrays, the same size and shape as the input x and y, giving the corresponding
    #           coordinates in the original image
    def display_to_original_coords(self,x,y):

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

        return x_out,y_out


    # Return the shape of the 'display' format image.
    # Outputs: display_shape - 2 element list [x pixels, y pixels]
    # Note this is the opposite way around to the 'array shape' which is [y pixels, x pixels]
    # since Python addresses image arrays [y,x]
    def get_display_shape(self):

        display_shape = [self.x_pixels,int(np.round(self.y_pixels*self.pixel_aspectratio))]

        for action in self.transform_actions:
            if action.lower() in ['rotate_clockwise_90','rotate_clockwise_270']:
                display_shape = reversed(display_shape)

        return tuple(display_shape)


    def get_original_shape(self):
        return (self.x_pixels,self.y_pixels)


    def display_to_original_pointpairs(self,pointpairs):

        if pointpairs is None:
            return None

        pp_out = copy.copy(pointpairs)

        for ipoint in range(pp_out.get_n_points()):
            for iview in range(pp_out.n_subviews):
                pp_out.image_points[ipoint][iview][:] = self.display_to_original_coords(*pp_out.image_points[ipoint][iview])

        return pp_out


    def original_to_display_pointpairs(self,pointpairs):

        if pointpairs is None:
            return None

        pp_out = copy.copy(pointpairs)

        for ipoint in range(pp_out.get_n_points()):
            for iview in range(pp_out.n_subviews):
                pp_out.image_points[ipoint][iview][:] = self.original_to_display_coords(*pp_out.image_points[ipoint][iview])

        return pp_out