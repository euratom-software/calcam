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

class CoordTransformer:
	
	def __init__(self,transform_actions=None):

		if transform_actions is not None:
			self.set_transform_actions(transform_actions)
		else:
			self.transform_actions = []

		self.x_pixels = None
		self.y_pixels = None

		self.pixel_aspectratio = 1.

	# Define the actions to perform on an image to change it from original to display coordinates.
	# Input: transform_actions: a list of strings, specifying image transform actions, in the order
	# they should be performed. See line 37 for allowed strings.
	def set_transform_actions(self,transform_actions):

		self.transform_actions = []
		for action in transform_actions:
			if action.lower() in ['flip_up_down','flip_left_right','rotate_clockwise_90','rotate_clockwise_180','rotate_clockwise_270']:
				self.transform_actions.append(action)
			elif action == '':
				pass			
			else:
				raise Exception('Unknown transformation action "' + action + '"')

	# Given an array containing an image in original coordinates, returns an array containing the image in display coordinates.
	# Inputs:	image - numpy ndarray containing the image in original coordinates.
	#			skip_resize - 
	#			binning - 
	# Returns: data_out - numpy ndarray containing the image in display coordinates.
	def original_to_display_image(self,image,skip_resize=False,binning=1):

		if image.shape[1] != self.x_pixels/binning or (image.shape[0] != self.y_pixels/binning and skip_resize == False) or (image.shape[0] != self.y_pixels*self.pixel_aspectratio/binning and skip_resize == True):
			raise Exception('This image is the wrong size!')

		data_out = image.copy()

		if not skip_resize:
			data_out = cv2.resize(data_out,(int(self.x_pixels/binning),int(self.y_pixels*self.pixel_aspectratio/binning)),interpolation=cv2.INTER_NEAREST)

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


		return data_out

	# Given an array containing an image in display coordinates, returns an array containing the image in original coordinates.
	# Inputs:	image - numpy ndarray containing the image in display coordinates.
	#			skip_resize - 
	#			binning - 
	# Returns: data_out - numpy ndarray containing the image in original coordinates.
	def display_to_original_image(self,image, skip_resize=False,binning=1):

		if image.shape[0] != self.get_display_shape()[1]/binning or image.shape[1] != self.get_display_shape()[0]/binning:
			raise Exception('This image is the wrong size!')

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
				
		if not skip_resize:
			data_out = cv2.resize(data_out,(self.x_pixels/binning,self.y_pixels/binning),interpolation=cv2.INTER_NEAREST)

		return data_out


	# Given pixel coordinates in original coordinates, translate these to display coordinates.
	# Inputs:	x,y - array-like objects containing the x and y pixel coordinates in the original image
	# Outputs: x_out, y_out: numpy arrays, the same size and shape as the input x and y, giving the corresponding
	#			coordinates in the 'display' format image.
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
	# Inputs:	x,y - array-like objects containing the x and y pixel coordinates in the display format image
	# Outputs: x_out, y_out: numpy arrays, the same size and shape as the input x and y, giving the corresponding
	#			coordinates in the original image
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

		display_shape = [self.x_pixels,int(self.y_pixels*self.pixel_aspectratio)]

		for action in self.transform_actions:
			if action.lower() in ['rotate_clockwise_90','rotate_clockwise_270']:
				display_shape = list(reversed(display_shape))

		return display_shape
