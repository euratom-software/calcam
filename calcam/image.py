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


'''
Image class for calcam.

This is used to store camera images used in calibrations. Provides methods used
by the rest of CalCam, and some possibly useful for the user.

Written by Scott Silburn
'''

import vtk
import os
import numpy as np
import matplotlib.path as mplPath
import sys
import cv2
from .coordtransformer import CoordTransformer
from scipy.io.netcdf import netcdf_file
from . import paths
import shutil
import filecmp
import inspect
import traceback

class Image():

    def __init__(self,FileName=None):
        self.data = None
        self.alpha = None
        self.clim = None
        self.fieldmask = None
        self.n_fields = 0
        self.name = None
        self.transform = CoordTransformer()
        self.postprocessor = None
        self.pixel_size = None

        if FileName is not None:
            self.load(FileName)



    # Return VTK image actor and VTK image reiszer objects for this image.
    def get_vtkActor(self,opacity=None):

        if self.data is None:
            raise Exception('No image data loaded!')

        ImageImporter = vtk.vtkImageImport()
		
        # Create a temporary floating point copy of the data, for scaling.
        # Also flip the data (vtk addresses y from bottom right) and put in display coords.
        Data = np.float32(np.flipud(self.transform.original_to_display_image(self.data)))
        clim = np.float32(self.clim)

        # Scale to colour limits and convert to uint8 for displaying.
        Data -= clim[0]
        Data /= (clim[1]-clim[0])
        Data *= 255.
        Data = np.uint8(Data)

        if self.postprocessor is not None:
            Data = self.postprocessor(Data)
            
        if opacity is not None:
            Alpha = np.uint8(np.ones(Data.shape[:2])*opacity*255)
            Data = np.dstack([Data,Alpha])
        if self.alpha is not None:
            Alpha = np.flipud(self.transform.original_to_display_image(self.alpha))
            Data = np.dstack([Data,Alpha])

        DataString = Data.tostring()
        ImageImporter.CopyImportVoidPointer(DataString,len(DataString))
        ImageImporter.SetDataScalarTypeToUnsignedChar()

        if len(self.data.shape) == 2:
            ImageImporter.SetNumberOfScalarComponents(1)
        else:
            ImageImporter.SetNumberOfScalarComponents(Data.shape[2])

        ImageImporter.SetDataExtent(0,Data.shape[1]-1,0,Data.shape[0]-1,0,0)
        ImageImporter.SetWholeExtent(0,Data.shape[1]-1,0,Data.shape[0]-1,0,0)

        Actor = vtk.vtkImageActor()
        Actor.InterpolateOff()
        mapper = Actor.GetMapper()
        mapper.SetInputConnection(ImageImporter.GetOutputPort())

        return Actor



    # Return VTK image actor and VTK image reiszer objects for this image.
    # This is older than get_vtkActor and is used in parts of the code not (yet)
    # changed to use vtkImageActor rather than vtkActor2D.
    def get_vtkobjects(self,opacity=None):

        if self.data is None:
            raise Exception('No image data loaded!')

        ImageImporter = vtk.vtkImageImport()
        
        # Create a temporary floating point copy of the data, for scaling.
        # Also flip the data (vtk addresses y from bottom right) and put in display coords.
        Data = np.float32(np.flipud(self.transform.original_to_display_image(self.data)))
        clim = np.float32(self.clim)

        # Scale to colour limits and convert to uint8 for displaying.
        Data -= clim[0]
        Data /= (clim[1]-clim[0])
        Data *= 255.
        Data = np.uint8(Data)

        if self.postprocessor is not None:
            Data = self.postprocessor(Data)
            
        if opacity is not None:
            Alpha = np.uint8(np.ones(Data.shape[:2])*opacity*255)
            Data = np.dstack([Data,Alpha])
        if self.alpha is not None:
            Alpha = np.flipud(self.transform.original_to_display_image(self.alpha))
            Data = np.dstack([Data,Alpha])

        DataString = Data.tostring()
        ImageImporter.CopyImportVoidPointer(DataString,len(DataString))
        ImageImporter.SetDataScalarTypeToUnsignedChar()

        if len(self.data.shape) == 2:
            ImageImporter.SetNumberOfScalarComponents(1)
        else:
            ImageImporter.SetNumberOfScalarComponents(Data.shape[2])

        ImageImporter.SetDataExtent(0,Data.shape[1]-1,0,Data.shape[0]-1,0,0)
        ImageImporter.SetWholeExtent(0,Data.shape[1]-1,0,Data.shape[0]-1,0,0)

        Resizer = vtk.vtkImageResize()
        Resizer.SetInputConnection(ImageImporter.GetOutputPort())
        Resizer.SetResizeMethodToOutputDimensions()
        Resizer.SetOutputDimensions((Data.shape[1],Data.shape[0],1))
        # jrh mod begin
        Resizer.InterpolateOff()

        mapper = vtk.vtkImageMapper()
        mapper.SetInputConnection(Resizer.GetOutputPort())
        mapper.SetColorWindow(255)
        mapper.SetColorLevel(127.5)

        Actor = vtk.vtkActor2D()
        Actor.SetMapper(mapper)
        Actor.GetProperty().SetDisplayLocationToBackground()

        return Actor,Resizer


    # Save the image to disk, along with all its calcam metadata
    # (e.g. transform from original to image coordinates.)
    # Inputs: savename - string to identify saved image.
    # If savename is not provided, saves as the image object's 'name' property, if it has one.
    def save(self,savename=None):

        if savename is None:
            savename = self.name

        if self.name != savename:
            self.name = savename

        f = netcdf_file(os.path.join(paths.images,savename + '.nc'),'w')
	
        setattr(f,'image_name',self.name)
        setattr(f,'image_transform_actions',"['" + "','".join(self.transform.transform_actions) + "']")
        setattr(f,'field_names',','.join(self.field_names))

        udim = f.createDimension('udim',self.data.shape[1])
        vdim = f.createDimension('vdim',self.data.shape[0])
        if len(self.data.shape) == 3:
                wdim = f.createDimension('wdim',3)
                data = f.createVariable('image_data','i1',('vdim','udim','wdim'))
                data[:,:,:] = self.data[:,:,:]
        else:
                data = f.createVariable('image_data','i1',('vdim','udim'))
                data[:,:] = self.data[:,:]

        if self.alpha is not None:
            alpha = f.createVariable('alpha_data','i1',('vdim','udim'))
            alpha[:,:] = self.alpha[:,:]
            alpha.units = 'DL'

        if self.pixel_size is not None:
            pxsize = f.createVariable('pixel_size','f4',())
            pxsize.assignValue(self.pixel_size)

        pairdim = f.createDimension('pairdim',2)

        fieldmask = f.createVariable('fieldmask','i1',('vdim','udim'))
        fieldmask[:,:] = self.fieldmask[:,:]

        clim = f.createVariable('calcam_clim','i1',('pairdim',))
        clim[:] = self.clim

        pixaspect = f.createVariable('pixel_aspect_ratio','f4',())
        pixaspect.assignValue(self.transform.pixel_aspectratio)

        data.units = 'DL'
        clim.units = 'DL'
        fieldmask.units = 'Field number'

        f.close()
		


    # Load an image previously saved using Image.save() in to this image instance.
    # Inputs: loadname - string to identify saved image.
    def load(self,loadname):

        f = netcdf_file(os.path.join(paths.images,loadname + '.nc'), 'r',mmap=False)

        # This is for dealing with "old" format save files.
        # If things are saved as 64 bit ints (why did I ever think that was sensible???)
        # let's convert things down to 8 bit int like a sensible person.
        # Also attempt some sort of recovery of any transparency info.
        if f.variables['image_data'].data.dtype == '>i4':
            data = f.variables['image_data'].data.astype('uint64')
            self.alpha = None
            if len(data.shape) == 3:
                if data.shape[2] == 4:
                    scale_factor = 255. / data[:,:,3].max()
                    self.alpha = np.int8(data[:,:,3] * scale_factor)

            scale_factor = 255. / data.max()
            self.data = np.uint8(data * scale_factor)
            self.fieldmask = f.variables['fieldmask'].data.astype('int8')
            clim = f.variables['calcam_clim'].data.astype('uint64')
            self.clim = np.uint8(clim * scale_factor)
            self.transform.pixel_aspectratio = f.variables['pixel_aspect_ratio'].data.astype('float32')

        else:
            # If the data is already saved as 8 bit ints, we can carry on happily :)
            self.data = f.variables['image_data'].data.astype('uint8')
            try:
                self.alpha = f.variables['alpha_data'].data.astype('uint8')
            except KeyError:
                self.alpha = None
            self.fieldmask = f.variables['fieldmask'].data.astype('uint8')
            self.clim = f.variables['calcam_clim'].data.astype('uint8')
            self.transform.pixel_aspectratio = f.variables['pixel_aspect_ratio'].data.astype('float32')

        try:
            self.pixel_size = f.variables['pixel_size'].data.astype('float32')
        except KeyError:
            self.pixel_size = None

        self.name = f.image_name

        if type(self.name) is bytes:
            self.name = self.name.decode('utf-8')

        self.n_fields = np.max(self.fieldmask) + 1

        try:
            fns = f.field_names
            if type(fns) is bytes:
                fns = fns.decode('utf-8')
            self.field_names = fns.split(',')
        except AttributeError:
            if self.n_fields == 1:
                self.field_names = ['Image']
            else:
                self.field_names = []
                for field in range(self.n_fields):
                    self.field_names.append('Sub FOV # {:d}'.format(field+1))

        self.transform.set_transform_actions(eval(f.image_transform_actions))
        self.transform.x_pixels = self.data.shape[1]
        self.transform.y_pixels = self.data.shape[0]

        f.close()
		

    # Export the image to an 8-bit regular image format (any that OpenCV can write)
    # Inputs: filename - filename, including extension, relative to the current working directory
    # Coords: whether to export the image 'right-way-up' (display coords) or as it comes from the camera (original coords)
    def export(self,filename,coords='Display'):
		
        # Scale the image to 8 bit
        im_out = 255. * (self.data - self.clim[0]) / (self.clim[1] - self.clim[0])
        im_out[im_out < 0] = 0
        im_out[im_out > 255] = 255
		
        if coords.lower()=='display':
            im_out = self.transform.original_to_display_image(im_out)

        # OpenCV can only save alpha data to PNGs
        if self.alpha is not None:
            if filename[-3:].lower() == 'png':
                if coords.lower()=='display':
                    alpha_out = self.transform.original_to_display_image(self.alpha)
                else:
                    alpha_out = self.alpha
                im_out = np.dstack(im_out,alpha_out)
            else:
                print('[Calcam image export] **Warning**: Transparency information will be lost when saving to this file format! (Use PNG instead)')

        # Re-shuffle colour channels because OpenCV expects BGR / BGRA
        im_out[:,:,:3] = im_out[:,:,2::-1]
        cv2.imwrite(filename,im_out)


'''
Function to create a calcam image from a 2D or 3D numpy array.

Inputs: Array - 2D array [y,x] containing a monochrome image or
                3D array [y,x,RGB(A)] containing a colour image (can have transparency).

Optional inputs: pixel_aspectatio - pixel aspect ratio (x/y)
                 transform_actions - list of actions to transform the image from the way round it is in the
                                      input array to 'right-way-up'. Can be modified later.
                 image_name - string specifying a name to give the image.

 Output: Calcam image object containing the image in the input array.

 IMPORTANT NOTE: this may keep a reference of thr original array,so any modifications within
 the input array after calling this function will be reflected in your calcam image object.
'''
def from_array(Array,pixel_aspectratio=1,transform_actions=None,image_name='New Image',pixel_size=None):

    if Array.min() < 0:
        print('[calcam.image.from_array] **Warning**: Image array with negative values passed; values will be shifted up to =>0.')
        Array = Array - Array.min()

    # If the array isn't already 8-bit int, make it 8-bit int...
    if Array.dtype != np.uint8:
        # If we're given a higher bit-depth integer, it's easy to downcast it.
        if Array.dtype == np.uint16 or Array.dtype == np.int16:
            Array = np.uint8(Array/2**8)
        elif Array.dtype == np.uint32 or Array.dtype == np.int32:
            Array = np.uint8(Array/2**24)
        elif Array.dtype == np.uint64 or Array.dtype == np.int64:
            Array = np.uint8(Array/2**56)
        # Otherwise, scale it in a floating point way to its own max & min
        # and strip out any transparency info (since we can't be sure of the scale used for transparency)
        else:
            if len(Array.shape) == 3:
                if Array.shape[2] == 4:
                    Array = Array[:,:,:-1]
                    print('[calcam.image.from_array] **Warning**: You passed an image with transparency with a non-integer data type! Transparency information will be thrown away.')
            Array = np.uint8(255.*(Array - Array.min())/(Array.max() - Array.min()))


    # Create a new image object
    NewImage = Image()
	
    # By default assume no split fields
    NewImage.n_fields = 1
    NewImage.field_names = ['Image']

    # If the image has an alpha channel, separate it out so we don't ruin it when colour scaling.
    if len(Array.shape) == 3:
        if Array.shape[2] == 4:
            NewImage.data = Array[:,:,:-1]
            NewImage.alpha = Array[:,:,-1]

    # Otherwise, just grab a reference to the input array for the image data
    if NewImage.data is None:
        NewImage.data = Array

    # Default colour limits are from max & min in frame, or full range if the image is blank
    if NewImage.data.min() != NewImage.data.max():
        NewImage.clim = [NewImage.data.min(),NewImage.data.max()]
    else:
        NewImage.clim = [0,255]

    # Define field 0 to cover the whole image
    NewImage.fieldmask = np.zeros(np.shape(NewImage.data)[0:2],dtype='int8')

    NewImage.pixel_size = pixel_size
    NewImage.transform.pixel_aspectratio = pixel_aspectratio

    NewImage.transform.x_pixels = Array.shape[1]
    NewImage.transform.y_pixels = Array.shape[0]

    if transform_actions is not None:
        NewImage.transform.set_transform_actions(transform_actions)

    if image_name is not None:
        NewImage.name = image_name

    return NewImage



'''
Parent class for CalCam image sources.
Currently basically nothing here...done mainly for future-proofing against features I haven't though of yet. 
'''
class ImageSource():
    def __init__(self):
        pass

    def __call__(self,*args, **kwargs):
        return self.get_image(*args,**kwargs)


'''
Image source for loading images from standard image files.
'''
class FileSource(ImageSource):


    def __init__(self):

        self.gui_display_name = 'Image file'

        self.gui_inputs = [
                            {'label': 'File Name' , 'type': 'filename' , 'filter':'Image Files (*.png *.jpg *.jpeg *.bmp *.jp2 *.tiff *.tif)' },
                          ]


    def get_image(self,Filename,pixel_aspectratio=1,transform_actions=None):
		
        # Get the image data from a file
        dat = cv2.imread(Filename)
        if dat is None:
            raise UserWarning('Could not read specified image file "' + Filename + '"')

        # If it's a colour image, fix the colour channel order (openCV loads BGRA, for some reason)
        if len(dat.shape) == 3:
            if dat.shape[2] == 3:
                dat[:,:,:3] = dat[:,:,2::-1]
			
        # Give the image a name which corresponds to the filename we loaded it from
        if '/' in Filename:
            name = ".".join(Filename.split('/')[-1].split('.')[:-1])
        elif '\\' in Filename:
            name = ".".join(Filename.split('\\')[-1].split('.')[:-1])
        else:
            name = ".".join(Filename.split('.')[:-1])

        return from_array(dat,pixel_aspectratio=pixel_aspectratio,transform_actions=transform_actions,image_name=name)




'''
Image source for loading saved Calcam images.
'''
class CalCamSaveSource(ImageSource):

    def __init__(self):
        self.gui_display_name = 'Previously Used Image'

        self.gui_inputs = [
                            {'label': 'Image Name' , 'type': 'choice' , 'choices': paths.get_save_list('Images') },
                          ]

    def get_image(self,savename):
        if savename != '':
            return Image(savename)
        else:
            raise UserWarning('No saved image name specified!')



'''
Stuff done at import:
Sets up image sources, and imports user defined ones.
'''
image_sources = []

from_file = FileSource()
image_sources.append(from_file)
image_sources.append(CalCamSaveSource())

example_file = os.path.join(paths.calcampath,'usercode_examples','image_source.py_')
user_files = [fname for fname in os.listdir(paths.image_sources) if fname.endswith('.py')]

# See if the user already has a CAD model definition example file, and if it's up to date. If not, create it.
# If the definitions might have changed, warn the user.
if 'Example.py' in user_files:
    is_current_version = filecmp.cmp(os.path.join(paths.image_sources,'Example.py'),example_file,shallow=False)
    if not is_current_version:
        shutil.copy2(example_file,os.path.join(paths.image_sources,'Example.py'))
        print('[Calcam Import] The latest image source definition example is different from your user copy. Your existing copy has been updated. If you get image source related errors, you may need to check and edit the CAD definition files in ' + paths.image_sources )
    user_files.remove('Example.py')
else:
    shutil.copy2(example_file,os.path.join(paths.image_sources,'Example.py'))
    print('[Calcam Import] Created image source definition example in ' + os.path.join(paths.image_sources,'Example.py'))


user_im_sources = []
# Go through all the python files which aren't examples, and import the CAD definitions
for def_filename in user_files:
    try:
        exec('import ' + def_filename[:-3] + ' as UserImageSource')
        classes = inspect.getmembers(UserImageSource, inspect.isclass)
        for iclass in classes:
            if inspect.getmodule(iclass[1]) is UserImageSource:
                image_sources.append( iclass[1]() )
        del UserImageSource
    except Exception as e:
        if __package__ == None:
            estack = traceback.extract_tb(sys.exc_info()[2])
            lineno = None
            for einf in estack:
                if def_filename in einf[0]:
                    lineno = einf[1]
            if lineno is not None:
                print('[Calcam Import] User image source {:s} not imported due to exception at line {:d}: {:s}'.format(def_filename,lineno,e))
            else:
                print('[Calcam Import] User image source {:s} not imported due to exception: {:s}'.format(def_filename,e))
