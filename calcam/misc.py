'''
* Copyright 2015-2021 European Atomic Energy Community (EURATOM)
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
A place to collect miscellaneous useful things which can
be used elsewhere in Calcam.
'''

import time
import getpass
import datetime
import socket
import importlib
import os
import sys
import subprocess

import numpy as np

# Current username and hostname
username = getpass.getuser()
hostname = socket.gethostname()


def get_formatted_time(timestamp=None):
    
    if timestamp is None:
        when = datetime.datetime.now()
    else:
        when = datetime.datetime.fromtimestamp(timestamp)
    
    return when.strftime('%H:%M on %Y-%m-%d')
    

def rotate_3d(vect,axis,angle):
    '''
    Rotate a given 3D coordinate about a given axis by a given angle.
	
    Parameters:
		
        vect (sequence) : 3 element sequence containing the point to rotate.
		
        axis (sequence) : 3 element sequence containing a vector defining the axis to rotate about.
		
        angle (float)   : Angle to rotate, in degrees
		
    Returns:
		
        np.array : 3 element array containing the rotated coordinates
	
    '''
    vect = np.array(vect,dtype=np.float64)
    vect_ = np.matrix(np.zeros([3,1]),dtype=np.float64)
    vect_[0,0] = vect[0]
    vect_[1,0] = vect[1]
    vect_[2,0] = vect[2]
    axis = np.array(axis)

    # Put angle in radians
    angle = angle * 3.14159 / 180.

    # Make sure the axis is normalised
    axis = axis / np.sqrt(np.sum(axis**2))

    # Make a rotation matrix!
    R = np.matrix(np.zeros([3,3]))
    R[0,0] = np.cos(angle) + axis[0]**2*(1 - np.cos(angle))
    R[0,1] = axis[0]*axis[1]*(1 - np.cos(angle)) - axis[2]*np.sin(angle)
    R[0,2] = axis[0]*axis[2]*(1 - np.cos(angle)) + axis[1]*np.sin(angle)
    R[1,0] = axis[1]*axis[0]*(1 - np.cos(angle)) + axis[2]*np.sin(angle)
    R[1,1] = np.cos(angle) + axis[1]**2*(1 - np.cos(angle))
    R[1,2] = axis[1]*axis[2]*(1 - np.cos(angle)) - axis[0]*np.sin(angle)
    R[2,0] = axis[2]*axis[0]*(1 - np.cos(angle)) - axis[1]*np.sin(angle)
    R[2,1] = axis[2]*axis[1]*(1 - np.cos(angle)) + axis[0]*np.sin(angle)
    R[2,2] = np.cos(angle) + axis[2]**2*(1 - np.cos(angle))

    return np.array( R * vect_)



class ColourCycle():
    '''
    A class to represent a colour cycle,
    used for when plotting / displaying multiple things.

    Note: this uses the same colours as the MatPlotLib 2 default
    colour cycle except with grey (C7) removed because usually grey doesn't
    stand out well enough.
    '''
    def __init__(self):

        self.colours = [(0.121,0.466,0.705),
                        (1,0.498,0.054),
                        (0.172,0.627,0.172),
                        (0.829,0.152,0.156),
                        (0.580,0.403,0.741),
                        (0.549,0.337,0.294),
                        (0.890,0.466,0.760),
                        (0.737,0.741,0.133),
                        (0.09,0.745,0.811),
                        ]

        self.extra_colours = []

        self.next_index = 0

        self.next = self.__next__

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.extra_colours) > 0:
            return self.extra_colours.pop()
        else:
            col = self.colours[self.next_index]
            self.next_index = self.next_index + 1
            if self.next_index > len(self.colours) - 1:
                self.next_index = 0 
            return col

    def queue_colour(self,colour):
        self.extra_colours.insert(0,colour)




class DodgyDict():
    '''
    Custom dictionary-like storage class.
    Behaves more-or-less like a dictionary to the user but without the requirement
    that the keys are hashable. This is a bodge so I can do things like use
    QTreeWidgetItems as keys.
    '''
    def __init__(self):

        self.keylist = []
        self.itemlist = []
        self.iter_index = 0
        self.next = self.__next__

    def __getitem__(self,key):
        for i,ikey in enumerate(self.keylist):
            if key == ikey:
                return self.itemlist[i]
        raise KeyError('No item with key "{:}"'.format(key))

    def __setitem__(self,key,value):

        for i,ikey in enumerate(self.keylist):
            if key == ikey:
                self.itemlist[i] = value
                return

        self.keylist.append(key)
        self.itemlist.append(value)

    def __delitem__(self,key):
        for i,ikey in enumerate(self.keylist):
            if key == ikey:
                self.itemlist.remove(self.itemlist[i])
                self.keylist.remove(key)
                return

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index > len(self.keys()) - 1: 
            raise StopIteration
        else:
            self.iter_index += 1
            return (self.keylist[self.iter_index-1],self.itemlist[self.iter_index-1])

    def keys(self):
        return self.keylist
        
        
        
class LoopProgPrinter:
    '''
    A little object for telling the user how a long,
    loopy calculation is progressing, using stdout.
    
    Includes simple time to completion prediction.
    '''
    def __init__(self):
        
        self.starttime = None
        self.startdatetime = None
        self.frac_done = 0.
        
        # Config parameters
        self.wait_time = 5
        self.min_remaining_length = 5
        self.start_printed = False
        self.end_printed = False
        
        
    def update(self,status):
        
        if status is None:
            return
            
        try:
            float(status)
            self.frac_done = status
            if self.starttime is None:
                self.starttime = time.time()
                self.startdatetime = datetime.datetime.now()
        except (TypeError, ValueError):
            print(status)
            return
            

        elapsed_time = time.time() - self.starttime

        if elapsed_time > self.wait_time and not self.start_printed and self.frac_done > 0:

                est_time = (elapsed_time / self.frac_done)
                
                if est_time - elapsed_time > self.min_remaining_length:
                    est_time_string = ''
                    if est_time > 3600:
                        est_time_string = est_time_string + '{:.0f} hr '.format(np.floor(est_time/3600))
                    if est_time > 600:
                        est_time_string = est_time_string + '{:.0f} min.'.format((est_time - 3600*np.floor(est_time/3600))/60)
                    elif est_time > 59:
                        est_time_string = est_time_string + '{:.0f} min {:.0f} sec.'.format(np.floor(est_time/60),est_time % 60)
                    else:
                        est_time_string ='{:.0f} sec.'.format(est_time)
                    print(self.startdatetime.strftime('Started on:         %Y-%m-%d at %H:%M:%S'))
                    print('Estimated duration: {:s}'.format(est_time_string))
                    
                    self.start_printed = True
                    
        elif self.frac_done == 1. and not self.end_printed:
             
            tot_time = time.time() - self.starttime
            time_string = ''
            if tot_time > 3600:
                time_string = time_string + '{:.0f} hr '.format(np.floor(tot_time / 3600))
            if tot_time >= 59:
                time_string = time_string + '{:.0f} min '.format(np.floor( (tot_time - 3600*np.floor(tot_time / 3600))  / 60))
            time_string = time_string + '{:.0f} sec. '.format( tot_time - 60*np.floor(tot_time / 60) )
            print('Completed in:       {:s}'.format(time_string))
            
            
            self.end_printed = True


def bin_image(arr, factor,binfunc=np.mean):
    """
    Bin an image by the given factor.

    Parameters:

        arr     (array)    :  Input image / array for binning
        factor  (int)      :  Factor by which to bin, e.g. factor=2 would be 2x2 binning.
        binfunc (callable) :  Function to bin with, Numpy ufunc-style. Default is mean.

    Returns:

        Array with the binned image.

    """
    factor = int(factor)

    if arr.shape[0] % factor or arr.shape[1] % factor:
        raise ValueError('The binning factor {:d} is not an integer factor of the array dimensions {:d}x{:d}!'.format(factor,arr.shape[1],arr.shape[0]))

    shape = (arr.shape[0]//factor, factor,
             arr.shape[1]//factor, factor)

    if len(arr.shape) == 2:
        return binfunc(binfunc(arr.reshape(shape), axis=-1), axis=1)
    elif len(arr.shape) == 3:
        out = np.zeros((arr.shape[0] // factor, arr.shape[1] // factor) + arr.shape[2:], dtype=arr.dtype)
        for channel in range(arr.shape[2]):
            out[:,:,channel] = bin_image(arr[:,:,channel],factor,binfunc)
        return out


def import_source(source_path):
    """
    Import a python module from specified python source file,
    and return a reference to the imported module. Closely based
    on the importlib documentation.

    Parameters:
        source_path (str)   : Path to the source to be imported.

    Returns:

        python module

    """
    unload_source(source_path)

    if os.path.isdir(source_path) and os.path.isfile(os.path.join(source_path, '__init__.py')):
        path_elements = source_path.split(os.path.sep)
        try:
            path_elements.remove('')
        except ValueError:
            pass
        modname = path_elements[-1]
        source_path = os.path.join(source_path, '__init__.py')
    elif source_path.endswith('.py'):
        modname = os.path.split(source_path)[-1][:-3]
    else:
        raise FileNotFoundError('Specified path "{:s}" is not a python source file or package directory.'.format(source_path))

    # Check we're not going to create a name clash
    modlist = sys.modules.keys()
    modname_ = modname
    i = 0
    while modname_ in modlist:
        i += 1
        modname_ = modname + '_{:d}'.format(i)


    spec = importlib.util.spec_from_file_location(modname_, source_path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[modname_] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[modname_]
        raise

    return module


def unload_source(source_path):
    """
    Kill any references held by sys.modules for a module with a given source path.
    """
    # Derive what module name import_source would have given it
    if os.path.isdir(source_path) and os.path.isfile(os.path.join(source_path, '__init__.py')):
        path_elements = source_path.split(os.path.sep)
        try:
            path_elements.remove('')
        except ValueError:
            pass
        modname = path_elements[-1]
    elif source_path.endswith('.py'):
        modname = os.path.split(source_path)[-1][:-3]
    else:
        return

    # Find a list of modules it could be
    matching_modules = [mod for mod in sys.modules.keys() if modname in mod]

    # Only remove references to ones which really have the right source path
    for module in matching_modules:
        try:
            if sys.modules[module].__path__[0] == source_path:
                del sys.modules[module]
        except AttributeError:
            if source_path in sys.modules[module].__file__:
                del sys.modules[module]


def open_file(path):
    """
    Cross-platform opening of a path in the OS's default way.
    From https://stackoverflow.com/questions/17317219/is-there-an-platform-independent-equivalent-of-os-startfile
    """
    if sys.platform == "win32":
        os.startfile(path)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, path])


def get_contour_intersection(contour,linestart,lineend,debug=False):
    """
    Returns the intersection of a line from the x-point to given
    strike point, and the wall.

    Parameters:

        linestart (2 element array) : [R,Z] position of X-point
        lineend (2 element array) : [R,Z] position of "strike point"
        debug (bool) : Whether to plot positions for debugging purposes

    Returns:

        NaN if the line which goes through linestart and strike point does not intersect the wall

        S coordinate in mm of the wall intersection if it exists.

    """
    Rs = contour[:,0]
    Zs = contour[:,1]

    Re = np.roll(Rs, 1)
    Ze = np.roll(Zs, 1)

    hs = np.hstack((Rs[:, np.newaxis], Zs[:, np.newaxis], np.ones((Rs.size, 1))))
    he = np.hstack((Re[:, np.newaxis], Ze[:, np.newaxis], np.ones((Rs.size, 1))))
    l_s = np.cross(hs, he)

    l = np.tile( np.cross([linestart[0],linestart[1],1],[lineend[0],lineend[1],1]), (l_s.shape[0],1) )

    res = np.cross(l,l_s)
    with np.errstate(divide='ignore',invalid='ignore'):
        Ri = res[:,0] / res[:,2]
        Zi = res[:,1] / res[:,2]

    dir_s = np.hstack((Ri[:,np.newaxis]- linestart[0],Zi[:,np.newaxis]- linestart[1]))
    dir = np.tile((np.array(lineend) - np.array(linestart))[np.newaxis,:],(dir_s.shape[0],1))
    dp = np.sum(dir_s * dir, axis=1)

    valid = (dp >= 0) & (Ri >= np.minimum(Rs,Re)-1e-5) & (Ri <= np.maximum(Rs,Re)+1e-5) & (Zi >= np.minimum(Zs,Ze)-1e-5) & (Zi <= np.maximum(Zs,Ze)+1e-5)

    if np.count_nonzero(valid) > 0:

        if np.count_nonzero(valid) > 1:
            dr = Ri[valid] - linestart[0]
            dz = Zi[valid] - linestart[1]
            d = np.sqrt(dr**2 + dz**2)
            mindex = np.argmin(d)
        else:
            mindex = 0


        if debug:
            import matplotlib.pyplot as plt
            plt.plot(Rs,Zs,'k')
            plt.plot(linestart[0],linestart[1],'x',label='linestart',markersize=10)
            plt.plot(Ri[valid][mindex],Zi[valid][mindex],'o',label='Detected intersection')
            plt.plot(lineend[0],lineend[1],'s',label='lineend')
            plt.legend()
            plt.show()

        return Ri[valid][mindex],Zi[valid][mindex]
    else:

        if debug:
            import matplotlib.pyplot as plt
            plt.plot(Rs,Zs,'k')
            plt.plot(linestart[0],linestart[1],'x',markersize=10,label='linestart')
            plt.plot(Ri,Zi,'o',label='All intersections')
            plt.plot(lineend[0],lineend[1],'s',label='lineend')
            plt.xlim([2.2,3.1])
            plt.ylim([-1.8,-1.3])
            plt.legend()
            plt.show()

        return np.nan