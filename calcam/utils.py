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


'''
A place to collect miscellaneous useful functions which can
be used elsewhere in Calcam.
'''

import numpy as np

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

    def __init__(self):

        self.colours = [(0.121,0.466,0.705),
                        (1,0.498,0.054),
                        (0.172,0.627,0.172),
                        (0.829,0.152,0.156),
                        (0.580,0.403,0.741),
                        (0.549,0.337,0.294),
                        (0.890,0.466,0.760),
                        (0.498,0.498,0.498),
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



# Custom dictionary-like storage class.
# Behaves more-or-less like a dictionary but without the requirement
# that the keys are hashable. Needed so I can do things like use
# QTreeWidgetItems as keys.
class DodgyDict():

    def __init__(self):

        self.keylist = []
        self.itemlist = []
        self.iter_index = 0
        self.next = self.__next__

    def __getitem__(self,key):
        for i,ikey in enumerate(self.keylist):
            if key == ikey:
                return self.itemlist[i]
        raise IndexError()

    def __setitem__(self,key,value):

        for i,ikey in enumerate(self.keylist):
            if key == ikey:
                self.itemlist[i] = value
                return

        self.keylist.append(key)
        self.itemlist.append(value)

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