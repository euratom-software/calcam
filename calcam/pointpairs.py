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
Class for storing 'point pairs': coordinate of matching points
on 2D camera images and 3D CAD models.

Written by Scott Silburn (scott.silburn@ukaea.uk)
"""
 

import numpy as np

# Simple class for storing results.
class PointPairs():
    
    def __init__(self,loadhandle=None):

        self.n_fields = 1
        self.image_points = []
        self.object_points = []

        if loadhandle is not None:
            self.load(loadhandle)


    def get_n_points(self):

        return len(self.object_points)


    # Save point pairs to csv file
    def save(self,savefile):

        # Construct and write the file header and column headings
        fieldheaders = 'World Coordinates [m],,,'
        xyheaders = ''
        for i in range(len(self.image_points[0])):
            fieldheaders = fieldheaders + ',Sub-view {:d},,'.format(i)
            xyheaders = xyheaders + ',,Image X,Image Y'
            
        savefile.write( '{:s}\nMachine X,Machine Y,Machine Z{:s}\n'.format(fieldheaders,xyheaders))

        n_subviews = len(self.image_points[0])

        # Write the point coordinates
        for i in range(len(self.object_points)):

            row = '{:.4f},{:.4f},{:.4f}'.format(self.object_points[i][0],self.object_points[i][1],self.object_points[i][2])
            for j in range(n_subviews):
                if self.image_points[i][j] is None:
                    row = row + ',,,'
                else:
                    row = row + ',,{:.1f},{:.1f}'.format(self.image_points[i][j][0],self.image_points[i][j][1])

            savefile.write(row + '\n')

        savefile.close()

    def load(self,savefile):

        # First row is useless headers
        next(savefile)

        # Read header info in second row...
        headrow = next(savefile)
        if not headrow.startswith('Machine X'):
            raise IOError('Header does not look like a Calcam point pairs file!')
        
        n_fields = int(np.floor( (len(headrow.split(',')) - 4) / 2.))

        # Actually read the point pairs
        self.image_points = []
        self.object_points = []
        
        for frow in savefile:
            row = frow.rstrip().split(',')
            self.object_points.append((float(row[0]),float(row[1]),float(row[2])))
            self.image_points.append([])
            for field in range(n_fields):
                if row[1 + (field+1)*3] != '':
                    self.image_points[-1].append( [float(row[1 + (field+1)*3]) , float(row[2 + (field+1)*3])] )
                else:
                    self.image_points[-1].append(None)