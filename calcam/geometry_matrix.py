# -*- coding: utf-8 -*-

"""
Loads in Rays from CalCam Ray tracing data.
Main function needs to be given parameters to create inversion mesh.
Given both of these, a geometry matrix is created assuming mesh cells
have 4 vertices and are joined by straight lines. The resultant geometry
matrix is saved to a file that can be accessed by other scripts.
This version looks at the JET divertor camera KL11.
Ray data has been saved in netCDT file so need to read this in.

This version uses a different method to find lengths through cells. Don't loop
over every cell 4 times! Loop over each R and z line and find intercepts with 
these and there positions. Then 10 by 5 grid has only 15 calculations rather 
than 4*10*5=200!!

Authors: Mark Smithies & James Harrison
Last Updated: 6th Jan 2018
"""

#import os
import numpy as np
import sys
#import matplotlib.pyplot as plt
#import pickle
#import pupynere
#import cv2

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

class RectangularGeometryMatrix:
    
    def __init__(self,Rmin=None,Rmax=None,Zmin=None,Zmax=None,nR=None,nZ=None,
                 filename=None,RayData=None):

        if ((RayData == None) & (filename == None)):
            raise Exception('RectangularGeometryMatrix: Calcam ray data needs to be specified')
        
        if ((Rmin == None) & (filename == None)):
            raise Exception('RectangularGeometryMatrix: Neither geometry matrix bounds nor filename specified')        
        
        # User has specified the bounds of the geometry matrix
        if ((Rmin is not None) & (Rmax is not None) & 
            (Zmin is not None) & (Zmax is not None) & 
            (nR is not None) & (nZ is not None)):
                
                self.Rmin              = np.min([Rmin,Rmax])
                self.Rmax              = np.max([Rmin,Rmax])
                self.Zmin              = np.max([Zmin,Zmax])
                self.Zmax              = np.min([Zmin,Zmax])
                self.nR                = nR+1
                self.nZ                = nZ+1
                self.Rgrid, self.Zgrid = self._SimpleGrid()
                self.RayData           = RayData
                self.LMatrixRows       = None
                self.LMatrixColumns    = None
                self.LMatrixValues     = None
                self.LMatrixShape      = None
        
        # User has specified a file name to load
        if ((filename is not None) & (RayData is not None)):
            self.LoadGeometryMatrix(filename)
            
    def _SimpleGrid(self):
        """Calculates a simple 2D radial and vertical mesh containing evenly
        spaced points"""
               
        nodes_r = np.linspace(self.Rmin,self.Rmax,self.nR)
        nodes_z = np.linspace(self.Zmin,self.Zmax,self.nZ)
        
        Rgrid, Zgrid = np.meshgrid(nodes_r,nodes_z)
        
        return Rgrid.flatten(), Zgrid.flatten()
      
    def _FindRayPath(self,x,y,EndPos,Origin):
        """Use pixel position x and y to find start and end coordinates of ray.
        Return x and y values along the ray. This can be used to plot ray in x-y
        space for debugging purposes if needed."""
          
        chord_x_values = np.linspace(Origin[0], EndPos[0], 100)
        chord_y_values = np.linspace(Origin[1], EndPos[1], 100)
        return chord_x_values, chord_y_values

    def _FindRz(self,x,y,t,Origin,DirectionVector):
        """Given a ray defined by pixel x-y, and a length along the ray t, 
        position in R-z space is returned. Used to plot rays in Rz space
        which can be used for debugging purposes."""
        
        R = np.sqrt( (Origin[0] + t*DirectionVector[0])**2 + (Origin[1] + t*DirectionVector[1])**2)
        z = Origin[2] + t*DirectionVector[2]
        return R, z    
    
    def _GetCellCorners(self,index):
        """Given cell index, this function returns the R and z values
        of each cell corner starting in top left and going clockwise.
        First corner is saved twice for ease of coding later on. Values
        returned in R array and z array."""
        
        R0 = self.Rgrid[index]
        z0 = self.Zgrid[index]
        R1 = self.Rgrid[index+1]
        z1 = self.Zgrid[index+1]
        R2 = self.Rgrid[index+1+self.nR-1]
        z2 = self.Zgrid[index+1+self.nR-1]
        R3 = self.Rgrid[index+self.nR-1]
        z3 = self.Zgrid[index+self.nR-1]
        R4 = self.Rgrid[index]
        z4 = self.Zgrid[index]
        R = [R0,R1,R2,R3,R4]
        z = [z0,z1,z2,z3,z4]
        
        return R,z

    
    def _SolveQuadratic(self,a,b,c):
        """Solves the quadratic equation given a, b and c and returns the 
        lower root followed by the higher root in an array."""
        
        t = np.zeros(2)
        A = 1.0/a
        t[0] = (-b - (b**2.0 - 4.0*a*c)**0.5)*0.5*A
        t[1] = (-b + (b**2.0 - 4.0*a*c)**0.5)*0.5*A
        return t

    def _FindMinMax(self,Value1,Value2):
        if Value1==Value2:
            ValueMin=Value1
            ValueMax=Value1
        elif Value1>Value2:
            ValueMax=Value1
            ValueMin=Value2
        else:
            ValueMax=Value2
            ValueMin=Value1
        return ValueMin,ValueMax
    
    def _FindInterceptSquare(self,SquareSide,R1,z1,R2,z2,DirectionVector,Origin,RayLen):
        """ This function is fed information for the equation of one line that
        makes up the cell. Check if ray intercepts this line and if this 
        intercept is within the cell corners. Also because the ray is curved
        in R-z space it could intercept a line twice. Check for this"""
        
        tIntercept=[]
        Intercept=False
        rmin,rmax = self._FindMinMax(R1,R2)
        zmin,zmax = self._FindMinMax(z1,z2)
        # Find intercepts for vertical sides of square
        if SquareSide==1 or SquareSide==3:
            r=R1
            c = -r*r + Origin[0]*Origin[0] + Origin[1]*Origin[1]
            b = 2*Origin[0]*DirectionVector[0] + 2*Origin[1]*DirectionVector[1]
            a = DirectionVector[0]**2 + DirectionVector[1]**2
            t = self._SolveQuadratic(a,b,c)
            zPossible = Origin[2] + t[0]*DirectionVector[2]
            if zPossible<zmax and zPossible>zmin and t[0]<=RayLen:
                tIntercept=np.append(tIntercept,t[0])
                Intercept=True    
            zPossible = Origin[2] + t[1]*DirectionVector[2] 
            if zPossible<zmax and zPossible>zmin and t[1]<=RayLen:
                tIntercept=np.append(tIntercept,t[1])
                Intercept=True
        # Find intercepts for horizontal sides of square
        if SquareSide==0 or SquareSide==2:
            t = (z1 - Origin[2])/DirectionVector[2]
            RPossibleSquared = (Origin[0] + t*DirectionVector[0])**2 + (Origin[1] + t*DirectionVector[1])**2        
            if RPossibleSquared<rmax*rmax and RPossibleSquared>rmin*rmin and t<=RayLen:
                tIntercept=t
                Intercept=True  
        return tIntercept,Intercept

    def _FindLength(self,tIntercept,tIntercepts,RayLen):
        """ Find length of ray given intercept points along ray within
        a single cell."""
        
        L=0
        if tIntercepts==1:
            L = RayLen-tIntercept[0]
        if tIntercepts==2:
            L = abs(tIntercept[1]-tIntercept[0])
        if tIntercepts==3:
            tIntercept = np.sort(tIntercept)
            L = tIntercept[1]-tIntercept[0] + RayLen-tIntercept[2]
        if tIntercepts==4:
            tIntercept = np.sort(tIntercept)
            L = tIntercept[1]-tIntercept[0] + tIntercept[3]-tIntercept[2]
        return L
    
    def _FindCellList(self,Origin,EndPos,RMin,RValues,ZValues,RCells,ZCells,LinePoints):
        z1 = EndPos[2]
        z2 = Origin[2]
        zmin,zmax = self._FindMinMax(z1,z2)
        end=False
        CellMin=0
        CellMax=RCells*ZCells
        CellList = []
        for i in range(int(LinePoints[1])):
            PointIndex=i*LinePoints[0]
            CellIndex=PointIndex-i
            if ZValues[CellIndex+i]>zmax:
                CellMin=CellIndex
            if ZValues[CellIndex+i]<zmin and end==False:
                CellMax=CellIndex
                end=True
        # Now loop over cell z range and eliminate cells with r positions < rmin
        for cell in range(int(CellMin),int(CellMax)):
            RowNumber = int(cell/RCells)
            index = cell + RowNumber + 1
            if RMin < RValues[index]:
                CellList = np.append(CellList,cell)
        return CellList
        
    def _FindRMin(self,x,y,Origin,DirectionVector):
        b = 2.0*(Origin[0]*DirectionVector[0]+Origin[1]*DirectionVector[1])
        c = DirectionVector[0]**2 + DirectionVector[1]**2
        tMin = 0.5*np.abs(b)/c
        RMin, z = self._FindRz(x,y,tMin,Origin,DirectionVector)
        return RMin, z
        
    def _ShowProgress(self,y,ypixels):
      
        update = False
        progress_string = "\r[Calcam CalcGeometryMatrix] Progress: "
        
        if y==int(ypixels/10):
            
            progress_string += "10%..."
            update = True
        if y==int(ypixels/5):
            progress_string += "20%..."
            update = True
        if y==int(3*ypixels/10):
            progress_string += "30%..."
            update = True
        if y==int(4*ypixels/10):
            progress_string += "40%..."
            update = True
        if y==int(ypixels*0.5):
            progress_string += "50%..."
            update = True
        if y==int(6*ypixels/10):
            progress_string += "60%..."
            update = True
        if y==int(7*ypixels/10):
            progress_string += "70%..."
            update = True
        if y==int(8*ypixels/10):
            progress_string += "80%..."
            update = True
        if y==int(9*ypixels/10):
            progress_string += "90%..."
            update = True
            
        if update:
            sys.stdout.write(progress_string)
            sys.stdout.flush()
    #------------------------------------------------------------------------
    # Start of main code which calculates geometry matrix given set of rays
    # and inversion mesh
    #-------------------------------------------------------------------------
    def CalcGeometryMatrix(self,filename=None,fill_frac=0.03):
        
        LinePoints = [0,0]
        LinePoints[0] = self.nR
        LinePoints[1] = self.nZ
        RCells = self.nR - 1
        ZCells = self.nZ - 1
        RSeparation = (self.Rmax-self.Rmin)/RCells
        ZSeparation = abs((self.Zmax-self.Zmin)/ZCells)
        RValues = self.Rgrid
        ZValues = self.Zgrid
       
        Origin = self.RayData.ray_start_coords[0,0]
        RayEndPoints = self.RayData.ray_end_coords
        
        tmp = np.shape(self.RayData.ray_start_coords)
        xpixels = tmp[1]
        ypixels = tmp[0]
        #xpixels = np.max(self.RayData.x)+1
        #ypixels = np.max(self.RayData.y)+1
        
        # Using this ray and mesh information, a geometry matrix can be defined
        TotalRays = ypixels*xpixels
        TotalInversionCells = int(RCells*ZCells)
        #CellInterceptFrequency = np.zeros(TotalInversionCells)
        maxcells = fill_frac*TotalRays*TotalInversionCells
        LMatrixRows = np.zeros(maxcells,dtype='int32')
        LMatrixColumns = np.zeros(maxcells,dtype='int32')
        LMatrixValues = np.zeros(maxcells,dtype='float32')
        LMatrixShape = np.zeros(2)
        LMatrixShape[0] = TotalRays
        LMatrixShape[1] = TotalInversionCells
        self.LMatrixShape = LMatrixShape        
        
        sys.stdout.write('[Calcam geometry matrix] Starting geometry matrix calculation.\n')
        # Loop over all rays
        Count=0
        for y in np.arange(ypixels):
            for x in np.arange(xpixels):
                if x==0:
                    print_progress(y,ypixels,'',' Complete', bar_length=50)
                    
                RayIndex = x + y*xpixels
                # Find Direction Vector for this ray
                EndPos = RayEndPoints[y][x]
                RayLen = np.sqrt( (EndPos[0] - Origin[0])**2 + (EndPos[1] - Origin[1])**2 + (EndPos[2] - Origin[2])**2 )
                DirectionVector = (EndPos - Origin)/RayLen
                RMin, ZatRMin = self._FindRMin(x,y,Origin,DirectionVector)
                #PlotRz(x,y,RayLen,Origin,DirectionVector,RValues,ZValues)
                # Array of tIntercepts for each cell
                tIntercepts = np.zeros(TotalInversionCells)
                # Array of actual intercept values
                tIntercept = np.zeros((TotalInversionCells,4))
                # Loop over vertical lines (different r values)       
                for i in np.arange(LinePoints[0]):
                    # if z value at i is < RMin then no intercepts
                    if RValues[i]>RMin:
                        RLine = RValues[i]
                        a = DirectionVector[0]**2 + DirectionVector[1]**2
                        b = 2*Origin[0]*DirectionVector[0] + 2*Origin[1]*DirectionVector[1]
                        c = -RLine**2 + Origin[0]**2 + Origin[1]**2
                        t = self._SolveQuadratic(a,b,c)
                        if t[0]<RayLen:
                            ZPossible = Origin[2] + t[0]*DirectionVector[2]
                            # Check zPossible is within inversion mesh
                            if ZPossible<ZValues[0] and ZPossible>ZValues[-1]:
                                # Only one cell associated with intercept at edges
                                if i!=0 and i!=LinePoints[0]-1:
                                    index = abs(int((ZPossible-ZValues[0])/ZSeparation))
                                    PointIndex = LinePoints[0]*index+i
                                    CellIndex = PointIndex-index
                                    tIntercept[CellIndex,tIntercepts[CellIndex]] = t[0]
                                    tIntercepts[CellIndex] += 1  
                                    CellIndex= PointIndex-index-1
                                    tIntercept[CellIndex,tIntercepts[CellIndex]] = t[0]
                                    tIntercepts[CellIndex] += 1  
                                elif i==0:
                                    index = abs(int((ZPossible-ZValues[0])/ZSeparation))
                                    PointIndex = LinePoints[0]*index+i
                                    CellIndex = PointIndex-index
                                    tIntercept[CellIndex,tIntercepts[CellIndex]] = t[0]
                                    tIntercepts[CellIndex] += 1  
                                elif i==LinePoints[0]-1:
                                    index = abs(int((ZPossible-ZValues[0])/ZSeparation))
                                    PointIndex = LinePoints[0]*index+i
                                    CellIndex = PointIndex-index-1
                                    tIntercept[CellIndex,tIntercepts[CellIndex]] = t[0]
                                    tIntercepts[CellIndex] += 1  
                        if t[1]<RayLen:
                            ZPossible = Origin[2] + t[1]*DirectionVector[2]
                            # Check zPossible is within inversion mesh
                            if ZPossible<ZValues[0] and ZPossible>ZValues[-1]:
                                # Only one cell associated with intercept at edges
                                if i!=0 and i!=LinePoints[0]-1:
                                    index = abs(int((ZPossible-ZValues[0])/ZSeparation))
                                    PointIndex = LinePoints[0]*index+i
                                    CellIndex = PointIndex-index
                                    tIntercept[CellIndex,tIntercepts[CellIndex]] = t[1]
                                    tIntercepts[CellIndex] += 1  
                                    CellIndex = PointIndex-index-1
                                    tIntercept[CellIndex,tIntercepts[CellIndex]] = t[1]
                                    tIntercepts[CellIndex] += 1  
                                elif i==0:
                                    index = abs(int((ZPossible-ZValues[0])/ZSeparation))
                                    PointIndex = LinePoints[0]*index+i
                                    CellIndex = PointIndex-index
                                    tIntercept[CellIndex,tIntercepts[CellIndex]] = t[1]
                                    tIntercepts[CellIndex] += 1  
                                elif i==LinePoints[0]-1:
                                    index = abs(int((ZPossible-ZValues[0])/ZSeparation))
                                    PointIndex = LinePoints[0]*index+i
                                    CellIndex = PointIndex-index-1
                                    tIntercept[CellIndex,tIntercepts[CellIndex]] = t[1]
                                    tIntercepts[CellIndex] += 1                          
    
                # Loop over horizontal lines (different z values)
                for j in np.arange(LinePoints[1]):
                    ZLine = ZValues[j*LinePoints[0]]
                    t = (ZLine - Origin[2])/DirectionVector[2]
                    if t<RayLen:
                        RPossible = ((Origin[0] + t*DirectionVector[0])**2 + (Origin[1] + t*DirectionVector[1])**2)**0.5       
                        # Check if RPossible is within inversion mesh
                        if RPossible<RValues[LinePoints[0]-1] and RPossible>RValues[0]:
                            # Find which cell the intercept is within.
                            # If j=0 or final line then intercept only has one associated cell
                            if j!=0 and j!=LinePoints[1]-1:
                                index = int((RPossible-RValues[0])/RSeparation)
                                PointIndex = LinePoints[0]*j + index
                                CellIndex = PointIndex - j
                                tIntercept[CellIndex,tIntercepts[CellIndex]] = t
                                tIntercepts[CellIndex] += 1
                                CellIndex = CellIndex-(LinePoints[0]-1)
                                tIntercept[CellIndex,tIntercepts[CellIndex]] = t
                                tIntercepts[CellIndex] += 1
                            elif j==0:
                                index = int((RPossible-RValues[0])/RSeparation)
                                PointIndex = LinePoints[0]*j + index
                                CellIndex = PointIndex - j
                                tIntercept[CellIndex,tIntercepts[CellIndex]] = t
                                tIntercepts[CellIndex] += 1
                            elif j == LinePoints[1]-1:
                                index = int((RPossible-RValues[0])/RSeparation)
                                PointIndex = LinePoints[0]*(j-1) + index
                                CellIndex = PointIndex - (j-1)
                                tIntercept[CellIndex,tIntercepts[CellIndex]] = t
                                tIntercepts[CellIndex] += 1
                            
                # Use tIntercept arrays to calculate lengths through each cell
                CellIndices = np.linspace(0,TotalInversionCells,TotalInversionCells+1)
                CellIndices = CellIndices[np.where(tIntercepts>0)]
                tIntercept = tIntercept[np.where(tIntercepts>0)]
                tIntercepts = tIntercepts[np.where(tIntercepts>0)]
                TotalL = 0
                if len(CellIndices>0):
                    Lengths = np.zeros(len(CellIndices))
                    for i in np.arange(len(Lengths)):
                        Lengths[i] = self._FindLength(tIntercept[i],tIntercepts[i],RayLen)
                        TotalL += Lengths[i]
                    for i in np.arange(len(CellIndices)):
                        #CellInterceptFrequency[CellIndices[i]]+=1
                        LMatrixRows[Count] = RayIndex
                        LMatrixColumns[Count] = CellIndices[i]
                        if CellIndices[i] == TotalInversionCells:
                            print('Error: Cell index out of range')
                            self._PlotRz(x,y,RayLen,Origin,DirectionVector,RValues,ZValues)
                        if LMatrixRows[Count] >TotalRays:
                            print('Error: Rayindex out of range')
                            self._PlotRz(x,y,RayLen,Origin,DirectionVector,RValues,ZValues)
                        LMatrixValues[Count] = Lengths[i]
                        Count+=1
        
        # Will want to convert this to sparse matrix so ignore zero values.
        self.LMatrixRows = LMatrixRows[np.where(LMatrixValues>0.0)]
        self.LMatrixColumns = LMatrixColumns[np.where(LMatrixValues>0.0)]
        self.LMatrixValues = LMatrixValues[np.where(LMatrixValues>0.0)]
        
        # Save the geometry matrix to a binary file
        if filename is not None:
            self.SaveGeometryMatrix(filename=filename)
        else:
            self.SaveGeometryMatrix()
            
        sys.stdout.write('\r[Calcam geometry_matrix] Geometry matrix calculation complete.\n')
        sys.stdout.flush()
    
    def calcRaysPerCell(self):
        """Returns the number of rays intersecting elements of the inversion mesh"""        
        
        if self.LMatrixShape is None:
            raise Exception('[Calcam geometry_matrix] Geometry matrix has not been calculated.')
        
        cellfreq = np.zeros(self.LMatrixShape[1])
    
        for i in np.arange(self.LMatrixShape[1]):
            cellfreq[i] = np.sum((self.LMatrixColumns == i)*1.0)

        return cellfreq
            
    def SaveGeometryMatrix(self,filename='geo_matrix',npformat=False,matformat=False):
        """Save a geometry matrix from a numpy or Matlab .mat file"""

        validGM = (self.LMatrixColumns is not None) & \
                  (self.LMatrixRows is not None) & \
                  (self.LMatrixValues is not None)
      
        if validGM:
    
            if npformat is True:
            
                np.savez_compressed(filename, \
                    columns = self.LMatrixColumns,\
                    rows = self.LMatrixRows,\
                    values = self.LMatrixValues,\
                    shape = self.LMatrixShape, \
                    grid_rmin = self.Rmin,\
                    grid_rmax = self.Rmax,\
                    grid_nr = self.nR,\
                    grid_zmin = self.Zmin,\
                    grid_zmax = self.Zmax,\
                    grid_nz = self.nZ,\
                    gridtype = 'RectangularGeometryMatrix')
                                
            if matformat is True:
                from scipy.io import savemat
                
                savemat(filename, \
                        mdict={'columns':   self.LMatrixColumns,\
                               'rows':      self.LMatrixRows,\
                               'values':    self.LMatrixValues,\
                               'shape':     self.LMatrixShape, \
                               'grid_rmin': self.Rmin,\
                               'grid_rmax': self.Rmax,\
                               'grid_nr':   self.nR,\
                               'grid_zmin': self.Zmin,\
                               'grid_zmax': self.Zmax,\
                               'grid_nz':   self.nZ,\
                               'gridtype':  'RectangularGeometryMatrix'})
            
    def LoadGeometryMatrix(self,filename=None,npformat=True,matformat=False):
        """Load a geometry matrix from a numpy or Matlab .mat file"""
        
        if (filename is not None):
            if npformat is True:
                dat = np.load(filename+'.npz')
                self.LMatrixRows = dat['rows']
                self.LMatrixColumns = dat['columns']
                self.LMatrixValues = dat['values']
                self.LMatrixShape = dat['shape']
                self.Rmin = dat['grid_rmin']
                self.Rmax = dat['grid_rmax']
                self.nR = dat['grid_nr']
                self.Zmin = dat['grid_zmin']
                self.Zmax = dat['grid_zmax']
                self.nZ = dat['grid_nz']
            if matformat is True:
                from scipy.io import loadmat
                dat = loadmat(filename+'.mat')
                self.LMatrixRows = dat['rows']
                self.LMatrixColumns = dat['columns']
                self.LMatrixValues = dat['values']
                self.LMatrixShape = dat['shape']
                self.Rmin = dat['grid_rmin']
                self.Rmax = dat['grid_rmax']
                self.nR = dat['grid_nr']
                self.Zmin = dat['grid_zmin']
                self.Zmax = dat['grid_zmax']
                self.nZ = dat['grid_nz']
        else:
            raise Exception('[Calcam geometry_matrix] Specify file name.')

class TriangularGeometryMatrix:
    def __init__(self,wall_r=None,wall_z=None,cut_start_r=None,cut_end_r=None, \
                 cut_start_z=None,cut_end_z=None,holes_r=None,holes_z=None,\
                 tri_area=0.0005,tridir=None,filename=None,RayData=None):
                     
        from numpy import arange, loadtxt, array
        from numpy import int as npint
        import subprocess
        import os
        
        self.RayData = RayData
        
        if filename is None:
            
            inputfile = 'tri_input'
            
            # Write the wall data to a triangle input file
            f = open(tridir+inputfile+'.poly','w')
            f.write(str(len(wall_r)+len(cut_start_r)*2)+'  2  1  0\n')
            for i in arange(len(wall_r)):
                f.write('{:<3d}{:10.4f}{:10.4f}\n'.format(i+1,wall_r[i],wall_z[i]))
                
            # write the cuts to the input file
            if (cut_start_r is not None) and (cut_end_r is not None) and \
               (cut_start_z is not None) and (cut_end_z is not None):
                ctr = len(wall_r)
                for i in arange(len(cut_start_r)):
                    f.write('{:<3d}{:10.4f}{:10.4f}\n'.format(ctr+i+1,cut_start_r[i],cut_start_z[i]))
                    ctr = ctr+1
                    f.write('{:<3d}{:10.4f}{:10.4f}\n'.format(ctr+i+1,cut_end_r[i],cut_end_z[i]))
            
                # write the polygon connection to the input file    
                f.write(str(len(wall_r)+len(cut_start_r)-1)+' 0\n')
                for i in arange(len(wall_r)-1):
                    f.write('{:<3d}{:4d}{:4d}\n'.format(i+1,i+1,i+2))
                    
                # write the cut lines to the input file
                ctr = len(wall_r)-1
                for i in arange(len(cut_start_r)):
                    f.write('{:<3d}{:4d}{:4d}\n'.format(ctr+i+1,ctr+i+2,ctr+i+3))
                    ctr = ctr+1
                
            if (holes_r is not None) and (holes_z is not None):
                # Write the holes
                f.write('{:<2d}\n'.format(len(holes_r)))
                for i in arange(len(holes_r)):
                    f.write('{:<2d}{:10.4f}{:10.4f}\n'.format(i+1,holes_r[i],holes_z[i]))
                    
            # Done :-)
            f.close()
                
            area = '{:f}'.format(tri_area)
            
            # Call triangle
            if tridir is None:
                # Get the current directory
                subprocess.call('triangle -pqa'+area+' '+inputfile,shell=True)
            else:
                wrkdir = os.getcwd()
                os.chdir(tridir)
                subprocess.call(tridir+'triangle -pqa'+area+' '+inputfile,shell=True)
                os.chdir(wrkdir)
        
            elefile = inputfile+'.1.ele'
            nodefile = inputfile+'.1.node'
            
            elements = array(loadtxt(tridir+elefile,skiprows=1),dtype=npint)
            nodes = loadtxt(tridir+nodefile,skiprows=1)
            
            self.tri_x = nodes[:,1]
            self.tri_y = nodes[:,2]
            self.tri_nodes = elements[:,1:4]-1
            
        else:
            self.LoadGeometryMatrix(filename)
        
    def fsign (self,p1x, p1y, p2x, p2y, p3x, p3y):
    
        return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y);
    
    
    def PointInTriangle (self,ptx, pty, v1x, v1y, v2x, v2y, v3x, v3y):
        """Determines whether at point (ptx,pty) is enclosed by the vertices
        (v1x,v1y) (v2x,v2y) (v3x,v3y)"""
    
        b1 = self.fsign(ptx, pty, v1x, v1y, v2x, v2y) < 0.0
        b2 = self.fsign(ptx, pty, v2x, v2y, v3x, v3y) < 0.0
        b3 = self.fsign(ptx, pty, v3x, v3y, v1x, v1y) < 0.0
    
        return ((b1 == b2) & (b2 == b3))
        
    def ray_point_dist_2d(self,line_start_x,line_start_y,line_end_x,line_end_y,pt_x,pt_y):
        """Calculates the distance along a line, defined by start and end points,
        and an arbitrary point (assumed to be along the line)"""
    
        line_dir_x = line_end_x-line_start_x
        line_dir_y = line_end_y-line_start_y
    
        ts1 = np.sqrt((pt_x-line_start_x)**2+(pt_y-line_start_y)**2) / \
              np.sqrt((line_end_x-line_start_x)**2+(line_end_y-line_start_y)**2)
                       
        # Check the sign is correct
        ts1 = ts1*np.sign((pt_x-line_start_x)*line_dir_x+(pt_y-line_start_y)*line_dir_y)
        
        return ts1
        
        
    def ray_line_intersect(self,ray_start,ray_dir,line_start,line_end):
        """Calculates the intersection between a 3D line of sight and a 2D 
        line defined in 2D cylindrical polar coordinates (R,Z)"""
        
        from numpy import sqrt,zeros,nan,where
        
        # Initialise parametric ray coefficients
        pax = ray_start[0]
        pay = ray_start[1]
        paz = ray_start[2]
        dpx = ray_dir[0]
        dpy = ray_dir[1]
        dpz = ray_dir[2]

        # Initialise parametric line coefficients
        lar = line_start[:,0]
        laz = line_start[:,1]
        dlr = line_end[:,0]-line_start[:,0]
        dlz = line_end[:,1]-line_start[:,1]

        a = -dlz**2*dpx**2 - dlz**2*dpy**2 + dlr**2*dpz**2
        b = 2*dlr*dlz*dpz*lar - 2*dlr**2*dpz*laz - 2*dlz**2*dpx*pax - 2*dlz**2*dpy*pay + 2*dlr**2*dpz*paz
        c = (dlz**2*lar**2 - 2*dlr*dlz*lar*laz + dlr**2*laz**2 - dlz**2*pax**2 - dlz**2*pay**2 + 2*dlr*dlz*lar*paz - 2*dlr**2*laz*paz + dlr**2*paz**2)
        
        tp0 = zeros((len(a)))*nan
        tl0 = zeros((len(a)))*nan
        tp1 = zeros((len(a)))*nan
        tl1 = zeros((len(a)))*nan

        d = b*b - 4*a*c

        # d < 0 means imaginary solution and hence no intersections, ignore this case

        # d > 0 means two real solutions and hence two intersections

        indx = where((d > 0) & (b < 0))
        if len(indx) > 0:
            q = -0.5 * (b[indx] - sqrt(d[indx]))
            tp0[indx] = q/a[indx]
            tl0[indx] = (-laz[indx] + paz + dpz * tp0[indx])/dlz[indx]
            tp1[indx] = c[indx] / q
            tl1[indx] = (-laz[indx] + paz + dpz * tp1[indx])/dlz[indx]
        
        indx = where((d > 0) & (b > 0))
        if len(indx) > 0:
            q = -0.5 * (b[indx] + sqrt(d[indx]))
            tp0[indx] = q/a[indx]
            tl0[indx] = (-laz[indx] + paz + dpz * tp0[indx])/dlz[indx]
            tp1[indx] = c[indx] / q
            tl1[indx] = (-laz[indx] + paz + dpz * tp1[indx])/dlz[indx]
            
        indx = where((d == 0) & (b < 0))
        if len(indx) > 0:
            q = -0.5 * (b[indx] - sqrt(d[indx]))
            tp0[indx] = q / a[indx]
            tl0[indx] = (-laz[indx] + paz + dpz * tp0[indx])/dlz[indx]
            
        indx = where((d == 0) & (b > 0))
        if len(indx) > 0:
            q = -0.5 * (b[indx] + sqrt(d[indx]))
            tp0[indx] = q / a[indx]
            tl0[indx] = (-laz[indx] + paz + dpz * tp0[indx])/dlz[indx]
            
        indx = where(dlz == 0)
        if len(indx) > 0:
            tp0[indx] = (-paz + laz[indx])/dpz
            hitr      = sqrt((pax+tp0[indx]*dpx)**2+(pay+tp0[indx]*dpy)**2)
            tl0[indx] = (-lar[indx] + hitr)/dlr[indx]
            
        return tl0, tp0, tl1, tp1
        
        
    def calc_tri_geomat(self,ray_orig,ray_dir,raylen,tri_start1,tri_end1,\
                        tri_start2,tri_end2,tri_start3,tri_end3,eps=1.0E-3):
        """Calculates each element of the geometry matrix by calculating the
        path a line of sight passes through a triangle, called by 
        CalcGeometryMatrix"""
        
        from numpy import where, arange, ndarray, array, sort, sqrt, isfinite, \
                          argmin, abs
        
        
        hit1 = self.ray_line_intersect(ray_orig,ray_dir,tri_start1,tri_end1)
        hit2 = self.ray_line_intersect(ray_orig,ray_dir,tri_start2,tri_end2)
        hit3 = self.ray_line_intersect(ray_orig,ray_dir,tri_start3,tri_end3)

        indx1 = ((hit1[0] > 0.0) & (hit1[0] < 1.0) & (hit1[1] > 0.0) & (hit1[1] < raylen))
        indx2 = ((hit1[2] > 0.0) & (hit1[2] < 1.0) & (hit1[3] > 0.0) & (hit1[3] < raylen))
                
        indx3 = ((hit2[0] > 0.0) & (hit2[0] < 1.0) & (hit2[1] > 0.0) & (hit2[1] < raylen))
        indx4 = ((hit2[2] > 0.0) & (hit2[2] < 1.0) & (hit2[3] > 0.0) & (hit2[3] < raylen))
        
        indx5 = ((hit3[0] > 0.0) & (hit3[0] < 1.0) & (hit3[1] > 0.0) & (hit3[1] < raylen))
        indx6 = ((hit3[2] > 0.0) & (hit3[2] < 1.0) & (hit3[3] > 0.0) & (hit3[3] < raylen))
        
        # Calculate the number of hits of the ray with each side of the triangle
        n_hits = indx1*1.0+indx2*1.0+indx3*1.0+indx4*1.0+indx5*1.0+indx6*1.0
        
        # Find the triangles the ray has intersected
        indx = where(n_hits > 0)
        
        # Store the indices of the triangles hit and the path length within them
        cellindx = ndarray(len(indx[0]))
        celllen  = ndarray(len(indx[0]))
        thits = np.zeros(6)
        
        for i in arange(len(indx[0])):
            index = indx[0][i]
            
            thit = array([hit1[1][index],hit1[3][index],hit2[1][index],hit2[3][index],\
                             hit3[1][index],hit3[3][index]])
                             
            thits = array([hit1[0][index],hit1[2][index],hit2[0][index],hit2[2][index],hit3[0][index],hit3[2][index]])
        
            thit = array([hit1[1][index],hit1[3][index],hit2[1][index],hit2[3][index],hit3[1][index],hit3[3][index]])        
        
            rayxhit = ray_orig[0]+thit*ray_dir[0]
            rayyhit = ray_orig[1]+thit*ray_dir[1]
            rayzhit = ray_orig[2]+thit*ray_dir[2]
            rayrhit = np.sqrt(rayxhit**2+rayyhit**2)
        
            tmp = ((thits > 0.0) & (thits < 1.0) & (thit > 0) & (thit < raylen))*1.0
        
            # Keep a tally of valid hits
            thitr = thit*tmp                     
            thitr[thitr <= 0.0] = 1.0E4
            thitr[thitr > raylen] = 1.0E4
            thitr = sort(thitr)
                             
            if n_hits[index] == 1:
                # Check if ray ended in cell
                rayxhit = ray_orig[0]+raylen*ray_dir[0]
                rayyhit = ray_orig[1]+raylen*ray_dir[1]
                rayzhit = ray_orig[2]+raylen*ray_dir[2]
                rayrhit = sqrt(rayxhit**2+rayyhit**2)
                endtri = self.PointInTriangle(rayrhit,rayzhit,tri_start1[index,0],tri_start1[index,1],\
                                              tri_start2[index,0],tri_start2[index,1],tri_start3[index,0],\
                                              tri_start3[index,1])
                # 1 hit, checking if ray ended in triangle
                if endtri == True:
                    # yep, ray ended in triangle
                    cell_len = raylen-thitr[0]
                else:
                    # nope, ray didnt end in triangle, triggering an error
                    
                    # Keep track of invalid hits
                    tmp = ((thits < 0.0) | (thits > 1.0))*1.0
                    thiti = thits*tmp
                    thiti[thiti == 0.0] = 1.0E4
                    thiti[~isfinite(thiti)] = 1.0E4
    
                    idx1 = argmin(abs(thiti))
                    idx2 = argmin(abs(1.0-thiti))
                    if abs(thits[idx1]) < abs(1.0-thits[idx2]):
                        idx = idx1
                    else:
                        idx = idx2
                    
                    cell_len = abs(thitr[0]-thit[idx])
                    
            if n_hits[index] == 2:
                cell_len = thitr[1]-thitr[0]
            
            if n_hits[index] == 3:
                # Check if ray ended in cell
                rayxhit = ray_orig[0]+raylen*ray_dir[0]
                rayyhit = ray_orig[1]+raylen*ray_dir[1]
                rayzhit = ray_orig[2]+raylen*ray_dir[2]
                rayrhit = sqrt(rayxhit**2+rayyhit**2)
                endtri = self.PointInTriangle(rayrhit,rayzhit,tri_start1[index,0],tri_start1[index,1],\
                                              tri_start2[index,0],tri_start2[index,1],tri_start3[index,0],\
                                              tri_start3[index,1])
                # 3 hits, checking if ray ended in triangle
                if endtri == True:
                    # yep, ray ended in triangle
                    cell_len = (raylen-thitr[2])+(thitr[1]-thitr[0])
                else:
                    # nope, ray didnt end in triangle - this could be due either
                    # there being two intersections near a vertex and one elsewhere
                    # or the valid intersections and the 4th just missed
                    
                    # Keep track of invalid hits
                    tmp = ((thits < 0.0) | (thits > 1.0))*1.0
                    thiti = thits*tmp
                    idx = np.where(thiti > 1.0)
                    thiti[idx] = thiti[idx]-1.0
                    thiti[thiti == 0.0] = 1.0E4
                    thiti[~isfinite(thiti)] = 1.0E4
                    
                    if np.min(abs(thiti)) < 0.1:
                        # Probably just missed a valid hit, find the closest
                        # invalid hit and use this
                        idx = argmin(abs(thiti))
                    
                        tmp = array([thit[idx],thitr[2],thitr[1],thitr[0]])
                        tmp = sort(tmp)
                
                        cell_len = (tmp[3]-tmp[2])+(tmp[1]-tmp[0])
                    else:
                        # Picked up a false positive, probably near a vertex,
                        # Just take the highest difference in path length between
                        # one intersection and the next
                        cell_len = np.max(np.diff(thitr[0:3]))
                        
            if n_hits[index] == 4:
                cell_len = (thitr[3]-thitr[2])+(thitr[1]-thitr[0])
                
            if n_hits[index] == 5:
                # Check if ray ended in cell
                rayxhit = ray_orig[0]+raylen*ray_dir[0]
                rayyhit = ray_orig[1]+raylen*ray_dir[1]
                rayzhit = ray_orig[2]+raylen*ray_dir[2]
                rayrhit = sqrt(rayxhit**2+rayyhit**2)
                endtri = self.PointInTriangle(rayrhit,rayzhit,tri_start1[index,0],tri_start1[index,1],\
                                              tri_start2[index,0],tri_start2[index,1],tri_start3[index,0],\
                                              tri_start3[index,1])
                # 5 hits, checking if ray ended in triangle
                if endtri == True:
                    # yep, ray ended in triangle
                    cell_len = (raylen-thitr[4])+(thitr[3]-thitr[2])+(thitr[1]-thitr[0])
                else:
                    # nope, ray didnt end in triangle - this could be due either
                    # there being two intersections near a vertex and one elsewhere
                    # or the valid intersections and the 6th just missed
                    
                    # Keep track of invalid hits
                    tmp = ((thits < 0.0) | (thits > 1.0))*1.0
                    thiti = thits*tmp
                    idx = np.where(thiti > 1.0)
                    thiti[idx] = thiti[idx]-1.0
                    thiti[thiti == 0.0] = 1.0E4
                    thiti[~isfinite(thiti)] = 1.0E4
                    
                    # Probably just missed a valid hit, find the closest
                    # invalid hit and use this
                    idx = argmin(abs(thiti))
                    
                    tmp = array([thit[idx],thitr[2],thitr[1],thitr[0]])
                    tmp = sort(tmp)
                
                    cell_len = (tmp[5]-tmp[4])+(tmp[3]-tmp[2])+(tmp[1]-tmp[0])
                
            if n_hits[index] == 6:
                cell_len = (thitr[5]-thitr[4])+(thitr[3]-thitr[2])+(thitr[1]-thitr[0])
        
            cellindx[i] = index
            celllen[i] = cell_len
        
        return cellindx, celllen
        
    def CalcGeometryMatrix(self,filename=None,fill_frac=0.1):
        """Calculate the geometry matrix"""
    
        from numpy import shape, zeros, arange, sqrt, vstack
        
        #raydata = calcam.RayData('89346d_400-408_2005_low')
        tmp = shape(self.RayData.ray_start_coords)
        xpixels = tmp[1]
        ypixels = tmp[0]
        
        TotalRays = ypixels*xpixels
        TotalInversionCells = len(self.tri_nodes)
        maxcells = fill_frac*TotalRays*TotalInversionCells
        LMatrixRows = zeros(maxcells,dtype='int32')
        LMatrixColumns = zeros(maxcells,dtype='int32')
        LMatrixValues = zeros(maxcells,dtype='float32')
        self.LMatrixShape = zeros(2)
        self.LMatrixShape[0] = TotalRays
        self.LMatrixShape[1] = TotalInversionCells
        
        tri_start1 = vstack((self.tri_x[self.tri_nodes[:,0]],self.tri_y[self.tri_nodes[:,0]])).transpose()
        tri_end1   = vstack((self.tri_x[self.tri_nodes[:,1]],self.tri_y[self.tri_nodes[:,1]])).transpose()
        
        tri_start2 = vstack((self.tri_x[self.tri_nodes[:,1]],self.tri_y[self.tri_nodes[:,1]])).transpose()
        tri_end2   = vstack((self.tri_x[self.tri_nodes[:,2]],self.tri_y[self.tri_nodes[:,2]])).transpose()
        
        tri_start3 = vstack((self.tri_x[self.tri_nodes[:,2]],self.tri_y[self.tri_nodes[:,2]])).transpose()
        tri_end3   = vstack((self.tri_x[self.tri_nodes[:,0]],self.tri_y[self.tri_nodes[:,0]])).transpose()
        
        # Loop over all rays
        sys.stdout.write('[Calcam geometry matrix] Starting geometry matrix calculation.\n')
        Count=0
        for y in arange(ypixels):
            for x in arange(xpixels):
                if x == 0:
                    print_progress(y,ypixels,'',' Complete', bar_length=50)
                    
                RayIndex = x + y*xpixels
                # Find Direction Vector for this ray
                Origin = self.RayData.ray_start_coords[y][x]
                EndPos = self.RayData.ray_end_coords[y][x]
                RayLen = sqrt((Origin[0]-EndPos[0])**2+(Origin[1]-EndPos[1])**2+(Origin[2]-EndPos[2])**2)
                DirectionVector = (EndPos - Origin)/RayLen
        
                gm_cellindex, gm_celllen = self.calc_tri_geomat(Origin,DirectionVector,RayLen,tri_start1,tri_end1,tri_start2,tri_end2,tri_start3,tri_end3)
        
                nel = len(gm_cellindex)        
                LMatrixRows[Count:Count+nel] = zeros(len(gm_cellindex))+RayIndex
                LMatrixColumns[Count:Count+nel] = gm_cellindex
                LMatrixValues[Count:Count+nel] = gm_celllen
                
                Count = Count+nel
                
        self.LMatrixRows = LMatrixRows[0:Count]
        self.LMatrixColumns = LMatrixColumns[0:Count]
        self.LMatrixValues = LMatrixValues[0:Count]
        sys.stdout.write('[Calcam geometry_matrix] Geometry matrix calculation complete.\n')
        
        if filename is not None:
            self.SaveGeometryMatrix(filename=filename)
            
    def calcRaysPerCell(self):
        """Returns the number of rays intersecting elements of the inversion mesh"""        
        
        if self.LMatrixShape is None:
            raise Exception('[Calcam geometry_matrix] Geometry matrix has not been calculated.')
        
        cellfreq = np.zeros(self.LMatrixShape[1])
    
        for i in np.arange(self.LMatrixShape[1]):
            cellfreq[i] = np.sum((self.LMatrixColumns == i)*1.0)

        return cellfreq
            
    def SaveGeometryMatrix(self,filename='geo_matrix',npformat=True,matformat=False):
        """Save a geometry matrix from a numpy or Matlab .mat file"""
      
        if npformat is True:
        
            if ((self.LMatrixColumns is not None) &
                (self.LMatrixRows is not None) &
                (self.LMatrixValues is not None)):
    
                np.savez_compressed(filename, \
                    columns   = self.LMatrixColumns,\
                    rows      = self.LMatrixRows,\
                    values    = self.LMatrixValues,\
                    shape     = self.LMatrixShape, \
                    tri_x     = self.tri_x, \
                    tri_y     = self.tri_y, \
                    tri_nodes = self.tri_nodes, \
                    gridtype  = 'TriangularGeometryMatrix')
                            
        if matformat is True:
            from scipy.io import savemat
            
            savemat(filename, \
                    mdict={'columns':   self.LMatrixColumns,\
                           'rows':      self.LMatrixRows,\
                           'values':    self.LMatrixValues,\
                           'shape':     self.LMatrixShape, \
                           'tri_x':     self.tri_x, \
                           'tri_y':     self.tri_y, \
                           'tri_nodes': self.tri_nodes, \
                           'gridtype':  'TriangularGeometryMatrix'})
                
    def LoadGeometryMatrix(self,filename=None,npformat=True,matformat=False):
        """Load a geometry matrix from a numpy or Matlab .mat file"""
      
        if (filename is not None):
            if npformat is True:
                dat = np.load(filename+'.npz')
                self.LMatrixColumns = dat['columns']
                self.LMatrixRows    = dat['rows']
                self.LMatrixValues  = dat['values']
                self.LMatrixShape   = dat['shape']
                self.tri_x          = dat['tri_x']
                self.tri_y          = dat['tri_y']
                self.tri_nodes      = dat['tri_nodes']
            if matformat is True:
                from scipy.io import loadmat
                dat = loadmat(filename+'.mat')
                self.LMatrixColumns = dat['columns']
                self.LMatrixRows    = dat['rows']
                self.LMatrixValues  = dat['values']
                self.LMatrixShape   = dat['shape']
                self.tri_x          = dat['tri_x']
                self.tri_y          = dat['tri_y']
                self.tri_nodes      = dat['tri_nodes']
        else:
            raise Exception('[Calcam geometry_matrix] Specify file name.')