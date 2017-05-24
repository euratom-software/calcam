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

Author: Mark Smithies & James Harrison
Last Updated: 15/11/16
"""

#import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
#import pupynere
#import cv2

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
        
        # TODO: Get the origin and direction from x, y!
        R = np.sqrt( (Origin[0] + t*DirectionVector[0])**2 + (Origin[1] + t*DirectionVector[1])**2)
        z = Origin[2] + t*DirectionVector[2]
        return R, z

    def PlotRz(self,x,y,RayLen,Origin,DirectionVector,RValues,ZValues):
        """For given ray, plot its path through R-z space. Also overplots
        chosen mesh onto Rz space by calling function PlotMesh(). Used for
        debugging purposes."""
        
        # Todo: Get the origin and direction from x, y!
        ChordPoints = 100
        t = np.linspace(0, RayLen,ChordPoints)
        R = np.zeros(ChordPoints)
        z = np.zeros(ChordPoints)
        for i in range(len(t)):
            R[i], z[i] = self._FindRz(x,y,t[i],Origin,DirectionVector)    
        plt.plot(R, z)
        plt.xlabel("R")
        plt.ylabel("z")
        self._PlotMesh(RValues,ZValues)
    
    def PlotMesh(self):
        "What it says on the tin really"
        plt.plot(self.Rgrid, self.Zgrid, 'rx')
        self._PlotChamber()
        plt.show()

    def _PlotChamber(self):
        wall_r = [3.2900,3.1990,3.1940,3.0600,3.0110,2.9630,2.9070,2.8880,2.8860,2.8900,2.9000, \
              2.8840,2.8810,2.8980,2.9870,2.9460,2.8700,2.8340,2.8140,2.7000,2.5760,2.5550, \
              2.5500,2.5220,2.5240,2.4370,2.4060,2.4180,2.4210,2.3980,2.4080,2.4130,2.4130, \
              2.4050,2.3600,2.2950,2.2940,2.2000,2.1390,2.0810,1.9120,1.8210,1.8080,1.8860, \
              1.9040,1.9160,2.0580,2.1190,2.1780,2.2430,2.3810,2.5750,2.8840,2.9750,3.1490, \
              3.3000,3.3980,3.4770,3.5620,3.6400,3.6410,3.7430,3.8240,3.8855,3.8925,3.8480, \
              3.7160,3.5780,3.3590,3.3090,3.2940,3.2900]
        
        wall_z = [-1.1520,-1.2090,-1.2140,-1.2980,-1.3350,-1.3350,-1.3830,-1.4230,-1.4760,-1.4980, \
              -1.5100,-1.5820,-1.6190,-1.6820,-1.7460,-1.7450,-1.7130,-1.7080,-1.6860,-1.6510, \
              -1.6140,-1.6490,-1.6670,-1.7030,-1.7100,-1.7110,-1.6900,-1.6490,-1.6020,-1.5160, \
              -1.5040,-1.4740,-1.4290,-1.3860,-1.3340,-1.3340,-1.3200,-1.2440,-1.0970,-0.9580, \
              -0.5130,-0.0230,0.4260,1.0730,1.1710,1.2320,1.6030,1.7230,1.8390,1.8940,1.9670, \
              2.0160,1.9760,1.9410,1.8160,1.7110,1.6430,1.5720,1.4950,1.4250,1.2830,1.0700, \
              0.8290,0.4950,0.2410,-0.0960,-0.4980,-0.7530,-1.0310,-1.0800,-1.1110,-1.1520]   
        plt.plot(wall_r,wall_z,'g')
    
    
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
        progress_string = "\r[CalcGeometryMatrix] Progress: "
        
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
    def CalcGeometryMatrix(self,filename=None):
        
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
        maxcells = 0.1*TotalRays*TotalInversionCells
        LMatrixRows = np.zeros(maxcells,dtype='float32')
        LMatrixColumns = np.zeros(maxcells,dtype='float32')
        LMatrixValues = np.zeros(maxcells,dtype='float32')
        LMatrixShape = np.zeros(2)
        LMatrixShape[0] = TotalRays
        LMatrixShape[1] = TotalInversionCells
        self.LMatrixShape = LMatrixShape        
        
        # Loop over all rays
        Count=0
        for y in np.arange(ypixels):
            for x in np.arange(xpixels):
                if x==0:
                    self._ShowProgress(y,ypixels)
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
            
        sys.stdout.write('\rGeometry matrix calculation complete.\n')
        sys.stdout.flush()
            
    def SaveGeometryMatrix(self,filename='geo_matrix'):
      
        if ((self.LMatrixColumns is not None) &
          (self.LMatrixRows is not None) &
          (self.LMatrixValues is not None)):
    
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
            
    def LoadGeometryMatrix(self,filename=None):
        
        if filename is None:
            raise Exception('LoadGeometryMatrix: no file name provided.')
            
        f = open(filename,'rb')
        try:
            dat = pickle.load(f)
        except:
            f.seek(0)
            dat = pickle.load(f,encoding='latin1')
        f.close()
        self.LMatrixRows    = dat[0]
        self.LMatrixColumns = dat[1]
        self.LMatrixValues  = dat[2]
        self.LMatrixShape   = dat[3]
        self.Rmin           = dat[4][0]
        self.Rmax           = dat[4][1]
        self.nR             = dat[4][2]
        self.Zmin           = dat[5][0]
        self.Zmax           = dat[5][1]
        self.nZ             = dat[5][2]
        self.RayData        = dat[6]