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
This module is for internal use in calcam.

It has variables containing the paths to various things e.g. the path the code is running from, 
paths where results files live etc. Changing stuff in here will change where the code looks for
saved results, etc.

Written by Scott Silburn.
"""

import os
import sys
import inspect
import shutil
import socket
import difflib

# The CalCam user data directory goes in the user's homedir
homedir = os.path.expanduser('~')

# Path where the calcam code is
calcampath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# These are the paths for different types of things
root = os.path.join(homedir,'calcam')
ui = os.path.join(calcampath,'ui')

pointpairs = os.path.join(root,'PointPairs')
fitresults = os.path.join(root,'FitResults')
raydata = os.path.join(root,'RayData')
images = os.path.join(root,'Images')
rois = os.path.join(root,'ROIs')
machine_geometry = os.path.join(root,'UserCode','machine_geometry')
image_sources =  os.path.join(root,'UserCode','image_sources')
code = os.path.join(root,'UserCode')
virtualcameras = os.path.join(root,'VirtualCameras')

# Check whether these actually exist, and create them if they don't.
if not os.path.isdir(pointpairs):
    print('[Calcam Setup] Creating user data directory at ' + pointpairs)
    os.makedirs(pointpairs)

if not os.path.isdir(fitresults):
    print('[Calcam Setup] Creating user data directory at ' + fitresults)
    os.makedirs(fitresults)

if not os.path.isdir(raydata):
    print('[Calcam Setup] Creating user data directory at ' + raydata)
    os.makedirs(raydata)

if not os.path.isdir(images):
    print('[Calcam Setup] Creating user data directory at ' + images)	
    os.makedirs(images)

if not os.path.isdir(machine_geometry):
    print('[Calcam Setup] Creating user code directory at ' + machine_geometry)
    os.makedirs(machine_geometry)

if not os.path.isdir(image_sources):
    print('[Calcam Setup] Creating user code directory at ' + image_sources)
    os.makedirs(image_sources)

if not os.path.isdir(rois):
    print('[Calcam Setup] Creating user data directory at ' + rois)
    os.makedirs(rois)

if not os.path.isdir(virtualcameras):
    print('[Calcam Setup] Creating user data directory at ' + virtualcameras)
    os.makedirs(virtualcameras)

# Add the user code directories to our python path.
sys.path.append(machine_geometry)
sys.path.append(image_sources)

def get_save_list(save_type):

    if save_type == 'PointPairs':
        extensions = ['.csv']
        location = pointpairs
    elif save_type == 'FitResults':
        extensions = ['.pickle','.csv']
        location = fitresults
    elif save_type == 'RayData':
        extensions = ['.nc']
        location = raydata
    elif save_type == 'Images':
        extensions = ['.nc']
        location = images
    elif save_type == 'ROIs':
        extensions = ['.csv']
        location = rois
    elif save_type == 'VirtualCameras':
        extensions = ['.pickle']
        location = virtualcameras
    else:
        raise Exception('Cannot list unknown item type ' + save_type)

    filelist = os.listdir(location)

    if save_type != 'ROIs':
        save_name_list = []
        for filename in filelist:
            for extension in extensions:
                if filename.lower().endswith(extension):
                    save_name_list.append(filename.replace(extension,''))
    else:
        save_name_list = []

        for filename in filelist:

            if os.path.isfile(os.path.join(location,filename)):
                for extension in extensions:
                    if filename.lower().endswith(extension):
                        save_name_list.append((filename.replace(extension,''),None))

            elif os.path.isdir(os.path.join(location,filename)):
                save_name_list.append((filename,None))
                filelist2 = os.listdir(os.path.join(location,filename))
                for filename2 in filelist2:
                    if os.path.isfile(os.path.join(location,filename,filename2)):
                        for extension in extensions:
                            if filename2.lower().endswith(extension):
                                save_name_list.append((filename2.replace(extension,''),filename))

                    elif os.path.isdir(os.path.join(location,filename,filename2)):
                        save_name_list.append((filename2,filename))
                        filelist3 = os.listdir(os.path.join(location,filename,filename2))
                        for filename3 in filelist3:
                            for extension in extensions:
                                if filename3.lower().endswith(extension):
                                    save_name_list.append((filename3.replace(extension,''),filename2))

    return save_name_list
    
    
def get_nearest_names(save_type,name):
    possibilities = get_save_list(save_type)
    return difflib.get_close_matches(name,possibilities)