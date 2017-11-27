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

def check_create_dirs():

    global pointpairs
    global fitresults
    global raydata
    global images
    global rois
    global machine_geometry
    global image_sources
    global code
    global virtualcameras

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
        os.makedirs(pointpairs)

    if not os.path.isdir(fitresults):
        os.makedirs(fitresults)

    if not os.path.isdir(raydata):
        os.makedirs(raydata)

    if not os.path.isdir(images):
        os.makedirs(images)

    if not os.path.isdir(machine_geometry):
        os.makedirs(machine_geometry)

    if not os.path.isdir(image_sources):
        os.makedirs(image_sources)

    if not os.path.isdir(rois):
        os.makedirs(rois)

    if not os.path.isdir(virtualcameras):
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



def change_save_location(new_path,migrate=True):

    src_filelists = []
    if migrate:
        src_filelists.append(os.walk(pointpairs))
        src_filelists.append(os.walk(fitresults))
        src_filelists.append(os.walk(raydata))
        src_filelists.append(os.walk(images))
        src_filelists.append(os.walk(rois))
        src_filelists.append(os.walk(machine_geometry))
        src_filelists.append(os.walk(image_sources))
        src_filelists.append(os.walk(code))
        src_filelists.append(os.walk(virtualcameras))

    global root
    old_root = root
    root = new_path

    try:
        check_create_dirs()
        if root != os.path.join(homedir,'calcam'):
            f = open(os.path.join(homedir,'calcam','.altpath.txt'),'w')
            f.write(root)
            f.close()
        else:
            if os.path.isfile(os.path.join(root,'.altpath.txt')):
                os.remove(os.path.join(root,'.altpath.txt'))
    except:
        root = old_root
        check_create_dirs()
        raise

    if migrate:
        for folderlist in src_filelists:
            for path,subdirs,files in folderlist:

                newpath = os.path.join(root,path.replace(old_root,'').strip(os.sep))

                for subdir in subdirs:
                    if not os.path.isdir(os.path.join(newpath,subdir)):
                        os.mkdir(os.path.join(newpath,subdir))
                for filename in files:
                    if not os.path.isfile(os.path.join(newpath,filename)):
                        os.rename( os.path.join(path,filename) , os.path.join(newpath,filename) )
                    else:
                        print('[Save Migration] Not moving {:s}: file already exists at new destination.'.format(os.path.join(path,filename)))



# The CalCam user data directory goes in the user's homedir
homedir = os.path.expanduser('~')

# Path where the calcam code is
calcampath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ui = os.path.join(calcampath,'ui')


root = os.path.join(homedir,'calcam')
if os.path.isfile(os.path.join(root,'.altpath.txt')):
    f = open(os.path.join(root,'.altpath.txt'),'r')
    root = f.readline().rstrip()
    f.close()

check_create_dirs()