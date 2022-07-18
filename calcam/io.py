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

import zipfile
import tempfile
import os
import shutil
import hashlib
import atexit

from .misc import import_source,unload_source

# Get a list of directory contents, including sub-directories.
# Returned list has absolute paths.
def listdir(path):

    filelist = []

    fnames = sorted(os.listdir(path))

    for fname in [ os.path.join(path,f) for f in fnames]:

        # Recursively do sub-directories
        if os.path.isdir(fname):
            filelist = filelist + listdir(fname)
        else:
            filelist.append(fname)

    return filelist


def md5_file(filename):

    blocksize = 65536
    hasher = hashlib.md5()
    with open(filename, 'rb') as f:
        buf = f.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(blocksize)
    return hasher.digest()


# Class for Zip file based save files.
class ZipSaveFile():

    def __init__(self,fname,mode='r',ignore_pyc=True):

        if 'w' in mode and 's' in mode:
            raise ValueError('Invalid mode "{:s}"'.format(mode))

        self.filename = os.path.abspath(fname)
        self.ignore_pyc = ignore_pyc
        self.mode = mode
        self.pypaths = []
        self.is_open = False
        self.open(self.mode)
        self.usermodule = None

        atexit.register(self.close)


    def open(self,mode):

        # Maybe we're re-opening the file, in which case
        # we need to close first.
        if self.is_open:
            self.close()

        self.mode = mode

        # Check the file exists, and that if we're going to try to write to it,
        # that we have the necessary permissions.
        if 'w' in self.mode:
            if os.path.isfile(self.filename):
                if not os.access(self.filename,os.W_OK):
                    raise IOError('Write permission denied on {:s}'.format(self.filename))
            elif os.path.isdir(self.filename):
                raise IOError('Specified filename already exists but is a directory!')
            else:
                savepath = os.path.split(self.filename)[0]
                if not os.access(savepath,os.W_OK):
                    raise IOError('Cannot write to directory {:s}'.format(savepath))

        if 'r' in self.mode and 'w' not in self.mode:
            if not os.path.isfile(self.filename):
                raise IOError('No such file: {:s}'.format(self.filename))


        # Create a temporary directory which we'll use 
        # to extract our ZIP while we work with its contents.
        self.tempdir = tempfile.mkdtemp()

        if 'r' in self.mode:

            try:
                with zipfile.ZipFile(self.filename,'r') as zf:

                    if 's' not in mode:
                        loadlist = zf.namelist()    
                    else:
                        loadlist = [name for name in zf.namelist() if not name.startswith('.large/')]

                    zf.extractall(self.tempdir,members=loadlist)
            
            except:
                if 'w' not in self.mode:
                    shutil.rmtree(self.tempdir)
                    raise

        self.file_handles = []
        self.is_open = True
        self.initial_hashes = self.get_hashes()


    def is_readonly(self):

        return not os.access(self.filename,os.W_OK)


    def get_hashes(self):

        if self.is_open:
            hashes = []
            for fname in self.list_contents():
                hashes.append( (fname,md5_file(os.path.join(self.tempdir,fname))) )

            return hashes
        else:
            raise Exception('File is not open!')


    def close(self,discard_changes=False):
        
        if self.is_open:
            for h in self.file_handles:
                h.close()

            # If we're in write mode, and the file contents have been modified since being loaded,
            # we need to re-save the ZIP file with the new contents.
            if 'w' in self.mode and not discard_changes and self.get_hashes() != self.initial_hashes:
                self.update()

            # Make sure we properly unload any user code
            try:
                unload_source(os.path.join(self.tempdir,'usercode'))
            except FileNotFoundError:
                pass

            self.is_open = False
            
            # Tidy up the temp directory after ourselves
            shutil.rmtree(self.tempdir)
            self.tempdir = None

        atexit.unregister(self.close)



    def update(self):

        with zipfile.ZipFile(self.filename,'w',zipfile.ZIP_DEFLATED,True) as zf:
            for fname in listdir(self.tempdir):
                zf.write(fname,os.path.relpath(fname,self.tempdir))


    # Open a file inside the zip for doing stuff with.
    # Reurns a file handle to the opened file.
    def open_file(self,fname,mode):

        if not self.is_open:
            self.open()

        if 'r' in mode and fname not in self.list_contents():
            raise IOError('File "{:s}" not in here!'.format(fname))

        if 'w' in mode and 'w' not in self.mode:
            raise IOError('File is open in read only mode!')

        h = open( os.path.join(self.tempdir,fname) , mode )

        self.file_handles.append(h)

        return h


    # If the is Python code in the zip either in usercode.py or a 
    # directory called usercode/, return it as a python module.
    def get_usercode(self):

        if not self.is_open:
            self.open()

        if 'r' not in self.mode or not self.is_open:
            raise IOError('File not open in read mode!')

        if os.path.join('usercode','__init__.py') in self.list_contents():
            return import_source(os.path.join(self.tempdir,'usercode'))
        elif 'usercode.py' in self.list_contents():
            return import_source(os.path.join(self.tempdir, 'usercode.py'))
        else:
            return None




    # Get a list of the files within.
    def list_contents(self):

        if not self.is_open:
            self.open()
        
        if self.ignore_pyc:
            return [os.path.relpath(fname,self.tempdir) for fname in listdir(self.tempdir) if not fname.endswith('.pyc')]
        else:
            return [os.path.relpath(fname,self.tempdir) for fname in listdir(self.tempdir)]


    # Add a file or directory to the archive.
    def add(self,from_path,to_path=None,replace=False):

        if not self.is_open:
            self.open()

        if 'w' not in self.mode:
            raise IOError('File is open in read-only mode!')

        if not (os.path.isdir(from_path) or os.path.isfile(from_path)):
            raise IOError('No such file or directory "{:s}"'.format(from_path))

        if to_path is None:
            dst_path = os.path.join(self.tempdir, from_path.split(os.sep)[-1] )
        else:
            dst_path = os.path.join(self.tempdir, to_path )

        if os.path.isdir(dst_path):
            if replace:
                shutil.rmtree(dist_path)
            else:
                raise IOError('This path already exists in this file! Use replace=True to allow overwriting.')
        elif os.path.isfile(dst_path):
            if replace:
                os.remove(dst_path)
            else:
                raise IOError('This file already exists in this file! Use replace=True to allow overwriting.')
        
        dst_folder = os.path.split(dst_path)[0]
        if not os.path.isdir(dst_folder):
            os.makedirs(dst_folder)

        if os.path.isfile(from_path):
            shutil.copy2(from_path,dst_path)
        elif os.path.isdir(from_path):
            shutil.copytree(from_path,dst_path)


    # Add user python code to the arvhive.
    def add_usercode(self,usercode_path,replace = False):

        if not self.is_open:
            self.open()

        if 'w' not in self.mode:
            raise IOError('File is open in read-only mode!')

        if self.get_usercode() and not replace:
                raise IOError('File already contains user code! Use replace=True to allow overwriting.')

        self.clear_usercode()
        
        if os.path.isfile(usercode_path):
            self.add(usercode_path,'usercode.py')
        else:
            self.add(usercode_path,'usercode')



    # Add user python code to the arvhive.
    def clear_usercode(self):

        if not self.is_open:
            self.open()

        if 'w' not in self.mode:
            raise IOError('File is open in read-only mode!')

        unload_source(os.path.join(self.tempdir,'usercode'))

        try:
            self.remove('usercode')
        except IOError:
            pass
        try:
            self.remove('usercode.py')
        except IOError:
            pass        


    # Remove a named file or folder from the archive
    def remove(self,fname):

        if 'w' not in self.mode:
            raise IOError('File is open in read-only mode!')

        fullpath = os.path.join(self.tempdir, fname)

        if os.path.isfile( fullpath ):
            os.remove(fullpath)
        elif os.path.isdir( fullpath ):
            shutil.rmtree( fullpath )
        else:
            raise IOError('File or directory "{:s}" not in here!'.format(fname))

    # Return the temporary path for manually playing with / using contents.
    def get_temp_path(self):

        return self.tempdir


    def mkdir(self,dirname):

        if not os.path.exists(dirname):
            os.makedirs(os.path.join(self.tempdir,dirname))


    # For context management
    def __enter__(self):

        if not self.is_open:
            self.open()

        return self


    # For context management
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            raise


    # If we forget to close the object properly
    # before it's garbage collected or whatever, make sure the 
    # temp files get cleaned up.
    def __del__(self):
        self.close()