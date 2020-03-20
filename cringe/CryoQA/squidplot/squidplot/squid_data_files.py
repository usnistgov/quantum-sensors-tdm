# -*- coding: utf-8 -*-
import os.path
import numpy as np


class squidfile(object):
    def __init__(self, filename):
        if not os.path.exists(filename):
            raise IOError('File \"%s\" not found'%filename)
        self.filename = filename
        self.shortfilename = os.path.basename(filename)
        if not self.check_for_npy_file():
            raise IOError('File not a numpy dump.')
        self.data = np.load(filename)
        shape = self.data.shape
        if len(shape) == 2:
            self.filetype = 'Modulation'
            self.columns = None
            self.pages = None
            self.channels = shape[0]
            self.numcurves = 1
        elif len(shape) == 4:
            self.filetype = 'Ic one column'
            self.columns = None
            self.pages = shape[0]
            self.channels = shape[1]
            self.numcurves = 2
        elif len(shape) == 5:
            self.filetype = 'Ic multi-column'
            self.columns = shape[0]
            self.pages = shape[1]
            self.channels = shape[2]
            self.numcurves = 2
        else:
            raise IOError('Unrecognized .npy file')
        
    def check_for_npy_file(self):
        '''.npy files have a signature'''
        f = open(self.filename,'rb')
        a = f.read(6)
        f.close()
#        print a
#        print a == '\x93NUMPY'
        if a == '\x93NUMPY':
            return True
        return False
    
    def get_data(self, column=None, page=None, channel=None):
        if self.filetype == 'Modulation':
            if channel == None:
                channel = 0
            return (self.data[channel],)
        elif self.filetype == 'Ic one column':
            if channel is None:
                channel = 0
            if page is None:
                page = 0
            data1 = self.data[page][channel].transpose()[0]
            data2 = self.data[page][channel].transpose()[1]
            return (data1,data2)
        elif self.filetype == 'Ic one column':
            if column is None:
                column = 0
            if channel is None:
                channel = 0
            if page is None:
                page = 0
            data1 = self.data[column][page][channel].transpose()[0]
            data2 = self.data[column][page][channel].transpose()[1]
            return (data1,data2)
        else:
            return None

            
             
            