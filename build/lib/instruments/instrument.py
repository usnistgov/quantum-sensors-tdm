'''
Created on Mar 3, 2009

@author: schimaf
'''

class Instrument(object):
    '''
    Base class for all controllable instruments. 
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        self.manufacturer = ""
        self.model_number = ""
        self.description  = ""
