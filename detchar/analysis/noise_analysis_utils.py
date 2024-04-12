''' noise_analysis_utils.py 

JH 04/2024 
'''

import matplotlib.pyplot as plt
import numpy as np 

class FitNoiseSingle():
    ''' 
    '''
    def __init__(self, f_hz,Ixx):
        self.f = f_hz # frequency in hz
        self.Ixx = Ixx # current noise density A^2/Hz

    def plot(self):
        plt.loglog(self.f,self.Ixx)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Current Noise Density (A$^2$/Hz)')
        plt.grid('on')

    def fit_func(self,f,a,b,c):
        return a*(1+(b/f)**2)**-1+c



    