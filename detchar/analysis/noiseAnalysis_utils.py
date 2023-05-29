'''noiseAnalysis_utils.py 

utility functions for analyzing noise data
@author JH 5/2023
'''

import numpy as np 

def noise_model_fit_func(x,a,b,c,d,e):
    return a(1+(b*x)**2)+c+d*x**e

