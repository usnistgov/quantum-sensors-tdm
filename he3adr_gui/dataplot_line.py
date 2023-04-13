'''
Created on Jul 27, 2009

@author: schimaf
'''

from matplotlib.lines import Line2D

import numpy as np

lineStyles={'solid':'','dotted':(2,2),'dashed':(20,5),'shortdash':(10,2),
    'longdash':(40,10),'dotdash':(2,2,20,2)}

symbolList = ['none', 'circle', 'plus', 'square', 'diamond', 
                    'cross', 'splus', 'scross', 'triangle']

class dataplot_line(Line2D):
    '''
    dataplot line
    '''

    def __init__(self, line, name, xdata, ydata, color='black',linewidth=1,
    linestyle='solid',symbol='none',pixel=1,fill='',outline=''):
        '''
        Constructor
        '''
        
        self.mpl_line = line
        self.name = name
        if line != None:
            self.mpl_line.set_label(name)
        self.hideVar = 1
        self.xdata = xdata
        self.ydata = ydata
        self.color=color
        self.linewidth=linewidth
        if linestyle not in list(lineStyles.keys()):
            linestyle='solid'
        self.linestyle=linestyle
        #self.dashes=self.dashlookup(linestyle)
        self.symbol=symbol
        self.pixel=pixel
        if fill == '':
            fill=color
        if outline == '':
            outline=color
        self.fill=fill
        self.outline=outline
        self.active = True
        self.visible = True

    def update_data(self, xdata, ydata):
        
        self.xdata = xdata
        self.ydata = ydata
        self.mpl_line.set_data(xdata, ydata)

    def addPoint(self, x, y):
        '''
        Add a single point to the plot line.
        '''
        self.xdata = np.append(self.xdata, x)
        self.ydata = np.append(self.ydata, y)
        
        self.update_data(self.xdata, self.ydata)

#===========================================================================  
    def hide(self):
#===========================================================================  
        #self.g.element_configure(self.name, hide='yes');
        self.active = False
        self.visible = False

#===========================================================================  
    def unhide(self):
#===========================================================================  
        #self.g.element_configure(self.name, hide='no');
        self.active = True
        self.visible = True

#===========================================================================  
    def delete(self):
#===========================================================================  
        self.xdata = []
        self.ydata = []
