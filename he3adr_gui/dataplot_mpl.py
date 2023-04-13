'''
Created on Jul 24, 2009

@author: schimaf
'''


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfFile
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import matplotlib.transforms as mtrans
import dataplot_line

class dataplot_mpl(QWidget):
    def __init__(self, parent=None, layout=None, width=5, height=4, dpi=100, x_label = "X Axis", y_label = "Y Axis", title="My Plot", scale_type='linear', addToolbar=False):

        QWidget.__init__(self, parent)

        self.parent = parent
        self.scale_type= scale_type
        self.hasToolbar = addToolbar

        # Create the figure
        self.figure = dataplot_figure(parent, width, height, dpi, x_label, y_label, title, scale_type)

        # Create list of colors
        self.colors = []

        color_red = QColor(Qt.red)
        color_yellow = QColor(Qt.yellow)
        color_green = QColor(Qt.green)
        color_blue = QColor(Qt.blue)
        color_cyan = QColor(Qt.cyan)
        color_magenta = QColor(Qt.magenta)

        # Assign colors
        self.color_array = [20, 21, 22, 23, 24, 25, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 42, 44, 45, 48]

        for repeat in range(12):
            self.colors.append(str(color_red.name()))
            self.colors.append(str(color_green.name()))
            self.colors.append(str(color_blue.name()))
            self.colors.append(str(color_magenta.name()))
            for color in self.color_array:
                #print str(QColor(QColor.colorNames()[color]).name())
                self.colors.append(str(QColor(QColor.colorNames()[color]).name()))

        self.num_curves = 0

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.figure)

        if addToolbar == True:
            self.addToolbar()
        else:
            self.toorbar = None
    
    def addToolbar(self):
        # Create the navbar
        self.toolbar = NavigationToolbar(self.figure, self.parent)

        self.autoscale_button = QToolButton()
        self.autoscale_button.setText('Autoscale')
        self.autoscale_button.setCheckable(True)
        self.autoscale_button.setChecked(True)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.autoscale_button)

        self.layout.addWidget(self.toolbar)

        self.hasToolbar = True

    def exportPDF(self, export_file=None):

        if export_file is None:
            new_file = QFileDialog.getSaveFileName()
            if (new_file != []):
                if new_file[-4:] == '.pdf':
                    export_file = str(new_file)
                else:
                    export_file = str(new_file) + '.pdf'
    
        #self.figure.print_figure(filename, dpi, facecolor, edgecolor, orientation, format)
        #self.figure.resize(800, 200)
        #print "export file", export_file
        self.figure.print_figure(export_file)

    def update(self):
        
        if self.hasToolbar == False or self.autoscale_button.isChecked() == True:
                self.figure.axes.relim()
                self.figure.axes.autoscale_view()
        self.figure.draw()

    def setAutoscale(self):
        self.figure.axes.autoscale_view(scalex=True, scaley=True)

    # This function does not work because autoscale seems to take over
    def setXAxisBounds(self, x_min, x_max):
        #print "setting X bounds"
        # The following command should work, but does not on Ubuntu 8.04. 
        #self.figure.axes.set_autoscalex_on(False)
        self.figure.axes.autoscale_view(scalex=False)
        self.figure.axes.set_xlim(xmin=x_min, xmax=x_max)
        self.figure.axes.relim()
    
    # This function does not work because autoscale seems to take over
    def setYAxisBounds(self, y_min, y_max):
        self.figure.axes.autoscale_view(scaley=False)
        self.figure.axes.set_ylim(ymin=y_min, ymax=y_max)

    def set_x_axis_label(self, label):
        self.figure.axes.set_xlabel(label)

    def set_y_axis_label(self, label):
        self.figure.axes.set_ylabel(label)
        
    def set_title(self, title):
        self.figure.axes.set_title(title)

    def set_labels(self, xlabel, ylabel, title):
        self.set_x_axis_label(xlabel)
        self.set_y_axis_label(ylabel)
        self.set_title(title)
        ax = self.figure.axes
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)
        ax.title.set_fontsize(10)


    def clear(self):
        '''
        Clear all lines from the plot
        '''
        
        self.figure.axes.clear()
        self.num_curves = 0

    def addLine(self, name, xdata, ydata, color=None,linewidth=1, linestyle='solid',symbol='none',pixel=1,fill='',outline=''):
        
        if color == None:
            # Pick acolor from the list of colors
            #print 'num_curves is ', self.num_curves, len(self.colors)
            color = self.colors[self.num_curves]
        
        newline, = self.figure.axes.plot(xdata, ydata, color)
        dp_line = dataplot_line.dataplot_line(newline, name, xdata, ydata, color,linewidth, linestyle,symbol,pixel,fill,outline)
        self.num_curves += 1
        
        return dp_line

    def showLegend(self, location='best', **kwargs):
        
        self.legend = self.figure.axes.legend(loc=location, **kwargs)

    def addSet(self,xdata,ydata,name='',color='',symbol='',linewidth=1,
    linestyle='solid',pixel=0,fill='',outline=''):

        label=name
        if label == '':
            label = 'data'+repr(self.data_number)
            self.figure.data_number = self.figure.data_number+1;
        if color == '':
            #color=self.getNextLineColor()
            color='black'
        if fill == '':
            fill=color
        if outline == '':
            outline=color
        self.figure.line = dataplot_line.dataplot_line(self, line=None, name=label, 
                     xdata=xdata, ydata=ydata,
                     color=color, linewidth=linewidth, linestyle=linestyle,
                     symbol=symbol, pixel=pixel, fill=fill, outline=outline)    
        self.figure.lines.append(self.figure.line)
        self.figure.data_number = self.figure.data_number+1;
        #self.updateConfigMenu()
        #self.updateLineMenu()
        return label

#class dataplot_mpl(FigureCanvas):
class dataplot_figure(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100, x_label="", y_label="", title="", scale_type=""):
        fig = Figure(figsize=(width, height), dpi=dpi)

        self.sources = [] 
        self.lines = []

        self.data_number = 0
        self.draw_x_intercept_when_clicked = False
        self.linked_field = None
        self.x_intercept_line = None

        #self.axes = fig.add_subplot(111)
        self.axes = fig.add_axes([0.2,0.2,0.75,0.6])
            # We want the axes retained every time plot() is called
            #self.axes.hold(True)
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        t = self.axes.set_title(title)
        t.set_transform(mtrans.ScaledTranslation(0, 0.1/72, fig.dpi_scale_trans) + t.get_transform())
            
        if scale_type == 'semilogx':
            self.axes.set_xscale('log')
        elif scale_type == 'semilogy':
            self.axes.set_yscale('log')
        elif scale_type == 'loglog':
            self.axes.set_xscale('log')
            self.axes.set_yscale('log')
        
        self.axes.grid(color='k', linestyle=':', linewidth=1)
        
        #fig.subplots_adjust(bottom=0.15)

        #self.parent.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Maximum)
        FigureCanvas.updateGeometry(self)

        self.mouse_move_id = self.mpl_connect('motion_notify_event', self.on_move)
        self.mouse_click_id = self.mpl_connect('button_press_event', self.on_click)
        
        self.clicked = False
        self.click_x = 0
        self.click_y = 0

    def on_move(self, event):
        # get the x and y pixel coords
        x, y = event.x, event.y
    
        if event.inaxes:
            ax = event.inaxes  # the axes instance
            #print 'data coords', event.xdata, event.ydata

    def on_click(self, event):
        # get the x and y coords, flip y from top to bottom
        if self.draw_x_intercept_when_clicked == True:
            x, y = event.x, event.y
            if event.button==1:
                if event.inaxes is not None:
                    self.clicked = True
                    self.click_x = event.xdata
                    self.click_y = event.ydata
                    #print 'clicked data coords', event.xdata, event.ydata
                    if self.linked_field != None:
                        self.linked_field.setText("%f" % self.click_x)
                    self.draw_x_intercept(self.click_x)

    def draw_x_intercept(self, x_value):
        # Draw a line for the linear region
        if self.draw_x_intercept_when_clicked == True:
            # Draw a line at self.click_x
            if self.x_intercept_line != None:
                # Remove the previous one
                self.axes.lines.remove(self.x_intercept_line)

            x = [x_value, x_value]
            y = self.axes.get_ylim()
            ymod = (y[0] - 0.01*y[0], y[1] - 0.01*y[1])
            #print "drawing x,y", x, y
            newline = self.axes.plot(x,ymod, color='black', linewidth=2.0)
            self.x_intercept_line = newline[0]
            self.draw()
            #print self.x_intercept_line

    def compute_initial_figure(self):
        pass
