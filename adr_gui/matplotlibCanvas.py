from __future__ import unicode_literals
from PyQt4 import QtGui, QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.pyplot import ylabel


class MplCanvas(FigureCanvasQTAgg):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(True)

        self.compute_initial_figure()

        #
        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def compute_initial_figure(self):
        pass
    
    def sizeHint(self):
        return QtCore.QSize(700,500) # this seems to be big enough to make the axes legible without dragging
    



class DynamicMplCanvas(MplCanvas):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, xlabel="time (s)", ylabel="data (arb)", title="a plot", max_points = 3000, **kwargs):
        MplCanvas.__init__(self, **kwargs)
        self.number_of_lines = 1
        self.x = [[]]
        self.y = [[]]
        self.style = "-o"
        self.max_points = max_points
        self.set_axis_labels(xlabel, ylabel, title)

         
    def set_axis_labels(self, xlabel="time (s)", ylabel="data (arb)", title="a plot"):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.update_figure()
    
    def add_point(self, x,y, line_number=0):
        self.x[line_number].append(x)
        self.y[line_number].append(y)
        if len(self.x[line_number]) > self.max_points:
            self.x[line_number] = self.x[line_number][1:]
            self.y[line_number] = self.y[line_number][1:]
        self.update_figure()
        
    def add_line(self):
        self.x.append([])
        self.y.append([])
        self.number_of_lines = len(self.x)       
        
    def clear_points(self, line_number=0):
        self.x[line_number] = []
        self.y[line_number] = []
        self.update_figure()
        
    def last_n_points(self,n, line_number=0):
        if len(self.x[line_number]) < n:
            return None
        else:
            return self.x[line_number][-n:], self.y[line_number][-n:]

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        self.axes.cla()
        for line_number in xrange(self.number_of_lines):
            self.axes.plot(self.x[line_number], self.y[line_number], self.style)
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_title(self.title)
        self.draw()
