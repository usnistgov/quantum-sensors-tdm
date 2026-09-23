from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class MplCanvas(FigureCanvasQTAgg):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called

        self.compute_initial_figure()

        #
        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
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
            self.x[line_number] = self.x[line_number][-self.max_points:]
            self.y[line_number] = self.y[line_number][-self.max_points:]
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
        for line_number in range(self.number_of_lines):
            self.axes.plot(self.x[line_number], self.y[line_number], self.style)
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_title(self.title)
        self.draw()


class SubplotsCanvas(FigureCanvasQTAgg):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, nrows, ncols, sharex=False, sharey=False, parent=None, width=5, height=4, dpi=100):
        fig,axs = plt.subplots(
            nrows, 
            ncols, 
            sharex=sharex, 
            sharey=sharey, 
            figsize=(width, height), 
            dpi=dpi, 
            constrained_layout=True,
            squeeze=False
        )
        self.fig=fig
        self.axs=axs
        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def sizeHint(self):
        return QtCore.QSize(700,500) #
    

class DynamicSubplotsCanvas(SubplotsCanvas):
    """A canvas that can be updated easily, Only one line per plot this time though,"""
    def __init__(
        self, 
        nrows, 
        ncols, 
        sharex=False, 
        sharey=False, 
        xlabel="time (s)", 
        ylabel="data (arb)", 
        max_points = 3000, 
        **kwargs
    ):
        SubplotsCanvas.__init__(
            self,
            nrows,
            ncols,
            sharex=sharex,
            sharey=sharey,
            **kwargs
        )
        self.nrows = len(self.axs)
        self.ncols = len(self.axs[0])
        self.x = [[[] for _ in range(self.ncols)] for _ in range(self.nrows)]
        self.y = [[[] for _ in range(self.ncols)] for _ in range(self.nrows)]
        self.xlabels = [[xlabel for _ in range(self.ncols)] for _ in range(self.nrows)]
        self.ylabels = [[ylabel for _ in range(self.ncols)] for _ in range(self.nrows)]
        self.style = "-o"
        self.max_points = max_points


    def set_axis_labels(self, row, col, xlabel, ylabel):
        self.xlabels[row][col] = xlabel
        self.ylabels[row][col] = ylabel
        self.update_figure()

    def add_point(self,row,col,x,y):
        self.x[row][col].append(x)
        self.y[row][col].append(y)
        if len(self.x[row][col]) > self.max_points:
            self.x[row][col] = self.x[row][col][-self.max_points:]
            self.y[row][col]= self.y[row][col][-self.max_points:]
        #allow adding multiple points before redraw

    def clear_points(self):
        self.x = [[[] for _ in range(self.ncols)] for _ in range(self.nrows)]
        self.y = [[[] for _ in range(self.ncols)] for _ in range(self.nrows)]
        self.update_figure()

    def last_n_points(self,row,col,n):
        if len(self.x[row][col]) < n:
            return None
        else:
            return self.x[row][col][-n:], self.y[row][col][-n:]

    def update_figure(self):
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axs[i][j]
                ax.cla()
                ax.plot(self.x[i][j], self.y[i][j], self.style)
                ax.set_xlabel(self.xlabels[i][j])
                ax.set_ylabel(self.ylabels[i][j])
        self.draw()