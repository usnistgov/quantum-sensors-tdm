from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
#
# Encapsulate the "proper" way to use threads in Qt. Useful for simple cases where
#  * you want to do some long-running work (like closing a heat switch)
#  * you don't want the GUI to freeze while the work is being done
#  * you want to notify the GUI when the work ends
#  * you don't need to cleanly pause/interrupt/stop the work
#
# Useful references for Qt threading:
#
# http://qt-project.org/wiki/Threads_Events_QObjects
# http://stackoverflow.com/a/22060122/274967
# https://mayaposch.wordpress.com/2011/11/01/how-to-really-truly-use-qthreads-the-full-explanation/
#

class Task(QObject):
    
    sig_finished = pyqtSignal()

    def __init__(self, work_func, finished_func):
        QObject.__init__(self)
        self.thread = QThread()

        self.moveToThread(self.thread)

        self.func = work_func

        self.sig_finished.connect(self.thread.quit)
        self.sig_finished.connect(finished_func)

        self.thread.started.connect(self.do_work)
    
    def isFinished(self):
        return self.thread.isFinished()
        
    def isRunning(self):
        return self.thread.isRunning()
        
    def start(self):
        self.thread.start()

    def do_work(self):
        try:
            self.func()
        finally:
            # xxx if an exception, do we need better error reporting
            # to user?
            self.sig_finished.emit()
            

#
# Encapsulate the "proper" way to use threads in Qt. Similar to
# the above Task class, but instead of doing the work in a single
# shot, this Task wakes up every so often to do some work, and
# supports sending a signal to interrupt the work. Useful
# long-running, complicated tasks like running a full fridge cycle
#
# WHen you create the TaskWaking object, you poass in a wake interval,
# a wake function, and a finished function.
#
#  * The "wake interval" is the interval (in seconds) at which you
#    wake up.
#
#  * The "wake function" is called each time the thread wakes up. A
#    "should_continue" value is passed in, which can be either True or
#    False. If the value is False, the task is
#    expected to clean itself up, as the task will no longer be woken.
#    
#    The wake function is also expected to return a True/False value,
#    with True indicating that some work remains to do, and False
#    indicating that the task is done.
#
# * The "finished function" should be a slot that is connected the
#   Task's sig_finished signal.


class TaskWaking(QObject):
    
    sig_stop = pyqtSignal()
    sig_finished = pyqtSignal()

    def __init__(self, wake_interval, wake_func, finished_func):
        QObject.__init__(self)
        self.thread = QThread()
        
        self.moveToThread(self.thread)
        
        self.wake_func = wake_func
        
        self.sig_finished.connect(self.thread.quit)
        self.sig_finished.connect(finished_func)
        
        self.sig_stop.connect(self.do_stop)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.wake_up)
        self.timer.start(wake_interval*1000)
    
    def start(self):
        self.thread.start()

    def stop(self):
        self.sig_stop.emit()
    
    def do_stop(self):
        self.wake_func(False)
        self.thread.quit()
        self.sig_finished.emit()
    
    def wake_up(self):
        should_continue = self.wake_func(True)
        print(('wake_up returned', should_continue))
        if not should_continue:
            self.thread.quit()
            self.sig_finished.emit()

