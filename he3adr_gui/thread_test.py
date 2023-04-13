import sys, time
from PyQt4.QtGui import QApplication
from PyQt4.QtCore import QObject, QTimer
from task import Task, TaskWaking

class ThreadTest(QObject):

    def __init__(self):
        QObject.__init__(self)


        self.timer = QTimer()
        self.timer.timeout.connect(self.my_update)
        self.timer.start(1000)
        self.counter = 0

        self.task = Task(self.worker, self.work_done)
        self.task_wake = TaskWaking(4, self.wake, self.work_done)

    def my_update(self):
        print(('my_update', self.counter, self.task.thread.isRunning(), self.task_wake.thread.isRunning()))
        if self.counter == 5:
            self.task_wake.start()
        if self.counter == 12:
            self.task_wake.stop()
        self.counter += 1

    def work_done(self):
        print('work done')

    def wake(self, state):
        print('wake start')
        time.sleep(2)
        print('wake end')
        return self.counter < 10

    def worker(self):
        print('worker start')
        time.sleep(10)
        print('worker end')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    test = ThreadTest()
    sys.exit(app.exec_())

