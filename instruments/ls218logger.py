import instruments
import time
import os
from lxml import etree


class MyLogger():
    def __init__(self):
        logDirectory = os.path.join(os.path.dirname(__file__),"logs")
        XML_CONFIG_FILE = "/etc/adr_system_setup.xml"
        if os.path.isfile(XML_CONFIG_FILE):
            f = open(XML_CONFIG_FILE, 'r')
            root = etree.parse(f)
            child = root.find("log_folder")
            if child is not None:
                value = child.text
                logDirectory = value
        if not os.path.isdir(logDirectory): 
            os.mkdir(logDirectory)

        self.filename = os.path.join(logDirectory,time.strftime("ls218_log_%Y%m%d_t%H%M%S.txt"))
        self.file = open(self.filename,"w")
        print(f"ls218_log log directory: {logDirectory}")
        print(f"ls218_log log filename: {self.filename}")

    def log(self,s):
	    self.file.write(s+"\n")
	    self.file.flush()

def _ls218_logger_entry_point():
    logger = MyLogger()
    ls = instruments.Lakeshore218()
    logger.log("time.time(), temp 3k (K), temp 50k (K)")
    PERIOD_S = 60

    while True:
        t3k = ls.getTemperature(2)
        t50k = ls.getTemperature(1)
        s = f"{time.time():.0f}, {t3k:.2f}, {t50k:.2f}"
        print(s)
        logger.log(s)
        time.sleep(PERIOD_S)

if __name__ == "__main__":
    _ls218_logger_entry_point()

