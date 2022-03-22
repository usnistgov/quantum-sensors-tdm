VERBOSITY_DEBUG = 50
VERBOSITY_INFO  = 40 
VERBOSITY_ERROR = 10 # lower numbers are shown more easily


class Logger():
    """
    GCO tried to use the logging package, but log.debug("abc", 4) doesnt work and the existing 
    code uses many statements line print("abc",4"), and I wanted to convert with a search and replace to 
    make sure I don't throw away any info
    """
    def __init__(self, verbosity=VERBOSITY_INFO, prepend=""):
        self.verbosity = verbosity
        self._always_raise_in_debug = False
        self.prepend = prepend

    def set_debug(self):
        self.verbosity = VERBOSITY_DEBUG

    def debug(self, *args):
        if self.verbosity >= VERBOSITY_DEBUG:
            if self.prepend:
                print("DEBUG:", self.prepend, *args)
            else:
                print("DEBUG:", *args)
        if self._always_raise_in_debug:
            raise Exception("for testing")

    def info(self, *args):
        if self.verbosity >= VERBOSITY_INFO:
            if self.prepend:
                print("INFO:", self.prepend, *args)
            else:
                print("INFO:", *args)

    def error(self, *args):
        if self.verbosity >= VERBOSITY_ERROR:
            if self.prepend:
                print("ERROR:", self.prepend, *args)
            else:
                print("ERROR:", *args)

        
    def child(self, prepend):
        return Logger(self.verbosity, self.prepend+prepend)