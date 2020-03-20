import socket

class NetworkPipe(socket.socket):
    """
    This class wraps a socket.socket object to provide a lock.
    Unfortunately, these locks are not yet implemented.  Try not to use threads yet!
    """
    
    def __init__(self, host, port, timeout = 0.1):
        socket.socket.__init__(self, socket.AF_INET, socket.SOCK_STREAM)
        self.connect((host, port))
        self.setblocking(True) # making it blocking seems to work much better than non-blocking for autotuning 8 columns
        self.settimeout(timeout)
        self.lock = None
        
    def __del__(self):
#        print 'DEBUG: Closing %s'%self
        del self

    # Eventually, we will want to 
    def get_lock(self):
        pass
    
    def release_lock(self):
        pass
