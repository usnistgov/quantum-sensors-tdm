import zmq

#monkey patch hack to fix hanging
#autotune was relaibly hanging with NDFB server, but I just ctrl-c
# it continued and worked fine, throwing an error from this __del__ function
# so lets just put an exception in there by default
def __del__(self):
    """deleting a Context should terminate it, without trying non-threadsafe destroy"""

    # Calling locals() here conceals issue #1167 on Windows CPython 3.5.4.
    locals()

    if not self._shadow and not zmq.sugar.context._exiting:
        raise Exception("ridiculous hack to prevent zmq hang")
        self.term()

zmq.sugar.context.Context.__del__ = __del__