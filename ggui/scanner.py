import npy_append_array
import collections
from dataclasses_json import dataclass_json
from dataclasses import dataclass
import numpy as np
import time
import sched

class ScanOne():
    """
    failval - if not None, then scan will return failval when there is an exception
                if None, then scan will throw the exception"""
    def __init__(self, name, dtype, callback, callback_args=(), callback_kwargs = {}, failval=None):
        self.name = name
        self.dtype = dtype
        self.callback = callback
        self.callback_args = callback_args
        self.callback_kwargs = callback_kwargs
        self.failval = failval

    def scan(self):
        if self.failval is None:
            return self.callback(*self.callback_args, **self.callback_kwargs)
        else:
            try:
                return self.callback(*self.callback_args, **self.callback_kwargs)
            except IOError:
                return self.failval        
    
class ScanMany():
    def __init__(self, scanones):
        self.scanones = scanones
        self.dtype = self._make_dtype(scanones)

    def _dtype_list(self, scanones):
        return [(so.name, so.dtype) for so in scanones]
    
    def _make_dtype (self, scanones):
        return np.dtype(self._dtype_list(scanones))
    
    def scan(self):
        # make this async?
        v = np.array([tuple(so.scan() for so in self.scanones),], dtype=self.dtype)
        return v

class ScanManyBuilder():
    def __init__(self):
        self.scanones = []

    def add(self, scanone):
        self.scanones.append(scanone)
    
    def build(self):
        return ScanMany(self.scanones)
    

class Scanner():
    def __init__(self, scanmany, filename):
        self.scanmany = scanmany
        self.log = npy_append_array.NpyAppendArray(filename, rewrite_header_on_append=True, delete_if_exists=True)
        self.n = 0 # tracks how many scans there have been
    
    def scan(self):
        result = self.scanmany.scan()
        self.log.append(result)
        self.n += 1
        # if self.n == 1:
        #     self.log.close()
        #     self.log = npy_append_array.NpyAppendArray(self.log.filename, rewrite_header_on_append=False) # stop rewriting the header
        self._last_result = result
        return result

    def getlastresult(self):
        return self._last_result

    def read(self):
        return np.load(self.log.filename, mmap_mode="r") # you cannot count on this to get the most recent row!
    
    def close(self):
        self.log.close()
    

if __name__ == "__main__":
    from instruments import cryocon24c_ser
    from instruments import lakeshore370_serial 
    builder = ScanManyBuilder()
    builder.add(ScanOne("time", "float64", time.time))
    cc = cryocon24c_ser.Cryocon24c_ser("cryocon1")
    builder.add(ScanOne("temp1", "float64", callback = cc.getTemperature, callback_args=(1,)))
    builder.add(ScanOne("temp2", "float64", callback = cc.getTemperature, callback_args=(2,)))
    builder.add(ScanOne("temp3", "float64", callback = cc.getTemperature, callback_args=(3,)))
    builder.add(ScanOne("temp4", "float64", callback = cc.getTemperature, callback_args=(4,)))
    ls = lakeshore370_serial.Lakeshore370()
    builder.add(ScanOne("temp5", "float64", callback = ls.getTemperature, callback_args=(1,)))
    builder.add(ScanOne("temp6", "float64", callback = ls.getTemperature, callback_args=(2,)))
    scanmany = builder.build()
    log = Scanner(scanmany, "yotest")

    for i in range(10):
        log.scan()
        print(log.read())

    log.close()
    print(f"{log.read()}")
    






