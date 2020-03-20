'''
Created on Apr 8, 2009

@author: schimaf
'''

class Lookup(dict):
    """
    a dictionary which can lookup value by key, or keys by value
    """
    def __init__(self, items=[]):
        """items can be a list of pair_lists or a dictionary"""
        dict.__init__(self, items)

    def get_key(self, value):
        """find the key(s) as a list given a value"""
        keys = [k for (k,v) in self.items() if v == value]
        if len(keys) == 0:
            raise Exception("found no keys for value {} in lookup: {}".format(value,self))
        return keys
        
    def get_value(self, key):
        """find the value given a key"""
        return self[key]
