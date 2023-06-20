class DetectorMap():
    ''' Map readout channels to detector characteristics '''
    def __init__(self,filename=None):
        ''' main data structure is self.map_dict, in which the keys are the row
            names, such as 'Row00'
        '''
        self.filename = filename
        self.map_dict = self.from_file(self.filename)
        self.keys = self.map_dict.keys()
        self._handle_none_in_dict()


    def from_file(self,filename):
        return self._parse_csv_file(filename)

    def _parse_csv_file(self,filename):
        f=open(filename,'r')
        lines = f.readlines()
        header = lines.pop(0)
        header = header[0:-1].split(',')
        #print(header)
        ncol = len(header)
        map_dict = {}
        for ii, line in enumerate(lines):
            line = line.rstrip().split(',')
            map_dict['Row%02d'%(int(line[0]))] = {}
            for jj in range(1,ncol):

                map_dict['Row%02d'%(int(line[0]))][str(header[jj])] = str(line[jj])
        map_dict = self._clean_map_dict(map_dict)
        return map_dict

    def _clean_map_dict(self,map_dict):
        for key, val in map_dict.items():
            f_low = map_dict[key].pop('f_low',None)
            f_high = map_dict[key].pop('f_high',None)
            f_low = self._handle_input(f_low,float)
            f_high = self._handle_input(f_high,float)
            if f_low == None or f_high == None:
                map_dict[key]['freq_edges_ghz'] = None
            else:
                map_dict[key]['freq_edges_ghz'] = [f_low,f_high]
            map_dict[key]['position'] = self._handle_input(map_dict[key]['position'],int)
        return map_dict

    def _handle_input(self,val,thetype=int):
        if val=='None':
            val = None
        else:
            val = thetype(val)
        return val

    def _handle_none_in_dict(self):
        for row in self.keys:
            for (subkey, val) in self.map_dict[row].items():
                if val == 'None':
                    self.map_dict[row][subkey] = None

    def get_devname_from_row_index(self,row_index):
        return self.map_dict['Row%02d'%row_index]['devname']

    def get_rowdict_from_keyval(self,key,val,dict=None):
        mydict={}
        if dict is not None:
            pass
        else:
            dict = self.map_dict

        for (row, subdict) in dict.items():
            if subdict[key] == val:
                mydict[row]=self.map_dict[row]
        return mydict

    def get_rowdict_from_keyval_list(self,search_list):
        ''' Returns the subset of map_dict that has a series of key,value pairs.
            Note that the order of [key,value] pairs in search_list matters.
            The 0th key,value pair in search_list will be performed first.
        '''
        mydict = self.map_dict
        for (key,val) in search_list:
            mydict = self.get_rowdict_from_keyval(key,val,mydict)
        return mydict

    def get_row_nums_from_keys(self,row_keys):
        mylist = []
        for row in row_keys:
            mylist.append(int(row.split('Row')[1]))
        return mylist

    def get_row_nums_from_keyval_list(self,search_list):
        newDict = self.get_rowdict_from_keyval_list(search_list)
        mylist = self.get_row_nums_from_keys(newDict.keys())
        return mylist

    def print_map(self):
        for key in self.map_dict.keys():
            print(key,":: ",self.map_dict[key])

    def get_mapval_for_row_index(self,row_index):
        return self.map_dict['Row%02d'%row_index]

    # are the methods below useful or obsolete?

    def get_onerow_device_dict(self,row_index):
        return {'Row%02d'%row_index:self.map_dict['Row%02d'%row_index]['devname']}

    def get_keys_and_indices_of_type(self,type_str='dark'):
        print('Warning.  Direct row to index mapping assumed (ie index of Row02 is 2)')
        idx=[]; keys=[]
        for key in self.map_dict.keys():
            if self.map_dict[key]['type']==type_str:
                keys.append(key)
                idx.append(int(key.split('Row')[1]))
        return keys, idx

if __name__ == "__main__":
    #dm = DetectorMap(filename='detector_map_run20210607.csv')
    dm = DetectorMap(filename='detector_map_run20201202.csv')
    #dm._handle_none_in_dict()
    dm.print_map()
    # row_list = dm.get_row_nums_from_keyval_list(search_list=[['position',1],['band','337']])
    # print(row_list)
