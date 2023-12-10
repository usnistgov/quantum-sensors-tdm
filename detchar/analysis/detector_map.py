''' detector_map.py 
    handle mapping readout channels to detectors and positions on a 4-pixel optical board
'''

class DetectorMap():
    ''' Map readout channels to detector characteristics '''
    def __init__(self,filename=None):
        ''' main data structure is self.map_dict, in which the keys are the row
            names, such as 'Row00'
        '''
        self.filename = filename
        self.map_dict = self.from_file(self.filename)
        self.keys = list(self.map_dict.keys())
        self._handle_none_in_dict()
        self.subkeys = list(self.map_dict[self.keys[0]].keys())


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

    def _convert_row_name_to_integer(self,row_name_list):
        mylist=[]
        for row in row_name_list:
            mylist.append(int(row.split('Row')[-1]))
        return mylist

    # --------------------------------------------------------

    def get_rowdict_from_keyval(self,key,val,dict=None):
        ''' return a dictionary of all rows that satisfy key=val.
            i.e. get_rowdict_from_keyval('position',1) would return a
            dictionary with all rows in position 1
        '''
        mydict={}
        if dict is None:
            dict = self.map_dict

        for (row, subdict) in dict.items():
            if subdict[key] == val:
                mydict[row]=self.map_dict[row]
        return mydict

    def rows_in_position(self,position_int,return_row_integers=True,exclude_squid_channels=True):
        ''' return a list of rows in position_int.  If return_row_integers=True,
            a list of integers is returned (i.e. 0 is returned for 'Row00').  If false,
            a list of strings like "RowXX" is returned.
        '''
        row_list = list(self.get_rowdict_from_keyval('position',position_int).keys())
        if exclude_squid_channels:
            sqdex = []
            for ii, row in enumerate(row_list):
                if self.map_dict[row]['type'] in ['optical','dark','Optical','Dark']:
                    sqdex.append(ii)
            row_list = [row_list[ii] for ii in sqdex] 
            
        if return_row_integers:
            row_list = self._convert_row_name_to_integer(row_list)
        return row_list

    def get_devname_from_row_index(self,row_index):
        return self.map_dict['Row%02d'%row_index]['devname']

    def bands_in_row_list(self,row_list):
        bands = []
        for row in row_list:
            bands.append(self.map_dict['Row%02d'%row]['band'])
        return set(bands)

    def get_rowdict_from_keyval_list(self,search_list):
        ''' Returns the subset of map_dict that has a series of key,value pairs within search_list.
            i.e. search_list =[[key1,val1],[key2,val2],...]
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

    def print_data_for_position(self,position=1):
        ''' method to print the rows and fields of each row for all within a spatial pixel (ie position) ''' 
        mydict = self.get_rowdict_from_keyval('position',position)
        header = 'Row'
        for key in self.subkeys:
            header = header + '\t'+key
        print(header)
        for row in mydict.keys():
            txt=row
            for key in self.subkeys:
                txt=txt+'\t'+str(mydict[row][key])
            print(txt)

    def get_row_from_position_band_pol(self,position,band,pol,fmt='int'):
        mydict = self.get_rowdict_from_keyval_list([['position',position],['band',band],['polarization',pol]])
        row_name = list(mydict.keys())[0]
        if fmt=='name':
            return row_name 
        elif fmt=='int':
            return int(row_name.split('Row')[-1])

    def get_bands_for_position(self,position):
        # get [low,high] band within the pixel
        bands = []
        for row in self.rows_in_position(position):
            bands.append(self.map_dict['Row%02d'%row]['band'])
        bands = list(set([y for y in bands if y != None]))
        bands = sorted([int(i) for i in bands])
        return bands

if __name__ == "__main__":
    #dm = DetectorMap(filename='detector_map_run20210607.csv')
    dm = DetectorMap(filename='detector_map_run20201202.csv')
    #dm._handle_none_in_dict()
    dm.print_map()
    # row_list = dm.get_row_nums_from_keyval_list(search_list=[['position',1],['band','337']])
    # print(row_list)
