class DetectorMap():
    ''' Class to map readout channels to detector characteristics '''
    def __init__(self,filename=None):
        self.filename = filename
        self.map_dict = self.from_file(self.filename)
        self.keys = self.map_dict.keys()

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
            line = line[0:-1].split(',')
            map_dict['Row%02d'%(int(line[0]))] = {}
            for jj in range(1,ncol):
                #print(jj)
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
        else: val = thetype(val)
        return val

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

    def print_map(self):
        for key in self.map_dict.keys():
            print(key,":: ",self.map_dict[key])

if __name__ == "__main__":
    dm = DetectorMap(filename='detector_map_run20210607.csv')
    dm.print_map()
