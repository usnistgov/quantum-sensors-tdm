# combine json files from multiple IV curves

import os
import json

directory = '/home/pcuser/data/uber_omt/20230421/'

# files = ['uber_omt_ColumnC_ivs_20230503_1683154594.json', # 90 to 170 mK in steps of 10
#          'uber_omt_ColumnC_ivs_20230504_1683223467.json', # 90, 100, then 170 to 190
#          'uber_omt_ColumnC_ivs_20230504_1683225935.json', # 200 mK
#         ]

# outfile = 'uber_omt_ColumnC_ivs_20230504_all_temps.json'

files = ['uber_omt_ColumnC_ivs_20230505_1683303799.json', # 300 350 400 mK
         'uber_omt_ColumnC_ivs_20230505_1683305444.json', # 450 and 500 mK
         'uber_omt_ColumnC_ivs_20230505_1683306467.json', # 550 and 600 mK
         'uber_omt_ColumnC_ivs_20230505_1683307714.json', # [0.46,0.47,0.48,0.49,0.51,0.52]
         'uber_omt_ColumnC_ivs_20230505_1683310351.json', # [0.505,0.512,0.515,0.516,0.517,0.518,0.519]
        ]

outfile = 'uber_omt_ColumnC_ivs_20230505_hot_temps.json'

def merge_json_files(files):
    '''
    files : list
        list of the files in directory
    '''
    # We could open the first file and find the keys so we can be key-agnostic
    result = {'set_temps_k':[], 
              'data':[]}
    print('Combining files from {}:'.format(directory))
    for ff in files:
        print(ff)
        with open(os.path.join(directory,ff),'r') as infile:
            thisjson = json.load(infile)
            for kk in thisjson:
                result[kk].extend(thisjson[kk])

    with open(os.path.join(directory,outfile),'w') as ofile:
        json.dump(result,ofile)
    print('Files combined into:',os.path.join(directory,outfile))    

merge_json_files(files)
