import pylab as pl
import numpy as np
import openpyxl as xl
import sys
#import time
import easyClient


client = easyClient.EasyClient()
client.setupAndChooseChannels()
#returns dataOut[col,row,frame,error=1/fb=2]
data = client.getNewData(minimumNumPoints=4096*4, exactNumPoints=True, divideNsamp=True)
row0 = data[0, 0, :, 1]

wb = xl.Workbook()
sheet_plots = wb.active
sheet_plots.title = 'Plots'
wb.create_sheet(index=1, title='Data')
sheet_data = wb.get_sheet_by_name('Data')

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            sheet_data['A' + str(k+1)] = data[i, j, k, 1].tolist()

print 'Size of Sheet'
print sheet_data.dimensions

print 'Data present in cell A1'
print sheet_data['A1'].value

wb.save("/home/pcuser/Documents/script_testing/yolo.xlsx")


